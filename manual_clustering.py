import torch
import os
import argparse
import torchvision
from orientation.utils import load_imgs, seed_everything, get_pca_params, get_attention_masks, extract_dino_features, get_feature_dict, test_probs, write_logs
from nerf.sd import StableDiffusion
from orientation.classifier import AddRandomNoise, OrientationClassifier

import re
import glob
from PIL import Image
def load_cross_imgs(path, resize=True):
    patch_h=40
    patch_w=40
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading images from", os.path.join(path, '*.png'))
    img_paths_list = glob.glob(os.path.join(path, '*.png'))
    img_paths_list.sort(key=lambda x: x.split('/')[-1].split('.')[0])
    # img_paths_list = img_paths_list[-n_cluster:]
    if resize:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(((patch_h * 14, patch_w * 14)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    imgs = [transform(Image.open(path).convert('RGB')) for path in img_paths_list]
    imgs = torch.stack(imgs).to(device)
    return imgs

def count_non_standard_png(directory):
    """
    Count the number of PNG files in a directory that are not named with a 5-digit pattern.
    
    Args:
        directory (str): The path to the directory to check.
    
    Returns:
        int: The count of PNG files not following the 5-digit naming convention.
    """
    if not os.path.exists(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")
    
    # Regex pattern for matching five-digit filenames
    pattern = re.compile(r"^\d{5}\.png$")
    
    non_standard_count = 0

    # Traverse all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png") and not pattern.match(filename):
            non_standard_count += 1
    
    return non_standard_count



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arguments for the classifier
    parser.add_argument("--text", type=str, default=None, help='text prompt')
    parser.add_argument('--clsdir', type=str, default=None, help='input directory of the classifier')
    parser.add_argument('--idx', type=str, default=None, help='index of template images')
    parser.add_argument('--hflip_idx', type=str, default=None, help='index of template images that needs horizontal flip')
    parser.add_argument('--type', type=str, default='pose', help='type of classifier')
    parser.add_argument('--infersize', type=int, default=16, help='inference size of dino features')
    parser.add_argument('--pca_threshold', type=float, default=0.5, help='threshold of pca mask after normalization')
    parser.add_argument('--mask_norm', type=str, default='stdsig', help='type of normalization for the PCA logits when calculating mask')
    parser.add_argument('--extractor', type=str, default='dinov2_vits14_reg', help='backbone of dino extractor')
    parser.add_argument('--temperature', type=float, default=0.01, help='classifier temperatures for softmax layer')
    parser.add_argument('--n_aug', type=int, default=200, help='number of times to augment for the template images')
    parser.add_argument('--note', type=str, default=None, help='additional notes to the workspace')
    parser.add_argument('--test', action='store_true', default=False, help='test all the images under the folder')
    # arguments for initializing stable diffusion
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help='stable diffusion time steps range')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help='stable diffusion version')
    parser.add_argument('--hf_key', type=str, default=None, help='hugging face Stable diffusion model key')

    parser.add_argument('--cross_modal', action='store_true', default=False, help='cross modality matching classifier')
    parser.add_argument('--crossdir', type=str, default=None, help='input directory of the cross modal images')
    parser.add_argument('--seq', type=int, nargs='+', default=None, help='sequence of template images')
    # parser.add_argument('--n_cross', type=int, default=2, help='number of times to augment for the template images')
    parser.add_argument('--scale', type=float, default=7.5, help='cfg scale')
    parser.add_argument('--n_samples', type=int, default=6, help="number of samples to generate")
    parser.add_argument('--gen_batch_size', type=int, default=1, help="batchsize of sampling")

    opt = parser.parse_args()
    
    if not opt.cross_modal:
        opt.idx = [int(num) for num in opt.idx.split(',')]
    else:
        assert opt.crossdir is not None
    
    seed_everything(0)

    # define paths
    workdir = os.path.join(opt.clsdir, opt.note)
    assert os.path.exists(workdir)
    pdir = os.path.join(opt.clsdir, opt.note, 'params', opt.type)
    os.makedirs(pdir, exist_ok=True)
    evaldir = os.path.join(pdir, 'eval')
    os.makedirs(evaldir, exist_ok=True)
    if opt.test:
        testdir = os.path.join(pdir, 'test')
        os.makedirs(testdir, exist_ok=True)

    # initialize stable diffusion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.usd = False # just use the simple mode
    sd_model = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)

    if opt.cross_modal:
        sd_model.generate_samples(opt.crossdir, opt.text, opt.scale, opt.n_samples, opt.gen_batch_size, pose_prompt=True, front=False)

    # define basic variables
    features = []

    if opt.cross_modal:
        n_cluster = count_non_standard_png(opt.crossdir)
    else:
        n_cluster = len(opt.idx)

    if opt.cross_modal:
        imgs = load_cross_imgs(opt.crossdir)
        template_imgs = imgs[-n_cluster:]
        # template_imgs = imgs
    else:
        # load file from folders and select the templates
        imgs = load_imgs(os.path.join(workdir, 'samples'), n_files=None)
        template_imgs = imgs[opt.idx]
        if opt.hflip_idx is not None:
            opt.hflip_idx = [int(num) for num in opt.hflip_idx.split(',')]
            assert template_imgs.shape[0] == len(opt.hflip_idx)
            for i, flag in enumerate(opt.hflip_idx):
                if flag:
                    template_imgs[i] = torch.flip(template_imgs[i], [2])
        
    if opt.seq is not None:
        template_imgs = template_imgs[opt.seq]
    
    # define augmentation
    cls_aug = torchvision.transforms.Compose([
        # torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(5, 15), sigma=(1.0, 5.0))], p=0.5),
        # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        torchvision.transforms.RandomApply([AddRandomNoise(mean=0.0, std=0.2)], p=0.5),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(1, 5)),
        torchvision.transforms.RandomGrayscale(p=0.2),
    ])
    if '3d' in opt.type:
        pixel_aug = torchvision.transforms.Compose([
            # torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 5.0))], p=1), # comandary blur
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # torchvision.transforms.RandomApply([AddRandomNoise(mean=0.0, std=0.2)], p=0.5),
            # torchvision.transforms.RandomGrayscale(p=0.2),
        ])
    else:
        pixel_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 5.0))], p=1), # comandary blur
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # torchvision.transforms.RandomApply([AddRandomNoise(mean=0.0, std=0.2)], p=0.5),
            # torchvision.transforms.RandomGrayscale(p=0.2),
        ])

    # augment for both cls and pixel similarity calculation
    imgs_pixel_aug = [pixel_aug(template_imgs) for i in range(opt.n_aug)]
    imgs_pixel_aug = torch.cat(imgs_pixel_aug)

    # calculate features for transformed images
    with torch.no_grad():
        features_pixel_aug, _features_cls_aug = extract_dino_features(imgs_pixel_aug, opt.extractor, resize=opt.infersize, cls=True)
        features_pixel_aug = features_pixel_aug.reshape(opt.n_aug, n_cluster, -1, opt.infersize, opt.infersize).mean(0)

    # extract original pixel and cls features
    features_pixel, _features_cls = extract_dino_features(template_imgs, opt.extractor, resize=opt.infersize, cls=True)

    all_attn_maps = get_attention_masks(sd_model, template_imgs, opt.text)
    torchvision.utils.save_image(all_attn_maps, os.path.join(evaldir, f'sd_scores.png'))
    
    # calculate cls
    imgs_cls_aug = [cls_aug(template_imgs) for i in range(opt.n_aug)]
    imgs_cls_aug = torch.cat(imgs_cls_aug)
    bg_features_pixel_aug, features_cls_aug = extract_dino_features(imgs_cls_aug, opt.extractor, resize=opt.infersize, cls=True)
    features_cls_aug = features_cls_aug.reshape(opt.n_aug, n_cluster, -1).mean(0)

    # train the PCA for foreground following [LINK HERE]
    masks, params = get_pca_params(features_pixel, all_attn_maps, bg_features_pixel_aug, opt.pca_threshold, opt.mask_norm)

    if opt.cross_modal:
        # opt.idx = [i + opt.n_samples for i in range(n_cluster)]
        opt.idx = [i for i in range(n_cluster)]


    feature_dict = get_feature_dict(n_cluster, opt.idx, features_pixel_aug, masks, features_cls_aug, opt.temperature, opt.extractor)

    # write to files
    torch.save(feature_dict, os.path.join(pdir, 'cfeat.pt'))
    torch.save(params, os.path.join(pdir, 'params.pt'))
    torchvision.utils.save_image(template_imgs, os.path.join(evaldir, f'template_imgs.png'), normalize=True)
    torchvision.utils.save_image(masks.to(torch.float32), os.path.join(evaldir, 'template_masks.png'))

    if opt.test:
        # test all the generated images
        if opt.cross_modal:
            # test_imgs = template_imgs
            test_imgs = imgs
        else:
            test_imgs = load_imgs(os.path.join(workdir, 'samples'), n_files=None)
        classifier = OrientationClassifier(workdir, opt.type, device)
        # calculate probability for each images
        if '3d' in opt.type:
            calculate_prob = classifier.calculate_prob_3d
        else:
            calculate_prob = classifier.calculate_prob_2d
        all_probs, all_cls_sims, all_pixel_sims, all_masks = test_probs(test_imgs, calculate_prob)
        # save image and write to files
        torchvision.utils.save_image(all_masks.to(torch.float), os.path.join(testdir, f'all_masks.png'), nrow=10)
        test_imgs_64 = torch.nn.functional.interpolate(test_imgs, 64)
        torchvision.utils.save_image(test_imgs_64.to(torch.float), os.path.join(testdir, f'all_imgs.png'), nrow=10, normalize=True)
        log_path = os.path.join(testdir, f'all_probs.txt')
        write_logs(log_path, all_probs, all_cls_sims, all_pixel_sims)
       