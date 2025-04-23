import torch
import glob
import torchvision
import os
import random
import numpy as np

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torch import einsum

# The inplementation is partially referenced to https://github.com/Junyi42/geoaware-sc

# set random seeds
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# deterministic normalization with min and max value
def normalize(X, min_value=None, max_value=None):
    max = 1
    min = 0
    if (min_value is None) or (max_value is None):
        min_value, _min_idx = torch.min(X, dim=0)
        max_value, _max_idx = torch.max(X, dim=0)
    X_std = (X - min_value) / (max_value - min_value)
    X_scaled = X_std * (max - min) + min
    return X_scaled, (min_value, max_value)

# element seperated minmax normalization with in a batch
def batchnorm_minmax(x):
    b = x.shape[0]
    x_temp = x.reshape(b, -1)
    x_min = x_temp.min(1)[0].view(b, *[1 for _ in range(len(x.shape) - 1)])
    x_max = x_temp.max(1)[0].view(b, *[1 for _ in range(len(x.shape) - 1)])
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

# ---------------------------------

# load images from the specified folder
def load_imgs(path, n_files=None, resize=True):
    patch_h=40
    patch_w=40
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading images from", os.path.join(path, '*.png'))
    img_paths_list = glob.glob(os.path.join(path, '*.png'))
    img_paths_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img_paths_list = img_paths_list[:n_files]
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

# get the feature_dict for saving parameters
def get_feature_dict(n_cluster, idx, features_pixel, masks, features_cls, temperature, extractor):
    feature_dict = {
        'flist': [],
        'num': n_cluster,
    }
    for i in range(len(idx)):
        cluster_dict = {
            'cfeat': features_pixel[i:i+1].detach().cpu(),
            'cmask': masks[i:i+1].detach().cpu(),
            'cnum': 1,
            'clabel': torch.tensor(idx[i]),
            'ctopk': torch.tensor(idx[i]),
            'ccls': features_cls[i:i+1],
        }
        feature_dict['flist'].append(cluster_dict)
    feature_dict['num'] = n_cluster
    feature_dict['temperature'] = temperature
    feature_dict['extractor'] = extractor
    return feature_dict

# write logs of probs
def write_logs(path, all_probs, all_cls_sims, all_pixel_sims):
    all_probs = all_probs.detach().cpu().numpy()
    all_cls_sims = all_cls_sims.detach().cpu().numpy()
    all_pixel_sims = all_pixel_sims.detach().cpu().numpy()
    with open(path, 'w') as f:
        f.write('final_probs:\n')
        for i, row in enumerate(all_probs):
            f.write(f'{i:03d} '+' '.join(map(str, row.tolist())) + '\n')
        f.write('all_sims:\n')
        for i, row in enumerate(all_cls_sims):
            f.write(f'{i:03d} '+' '.join(map(str, row.tolist())) + '\n')
        f.write('all_dis:\n')
        for i, row in enumerate(all_pixel_sims):
            f.write(f'{i:03d} '+' '.join(map(str, row.tolist())) + '\n')

# adaptively get the pca paramerters and the foreground mask
def get_pca_params(features, sd_scores, aug_features=None, threshold=0.5, norm_type='minmax'):
    if aug_features is None:
        aug_features = features
    b, c, h, w = features.shape
    b_aug = aug_features.shape[0]
    # Reshape the features to [b*h*w, c] (and features_aug to [b_aug*c*h, c])    
    features = features.permute(0, 2, 3, 1).reshape(b * h * w, -1)
    aug_features = aug_features.permute(0, 2, 3, 1).reshape(b_aug * h * w, -1)
    # Get PCA parameters with the augmented features
    _fg_U, _fg_S, fg_V = torch.pca_lowrank(aug_features, q=10, center=True, niter=10)
    # Extract PCA scores for both features
    pca_aug_features = torch.matmul(aug_features, fg_V[:, :1])
    pca_features = torch.matmul(features, fg_V[:, :1])
    # Construct a dict for return value
    norm_params = {
        'type': norm_type
    }
    if norm_type == "minmax":
        pca_aug_features, (pca_min, pca_max) = normalize(pca_aug_features)
        norm_params['min'] = pca_min.detach().cpu()
        norm_params['max'] = pca_max.detach().cpu()
    elif norm_type == "stdsig":
        pca_mean = pca_aug_features.mean()
        pca_std = pca_aug_features.std()
        pca_features_norm = (pca_aug_features - pca_mean) / pca_std
        pca_aug_features = torch.nn.functional.sigmoid(pca_features_norm)
        norm_params['mean'] = pca_mean.detach().cpu()
        norm_params['std'] = pca_std.detach().cpu()
    # Use masks extracted from the similarity map of diffusion model to classify the foreground and the background
    sd_masks = torch.nn.functional.interpolate(sd_scores, [h, w], mode='bilinear')
    sd_masks_norm = (sd_masks - sd_masks.mean()) / sd_masks.std()
    sd_masks = (sd_masks_norm > 0.)
    features_mask = pca_features.reshape(b, h, w).unsqueeze(1) > 0.5
    if (sd_masks * features_mask).sum() > (sd_masks * ~features_mask).sum(): # sim binary is foreground gt
        neg_fg = False
        pca_features_fg = (pca_features[:, 0] > threshold)
    else:
        neg_fg = True
        pca_features_fg = (pca_features[:, 0] <= (1 - threshold))
    # get foreground for template images
    pca_features_fg = pca_features_fg.reshape(b, h, w).unsqueeze(1)
    params = {
        'pca_matrix': fg_V.detach().cpu(),
        'norm_params': norm_params,
        'neg_fg': neg_fg,
        'thresh': threshold,
        'infersize': h,
    }
    return pca_features_fg, params

# get the attention mask of stable diffusion
def get_attention_masks(sd_model, template_imgs, text):
    # register hook for attention layers of sd
    def hook_attn(module, args, kwargs, output):
        with torch.no_grad():
            x = args[0]
            encoder_hidden_states = kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs.keys() else None
            h = module.heads
            scale = module.scale
            q = module.to_q(x)
            k = module.to_k(encoder_hidden_states)
            kqhs.append((torch.clone(k).detach(), torch.clone(q).detach(), h, scale))
    module_list = [(n, m) for n, m in sd_model.unet.named_modules() if m.__class__.__name__ == 'Attention' if m.cad is not None]
    for _n, m in module_list:
        m.register_forward_hook(hook_attn, with_kwargs=True)

    def cal_sims(k, q, h, scale, prompt, batchsize, init_shape):
        plen = len(prompt.split(' ')) + 1
        with torch.no_grad():
            q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))
            sim = einsum('b i d, b j d -> b i j', q, k) * scale # batch * head, num_pixel, num_text
            sim = sim.softmax(dim=-1)
            sim = sim[:, :, 1:plen]
            sim = rearrange(sim, '(b h) n d -> b h n d', h=h)
            sim = sim.mean(dim=1) # mean the heads -> batch * 2, num_pixel, num_text
            sim = sim[batchsize:] # get the conditoinal sims [unconditional * batchsize, conditional * batchsize] -> [batch, num_pixel, num_text]
            num_pixels = sim.shape[1]
            w = int(num_pixels ** 0.5)
            # sim = (sim - sim.min()) / (sim.max() - sim.min())
            sim = sim.permute(0, 2, 1).reshape(batchsize, -1, w, w)
            sim = torchvision.transforms.functional.resize(sim, init_shape) # [batchsize, n_tokens, init_w, init_h]
            return sim

    all_attn_maps = []
    for i, timg in enumerate(tqdm(template_imgs)):
        kqhs = []
        timg = torch.tensor(timg.unsqueeze(0)).to(sd_model.device)
        sd_model.back_attn(timg, text, 10)
        batch_size=1
        attns = [cal_sims(k, q, h, scale, text, batch_size, (64, 64)) for k, q, h, scale in kqhs]
        attns = torch.stack(attns)
        attns = rearrange(attns, '(n l) b t h w -> n l b t h w', n=1)
        attn_maps = attns.mean(3).squeeze(0)[2:14].mean(0)
        attn_maps = (attn_maps - attn_maps.min()) / (attn_maps.max() - attn_maps.min()) # (1, 1, 64, 64)
        all_attn_maps.append(attn_maps)
    all_attn_maps = torch.stack(all_attn_maps)
    return all_attn_maps

# extract the dino features for all images
def extract_dino_features(imgs, version='dinov2_vits14', resize=None, cls=False):
    bs = 20
    all_features = []
    all_cls = []
    all_imgs = imgs.split(bs)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    model = torch.hub.load(os.path.join(os.path.expanduser('~'), '.cache/torch/hub/facebookresearch_dinov2_main/'), version, source='local').to(device)

    # Get feature from patches
    with torch.no_grad():
        for imgs_batch in all_imgs:
            features_dict = model.forward_features(imgs_batch)
            features = features_dict['x_norm_patchtokens'] # [b, h*w, c]
            all_features.append(features)
            if cls:
                all_cls.append(features_dict['x_norm_clstoken'])

    features = torch.cat(all_features)
    patch_h = 40
    patch_w = 40
    features = rearrange(features, 'b (h w) c -> b c h w', h=patch_h)
    if cls:
        all_cls = torch.cat(all_cls)

    if resize is not None:
        features = torch.nn.functional.interpolate(features, resize, mode='bilinear')

    del model
    torch.cuda.empty_cache()

    if cls:
        return features, all_cls
    return features

# calculate probability for each images
def test_probs(test_imgs, calculate_prob):
    all_probs = []
    all_cls_sims = []
    all_pixel_sims = []
    all_masks = []
    with torch.no_grad():
        for i in range(test_imgs.shape[0]):
            prob, cls_sim, pixel_sim, mask = calculate_prob(test_imgs[i:i+1])
            all_probs.append(prob)
            all_cls_sims.append(cls_sim)
            all_pixel_sims.append(pixel_sim)
            all_masks.append(mask)
    all_probs = torch.cat(all_probs)
    all_cls_sims = torch.cat(all_cls_sims)
    all_pixel_sims = torch.cat(all_pixel_sims)
    all_masks = torch.cat(all_masks)
    return all_probs, all_cls_sims, all_pixel_sims, all_masks

# ---------------------------------

# differential extract dino features
def get_features(model, imgs, resize=None, cls=False):
    patch_h = 40
    imgs = torch.nn.functional.interpolate(imgs, patch_h * 14, mode='bilinear')
    features_dict = model.forward_features(imgs)
    features = features_dict['x_norm_patchtokens'] # [b, h*w, c]
    features = rearrange(features, 'b (h w) c -> b c h w', h=patch_h)
    if resize is not None:
        features = torch.nn.functional.interpolate(features, resize, mode='bilinear')
    if cls:
        cls_token = features_dict['x_norm_clstoken']
        return features, cls_token
    return features

# get the pca masks
def get_pca_masks(features, pca_matrix, norm_params, neg_fg, thresh, **kwargs):
    b, c, h, w = features.shape
    features = features.permute(0, 2, 3, 1).reshape(b * h * w, -1)
    pca_matrix = pca_matrix.to(features.device)
    pca_features = torch.matmul(features, pca_matrix[:, :1])

    norm_type = norm_params['type']

    if norm_type == "minmax":
        pca_features, (_pca_min, _pca_max) = normalize(pca_features, min_value=norm_params['min'], max_value=norm_params['max'])
    elif norm_type == "stdsig":
        pca_features_norm = (pca_features - norm_params['mean']) / norm_params['std']
        pca_features_sig = torch.nn.functional.sigmoid(pca_features_norm)
        pca_features = pca_features_sig

    if neg_fg:
        pca_features_fg = (pca_features[:, 0] <= (1 - thresh))
    else:
        pca_features_fg = (pca_features[:, 0] > thresh)
    
    pca_features_fg = pca_features_fg.reshape(b, h, w).unsqueeze(1)
    return pca_features_fg

# get the color map representing the relevant coordinate system of different images
def get_colormap(input_mask, image_shape):
    b, c, h, w = image_shape
    temp_mask = torch.zeros_like(input_mask).to(torch.float)
    for i in range(b):
        minvalues = torch.nonzero(input_mask[i] > 0)[:, -2:].min(0)[0]
        _min_x, min_y = minvalues[0], minvalues[1]
        maxvalues = torch.nonzero(input_mask[i] > 0)[:, -2:].max(0)[0] # [b, 2]
        _max_x, max_y = maxvalues[0], maxvalues[1] # [b]
        temp_mask[i:i+1,:,:,min_y:max_y+1] = torch.linspace(-0.5, 0.5, max_y + 1 - min_y).reshape(1,1,1,-1).repeat(1,1,h,1)
    return temp_mask

def minmax_color_map(input_mask, image_shape):
    b, c, h, w = image_shape
    minvalues = torch.nonzero(input_mask > 0)[:, 2:].min(0)[0]
    min_x, min_y = minvalues[0], minvalues[1]
    maxvalues = torch.nonzero(input_mask > 0)[:, 2:].max(0)[0] # [b, 2]
    max_x, max_y = maxvalues[0], maxvalues[1] # [b]
    temp_mask = torch.zeros_like(input_mask).to(torch.float)
    temp_mask[:,:,:,min_y:max_y+1] = torch.linspace(-0.5, 0.5, max_y + 1 - min_y).reshape(1,1,1,-1).repeat(1,1,h,1)

    return temp_mask

# calculate the pixel distance given dino features
def get_pixel_distance(f0s, f1, m0s, m1):
    '''
    f0, f1: [b c h w]
    m1, m2: [b 1 h w]
    '''
    b0, c, h, w = f0s.shape
    b, c, h, w = f1.shape

    f0s = f0s.unsqueeze(1).repeat(1, b, 1, 1, 1) # [b0, b, c, h, w]
    m0s = m0s.unsqueeze(1).repeat(1, b, 1, 1, 1) # [b0, b, 1, h, w]

    m1_ori = m1.clone().detach()
    f1 = batchnorm_minmax(f1)
    f1 = rearrange(f1, 'b c h w -> b (h w) c')
    m1 = m1.reshape(b, -1).to(torch.bool) # [b, h*w]
    
    mean_dis = []
    for i in range(b0):
        f0 = f0s[i]
        m0 = m0s[i]

        m0_ori = m0.clone().detach()
        f0 = (f0 - f0.min()) / (f0.max() - f0.min())
        f0 = rearrange(f0, 'b c h w -> b (h w) c')
        m0 = m0.reshape(b, -1).to(torch.bool) # [b, h*w]

        distances = torch.cdist(f1, f0) # [b, h*w, h*w]

        with torch.no_grad():
            rmask0 = minmax_color_map(m0_ori, m0_ori.shape)
            rmask1 = get_colormap(m1_ori, m1_ori.shape)

        scores = []
        T = 0.01
        for i in range(b):
            ri0 = rmask0[i].reshape(h * w)[m0[i]]
            ri1 = rmask1[i].reshape(h * w)[m1[i]]
            di = - distances[i][m1[i]][:, m0[i]] # [1(A), 0(B)]
            di_soft = torch.nn.functional.softmax(di/T, dim=0) # [sum(A) = 1] [0(B)]
            mati1, mati0 = torch.meshgrid(ri1, ri0)
            s = (di_soft * torch.abs(mati1 - mati0)).sum(0)
            scores.append(s.mean())
        mean_dis.append(torch.stack(scores))
    mean_dis = torch.stack(mean_dis)

    return mean_dis