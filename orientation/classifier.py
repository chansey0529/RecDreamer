import torch
import os
from orientation.utils import get_features, get_pca_masks, get_pixel_distance
import torchvision

class AddRandomNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise

class OrientationClassifier(torch.nn.Module):
    def __init__(self, stage0_path, cls_config, device):
        super(OrientationClassifier, self).__init__()
        self.load_configs(stage0_path, cls_config, device)
        self.blur = torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(25, 25), sigma=(15.0, 15.0))], p=1)

    def load_configs(self, stage0_path, cls_config, device):
        feature_dict = torch.load(os.path.join(stage0_path, f'params/{cls_config}/cfeat.pt'))
        self.params = torch.load(os.path.join(stage0_path, f'params/{cls_config}/params.pt'))

        feats = []
        masks = []
        nums = []
        cls = []
        
        for d in feature_dict['flist']:
            feats.append(d['cfeat'])
            masks.append(d['cmask'])
            nums.append(d['cnum'])
            cls.append(d['ccls'])
        
        self.cfeatures = torch.cat(feats).to(device)
        self.cmasks = torch.cat(masks).to(device)
        self.ccls = torch.cat(cls).to(device)

        self.n_cluster = len(self.cfeatures)
        self.temperature = feature_dict['temperature']
        
        # load feature extractor
        self.dinov2 = torch.hub.load(os.path.join(os.path.expanduser('~'), '.cache/torch/hub/facebookresearch_dinov2_main/'), feature_dict['extractor'], source='local').to(device)
        self.device = device

    def calculate_prob_2d(self, decoded_imgs, debug=True):
        b = decoded_imgs.shape[0]
        decoded_imgs_blur = self.blur(decoded_imgs) # gaussian blur the input images to avoid overfitting of pixel similarity
        decoded_input = torch.cat([decoded_imgs, decoded_imgs_blur])
        # extract features from the images
        all_decoded_features, all_tokens = get_features(self.dinov2, decoded_input, resize=self.params['infersize'], cls=True)
        decoded_features = all_decoded_features[b:]
        tokens = all_tokens[:b]
        # get the foreground mask
        with torch.no_grad():
            decoded_masks = get_pca_masks(decoded_features, **self.params)
            if decoded_masks.sum() == 0:
                decoded_masks = torch.ones_like(decoded_masks)
        # calculate the cosine similarity of the [cls] token
        tokens_expand = tokens.unsqueeze(1).repeat(1, self.n_cluster, 1).reshape(b*self.n_cluster, -1) # [x1,x1,x1,x1,x2,x2,x2,x2,...]
        ccls_expand = self.ccls.unsqueeze(0).repeat(b,1,1).reshape(b*self.n_cluster, -1)
        cls_sim = torch.nn.functional.cosine_similarity(tokens_expand, ccls_expand).reshape(b, self.n_cluster)
        cls_sim = (cls_sim + 1) / 2
        # calculate the mean pixel distance
        pixel_dis = get_pixel_distance(decoded_features, self.cfeatures, decoded_masks, self.cmasks)
        pixel_sim = 1 - pixel_dis

        # calculte the joint similairty and normalize the probability
        joint_sim = pixel_sim * cls_sim
        joint_sim_norm = joint_sim / joint_sim.sum(1, keepdim=True)
        prob = torch.nn.functional.softmax(joint_sim_norm/self.temperature, dim=1)

        if debug:
            return prob, cls_sim.detach(), pixel_sim.detach(), decoded_masks.detach()
        else:
            return prob

    def calculate_prob_3d(self, decoded_imgs, debug=True):
        b = decoded_imgs.shape[0]
        # extract features from the images
        decoded_features, tokens = get_features(self.dinov2, decoded_imgs, resize=self.params['infersize'], cls=True)
        # get the foreground mask
        with torch.no_grad():
            decoded_masks = get_pca_masks(decoded_features, **self.params)
            if decoded_masks.sum() == 0:
                decoded_masks = torch.ones_like(decoded_masks)
        # calculate the cosine similarity of the [cls] token
        tokens_expand = tokens.unsqueeze(1).repeat(1, self.n_cluster, 1).reshape(b*self.n_cluster, -1) # [x1,x1,x1,x1,x2,x2,x2,x2,...]
        ccls_expand = self.ccls.unsqueeze(0).repeat(b,1,1).reshape(b*self.n_cluster, -1)
        cls_sim = torch.nn.functional.cosine_similarity(tokens_expand, ccls_expand).reshape(b, self.n_cluster)
        cls_sim = (cls_sim + 1) / 2
        # calculate the mean pixel distance
        pixel_dis = get_pixel_distance(decoded_features, self.cfeatures, decoded_masks, self.cmasks)
        pixel_sim = 1 - pixel_dis

        # calculte the joint similairty and normalize the probability
        joint_sim = pixel_sim * cls_sim
        joint_sim_norm = joint_sim / joint_sim.sum(1, keepdim=True)
        prob = torch.nn.functional.softmax(joint_sim_norm/self.temperature, dim=1)

        if debug:
            return prob, cls_sim.detach(), pixel_sim.detach(), decoded_masks.detach()
        else:
            return prob
        

    def calculate_orient(self, decoded_imgs, debug=True):
        b = decoded_imgs.shape[0]
        # extract features from the images
        decoded_features, tokens = get_features(self.dinov2, decoded_imgs, resize=self.params['infersize'], cls=True)
        # get the foreground mask
        with torch.no_grad():
            decoded_masks = get_pca_masks(decoded_features, **self.params)
            if decoded_masks.sum() == 0:
                decoded_masks = torch.ones_like(decoded_masks)

        # calculate the mean pixel distance
        pixel_dis = get_pixel_distance(decoded_features, self.cfeatures, decoded_masks, self.cmasks)
        pixel_sim = 1 - pixel_dis

        # calculte the joint similairty and normalize the probability
        joint_sim = pixel_sim
        joint_sim_norm = joint_sim / joint_sim.sum(1, keepdim=True)
        prob = torch.nn.functional.softmax(joint_sim_norm/self.temperature, dim=1)
        if debug:
            return prob, None, pixel_sim.detach(), decoded_masks.detach()
        else:
            return prob

    def calculate_tex(self, decoded_imgs, debug=True):
        b = decoded_imgs.shape[0]
        # extract features from the images
        decoded_features, tokens = get_features(self.dinov2, decoded_imgs, resize=self.params['infersize'], cls=True)
        # get the foreground mask
        with torch.no_grad():
            decoded_masks = get_pca_masks(decoded_features, **self.params)
            if decoded_masks.sum() == 0:
                decoded_masks = torch.ones_like(decoded_masks)
        # calculate the cosine similarity of the [cls] token
        tokens_expand = tokens.unsqueeze(1).repeat(1, self.n_cluster, 1).reshape(b*self.n_cluster, -1) # [x1,x1,x1,x1,x2,x2,x2,x2,...]
        ccls_expand = self.ccls.unsqueeze(0).repeat(b,1,1).reshape(b*self.n_cluster, -1)
        cls_sim = torch.nn.functional.cosine_similarity(tokens_expand, ccls_expand).reshape(b, self.n_cluster)
        cls_sim = (cls_sim + 1) / 2


        # calculte the joint similairty and normalize the probability
        joint_sim = cls_sim
        joint_sim_norm = joint_sim / joint_sim.sum(1, keepdim=True)
        prob = torch.nn.functional.softmax(joint_sim_norm/self.temperature, dim=1)

        if debug:
            return prob, cls_sim.detach(), None, decoded_masks.detach()
        else:
            return prob

    def init_adaptive_pc(self, n_tinterval, ema_kiters, ema_previous):
        self.n_tinterval = n_tinterval
        self.pc_adaptive = torch.ones([n_tinterval, self.n_cluster]).to(self.device) / self.n_cluster
        ema_kiters = torch.tensor([ema_kiters])
        ema_previous = torch.tensor([ema_previous])
        self.alpha_ema = 1 - torch.exp(torch.log(ema_previous) / ema_kiters).to(self.device)

    def update_pc(self, prob, idx):
        self.pc_adaptive[idx, :] = prob.detach() * self.alpha_ema + self.pc_adaptive[idx, :] * (1 - self.alpha_ema)