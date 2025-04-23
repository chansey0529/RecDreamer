from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 

from orientation.utils import *

def get_test_timesteps(n):
    return torch.linspace(1, 1000 // n * (n - 1) + 1, n).flip(dims=[0]).to(torch.int)

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        gt_grad = torch.nan_to_num(gt_grad)
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None

class DiffSpecifyGradient_coef(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, grad_vsd, coef):
        ctx.save_for_backward(grad_vsd, coef)
        # ctx.save_for_backward(coef)
        return input_tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_usd):
        grad_vsd, coef, = ctx.saved_tensors
        b, c, h, w = grad_usd.shape
        # scaling the usd grad to the vsd scale, in Dreamfusion implementation, the sds/vsd grad is scaled up with factor b * c * h * w compared with implementation of https://github.com/threestudio-project/threestudio
        grad_usd_scaled = grad_usd * b * c * h * w

        # normalize the grad of vsd
        norm_vsd = torch.linalg.norm(grad_vsd)
        norm_usd_scaled = torch.linalg.norm(grad_usd_scaled)
        if norm_usd_scaled > norm_vsd:
            grad_usd_scaled = grad_usd_scaled / norm_usd_scaled * norm_vsd * coef
        grad_total = grad_usd_scaled + grad_vsd
        grad_total = torch.nan_to_num(grad_total)
        batch_size = len(grad_total)
        return grad_total / batch_size, None, None

class DiffSpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, grad_vsd):
        ctx.save_for_backward(grad_vsd)
        return input_tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_usd):
        grad_vsd, = ctx.saved_tensors
        b, c, h, w = grad_usd.shape
        # scaling the usd grad to the vsd scale, in Dreamfusion implementation, the sds/vsd grad is scaled up with factor b * c * h * w compared with implementation of [LINK]
        grad_usd_scaled = grad_usd * b * c * h * w

        # normalize the grad of vsd
        norm_vsd = torch.linalg.norm(grad_vsd)
        norm_usd_scaled = torch.linalg.norm(grad_usd_scaled)
        if norm_usd_scaled > norm_vsd:
            grad_usd_scaled = grad_usd_scaled / norm_usd_scaled * norm_vsd
        # print(norm_vsd, norm_usd_scaled, torch.linalg.norm(grad_usd_scaled))
        grad_total = grad_usd_scaled + grad_vsd
        grad_total = torch.nan_to_num(grad_total)
        batch_size = len(grad_total)
        return grad_total / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True




class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        # if is_xformers_available():
        #     self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")


        self.scheduler_test = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        num_inference_steps=50
        self.scheduler_test.set_timesteps(num_inference_steps, self.device)
        

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, self.device)

        # self.scheduler.to(self.device)
        self.scheduler.betas = self.scheduler.betas.to(device)
        self.scheduler.alphas = self.scheduler.alphas.to(device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def cal_usd_loss(self, hat_imgs, t, training=False): # whether the noisy_latents can be extended to the t0 space
        w = (1 - self.alphas[t])
        coef = torch.sqrt((1 - self.alphas[t]) / self.alphas[t])
        probs = self.classifier.calculate_prob_3d(hat_imgs, debug=False)
        pc_tidx = self.classifier.n_tinterval - t // (1000 // self.classifier.n_tinterval) - 1

        mean_cal = probs / self.classifier.pc_adaptive[pc_tidx]
        
        loss_usd = - torch.log(mean_cal.mean()) * w * coef
        # ema update for mean pc
        if training:
            self.classifier.update_pc(probs, pc_tidx)
        return loss_usd

    def cal_control_loss(self, hat_imgs, t, dirs, training=False): # whether the noisy_latents can be extended to the t0 space
        w = (1 - self.alphas[t])
        coef = torch.sqrt((1 - self.alphas[t]) / self.alphas[t])
        probs = self.classifier.calculate_prob_3d(hat_imgs, debug=False)
        
        loss_usd = - torch.log(torch.gather(probs, dim=1, index=dirs.unsqueeze(1))).mean() * w * coef
        return loss_usd

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, q_unet=None, pose=None, shading=None, grad_clip=None, as_latent=False, t_schedule=[None,None], step_usd=True, dirs=None, norm_coef=None):
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        assert len(t_schedule) == 2
        if as_latent:
            if pred_rgb.shape[2] == 64:
                latents = pred_rgb
            else:
                latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        elif self.opt.latent == True:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        

        # sample t according to the time scheduler
        min_step, max_step = t_schedule
        min_step = min_step if min_step is not None else self.min_step
        max_step = max_step if max_step is not None else self.max_step
        t = torch.randint(min_step, max_step, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if self.opt.sds is False:
                if q_unet is not None:
                    if pose is not None:
                        noise_pred_q = q_unet(latents_noisy, t, c = pose, shading = shading).sample
                    else:
                        noise_pred_q = q_unet(latents_noisy, t, c = pose).sample
                        # raise NotImplementedError()

                    if self.opt.v_pred:
                        sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                        while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                        while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy


        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        if q_unet is None or self.opt.sds:
            grad = w * (noise_pred - noise)
        else:
            grad = w * (noise_pred - noise_pred_q)


        usd_loss = None
        if self.opt.usd:
            if step_usd:
                # calculate coef and weight for grad update
                # w = (1 - self.alphas[t])
                # coef = torch.sqrt((1 - self.alphas[t]) / self.alphas[t])
                if norm_coef is None:
                    latents_usd = DiffSpecifyGradient.apply(latents, grad)
                else:
                    latents_usd = DiffSpecifyGradient_coef.apply(latents, grad, norm_coef)

                usd_noise = torch.randn_like(latents)
                usd_latents_noisy = self.scheduler.add_noise(latents_usd, usd_noise, t)
                usd_latent_model_input = torch.cat([usd_latents_noisy] * 2)
                usd_noise_pred = self.unet(usd_latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance (high scale from paper!)
                usd_noise_pred_uncond, usd_noise_pred_text = usd_noise_pred.chunk(2)
                usd_noise_pred = usd_noise_pred_uncond + guidance_scale * (usd_noise_pred_text - usd_noise_pred_uncond)

                usd_hat_latents = self.scheduler.step(usd_noise_pred, t, usd_latents_noisy).pred_original_sample
                usd_hat_imgs = self.decode_latents(usd_hat_latents, diff=True)

                if self.opt.pose_control and dirs is not None:
                    loss = self.cal_control_loss(usd_hat_imgs, t, dirs, training=True)
                else:
                    loss = self.cal_usd_loss(usd_hat_imgs, t, training=True)
                if torch.any(torch.isnan(loss)):
                    print('nan!!!!!')
                usd_loss = loss
            else:
                with torch.no_grad():
                    # reserve this part for calculating statistics of distribution
                    latents_usd = latents
                    usd_noise = torch.randn_like(latents)
                    usd_latents_noisy = self.scheduler.add_noise(latents_usd, usd_noise, t)
                    usd_latent_model_input = torch.cat([usd_latents_noisy] * 2)
                    usd_noise_pred = self.unet(usd_latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance (high scale from paper!)
                    usd_noise_pred_uncond, usd_noise_pred_text = usd_noise_pred.chunk(2)
                    usd_noise_pred = usd_noise_pred_uncond + guidance_scale * (usd_noise_pred_text - usd_noise_pred_uncond)

                    usd_hat_latents = self.scheduler.step(usd_noise_pred, t, usd_latents_noisy).pred_original_sample
                    usd_hat_imgs = self.decode_latents(usd_hat_latents, diff=True)

                    _loss = self.cal_usd_loss(usd_hat_imgs, t, training=True)
                    usd_loss = torch.zeros([1]).to(self.device)

                loss = SpecifyGradient.apply(latents, grad)
        else:
            loss = SpecifyGradient.apply(latents, grad)
            usd_loss = torch.zeros([1]).to(self.device)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)


        # if grad_clip is not None:
        #     grad_total = grad_total.clamp(-grad_clip, grad_clip)


        # grad_total = torch.nan_to_num(grad_total)
        

        pseudo_loss = torch.mul((w*noise_pred).detach(), latents.detach()).detach().sum()

        return loss, usd_loss, pseudo_loss, latents

    @torch.no_grad()
    def eval_step_usd(self, text_embeddings, pred_rgb, guidance_scale):
        test_timesteps = get_test_timesteps(self.classifier.n_tinterval)
        noisy_probs = []
        noisy_hat_imgs = []
        for i, test_t in enumerate(test_timesteps):
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, test_t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, test_t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            hat_latents = self.scheduler.step(noise_pred, test_t, latents_noisy).pred_original_sample
            hat_imgs = self.decode_latents(hat_latents, diff=True)
            probs = self.classifier.calculate_prob_3d(hat_imgs, debug=False)
            noisy_probs.append(probs)
            noisy_hat_imgs.append(hat_imgs)
        noisy_probs = torch.cat(noisy_probs)
        noisy_hat_imgs = torch.cat(noisy_hat_imgs)
        return noisy_probs, noisy_hat_imgs

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, return_all=False):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        results = []
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                result_dict = self.scheduler.step(noise_pred, t, latents)
                latents = result_dict['prev_sample']

                if return_all:
                    results.append(result_dict['pred_original_sample'])

        if return_all:
            return latents, results
        
        return latents

    def decode_latents(self, latents, diff=False):

        latents = 1 / 0.18215 * latents
        
        if diff:
            imgs = self.vae.decode(latents).sample
        else:
            with torch.no_grad():
                imgs = self.vae.decode(latents).sample

                imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    @torch.no_grad()
    def back_attn(self, imgs, prompts, t, negative_prompts='', num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps)
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        # latents = self.encode_imgs(imgs) # imgs [0, 1]

        t = torch.tensor([t], dtype=torch.long, device=self.device)

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2)
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        with torch.autocast('cuda'):
            _noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def generate_samples(self, path, text, scale, n_samples, gen_batch_size, pose_prompt=True, front=True):
        base_count = 0
        if pose_prompt:
            if front:
                print('Use explicit front prompts.')
                prompt_augment = [', from front view', ', from side view', ', from back view']
            else:
                prompt_augment = ['', ', from side view', ', from back view']
        else:
            prompt_augment = ['']
        num_batch = n_samples // gen_batch_size
        num_per_prompt = num_batch // len(prompt_augment)
        for batch_count in range(num_batch):
            aug_idx = batch_count // num_per_prompt
            imgs = self.prompt_to_img([text + prompt_augment[aug_idx]] * gen_batch_size, [''] * gen_batch_size, guidance_scale=scale)
            for i in range(gen_batch_size):
                img = imgs[i]
                img = Image.fromarray(img.astype(np.uint8))
                img.save(os.path.join(path, f'{base_count:05}.png'))
                base_count += 1



if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
    plt.savefig('temp.png')
