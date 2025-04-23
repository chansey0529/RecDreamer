import torch
import argparse
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.network_particle import NeRFNetwork


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=50, help="evaluate on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='exp/')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=1e5, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet")
    parser.add_argument('--tet_grid_size', type=int, default=256, help="tet grid size")
    parser.add_argument('--init_ckpt', type=str, default='', help="ckpt to init dmtet")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true', default=True, help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='softplus', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=0.1, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=10, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.5, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='particle', choices=['grid', 'vanilla', 'particle'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adam', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--val_radius', type=float, default=1.8, help="valid camera radius")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_false', default=True, help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--val_theta', type=float, default=60, help="Angle when validating")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[0, 120], help="training camera up-down theta range")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=10, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_lap', type=float, default=0.5, help="loss scale for mesh laplacian")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--tri_res', type=int, default=64, help="resolution of triple plane")
    parser.add_argument('--num_layers', type=int, default=1, help="num layers of MLP decoder")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dims of MLP decoder")
    parser.add_argument('--decoder_act', type=str, default="relu", choices=["relu", "softplus"], help="hidden dims of MLP decoder")
    parser.add_argument('--per_iter', type=int, default=100, help="iters per epoch")

    parser.add_argument('--K', type=int, default=1, help="K unet iters per particle optimization iters")
    parser.add_argument('--K2', type=int, default=1, help="1 unet iters per K2 iters")

    parser.add_argument('--unet_bs', type=int, default=1, help="batch size of unet")
    parser.add_argument('--unet_lr', type=float, default=0.0001, help="learning rate of unet")
    parser.add_argument('--val_size', type=int, default=7, help="size of val set")
    parser.add_argument('--val_nz', type=int, default=5, help="number of z of val set")
    parser.add_argument('--scale', type=float, default=100, help="guidance scale")

    parser.add_argument('--q_iter', type=int, default=0, help="time to start using q")
    parser.add_argument('--q_rate', type=float, default=1, help="strength of H(q) loss")
    parser.add_argument('--latent', type=bool, default=False, help="wheather to render in latent mode")
    parser.add_argument('--q_cond', type=bool, default=True, help="use q with pose condition")
    parser.add_argument('--uncond_p', type=float, default=0.1, help="probability of uncond classfier free guidance")

    parser.add_argument('--v_pred', type=bool, default=True, help="use v prediction")
    parser.add_argument('--n_particles', type=int, default=1, help="num of particles")
    parser.add_argument('--cube', type=bool, default=True, help="use cube marching box")
    parser.add_argument('--no_textureless', type=bool, default=False, help="no using of textureless")
    parser.add_argument('--no_lambertian', type=bool, default=False, help="no using of lambertian")
    parser.add_argument('--iter512', type=int, default=-1, help="the time to change into 512")
    parser.add_argument('--buffer_size', type=int, default=-1, help="the size of replay buffer")
    parser.add_argument('--sphere_mask', type=bool, default=False, help="bound the sigmas in a sphere of radius [bound]")
    parser.add_argument('--pre_noise', type=bool, default=True, help="Add noise to sigma during training")
    parser.add_argument('--desired_resolution', type=int, default=2048, help="resolution of hashgrid")
    parser.add_argument('--mesh_idx', type=int, default=-1, help="saving this mesh")
    parser.add_argument('--flip_sigma', type=bool, default=False, help="flip the sigmas")
    parser.add_argument('--set_ws', type=str, default='', help="")
    parser.add_argument('--upper_clip', type=float, default=-1, help="make upper sigma zeros")
    parser.add_argument('--side_clip', type=float, default=-1, help="make side sigma zeros")
    parser.add_argument('--dynamic_clip', type=bool, default=False, help="clip the gradient")
    parser.add_argument('--p_normal', type=float, default=0, help="probability to use normal shading")
    parser.add_argument('--p_textureless', type=float, default=0, help="probability to use textureless shading")
    parser.add_argument('--normal', type=bool, default=False, help="optimize with normal")
    parser.add_argument('--upper_clip_m', type=float, default=-100, help="make upper sigma zeros in training")
    parser.add_argument('--complex_bg', type=bool, default=False, help="")
    parser.add_argument('--normal_iters', type=int, default=-1, help="warm up iters using only normals")
    parser.add_argument('--t5_iters', type=int, default=5000, help="change tmax to 500 after this")
    parser.add_argument('--lora', type=bool, default=True, help="Use lora as variational score.")
    parser.add_argument('--sds', type=bool, default=False, help="use SDS instead of VSD")
    parser.add_argument('--finetune', type=bool, default=False, help="only finetune texture")
    parser.add_argument('--note', type=str, default='', help="")

    ### usd iters
    parser.add_argument('--simple_dir', action='store_true', help="save a simple directory with note and date")
    parser.add_argument('--name', type=str, default=None, help="name of the templates")
    parser.add_argument('--usd', action='store_true', help="allow uniform score distillation")
    parser.add_argument('--ema_kiters', type=int, default=100, help="number of iters that poccesses 1 - ema_previous proportion of weight")
    parser.add_argument('--ema_previous', type=int, default=0.1, help="proportion of weight previous the nearest ema_kiters")
    parser.add_argument('--n_tinterval', default=10, type=int, help="number of diffusion timestep intervals")
    parser.add_argument('--log_interval', type=int, default=1, help="log the mean probability for each log_interval steps")
    parser.add_argument('--log_particle', action='store_true', help="log the probability for each particles")
    parser.add_argument('--lazy_activation', type=int, default=None, help="lazy activation of usd loss, enable a smooth estimation of probability")
    parser.add_argument('--t_schedule', type=str, default='t5', help="time scheduling")
    parser.add_argument('--cls_ckpt', type=str, default='', help="ckpt to classifier")

    ### extensive functions
    parser.add_argument('--q_prob', nargs='+', type=float, default=None, help='a list of probability')
    parser.add_argument('--pose_control', action='store_true', help="add pose control to the generation")
    parser.add_argument('--norm_coef', default=None, type=float, help="custom norm coefficient")
    parser.add_argument('--norm_iters', default=5000, type=int, help="apply norm coefficient before specific iters")
    
    # zero-stage
    parser.add_argument('--t2i', action='store_true', help="generate rgb samples")
    parser.add_argument('--n_samples', type=int, default=200, help="number of samples to generate")
    parser.add_argument('--gen_batch_size', type=int, default=4, help="batchsize of sampling")
    parser.add_argument('--pose_prompt', action='store_true', help="augment the distribution by adding back view at stage 0")
    parser.add_argument('--front', action='store_true', help="augment the distribution by adding back view at stage 0")
    


    opt = parser.parse_args()

    assert opt.p_normal == 0

    if opt.t2i:
        if len(opt.note) > 0:
            opt.workspace = os.path.join(opt.workspace, opt.note)
        else:
            opt.workspace += str(time.strftime('%Y-%m-%d', time.localtime()))+"-"+str(opt.text).replace(" ", "-")

        assert (opt.n_samples % opt.gen_batch_size) == 0
    else:
        ori_workspace = opt.workspace
        if opt.usd:
            # assert opt.sds is False
            assert opt.cls_ckpt != ''
            opt.workspace += 'usd'
            assert opt.name is not None
            opt.workspace += f'-{opt.name}'
            opt.dir_text = False
        if opt.dmtet:
            # parameters for finetuning
            opt.h = 512
            opt.w = 512
            opt.t_range = [0.02, 0.50]
            # opt.fovy_range = [60, 90]
            opt.fovy_range = [30, 60]

        if opt.albedo:
            opt.albedo_iters = opt.iters
            albedostr = "albedo"
        else:
            albedostr = "shading-"+str(opt.albedo_iters)

        opt.val_nz = opt.n_particles

        opt.workspace += '-' + str(time.strftime('%Y-%m-%d', time.localtime()))+"-"+str(opt.text).replace(" ", "-")
        if opt.latent == True:
            opt.workspace += "-latent"
            opt.H = 64
            opt.W = 64
        opt.workspace += "-scale-"+str(opt.scale) + "-lr-"+str(opt.lr) 
        opt.workspace += "-" + albedostr+"-le-"+str(opt.lambda_entropy)

        if opt.w != 64:
            assert opt.w == opt.h
            opt.workspace += "-render-" +str(opt.w)
        if opt.cube:
            opt.workspace += "-cube"
        if opt.no_textureless:
            opt.workspace += "-no_textless"
        if opt.suppress_face:
            opt.workspace += "-supface"
        if opt.iter512 != -1:
            opt.workspace += "-iter512-"+str(opt.iter512)
        if opt.buffer_size != -1:
            opt.workspace += "-buffsize-"+str(opt.buffer_size)
        if opt.sphere_mask:
            opt.workspace += "-sphere_mask"
        if opt.bound != 1:
            opt.workspace += "-bound-"+str(opt.bound)
        if opt.sd_version != "1.5":
            opt.workspace += "-sd-"+str(opt.sd_version)        
        if opt.lambda_opacity != 0:
            opt.workspace += "-lo-" + str(opt.lambda_opacity)
        if opt.desired_resolution != 2048:
            opt.workspace += "-g-"+str(opt.desired_resolution)  
        if opt.t5_iters != -1:
            opt.workspace +=  "-"+str(opt.t5_iters)
        if opt.sds:
            opt.workspace += "-sds"
        if opt.normal:
            opt.workspace += "-normal"
        if opt.finetune:
            opt.workspace += "-finetune"
        if opt.num_layers != 1:
            opt.workspace += "-nlayers-" + str(opt.num_layers)
        if opt.density_thresh != 0.1:
            opt.workspace += "-dth-" + str(opt.density_thresh)
        opt.workspace += "-tet-"+str(opt.tet_grid_size)
        if opt.lambda_normal != 0:
            opt.workspace += "-lnorm-" + str(opt.lambda_normal)
        if opt.p_textureless != 0:
            opt.workspace += "-ptext-" + str(opt.p_textureless)
        opt.workspace += "-" + opt.note

        if opt.set_ws != "":
            opt.workspace = opt.set_ws
        # use simple dir
        if opt.note != '' and opt.simple_dir:
            folder_name = str(time.strftime('%Y-%m-%d', time.localtime()))+"-"+opt.note
            if opt.seed is not None:
                folder_name = folder_name + f"-seed-{opt.seed}"
            opt.workspace = os.path.join(ori_workspace, folder_name)

    if opt.seed is not None:
        seed_everything(opt.seed)

    # use text-to-image mode to generate samples
    if opt.t2i:
        sample_path = os.path.join(opt.workspace, 'samples')
        os.makedirs(sample_path, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from nerf.sd import StableDiffusion
        assert not (opt.n_samples % opt.gen_batch_size)
        opt.usd = False
        sd_model = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)
        sd_model.generate_samples(sample_path, opt.text, opt.scale, opt.n_samples, opt.gen_batch_size, pose_prompt=opt.pose_prompt, front=opt.front)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt).to(device)
            
        if opt.dmtet and opt.init_ckpt != '':
            if opt.finetune:
                opt.ckpt = opt.init_ckpt
                model.set_idx()
            else:
                state_dict = torch.load(opt.init_ckpt, map_location=device)
                model.load_state_dict(state_dict['model'], strict=False)
                if opt.cuda_ray:
                    model.mean_density = state_dict['mean_density']
                model.set_idx()
                model.init_tet()
        elif opt.init_ckpt != '':
            state_dict = torch.load(opt.init_ckpt, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
        print(model)

        if opt.test:
            guidance = None # no need to load guidance model at test
            from nerf.sd import StableDiffusion

            guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)

            trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
            trainer.model.set_idx(opt.mesh_idx)

            if opt.save_mesh:
                trainer.save_mesh()
            else:
                test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.per_iter).dataloader()
                trainer.test(test_loader, name = "test", idx = opt.mesh_idx, shading = "albedo", write_video=False)    
                trainer.test(test_loader, name = "test", idx = opt.mesh_idx, shading = "textureless", write_video=False)
                trainer.test(test_loader, name = "test", idx = opt.mesh_idx, shading = "normal", write_video=False)
        else:
            ds = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.per_iter, q_prob=opt.q_prob, control=opt.pose_control)
            train_loader = ds.dataloader()

            if opt.optim == 'adan':
                from optimizer import Adan
                # Adan usually requires a larger LR
                optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            else:
                optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr, finetune = opt.finetune), betas=(0.9, 0.99), eps=1e-15)

            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

            if opt.guidance == 'stable-diffusion':
                from nerf.sd import StableDiffusion
                guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)
            elif opt.guidance == 'clip':
                from nerf.clip import CLIP
                guidance = CLIP(device)
            else:
                raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')
            
            torch.cuda.empty_cache()

            trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
            if opt.pose_control:
                ds.n_cluster = trainer.classifier.n_cluster
            trainer.model.set_idx(opt.mesh_idx)
            trainer.test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.train_loader512 = NeRFDataset(opt, device=device, type='train', H=512, W=512, size=opt.per_iter).dataloader()

            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.val_size).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(ds, train_loader, valid_loader, max_epoch)
