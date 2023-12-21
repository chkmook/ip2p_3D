import os

import time
import json

import imageio
import argparse

from PIL import Image
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers.optimization import get_scheduler

from dataset import Dataset3D, load_infer_data
from IP2P3D import IP2P3D, seed_everything


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int ,default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume_pt', type=str, default=None)

    parser.add_argument('--data_path', type=str)
    parser.add_argument("--w_size", type=int, default=512)
    parser.add_argument("--h_size", type=int, default=512)
    parser.add_argument('--save_path', type=str, default='./outputs_training')

    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--sample_frame_rate', type=int, default=3)

    parser.add_argument('--train_prompt', type=str, default='')
    parser.add_argument('--infer_prompt', nargs='+', type=str, default=['Give him a red checkered shirt',
                                                                        'Give him a blue striped shirt'])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)
    
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--lower_bound", type=float, default=0.02)
    parser.add_argument("--upper_bound", type=float, default=0.98)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument('--ip2p_use_full_precision', action='store_true')
    parser.add_argument('--consistency_decoder', action='store_true')

    # Training parameters
    parser.add_argument('--max_train_steps', type=int, default=500)
    parser.add_argument('--val_steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--trainable_modules', nargs='+', type=str, default=['attn1.to_q', 'attn2.to_q', 'attn_temp'])
    parser.add_argument('--trainable_modules', nargs='+', type=str, default=['attn1.to_q', 'attn2.to_q'])
    # AdamW parameters
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    # Scheduler parameters
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--lr_warmup_steps', type=int, default=0)

    opt = parser.parse_args()

    # create saving directory
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)
    opt.time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    saving_dir = os.path.join(opt.save_path, opt.time_now)
    os.makedirs(saving_dir, exist_ok=True)
    # create subdirectories
    os.makedirs(f"{saving_dir}/samples", exist_ok=True)
    os.makedirs(f"{saving_dir}/inv_latents", exist_ok=True)
    os.makedirs(f"{saving_dir}/checkpoints", exist_ok=True)
    with open(f"{saving_dir}/config.json", 'w') as f:
        json.dump(vars(opt), f, indent=4)

    # set device
    opt.device = torch.device(f'cuda:{opt.device}')
    seed_everything(opt.seed)

    # load model
    model = IP2P3D(opt.batch, opt.device, ip2p_use_full_precision=opt.ip2p_use_full_precision, use_temp_attn=True)
    model.requires_grad_(False)
    model.unet.requires_grad_(True)

    # set optimizer parameters
    optim_params = []
    for name, module in model.unet.named_modules():
        if name.endswith(tuple(opt.trainable_modules)):
            for params in module.parameters(): optim_params.append(params)

    # set optimizer
    optimizer = torch.optim.AdamW(optim_params, lr = opt.lr, betas = (opt.adam_beta1, opt.adam_beta2),
                                  weight_decay = opt.adam_weight_decay, eps = opt.adam_epsilon)

    # load dataset
    datset = Dataset3D(opt.data_path, n_sample_frames=opt.batch, sample_frame_rate=opt.sample_frame_rate,
                       w_size = opt.w_size, h_size = opt.h_size)
    dataloader = torch.utils.data.DataLoader(datset, batch_size=1)

    # Scheduler
    lr_scheduler = get_scheduler(opt.lr_scheduler, optimizer = optimizer, num_warmup_steps = opt.lr_warmup_steps,
                                 num_training_steps = opt.max_train_steps)

    first_step = 0
    if opt.resume_pt is not None:
        # first_step = 원래 하다 만데
        raise NotImplementedError("나중에")

    for step in tqdm(range(first_step, opt.max_train_steps), desc='Steps'):

        # validation step
        if step % opt.val_steps == 0:
            images, _, _ = load_infer_data(data_dir = opt.data_path, batch = opt.batch,
                                           w_size = opt.w_size, h_size = opt.h_size,
                                           device = opt.device)
            for prompt_i, infer_prompt in enumerate(opt.infer_prompt):
                with torch.no_grad():
                    # load data for inference
                    # encode prompt
                    text_embedding = model.pipe._encode_prompt(infer_prompt, device=opt.device,
                                                               num_images_per_prompt=opt.batch,
                                                               do_classifier_free_guidance=True)
                    if opt.ip2p_use_full_precision: text_embedding = text_embedding.float()
                    # inference
                    edited = model.edit_sequence(text_embedding, images,
                                                 image_guidance_scale = opt.image_guidance_scale,
                                                 diffusion_steps = opt.diffusion_steps,
                                                 lower_bound = opt.lower_bound,
                                                 upper_bound = opt.upper_bound)
                    f"{saving_dir}/samples"

                    # convert to numpy array and save
                    edited = 255.0 * rearrange(edited, "b c h w ->b h w c")
                    edited = [Image.fromarray(img.type(torch.uint8).cpu().numpy()) for img in edited]
                    
                    # make it to gif file
                    sample_dir = os.path.join(f"{saving_dir}/samples", f'step_{step:04d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    edited[0].save(f'{sample_dir}/{prompt_i}_output.gif', save_all=True, append_images=edited[1:], duration=300, loop=0)

            # save model
            if step > 0:
                torch.save(model.state_dict(), f"{saving_dir}/checkpoints/model_{step:04d}.pt")

        # train step
        model.unet.train()

        batch = next(iter(dataloader))
        frames = batch["images"].squeeze(0).to(opt.device)

        with torch.no_grad():
            timesteps = torch.randint(0, opt.num_train_timesteps, [1], dtype=torch.long, device=opt.device)

            # prepare image and image_cond latents
            latents = model.imgs_to_latent(frames).detach()
            noise = torch.randn_like(latents)
            noise_input = model.scheduler.add_noise(latents, noise, timesteps)
            noise_input = torch.cat([noise_input, latents], dim=1)

            # encode prompt
            text_embedding = model.pipe._encode_prompt(opt.train_prompt, device=opt.device,
                                                    num_images_per_prompt=opt.batch,
                                                    do_classifier_free_guidance=False)
            if opt.ip2p_use_full_precision: text_embedding = text_embedding.float()

        noise_pred = model.unet(noise_input, timesteps, text_embedding, return_dict=False)[0]
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
    
    torch.save(model.state_dict(), f"{saving_dir}/model_final.pt")