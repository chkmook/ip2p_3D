import os

import math
import time
import json

import imageio
import argparse
import numpy as np

from PIL import Image
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers.optimization import get_scheduler

from dataset import Dataset3D
from IP2P3D import IP2P3D, seed_everything




def load_infer_data(data_dir = '.', batch = 12, sample_frame_rate = 3,
                    index = 0, w_size = None, h_size = None,
                    device = torch.device('cpu')):
    
    img_dir = os.path.join(data_dir, "images")
    image_list = [img for img in os.listdir(img_dir) if img.endswith(".jpg") or img.endswith(".png")]
    image_list.sort()
    images = [os.path.join(img_dir, img) for img in image_list]

    sample_index = [index + i * sample_frame_rate for i in range(batch)]
    sample_index = [i % len(images) for i in sample_index]
        
    frame_paths = [images[i] for i in sample_index]
    frames = [Image.open(frame_path) for frame_path in frame_paths]

    o_width, o_height = frames[0].size

    if h_size is None or w_size is None:
        factor = math.ceil(min(o_width, o_height) / 64) * 64 / min(o_width, o_height)
        width = int((o_width * factor) // 64) * 64
        height = int((o_height * factor) // 64) * 64
    else:
        if h_size%64 != 0 or w_size%64 != 0:
            raise ValueError("h_size and w_size must be multiples of 64")
        width = w_size
        height = h_size

    # resize images in frames
    frames = [frame.resize((width, height), resample=Image.BILINEAR) for frame in frames]
    frames = [torch.from_numpy(np.array(img, dtype="uint8").astype("float32")) / 255.0 for img in frames]

    frames = torch.stack(frames)
    frames = rearrange(frames, "f h w c -> f c h w")

    return frames.to(device), [o_height, o_width], [h_size, w_size]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int ,default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default=None)

    parser.add_argument('--data_path', type=str)
    parser.add_argument("--w_size", type=int, default=512)
    parser.add_argument("--h_size", type=int, default=512)
    parser.add_argument('--save_path', type=str, default='./outputs_training')

    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--sample_frame_rate', type=int, default=3)

    parser.add_argument('--train_prompt', type=str, default='')
    parser.add_argument('--infer_prompt', type=str, default='Give him a red checkered jacket')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)
    
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--lower_bound", type=float, default=0.02)
    parser.add_argument("--upper_bound", type=float, default=0.98)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument('--ip2p_use_full_precision', action='store_true')
    parser.add_argument('--consistency_decoder', action='store_true')

    # Training parameters
    parser.add_argument('--max_train_epochs', type=int, default=100)
    parser.add_argument('--val_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--trainable_modules', nargs='+', type=str, default=['attn1.to_q', 'attn2.to_q', 'attn_temp'])
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
    max_train_steps = len(datset) - opt.sample_frame_rate

    # Scheduler
    lr_scheduler = get_scheduler(opt.lr_scheduler, optimizer = optimizer, num_warmup_steps = opt.lr_warmup_steps,
                                 num_training_steps = opt.max_train_epochs*max_train_steps)

    # load data for inference
    images, o_size, new_size = load_infer_data(data_dir = opt.data_path, batch = opt.batch,
                                               w_size = opt.w_size, h_size = opt.h_size, device = opt.device)

    
    first_epoch = 0
    if opt.resume_ckpt is not None:
        # first_step = 원래 하다 만데
        raise NotImplementedError("나중에")

    for epoch in tqdm(range(first_epoch, opt.max_train_epochs), desc='Epochs'):

        # validation
        if epoch % opt.val_epochs == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                # load data for inference
                images, o_size, new_size = load_infer_data(data_dir = opt.data_path,
                                                           batch = opt.batch,
                                                           w_size = opt.w_size,
                                                           h_size = opt.h_size,
                                                           device = opt.device)
                # encode prompt
                text_embedding = model.pipe._encode_prompt(opt.infer_prompt, device=opt.device,
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
                # resize to original image size (often not necessary)
                if (edited.size()[2:] != o_size):
                    edited = torch.nn.functional.interpolate(edited, size=o_size, mode='bilinear')

                # convert to numpy array and save
                edited = 255.0 * rearrange(edited, "b c h w ->b h w c")
                edited = [Image.fromarray(img.type(torch.uint8).cpu().numpy()) for img in edited]
                
                # make it to gif file
                sample_dir = os.path.join(f"{saving_dir}/samples", f'epoch_{epoch}')
                os.makedirs(sample_dir, exist_ok=True)
                for i, img in enumerate(edited):
                    img.save(os.path.join(sample_dir, f'{str(i).zfill(4)}.png'))
                imageio.mimsave(f'{sample_dir}/output.gif', edited, duration=1)

        # train
        model.train()
        epoch_loss = 0.0
        # encode prompt
        text_embedding = model.pipe._encode_prompt(opt.train_prompt, device=opt.device,
                                                   num_images_per_prompt=opt.batch,
                                                   do_classifier_free_guidance=False)
        if opt.ip2p_use_full_precision: text_embedding = text_embedding.float()

        for step, batch in enumerate(dataloader):
            print(step)
            for name, params in model.named_parameters():
                print(name, params.requires_grad)
            
            images = batch["images"].squeeze(0).to(opt.device)

            # prepare image and image_cond latents
            with torch.no_grad():
                timesteps = torch.randint(0, opt.num_train_timesteps, [1], dtype=torch.long, device=opt.device)

                latents = model.imgs_to_latent(images).detach()
                noise = torch.randn_like(latents)

                noise_input = model.scheduler.add_noise(latents, noise, timesteps)
                noise_input = torch.cat([noise_input, latents], dim=1)

            noise_pred = model.unet(noise_input, timesteps, text_embedding, return_dict=False)[0]
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            # epoch_loss += loss.detach().item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            
            if step >= max_train_steps:
                break

        epoch_loss /= max_train_steps
        print(f"Epoch [{epoch+1}/{opt.max_train_epochs}], Epoch Loss: {epoch_loss}")
    