import os
import torch
import argparse
import PIL
import numpy as np
import glob
import mediapy as media
from ip2p_3D import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_prompt', type=str, default='')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--device', type=int ,default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--min_step", type=int, default=20)
    parser.add_argument("--max_step", type=int, default=500)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--consistency_decoder', action='store_true')
    parser.add_argument("--rows", default=3, type=int)
    parser.add_argument("--cols", default=4, type=int)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)
    
    # prepare datasets
    files = sorted(glob.glob(os.path.join(opt.data_path, "*")))
    images = []
    for file in files[:12]:
        image = PIL.Image.open(file).convert('RGB')
        opt.w, opt.h = image.size
        images.append(image)
        
    opt.device = torch.device(f'cuda:{opt.device}')

    print(opt)

    model = IP2P3D(opt)

    opt.save_name = f'prompt_{opt.tgt_prompt}.png'

    print("Start editing...")
    img = model.edit_sequence(images,
                              opt.tgt_prompt,
                              opt.guidance_scale,
                              opt.image_guidance_scale,
                              opt.diffusion_steps)
    print("Done!")
    
    # save image
    print("Saving image...")
    img = [image.resize((opt.w, opt.h)) for image in img]
    img_grid = image_grid(img, rows=opt.rows, cols=opt.cols)
    img_grid.save(os.path.join(opt.save_path, opt.save_name))
    print("Done!")
    
    # save video
    print("Saving video...")
    vid = [np.array(image)[None, :] for image in img]
    vid = np.concatenate(vid, axis=0).astype(np.float32) / 255.0
    media.write_video(os.path.join(opt.save_path, f'prompt_{opt.tgt_prompt}.mp4'), vid, fps=10)