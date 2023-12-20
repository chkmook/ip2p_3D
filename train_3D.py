import os
import math
import copy
import argparse
import numpy as np

from PIL import Image
from einops import rearrange

import torch
import torchvision.transforms as T

from ip2p_3D import IP2P3D, seed_everything

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

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str, default='./outputs')

    parser.add_argument('--tgt_prompt', type=str, default='')
    
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--lower_bound", type=float, default=0.02)
    parser.add_argument("--upper_bound", type=float, default=0.98)
    parser.add_argument("--diffusion_steps", type=int, default=20)

    parser.add_argument("--w_size", type=int, default=512)
    parser.add_argument("--h_size", type=int, default=512)

    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=5.)

    parser.add_argument('--ip2p_use_full_precision', action='store_true')

    parser.add_argument('--batch', type=int, default=12)

    parser.add_argument('--consistency_decoder', action='store_true')

    opt = parser.parse_args()
    opt.device = torch.device(f'cuda:{opt.device}')

    print(opt)

    seed_everything(opt.seed)
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)

    opt.prompt_dir = f'prompt_{opt.tgt_prompt}'
    saving_dir = os.path.join(opt.save_path, opt.prompt_dir)
    os.makedirs(saving_dir, exist_ok=True)


    # load data
    images, o_size, new_size = load_infer_data(data_dir = opt.data_path,
                                               batch = opt.batch,
                                               w_size = opt.w_size,
                                               h_size = opt.h_size,
                                               device = opt.device)

    saving_dir_ip2p = os.path.join(saving_dir, 'ip2p')
    # load model
    model = IP2P3D(opt.device, ip2p_use_full_precision=opt.ip2p_use_full_precision)
    print(model.unet)

    # encode prompt
    text_embedding = model.pipe._encode_prompt(
        opt.tgt_prompt, device=opt.device, num_images_per_prompt=opt.batch,
        do_classifier_free_guidance=True, negative_prompt=""
    )
    
    if opt.ip2p_use_full_precision: text_embedding = text_embedding.float()
    
    for name, module in model.unet.named_modules():
        if name.endswith('transformer_blocks'):
            print(name, len(module))
            import pdb; pdb.set_trace()