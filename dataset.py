import os
import math
import json

import numpy as np

from einops import rearrange
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset


# camera_res_scale_factor: float = 1.0

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data

def compute_w2c(T):
    rot_matrix = T[:3, :3].t()
    C = T[:3, -1]
    
    trans_matrix = -rot_matrix @ C
    
    w2c = torch.eye(4)
    w2c[:3, :3] = rot_matrix
    w2c[:3, -1] = trans_matrix

    return w2c

class Dataset3D(Dataset):
    def __init__(
            self,
            data_dir: str,
            n_sample_frames: int = 12,
            sample_frame_rate: int = 1,
            w_size: int = None,
            h_size: int = None,
    ):
        self.data_dir = data_dir
        
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        
        img_dir = os.path.join(data_dir, "images")
        image_list = [img for img in os.listdir(img_dir) if img.endswith(".jpg") or img.endswith(".png")]
        image_list.sort()
        self.images = [os.path.join(img_dir, img) for img in image_list]

        self.w_size = w_size
        self.h_size = h_size

        cams = os.path.join(data_dir, "transforms.json")
        self.cams = read_json_file(cams)['frames']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        n_sample_frames = self.n_sample_frames
        
        sample_index = [index + i * np.random.randint(1, self.sample_frame_rate + 1) for i in range(n_sample_frames)]
        sample_index = [i % len(self.images) for i in sample_index]
        
        frame_paths = [self.images[i] for i in sample_index]
        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

        o_width, o_height = frames[0].size

        if self.h_size is None or self.w_size is None:
            factor = math.ceil(min(o_width, o_height) / 64) * 64 / min(o_width, o_height)
            width = int((o_width * factor) // 64) * 64
            height = int((o_height * factor) // 64) * 64
        else:
            if self.h_size%64 != 0 or self.w_size%64 != 0:
                raise ValueError("h_size and w_size must be multiples of 64")
            width = self.w_size
            height = self.h_size

        # resize images in frames
        frames = [frame.resize((width, height), resample=Image.BILINEAR) for frame in frames]
        frames = [torch.from_numpy(np.array(img, dtype="uint8").astype("float32")) / 255.0 for img in frames]
        frames = torch.stack(frames)
        frames = rearrange(frames, "f h w c -> f c h w")

        cam = {}
        
        return {'images': frames, 'cam': cam, 'idx': sample_index}