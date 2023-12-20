import os
import math
import json

import numpy as np

from einops import rearrange
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset

import copy


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
    ):
        self.data_dir = data_dir
        
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate

        self.sample_N = 10
        
        img_dir = os.path.join(data_dir, "images")
        image_list = [img for img in os.listdir(img_dir) if img.endswith(".jpg") or img.endswith(".png")]
        image_list.sort()
        self.images = [os.path.join(img_dir, img) for img in image_list]

        cams = os.path.join(data_dir, "transforms.json")
        self.cams = read_json_file(cams)['frames']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # sample_frame_rate = np.random.randint(1, self.sample_frame_rate + 1)
        sample_frame_rate = self.sample_frame_rate
        n_sample_frames = self.n_sample_frames
        
        sample_index = [index + i * sample_frame_rate for i in range(n_sample_frames)]
        sample_index = [i % self.__len__ for i in sample_index]
        
        frame_paths = [self.images[i] for i in sample_index]
        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

        o_width, o_height = frames[0].size
        factor = math.ceil(min(o_width, o_height) / 64) * 64 / min(o_width, o_height)

        width = int((o_width * factor) // 64) * 64
        height = int((o_height * factor) // 64) * 64

        # resize images in frames
        frames = [ImageOps.fit(frame, (width, height)) for frame in frames]
        frames = [torch.tensor(np.array(img)).float() for img in frames]
        frames = torch.stack(frames)
        frames = rearrange(frames, "f h w c -> f c h w")

        data = 2*frames / 255 - 1

        return data