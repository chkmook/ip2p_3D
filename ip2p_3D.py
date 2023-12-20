# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
import copy
import tqdm

from dataclasses import dataclass
from typing import Union, Any, Dict, Optional

import torch
from torch import Tensor, nn
from jaxtyping import Float

from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from transformers import logging


logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class IP2P3D(nn.Module):
    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        # to make 3D model compatible with 2D model
        for name, module in pipe.unet.named_modules():
            if name.endswith('transformer_blocks'):
                module[0].attn1 = copy.deepcopy(module[0].attn1)
                module[0].temp_attn = copy.deepcopy(module[0].attn1)
                module.forward = lambda hidden, **kwargs: forward_3D(module[0], hidden, **kwargs)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae


    def edit_sequence(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image_cond: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:

        # select t, set multi-step diffusion
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)


        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image_cond)
            image_cond_latents = self.prepare_image_latents(image_cond)


        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore


        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        tqdm_bar = tqdm.tqdm(self.scheduler.timesteps)
        for i, t in enumerate(tqdm_bar):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)
        return decoded_img
    

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        latents = 1 / CONST_SCALE * latents
        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        imgs = 2 * imgs - 1
        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE
        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        imgs = 2 * imgs - 1
        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        return image_latents

    def forward(self):
        raise NotImplementedError








def forward_3D(model, hidden_states, **kwargs) -> torch.FloatTensor:
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if model.use_ada_layer_norm:
        norm_hidden_states = model.norm1(hidden_states, timestep)
    elif model.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = model.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif model.use_layer_norm:
        norm_hidden_states = model.norm1(hidden_states)
    elif model.use_ada_layer_norm_single:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            model.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = model.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Incorrect norm used")

    if model.pos_embed is not None:
        norm_hidden_states = smodelelf.pos_embed(norm_hidden_states)

    # 1. Retrieve lora scale.
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = model.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if model.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if model.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif model.use_ada_layer_norm_single:
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = model.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if model.attn2 is not None:
        if model.use_ada_layer_norm:
            norm_hidden_states = model.norm2(hidden_states, timestep)
        elif model.use_ada_layer_norm_zero or model.use_layer_norm:
            norm_hidden_states = model.norm2(hidden_states)
        elif model.use_ada_layer_norm_single:
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        else:
            raise ValueError("Incorrect norm")

        if model.pos_embed is not None and model.use_ada_layer_norm_single is False:
            norm_hidden_states = model.pos_embed(norm_hidden_states)

        attn_output = model.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    if not model.use_ada_layer_norm_single:
        norm_hidden_states = model.norm3(hidden_states)

    if model.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if model.use_ada_layer_norm_single:
        norm_hidden_states = model.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if model._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(
            model.ff, norm_hidden_states, model._chunk_dim, model._chunk_size, lora_scale=lora_scale
        )
    else:
        ff_output = model.ff(norm_hidden_states, scale=lora_scale)

    if model.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif model.use_ada_layer_norm_single:
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states