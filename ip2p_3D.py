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
from einops import rearrange

from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.models.attention import Attention
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
    def __init__(self,
                 batch: int,
                 device: Union[torch.device, str],
                 num_train_timesteps: int = 1000,
                 ip2p_use_full_precision=False,
                 use_temp_attn=False) -> None:
        super().__init__()

        self.batch = batch

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        # improve memory performance
        pipe.enable_attention_slicing()

        # to make 3D model compatible with 2D model
        if use_temp_attn:
            for name, module in pipe.unet.named_modules():
                if name.endswith('transformer_blocks'):
                    module[0].attn_temp = copy.deepcopy(module[0].attn1)
                    nn.init.zeros_(module[0].attn_temp.to_out[0].weight.data)
                    module[0].norm4 = copy.deepcopy(module[0].norm3)

                    module[0].attn1.processor = SpatialTemporalProcessor(4, self.batch)

                    module[0].forward = forward_3D.__get__(module[0], type(module[0]))

        self.pipe = pipe

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


# https://github.com/huggingface/diffusers/blob/ff43dba7eaee6d2055d299cf58183b8d19a35daa/src/diffusers/models/attention_processor.py#L1609
class SpatialTemporalProcessor:
    def __init__(self,
                 slice_size: int,
                 video_length: int,):
        self.video_length = video_length
        self.slice_size = slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # key and value for Spatial Temporal attn
        former_index = torch.arange(self.video_length) - 1
        former_index[0] = 0
        
        key = rearrange(key, "(b f) d c -> b f d c", f=self.video_length)
        key = torch.cat([key[:, [0] * self.video_length], key[:, former_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=self.video_length)
        value = torch.cat([value[:, [0] * self.video_length], value[:, former_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")
        #########################################

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



# https://github.com/huggingface/diffusers/blob/ff43dba7eaee6d2055d299cf58183b8d19a35daa/src/diffusers/models/attention.py#L96
def forward_3D(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
    # Notice that normalization is always applied before the real computation in the following blocks.l

    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.use_layer_norm:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.use_ada_layer_norm_single:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)


    # 1. Retrieve lora scale.
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0


    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    
    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.use_ada_layer_norm_single:
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])


    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero or self.use_layer_norm:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.use_ada_layer_norm_single:
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
    
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states


    # 4. Feed-forward
    if not self.use_ada_layer_norm_single:
        norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.use_ada_layer_norm_single:
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(
            self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
        )
    else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.use_ada_layer_norm_single:
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states


    # 5. temporal attention
    if self.attn_temp is not None:
        hidden_states = rearrange(hidden_states, "f d c -> d f c")

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm4(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero or self.use_layer_norm:
            norm_hidden_states = self.norm4(hidden_states)
        elif self.use_ada_layer_norm_single:
            norm_hidden_states = hidden_states
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn_temp(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states
        hidden_states = rearrange(hidden_states, "d f c -> f d c")

    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states