from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from tqdm import tqdm
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as TT
import numpy as np
import PIL

import tqdm

##################################################################

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def PIL2Tensor(image):
    w, h = image[0].size
    w, h = (x - x % 512 for x in (w, h))  # resize to integer multiple of 8
    image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
    # image = [np.array(i.resize((512, 512)))[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        image = [np.array(i.resize((w, h), resample=PIL.Image.Resampling.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

##################################################################

CONST_SCALE = 0.18215

class IP2P3D(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = opt.device
        model_key = 'timbrooks/instruct-pix2pix'
        self.fp16 = opt.fp16
        self.precision_t = torch.float16 if self.fp16 else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_key, 
                                                                           torch_dtype=self.precision_t, 
                                                                           safety_checker=None)
        self.pipe.to(self.device)
        print(f'[INFO] loading InstructPix2Pix...')
        
        # improve memory performance trading compute time
        self.pipe.enable_attention_slicing()
        
        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        
        if self.fp16:
            self.unet.half()
            self.vae.half()

        self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.scheduler.set_timesteps(100)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore
        
        if opt.consistency_decoder:
            from consistencydecoder import ConsistencyDecoder
            self.decoder_consistency = ConsistencyDecoder(device=opt.device)
        print(f'[INFO] loaded InstructPix2Pix!')

    @torch.no_grad()
    def encode_latents(self, images):
        images = preprocess(images).to(self.device)
        if self.fp16:
            images = images.half()
        posterior = self.vae.encode(images).latent_dist.sample()
        latent = posterior * CONST_SCALE
        return latent
           
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / CONST_SCALE * latents
        if self.opt.consistency_decoder:
            imgs = self.decoder_consistency(latents)
        else:
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


    @torch.no_grad()
    def edit_sequence(self,
                      sequence,
                      tgt_prompt=None,
                      guidance_scale=7.5,
                      image_guidance_scale=1.5,
                      diffusion_steps=100):
        
        # select t, set multi-step diffusion
        T = torch.randint(self.opt.min_step, self.opt.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # set scheduler
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)
        
        batch_size = len(sequence)

        # prepare text embeddings
        tgt_prompt = tgt_prompt if tgt_prompt is not None else self.opt.tgt_prompt
        if isinstance(tgt_prompt, str):
            tgt_prompts = [tgt_prompt] * batch_size
        txt_embed = self.pipe._encode_prompt(
            tgt_prompts, device=self.device, num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )
        src_latents = self.encode_latents(sequence)
    
        uncond_image_latents = torch.zeros_like(src_latents)
        image_cond_latents = torch.cat([src_latents, src_latents, uncond_image_latents], dim=0)

        # add noise
        noise = torch.randn_like(src_latents)
        latents = self.scheduler.add_noise(src_latents, noise, self.scheduler.timesteps[0])
        
        tqdm_bar = tqdm.tqdm(range(diffusion_steps))
        for i, t in enumerate(tqdm_bar):
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=txt_embed).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        with torch.no_grad():
            imgs = self.decode_latents(latents)
            imgs = [TT.ToPILImage()(image.cpu()) for image in imgs]
        return imgs

