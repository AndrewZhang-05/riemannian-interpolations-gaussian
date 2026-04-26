"""Lightweight wrapper around HuggingFace diffusers' StableDiffusionPipeline.

Adapted from GeodesicDiffusion/model/{pipeline,utils}.py with seaborn/matplotlib/sklearn
dependencies stripped — we only keep the VAE/DDIM helpers and the deterministic
DDIM-inversion path needed for our experiments.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


# ---------------------------------------------------------------------------
# Image <-> tensor helpers (verbatim from GeodesicDiffusion/model/utils.py).
# ---------------------------------------------------------------------------

def load_image(img: Image.Image, device, resize_dims=(512, 512)):
    img = img.convert("RGB")
    img = img.resize(resize_dims)
    img = 2.0 * np.array(img).astype(np.float32) / 255.0 - 1.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return img


def load_image_batch(imags, device, resize_dims=(512, 512)):
    return torch.cat([load_image(im, device, resize_dims) for im in imags], dim=0)


def output_image(img):
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    return Image.fromarray((255 * img).astype(np.uint8))


def output_image_batch(imgs):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = np.clip((imgs + 1.0) / 2.0, 0.0, 1.0)
    imgs = (255 * imgs).astype(np.uint8)
    return [Image.fromarray(im) for im in imgs]


# ---------------------------------------------------------------------------
# Pipeline (copied from GeodesicDiffusion/model/pipeline.py, simplified).
# ---------------------------------------------------------------------------

class SimpleDiffusionPipeline(StableDiffusionPipeline):
    """DDIM pipeline with deterministic inversion. Supports batched calls."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, image_encoder, requires_safety_checker,
        )
        self.generator = torch.Generator(device=self.device)

    def set_seed(self, seed):
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    # --- VAE encode / decode ---
    def img2latent(self, image: Image.Image):
        img_tensor = load_image(image, self.device)
        return self.vae.config.scaling_factor * self.vae.encode(img_tensor)["latent_dist"].mean

    def img2latent_batch(self, images):
        img_tensor = load_image_batch(images, self.device)
        latents = [self.vae.encode(img_tensor[i:i + 1])["latent_dist"].mean for i in range(img_tensor.shape[0])]
        return self.vae.config.scaling_factor * torch.cat(latents, dim=0)

    def latent2img(self, latent):
        latent = (1 / self.vae.config.scaling_factor) * latent
        return output_image(self.vae.decode(latent)["sample"])

    def latent2img_batch(self, latents):
        latents = (1 / self.vae.config.scaling_factor) * latents
        decoded = [self.vae.decode(latents[i:i + 1])["sample"] for i in range(latents.shape[0])]
        return output_image_batch(torch.cat(decoded, dim=0))

    # --- prompt embedding ---
    def prompt2embed(self, prompt_text: str):
        token = self.tokenizer(
            prompt_text,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).input_ids
        return self.text_encoder(token.to(self.device))[0]

    # --- noise prediction ---
    def noise_pred(self, latent, t, prompt_embed):
        return self.unet(latent, t, encoder_hidden_states=prompt_embed).sample

    def noise_pred_cfg(self, latent, t, prompt_embed, guidance_scale=0.5):
        latent_in = torch.cat([latent] * 2)
        pred = self.unet(latent_in, t, encoder_hidden_states=prompt_embed).sample
        pred_uncond, pred_cond = pred.chunk(2)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    # --- forward / backward / inversion in latent space ---
    def latent_forward(self, latent, noise_level=1, ep0=None):
        t = self.get_t(noise_level, return_single=True)
        if t.item() == 0:
            return latent
        noise = ep0 if ep0 is not None else torch.randn(latent.shape, device=self.device, generator=self.generator)
        return self.scheduler.add_noise(latent, noise, t)

    def latent_backward(self, latent, prompt_embed, noise_level=1, guidance_scale=0, eta=0.0, show_progress=True):
        assert (guidance_scale > 0 and prompt_embed.shape[0] == 2 * latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed.shape[0] == latent.shape[0])
        time_steps = self.get_t(noise_level)
        extra = self.prepare_extra_step_kwargs(self.generator, eta=eta)
        iterator = tqdm(time_steps, leave=False) if show_progress else time_steps
        for t in iterator:
            if guidance_scale > 0:
                noise_pred = self.noise_pred_cfg(latent, t, prompt_embed, guidance_scale)
            else:
                noise_pred = self.noise_pred(latent, t, prompt_embed)
            latent = self.scheduler.step(noise_pred, t, latent, **extra).prev_sample
        return latent

    def latent_forward_inversion(self, latent, prompt_embed, noise_level=1, guidance_scale=0, show_progress=True):
        """Deterministic DDIM inversion: maps a clean latent to its corresponding noise at `noise_level`."""
        assert (guidance_scale > 0 and prompt_embed.shape[0] == 2 * latent.shape[0]) \
            or (guidance_scale == 0 and prompt_embed.shape[0] == latent.shape[0])
        time_steps = list(reversed(self.get_t(noise_level)))
        iterator = tqdm(time_steps, leave=False) if show_progress else time_steps
        for t in iterator:
            t_prev = max(int(t) - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 0)
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev]
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            if guidance_scale > 0:
                eps = self.noise_pred_cfg(latent, t, prompt_embed, guidance_scale=guidance_scale)
            else:
                eps = self.noise_pred(latent, t, prompt_embed)
            x0 = (latent - (1 - alpha_prod_t_prev) ** 0.5 * eps) / (alpha_prod_t_prev ** 0.5)
            latent = alpha_prod_t ** 0.5 * x0 + (1 - alpha_prod_t) ** 0.5 * eps
        return latent

    # --- helpers ---
    def get_t(self, noise_level, return_single=False):
        if noise_level == 0:
            return torch.tensor(0, device=self.device) if return_single else torch.tensor([], device=self.device)
        time_steps = self.scheduler.timesteps
        time_stamp = max(int(len(time_steps) * noise_level), 1)
        t = time_steps[-time_stamp]
        return t if return_single else time_steps[-time_stamp:]


# Stability AI deprecated stabilityai/stable-diffusion-2-1-base in late 2025.
# Manojb/stable-diffusion-2-1-base is a community re-upload whose model card text
# matches Stability's original release verbatim and whose diffusers folder layout is
# correct. We pin to a specific commit so future edits to that repo don't change our weights.
DEFAULT_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEFAULT_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"


def load_pipe(
    device="cuda",
    model_id=DEFAULT_MODEL_ID,
    revision=DEFAULT_REVISION,
    num_inference_steps=50,
):
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", revision=revision)
    pipe = SimpleDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, revision=revision, torch_dtype=torch.float32
    )
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.to(device)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe
