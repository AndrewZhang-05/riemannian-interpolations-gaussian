"""Latent-space interpolation methods.

Each method takes:
    z0, z1: (B, 4, 64, 64) — endpoint latents at tau=600 (DDIM-inverted)
    t_grid: (K,) in [0, 1]
and returns:
    (B, K, 4, 64, 64) — interpolated latents at tau=600

All methods operate purely in noise space. Decoding back to pixels (DDIM
denoising + VAE) is the caller's job; baselines that operate in noise space
(LERP, SLERP, NoiseDiffusion, Be Tangential, ours) share the same decoder so
the only varying factor across methods is the interpolation itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

Interpolator = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def lerp(z0: torch.Tensor, z1: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    t = t_grid.view(1, -1, 1, 1, 1)
    return (1 - t) * z0.unsqueeze(1) + t * z1.unsqueeze(1)


def slerp(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t_grid: torch.Tensor,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Spherical linear interpolation in the flattened latent space.

    For tau=600 latents on the Gaussian-annulus shell of radius ~sqrt(D), this
    keeps the interpolated norm roughly constant — the canonical fix to LERP's
    "norm collapses to zero at t=0.5" pathology.

    Falls back to LERP when endpoints are near-parallel (|cos theta| above
    `dot_threshold`), where 1/sin(theta) is ill-conditioned.
    """
    z0_flat = z0.flatten(1)
    z1_flat = z1.flatten(1)
    n0 = z0_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n1 = z1_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_theta = (z0_flat * z1_flat).sum(dim=1, keepdim=True) / (n0 * n1)
    cos_theta = cos_theta.clamp(-1.0, 1.0).view(-1, 1, 1, 1, 1)

    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp_min(1e-12)
    t = t_grid.view(1, -1, 1, 1, 1)
    z0e = z0.unsqueeze(1)
    z1e = z1.unsqueeze(1)

    alpha = torch.sin((1.0 - t) * theta) / sin_theta
    beta = torch.sin(t * theta) / sin_theta
    spherical = alpha * z0e + beta * z1e
    linear = (1.0 - t) * z0e + t * z1e

    use_linear = cos_theta.abs() > dot_threshold
    return torch.where(use_linear, linear, spherical)


INTERPOLATORS: dict[str, Interpolator] = {
    "lerp": lerp,
    "slerp": slerp,
}


class LearnedInterpolator:
    """Eval-time wrapper around a trained GeoPathUNet checkpoint.

    Applies the same interpolant formula as training-time
    (scripts/train_interpolant.py:97):

        x_{t,eta} = (1 - t) x_0 + t x_1 + 2 t (1 - t) phi(x_0, x_1, t)

    Used for both the "ours" method (trained with G = G_xt + G_eps) and the
    "be_tangential" ablation (trained with G = G_xt only via --no_annulus).
    The forward pass is identical; only the training metric differs.
    """

    def __init__(self, net: nn.Module, device: str):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str) -> "LearnedInterpolator":
        from geopath_networks.unet import GeoPathUNet

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = GeoPathUNet(**ckpt["args"])
        net.load_state_dict(ckpt["weight"])
        return cls(net, device)

    def __call__(
        self, z0: torch.Tensor, z1: torch.Tensor, t_grid: torch.Tensor
    ) -> torch.Tensor:
        B = z0.shape[0]
        K = t_grid.shape[0]
        feat = z0.shape[1:]
        z0r = z0.unsqueeze(1).expand(B, K, *feat)
        z1r = z1.unsqueeze(1).expand(B, K, *feat)
        t = t_grid.view(1, K, *([1] * len(feat))).expand(B, K, *([1] * len(feat)))
        with torch.no_grad():
            phi = self.net(
                z0r.reshape(-1, *feat),
                z1r.reshape(-1, *feat),
                t.reshape(-1),
            ).view(B, K, *feat)
        return (1.0 - t) * z0r + t * z1r + 2.0 * t * (1.0 - t) * phi
