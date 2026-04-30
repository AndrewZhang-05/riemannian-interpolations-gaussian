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

from typing import Callable

import torch

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
