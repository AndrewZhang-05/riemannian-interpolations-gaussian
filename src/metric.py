"""Riemannian metric on τ=600 SD-latent space.

g(x_t, v) = ‖J_{x_t} s_θ(x_t, τ) v‖²  +  G_ε(x_t) · ‖v‖²
                                       └────────────────┘
                                conformal Gaussian-annulus term

with G_ε(x) = ((‖x‖₂ − μ_ε) / σ_ε)².

The Jacobian–vector product J·v is computed with one of two backends:

- ``"jvp"``  — :func:`torch.func.jvp` under the **math** SDPA backend. Exact, but
  the math backend is the only PyTorch SDPA implementation that supports
  forward-mode AD as of torch 2.5; it is roughly 2–3× slower in forward and
  ~5–6× slower per JVP than Flash SDPA.

- ``"fd"``   — finite-difference J·v ≈ (s(x+εv) − s(x−εv)) / 2ε. Two SD-UNet
  forward passes per JVP under any (Flash) SDPA backend; introduces O(ε²)
  approximation error and is sensitive to ε.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.score import Score_Distillation


@dataclass
class AnnulusStats:
    mu: float
    sigma: float

    @classmethod
    def load(cls, path: str | Path) -> "AnnulusStats":
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(mu=float(d["mu"]), sigma=float(d["sigma"]))


class RiemannianMetric:
    def __init__(
        self,
        score: Score_Distillation,
        annulus: AnnulusStats,
        embed_cond: torch.Tensor,
        jvp_backend: str = "jvp",
        fd_eps: float = 1e-3,
        chunk_size: Optional[int] = None,
        use_annulus: bool = True,
        fd_normalize: bool = False,
    ):
        if jvp_backend not in ("jvp", "fd"):
            raise ValueError(f"jvp_backend must be 'jvp' or 'fd', got {jvp_backend!r}")
        if embed_cond.dim() != 3 or embed_cond.shape[0] != 1:
            raise ValueError(
                f"embed_cond must have shape (1, T, D); got {tuple(embed_cond.shape)}"
            )
        self.score = score
        self.mu = annulus.mu
        self.sigma = annulus.sigma
        self.embed_cond = embed_cond
        self.jvp_backend = jvp_backend
        self.fd_eps = fd_eps
        self.chunk_size = chunk_size
        # When False, drop G_eps and reduce the metric to G = G_{x_t} only —
        # i.e., the Be-Tangential ablation. Score-Jacobian term is unchanged.
        self.use_annulus = use_annulus
        # When True, FD perturbs along v_hat = v/||v|| and rescales by ||v||.
        # Output is mathematically the same J*v, but eps controls the
        # perturbation magnitude in x-space directly, independent of ||v||.
        self.fd_normalize = fd_normalize

    def _score_fn(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embed_cond.expand(x.shape[0], -1, -1)
        return self.score.grad_compute(x, embed)

    def _jvp_chunk(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.jvp_backend == "jvp":
            with sdpa_kernel([SDPBackend.MATH]):
                _, jv = torch.func.jvp(self._score_fn, (x,), (v,))
            return jv
        eps = self.fd_eps
        if self.fd_normalize:
            v_norms = v.flatten(1).norm(p=2, dim=1).view(-1, *[1] * (v.dim() - 1))
            v_unit = v / v_norms.clamp_min(1e-12)
            s_plus = self._score_fn(x + eps * v_unit)
            s_minus = self._score_fn(x - eps * v_unit)
            return (s_plus - s_minus) / (2 * eps) * v_norms
        s_plus = self._score_fn(x + eps * v)
        s_minus = self._score_fn(x - eps * v)
        return (s_plus - s_minus) / (2 * eps)

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """J·v with shape == v. Chunks along leading batch dim if requested."""
        if self.chunk_size is None or x.shape[0] <= self.chunk_size:
            return self._jvp_chunk(x, v)
        out = torch.empty_like(v)
        for i in range(0, x.shape[0], self.chunk_size):
            j = i + self.chunk_size
            out[i:j] = self._jvp_chunk(x[i:j], v[i:j])
        return out

    def annulus_factor(self, x: torch.Tensor) -> torch.Tensor:
        norms = x.flatten(1).norm(p=2, dim=1)
        return ((norms - self.mu) / self.sigma).pow(2)

    def kinetic(self, z_t: torch.Tensor, z_t_dot: torch.Tensor) -> torch.Tensor:
        """Per-step kinetic energy.

        z_t, z_t_dot: ``(B, K, 4, 64, 64)``. Returns ``(B, K)``.
        """
        if z_t.shape != z_t_dot.shape:
            raise ValueError(f"shape mismatch: {z_t.shape} vs {z_t_dot.shape}")
        B, K = z_t.shape[:2]
        feat = z_t.shape[2:]
        x = z_t.reshape(B * K, *feat)
        v = z_t_dot.reshape(B * K, *feat)

        Jv = self.jvp(x, v)
        score_term = Jv.flatten(1).pow(2).sum(dim=1)
        if not self.use_annulus:
            return score_term.view(B, K)
        ann = self.annulus_factor(x)
        eucl = v.flatten(1).pow(2).sum(dim=1)
        return (score_term + ann * eucl).view(B, K)
