"""Per-pair geodesic optimization in DDIM-inverted noise space.

Initializes a discrete path (B, N+1, ...) via SLERP between endpoints, then
optimizes only the interior points (B, N-1, ...) to minimize the discretized
Riemannian energy

    E_disc = (1/(2 Δu)) * sum_i [ ||s(z_{i+1}) - s(z_i)||^2
                                  + λ_m * m(z_i) * ||z_{i+1} - z_i||^2 ]

where Δu = 1/N and λ_m balances the two terms. The metric is
g(v, v) = ||J_x v||^2 + λ_m m(x) ||v||^2; m enters the energy unsquared so the
discrete loss matches RiemannianMetric.kinetic (which already squares inside
annulus_factor). For the canonical "annulus" choice,
m(x) = G_ε(x) = ((||x|| − μ_ε)/σ_ε)^2.

The 1/(2Δu) prefactor is kept so the loss approximates the continuous energy
E[γ] = (1/2) ∫ g(γ, γ') du and stays invariant under choice of N.

Compared to the amortized-training path (scripts/train_interpolant.py), this
bypasses the SGD loop entirely — each pair is optimized independently with
Adam + cosine LR decay. Pairs are batched together for SD-UNet throughput.
The score-Jacobian term is approximated by a finite difference of scores at
adjacent path points, so no JVP machinery is required.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from tqdm import tqdm

from src.interpolators import slerp
from src.metric import AnnulusStats

# A "scalar field" m(x): a callable that takes (B, C, H, W) and returns (B,).
MFunction = Callable[[torch.Tensor], torch.Tensor]


def annulus_m(stats: AnnulusStats) -> MFunction:
    """m(x) = ((||x|| - mu) / sigma)^2 = G_ε(x). Matches RiemannianMetric.annulus_factor."""
    mu, sigma = stats.mu, stats.sigma

    def m(x: torch.Tensor) -> torch.Tensor:
        return ((x.flatten(1).norm(p=2, dim=1) - mu) / sigma).pow(2)

    return m


def _score_chunked(
    score_module,
    z: torch.Tensor,
    embed_cond: torch.Tensor,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """Apply score_module.grad_compute over z, chunked. Preserves autograd graph."""
    if chunk_size is None or z.shape[0] <= chunk_size:
        embed = embed_cond.expand(z.shape[0], -1, -1)
        return score_module.grad_compute(z, embed)
    out = []
    for i in range(0, z.shape[0], chunk_size):
        zi = z[i:i + chunk_size]
        embed = embed_cond.expand(zi.shape[0], -1, -1)
        out.append(score_module.grad_compute(zi, embed))
    return torch.cat(out, dim=0)


@torch.enable_grad()
def optimize_paths(
    z0: torch.Tensor,
    z1: torch.Tensor,
    score_module,
    m_fn: MFunction,
    embed_cond: torch.Tensor,
    *,
    num_segments: int = 10,
    num_iters: int = 500,
    lr: float = 1e-3,
    lr_min: float = 1e-4,
    lambda_m: float = 1.0,
    score_chunk_size: Optional[int] = None,
    progress: bool = False,
) -> torch.Tensor:
    """Optimize geodesic paths for a batch of B pairs simultaneously.

    z0, z1: (B, C, H, W) — endpoints (frozen).
    lambda_m: weight on the annulus term relative to the score-Jacobian term.
    Returns: (B, N+1, C, H, W) — full optimized path including endpoints.
    """
    B = z0.shape[0]
    N = num_segments
    feat = z0.shape[1:]
    device = z0.device

    # SLERP init for the interior points (endpoints stay z0, z1).
    t_init = torch.linspace(0.0, 1.0, N + 1, device=device)
    with torch.no_grad():
        z_path0 = slerp(z0, z1, t_init)  # (B, N+1, ...)
    z_interior = z_path0[:, 1:-1].clone().detach().requires_grad_(True)  # (B, N-1, ...)

    optimizer = torch.optim.Adam([z_interior], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=lr_min,
    )

    z0_e = z0.unsqueeze(1)
    z1_e = z1.unsqueeze(1)
    # 1/(2Δu) = N/2; preserves the continuous-energy scale under N changes.
    energy_prefactor = N / 2.0

    iterator = range(num_iters)
    if progress:
        iterator = tqdm(iterator, desc="path-opt", leave=False)

    for _ in iterator:
        z_full = torch.cat([z0_e, z_interior, z1_e], dim=1)              # (B, N+1, ...)
        z_flat = z_full.reshape(B * (N + 1), *feat)
        scores = _score_chunked(score_module, z_flat, embed_cond, score_chunk_size)
        scores = scores.view(B, N + 1, *feat)

        delta_s = scores[:, 1:] - scores[:, :-1]                          # (B, N, ...)
        delta_z = z_full[:, 1:] - z_full[:, :-1]                          # (B, N, ...)

        # m enters unsquared; annulus_m already returns G_ε = ((||x||−μ)/σ)^2.
        m_at = m_fn(z_full[:, :-1].reshape(B * N, *feat)).view(B, N)
        L_s = delta_s.flatten(2).pow(2).sum(dim=2)                        # (B, N)
        L_m = m_at * delta_z.flatten(2).pow(2).sum(dim=2)                 # (B, N)
        loss = energy_prefactor * (L_s + lambda_m * L_m).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        z_full = torch.cat([z0_e, z_interior.detach(), z1_e], dim=1)
    return z_full


class PathOptimizer:
    """Eval-time interpolator that runs per-pair geodesic optimization.

    Plugs into scripts/eval_interpolation.py with the same callable signature
    as lerp/slerp/LearnedInterpolator: ``__call__(z0, z1, t_grid)`` -> ``(B, K, ...)``.

    The number of path points K is taken from ``t_grid.shape[0]``; the optimizer
    uses N = K-1 segments and ignores the actual t_grid values (the optimized
    path's natural parameterization is uniform in segment count).
    """

    def __init__(
        self,
        score_module,
        m_fn: MFunction,
        embed_cond: torch.Tensor,
        *,
        num_iters: int = 500,
        lr: float = 1e-3,
        lr_min: float = 1e-4,
        lambda_m: float = 1.0,
        score_chunk_size: Optional[int] = None,
        progress: bool = False,
    ):
        self.score_module = score_module
        self.m_fn = m_fn
        self.embed_cond = embed_cond
        self.num_iters = num_iters
        self.lr = lr
        self.lr_min = lr_min
        self.lambda_m = lambda_m
        self.score_chunk_size = score_chunk_size
        self.progress = progress

    def __call__(
        self, z0: torch.Tensor, z1: torch.Tensor, t_grid: torch.Tensor,
    ) -> torch.Tensor:
        N = t_grid.shape[0] - 1
        return optimize_paths(
            z0, z1,
            self.score_module, self.m_fn, self.embed_cond,
            num_segments=N,
            num_iters=self.num_iters,
            lr=self.lr,
            lr_min=self.lr_min,
            lambda_m=self.lambda_m,
            score_chunk_size=self.score_chunk_size,
            progress=self.progress,
        )
