"""Calibrate the Gaussian-annulus stats (μ_ε, σ_ε) from cached τ=600 latents.

Loads N random latents from the SD 2.1-base τ=600 cache and saves the mean and
standard deviation of their L2 norms — these define the G_ε term:
    G_ε(x) = ((‖x‖₂ − μ_ε) / σ_ε)²
"""
import argparse
from pathlib import Path

import torch

DEFAULT_LATENT_CACHE = Path("data/celeba_hq/train_celebahq_sd21_tau600.pt")
DEFAULT_OUT = Path("data/celeba_hq/annulus_stats_sd21_tau600.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_LATENT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    latents = torch.load(args.cache, map_location="cpu")
    print(f"Loaded {tuple(latents.shape)} latents from {args.cache}")

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(latents.shape[0], generator=g)[: args.num_samples]
    sample = latents[idx]

    norms = sample.flatten(1).norm(p=2, dim=1)
    mu = norms.mean().item()
    sigma = norms.std().item()
    D = sample.flatten(1).shape[1]

    print(f"  N = {len(sample)}, D = {D}, sqrt(D) = {D ** 0.5:.2f}")
    print(f"  μ_ε = {mu:.4f}")
    print(f"  σ_ε = {sigma:.4f}")
    print(f"  μ_ε / sqrt(D) = {mu / D ** 0.5:.4f}  (≈1 if z_τ is approximately N(0, I))")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mu": mu,
            "sigma": sigma,
            "num_samples": args.num_samples,
            "cache_path": str(args.cache),
            "seed": args.seed,
        },
        args.out,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
