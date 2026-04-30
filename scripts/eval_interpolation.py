"""Evaluate an interpolation method on a fixed set of CelebA-HQ pairs.

Pipeline:
  cached tau=600 latents  ->  interpolate (method-specific)  ->  DDIM decode
  ->  VAE decode  ->  save K frames per pair to disk.

All baselines that operate in noise space share this decoder so the only
varying factor across methods is the interpolation itself.

Outputs:
    {out_dir}/{method}/pair_index.pt           # (idx0, idx1) tensors + seed
    {out_dir}/{method}/pair_NNNNN/frame_KK.png # decoded 512x512 image

Usage:
    # smoke: 8 pairs, 10 frames
    python scripts/eval_interpolation.py --method lerp --smoke
    # full eval set
    python scripts/eval_interpolation.py --method lerp --num-pairs 5000
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusion_pipeline import load_pipe
from src.interpolators import INTERPOLATORS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(INTERPOLATORS.keys()))
    ap.add_argument("--latent-cache", type=Path,
                    default=REPO_ROOT / "data/celeba_hq/train_celebahq_sd21_tau600.pt")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/eval")
    ap.add_argument("--num-pairs", type=int, default=5000)
    ap.add_argument("--num-frames", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42,
                    help="Eval-pair seed. Same across methods for direct comparison.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau", type=int, default=600)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--pairs-per-batch", type=int, default=4,
                    help="How many pairs to interpolate together per outer iteration.")
    ap.add_argument("--decode-chunk", type=int, default=16,
                    help="Latents per latent_backward call (caps VRAM).")
    ap.add_argument("--smoke", action="store_true",
                    help="Override num_pairs=8 and switch to data/celeba_hq/smoke.pt.")
    args = ap.parse_args()

    if args.smoke:
        args.num_pairs = 8
        smoke_cache = REPO_ROOT / "data/celeba_hq/smoke.pt"
        if smoke_cache.exists():
            args.latent_cache = smoke_cache

    print(f"[load] {args.latent_cache}")
    latents = torch.load(args.latent_cache, map_location="cpu", weights_only=False).float()
    print(f"  latents shape: {tuple(latents.shape)}")
    if args.num_pairs > latents.shape[0]:
        print(f"[note] num_pairs={args.num_pairs} > N={latents.shape[0]}; clamping")
        args.num_pairs = latents.shape[0]

    g = torch.Generator().manual_seed(args.seed)
    idx0 = torch.randint(0, latents.shape[0], (args.num_pairs,), generator=g)
    idx1 = torch.randint(0, latents.shape[0], (args.num_pairs,), generator=g)
    # Resample collisions so every pair has distinct endpoints. Negligible at
    # the full eval scale (5000 pairs over 30k latents) but matters for smoke.
    while (collisions := (idx0 == idx1)).any():
        idx1[collisions] = torch.randint(0, latents.shape[0], (int(collisions.sum()),), generator=g)

    print(f"[load] SD pipeline on {args.device}")
    pipe = load_pipe(args.device, num_inference_steps=args.num_inference_steps)
    pipe.unet.eval()
    embed_empty = pipe.prompt2embed("")

    # noise_level for tau=args.tau (mirrors scripts/cache_celeba_latents.py:74-78).
    step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    K_steps = math.ceil(args.tau / step_ratio) + 1
    noise_level = K_steps / pipe.scheduler.num_inference_steps
    print(f"[cfg] tau={args.tau} -> K_steps={K_steps}, noise_level={noise_level:.4f}")
    print(f"[cfg] method={args.method}, num_pairs={args.num_pairs}, frames={args.num_frames}")

    interp_fn = INTERPOLATORS[args.method]
    method_dir = args.out_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"idx0": idx0, "idx1": idx1, "seed": args.seed,
         "num_frames": args.num_frames, "tau": args.tau,
         "latent_cache": str(args.latent_cache)},
        method_dir / "pair_index.pt",
    )

    t_grid = torch.linspace(0, 1, args.num_frames, device=args.device)

    pbar = tqdm(range(0, args.num_pairs, args.pairs_per_batch), desc=f"{args.method}")
    for batch_start in pbar:
        b = min(args.pairs_per_batch, args.num_pairs - batch_start)
        z0 = latents[idx0[batch_start:batch_start + b]].to(args.device)
        z1 = latents[idx1[batch_start:batch_start + b]].to(args.device)

        with torch.no_grad():
            z_t = interp_fn(z0, z1, t_grid)            # (b, K, 4, 64, 64)
            z_flat = z_t.reshape(-1, 4, 64, 64)        # (b*K, 4, 64, 64)

            decoded_chunks = []
            for s in range(0, z_flat.shape[0], args.decode_chunk):
                chunk = z_flat[s:s + args.decode_chunk]
                embed = embed_empty.repeat(chunk.shape[0], 1, 1)
                clean = pipe.latent_backward(
                    chunk, embed,
                    noise_level=noise_level,
                    guidance_scale=0,
                    show_progress=False,
                )
                decoded_chunks.append(clean)
            clean_latents = torch.cat(decoded_chunks, dim=0)
            images = pipe.latent2img_batch(clean_latents)

        for i in range(b):
            pair_dir = method_dir / f"pair_{batch_start + i:05d}"
            pair_dir.mkdir(exist_ok=True)
            for k in range(args.num_frames):
                images[i * args.num_frames + k].save(pair_dir / f"frame_{k:02d}.png")

    print(f"[done] saved {args.num_pairs} pairs x {args.num_frames} frames to {method_dir}")


if __name__ == "__main__":
    main()
