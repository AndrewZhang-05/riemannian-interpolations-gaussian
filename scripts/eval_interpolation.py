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
from src.interpolators import INTERPOLATORS, LearnedInterpolator
from src.metric import AnnulusStats
from src.path_optimizer import PathOptimizer, annulus_m
from src.score import Score_Distillation


def main() -> None:
    ap = argparse.ArgumentParser()
    method_choices = sorted(INTERPOLATORS.keys()) + ["learned", "path_opt"]
    ap.add_argument("--method", required=True, choices=method_choices,
                    help="Method name. 'learned' loads --checkpoint; 'path_opt' runs per-pair "
                         "geodesic optimization (no training).")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to a trained interpolant .model file (required when --method learned).")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Subdir under --out-dir for outputs. Defaults to --method, or for "
                         "--method learned, the parent dir name of --checkpoint.")
    ap.add_argument("--latent-cache", type=Path,
                    default=REPO_ROOT / "data/celeba_hq/train_celebahq_sd21_tau600.pt")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/eval")
    ap.add_argument("--num-pairs", type=int, default=5000)
    ap.add_argument("--num-frames", type=int, default=11,
                    help="Path points including endpoints. Default 11 = 9 interior + 2 endpoints, "
                         "matching the CelebA-HQ paper protocol.")
    # path_opt-only knobs
    ap.add_argument("--annulus-stats", type=Path,
                    default=REPO_ROOT / "data/celeba_hq/annulus_stats_sd21_tau600.pt",
                    help="Annulus stats for path_opt's m_fn = G_ε.")
    ap.add_argument("--path-iters", type=int, default=500,
                    help="path_opt: Adam iterations per pair-batch.")
    ap.add_argument("--path-lr", type=float, default=1e-3)
    ap.add_argument("--path-lr-min", type=float, default=1e-4)
    ap.add_argument("--lambda-m", type=float, default=1.0,
                    help="path_opt: weight on the annulus term relative to the score-Jacobian term.")
    ap.add_argument("--score-chunk-size", type=int, default=16,
                    help="path_opt: chunk size for the SD UNet score evaluation.")
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

    if args.method == "learned" and args.checkpoint is None:
        ap.error("--checkpoint is required when --method=learned")
    if args.method != "learned" and args.checkpoint is not None:
        print(f"[note] --checkpoint ignored for method={args.method}")

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
    print(f"[cfg] num_pairs={args.num_pairs}, frames={args.num_frames}")

    if args.method == "learned":
        interp_fn = LearnedInterpolator.from_checkpoint(args.checkpoint, args.device)
        out_name = args.out_name or args.checkpoint.parent.name
        print(f"[method] learned from {args.checkpoint} -> out_name={out_name}")
    elif args.method == "path_opt":
        score = Score_Distillation(
            pipe, time_step=args.tau,
            grad_guidance_0=1, grad_guidance_1=1,
            grad_weight_type="uniform", grad_sample_type="ori_step",
        )
        annulus = AnnulusStats.load(args.annulus_stats)
        m_fn = annulus_m(annulus)
        interp_fn = PathOptimizer(
            score_module=score, m_fn=m_fn, embed_cond=embed_empty,
            num_iters=args.path_iters,
            lr=args.path_lr, lr_min=args.path_lr_min,
            lambda_m=args.lambda_m,
            score_chunk_size=args.score_chunk_size,
            progress=False,
        )
        out_name = args.out_name or "path_opt"
        print(f"[method] path_opt: iters={args.path_iters} lr={args.path_lr}->{args.path_lr_min} "
              f"lambda_m={args.lambda_m} score_chunk={args.score_chunk_size} "
              f"μ_ε={annulus.mu:.3f} σ_ε={annulus.sigma:.3f}")
    else:
        interp_fn = INTERPOLATORS[args.method]
        out_name = args.out_name or args.method
    method_dir = args.out_dir / out_name
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
