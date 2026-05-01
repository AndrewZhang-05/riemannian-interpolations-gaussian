"""Compare path_opt with vs without the annulus term on the same set of pairs.

Mirrors the CelebA-HQ protocol in
ClaudeInstructions/intermediate_optimization_eval.md:
- 100 random endpoint pairs (gender-stratification + LPIPS<0.6 filter TODO)
- 9 interior frames + 2 endpoints per pair  (num_frames=11)
- 500 Adam iters per pair-batch with cosine LR decay 1e-3 -> 1e-4

Two configs run on the same indices, same SD pipeline, same RNG seed:
  lambda1: λ_m = 1.0   full metric  (ours,  ‖J·v‖² + G_ε‖v‖²)
  lambda0: λ_m = 0.0   ablation     (Be Tangential, ‖J·v‖² only)

Reports the four metrics from the eval spec:
  PPL  Perceptual Path Length:     Σ LPIPS(x_t, x_{t+1})  per pair  (lower=smoother)
  PDV  Perceptual Distance Var:    std LPIPS(x_t, x_{t+1}) per pair (lower=more uniform)
  FID  Fréchet Inception Distance: 900 interior frames vs 200 reconstructed endpoints
  RE   Reconstruction Error:       MSE between original endpoint and DDIM round-trip
       (config-independent — endpoints frozen — included as a sanity check)

Outputs:
  {out_dir}/pair_index.pt
  {out_dir}/{lambda1,lambda0}/path_latents.pt           # (P, K, 4, 64, 64)
  {out_dir}/{lambda1,lambda0}/pair_NNNNN/frame_KK.png   # decoded 512x512
  {out_dir}/summary.json
  {out_dir}/comparison_pair_NNNNN.png                   # qualitative side-by-side

Usage:
  # full ablation
  python scripts/compare_annulus_ablation.py --num-pairs 100 --path-iters 500
  # smoke
  python scripts/compare_annulus_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusion_pipeline import load_pipe
from src.metric import AnnulusStats
from src.path_optimizer import PathOptimizer, annulus_m
from src.score import Score_Distillation


# ---------------------------------------------------------------------------
# Pair sampling.
# ---------------------------------------------------------------------------

def sample_pairs(num_latents: int, num_pairs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx0 = torch.randint(0, num_latents, (num_pairs,), generator=g)
    idx1 = torch.randint(0, num_latents, (num_pairs,), generator=g)
    while (collisions := (idx0 == idx1)).any():
        idx1[collisions] = torch.randint(
            0, num_latents, (int(collisions.sum()),), generator=g
        )
    return idx0, idx1


# ---------------------------------------------------------------------------
# Run one (label, lambda_m) configuration end-to-end.
# ---------------------------------------------------------------------------

def run_config(
    label: str,
    lambda_m: float,
    pairs,
    latents,
    pipe,
    score,
    m_fn,
    embed_empty,
    args,
    decode_kwargs,
    out_root: Path,
) -> torch.Tensor:
    optimizer = PathOptimizer(
        score_module=score, m_fn=m_fn, embed_cond=embed_empty,
        num_iters=args.path_iters, lr=args.path_lr, lr_min=args.path_lr_min,
        lambda_m=lambda_m, score_chunk_size=args.score_chunk_size, progress=False,
    )
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = torch.empty(args.num_pairs, args.num_frames, 4, 64, 64,
                            dtype=torch.float32)
    t_grid = torch.linspace(0, 1, args.num_frames, device=args.device)

    idx0, idx1 = pairs
    for batch_start in tqdm(range(0, args.num_pairs, args.pairs_per_batch), desc=label):
        b = min(args.pairs_per_batch, args.num_pairs - batch_start)
        z0 = latents[idx0[batch_start:batch_start + b]].to(args.device)
        z1 = latents[idx1[batch_start:batch_start + b]].to(args.device)

        z_t = optimizer(z0, z1, t_grid)                   # (b, K, 4, 64, 64)
        all_paths[batch_start:batch_start + b] = z_t.detach().float().cpu()

        z_flat = z_t.reshape(-1, 4, 64, 64)
        decoded_chunks = []
        with torch.no_grad():
            for s in range(0, z_flat.shape[0], args.decode_chunk):
                chunk = z_flat[s:s + args.decode_chunk]
                embed = embed_empty.repeat(chunk.shape[0], 1, 1)
                clean = pipe.latent_backward(chunk, embed, **decode_kwargs)
                decoded_chunks.append(clean)
            clean_latents = torch.cat(decoded_chunks, dim=0)
            images = pipe.latent2img_batch(clean_latents)

        for i in range(b):
            pair_dir = out_dir / f"pair_{batch_start + i:05d}"
            pair_dir.mkdir(exist_ok=True)
            for k in range(args.num_frames):
                images[i * args.num_frames + k].save(pair_dir / f"frame_{k:02d}.png")

    torch.save(all_paths, out_dir / "path_latents.pt")
    return all_paths


# ---------------------------------------------------------------------------
# Metric computation: PPL, PDV, FID, RE.
# ---------------------------------------------------------------------------

_TO_TENSOR = transforms.ToTensor()


def _load_pair_frames(pair_dir: Path, num_frames: int) -> torch.Tensor:
    """Load the K decoded frames for one pair as a (K, 3, H, W) tensor in [0, 1]."""
    return torch.stack([
        _TO_TENSOR(Image.open(pair_dir / f"frame_{k:02d}.png").convert("RGB"))
        for k in range(num_frames)
    ])


def compute_ppl_pdv(
    config_dir: Path,
    num_pairs: int,
    num_frames: int,
    device: str,
    lpips_net,
) -> dict:
    """PPL = Σ LPIPS(x_t, x_{t+1}); PDV = std of the same.

    `lpips_net` is the underlying NoTrainLpips network (no aggregation).
    Frames come in [0, 1]; we shift to [-1, 1] before calling.
    """
    ppl, pdv = [], []
    with torch.no_grad():
        for p in tqdm(range(num_pairs), desc="LPIPS"):
            frames = _load_pair_frames(config_dir / f"pair_{p:05d}", num_frames).to(device)
            a, b = frames[:-1], frames[1:]                     # (K-1, 3, H, W) each
            d = lpips_net(a * 2.0 - 1.0, b * 2.0 - 1.0).flatten()  # (K-1,)
            ppl.append(d.sum().item())
            pdv.append(d.std().item() if d.numel() > 1 else 0.0)
    return {
        "ppl_mean": float(torch.tensor(ppl).mean()),
        "ppl_std":  float(torch.tensor(ppl).std()),
        "pdv_mean": float(torch.tensor(pdv).mean()),
        "pdv_std":  float(torch.tensor(pdv).std()),
    }


def compute_fid(
    config_dir: Path,
    num_pairs: int,
    num_frames: int,
    device: str,
    fid_metric,
    chunk: int = 32,
) -> dict:
    """FID(900 interior frames || 200 reconstructed endpoints).

    The 200 endpoints come from frame_00 and frame_{K-1} of each pair, which
    are identical between configs (endpoints are frozen during optimization).
    """
    fid_metric.reset()

    # Real = endpoints (decoded reconstructions of the inverted latents).
    real_buf = []
    for p in tqdm(range(num_pairs), desc="FID/real"):
        pdir = config_dir / f"pair_{p:05d}"
        for k in (0, num_frames - 1):
            real_buf.append(_TO_TENSOR(Image.open(pdir / f"frame_{k:02d}.png").convert("RGB")))
            if len(real_buf) >= chunk:
                fid_metric.update(torch.stack(real_buf).to(device), real=True)
                real_buf.clear()
    if real_buf:
        fid_metric.update(torch.stack(real_buf).to(device), real=True)

    # Fake = interior frames.
    fake_buf = []
    for p in tqdm(range(num_pairs), desc="FID/fake"):
        pdir = config_dir / f"pair_{p:05d}"
        for k in range(1, num_frames - 1):
            fake_buf.append(_TO_TENSOR(Image.open(pdir / f"frame_{k:02d}.png").convert("RGB")))
            if len(fake_buf) >= chunk:
                fid_metric.update(torch.stack(fake_buf).to(device), real=False)
                fake_buf.clear()
    if fake_buf:
        fid_metric.update(torch.stack(fake_buf).to(device), real=False)

    fid = float(fid_metric.compute().item())
    return {"fid": fid}


def compute_reconstruction_error(
    config_dir: Path,
    pairs,
    num_pairs: int,
    num_frames: int,
    img_dir: Path | None,
) -> dict:
    """RE = mean MSE between original CelebA-HQ image and DDIM-roundtrip endpoint.

    Returns NaN if originals are not on disk. Same value across configs
    (endpoints are frozen).
    """
    if img_dir is None or not img_dir.exists():
        return {"re_mean": float("nan"), "re_std": float("nan"), "re_note": f"img_dir {img_dir} not found; skipping"}

    idx0, idx1 = pairs
    tf = transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor(),
    ])
    errs = []
    for p in tqdm(range(num_pairs), desc="RE"):
        pdir = config_dir / f"pair_{p:05d}"
        for k, img_idx in [(0, idx0[p].item()), (num_frames - 1, idx1[p].item())]:
            orig_path = img_dir / f"{img_idx}.jpg"
            if not orig_path.exists():
                continue
            orig = tf(Image.open(orig_path).convert("RGB"))
            recon = _TO_TENSOR(Image.open(pdir / f"frame_{k:02d}.png").convert("RGB"))
            errs.append(((orig - recon) ** 2).mean().item())
    if not errs:
        return {"re_mean": float("nan"), "re_std": float("nan"), "re_note": "no originals matched"}
    t = torch.tensor(errs)
    return {"re_mean": float(t.mean()), "re_std": float(t.std()), "re_n": len(errs)}


# ---------------------------------------------------------------------------
# Qualitative grid: λ=1 above λ=0 for the first few pairs.
# ---------------------------------------------------------------------------

def make_comparison_grid(out_root: Path, num_pairs_to_show: int, num_frames: int,
                         cell_size: int = 128) -> None:
    label_dirs = [("lambda1", out_root / "lambda1"),
                  ("lambda0", out_root / "lambda0")]
    for pair_idx in range(num_pairs_to_show):
        rows = []
        for _, ldir in label_dirs:
            pdir = ldir / f"pair_{pair_idx:05d}"
            tiles = [Image.open(pdir / f"frame_{k:02d}.png").resize((cell_size, cell_size))
                     for k in range(num_frames)]
            row = Image.new("RGB", (cell_size * num_frames, cell_size), "white")
            for k, t in enumerate(tiles):
                row.paste(t, (k * cell_size, 0))
            rows.append(row)
        gap = 6
        grid = Image.new("RGB",
                         (cell_size * num_frames, cell_size * len(rows) + gap * (len(rows) - 1)),
                         "white")
        for r, row in enumerate(rows):
            grid.paste(row, (0, r * (cell_size + gap)))
        grid.save(out_root / f"comparison_pair_{pair_idx:05d}.png")


# ---------------------------------------------------------------------------
# Entry.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", type=Path,
                    default=REPO_ROOT / "data/celeba_hq/train_celebahq_sd21_tau600.pt")
    ap.add_argument("--annulus-stats", type=Path,
                    default=REPO_ROOT / "data/celeba_hq/annulus_stats_sd21_tau600.pt")
    ap.add_argument("--orig-img-dir", type=Path,
                    default=Path("/scratch/azhang/data/celebahq/CelebAMask-HQ/CelebA-HQ-img"),
                    help="Optional. If present, used to compute RE; otherwise RE is NaN.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs/compare_annulus")
    ap.add_argument("--num-pairs", type=int, default=100)
    ap.add_argument("--num-frames", type=int, default=11,
                    help="9 interior + 2 endpoints per the eval spec.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau", type=int, default=600)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--pairs-per-batch", type=int, default=4)
    ap.add_argument("--decode-chunk", type=int, default=16)
    ap.add_argument("--score-chunk-size", type=int, default=16)
    ap.add_argument("--path-iters", type=int, default=500)
    ap.add_argument("--path-lr", type=float, default=1e-3)
    ap.add_argument("--path-lr-min", type=float, default=1e-4)
    ap.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="vgg")
    ap.add_argument("--num-grid-pairs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="8 pairs x 30 iters using smoke.pt cache, for quick validation.")
    ap.add_argument("--skip-metrics", action="store_true",
                    help="Skip post-hoc metric computation (just produce frames).")
    args = ap.parse_args()

    if args.smoke:
        args.num_pairs = 8
        args.path_iters = 30
        smoke_cache = REPO_ROOT / "data/celeba_hq/smoke.pt"
        if smoke_cache.exists():
            args.latent_cache = smoke_cache

    print(f"[load] {args.latent_cache}")
    latents = torch.load(args.latent_cache, map_location="cpu", weights_only=False).float()
    print(f"  shape: {tuple(latents.shape)}")
    if args.num_pairs > latents.shape[0]:
        print(f"[note] num_pairs={args.num_pairs} > N={latents.shape[0]}; clamping")
        args.num_pairs = latents.shape[0]

    idx0, idx1 = sample_pairs(latents.shape[0], args.num_pairs, args.seed)
    pairs = (idx0, idx1)

    print(f"[load] SD pipeline on {args.device}")
    pipe = load_pipe(args.device, num_inference_steps=args.num_inference_steps)
    pipe.unet.eval()
    embed_empty = pipe.prompt2embed("")

    step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    K_steps = math.ceil(args.tau / step_ratio) + 1
    noise_level = K_steps / pipe.scheduler.num_inference_steps
    decode_kwargs = dict(noise_level=noise_level, guidance_scale=0, show_progress=False)
    print(f"[cfg] tau={args.tau} -> K_steps={K_steps}, noise_level={noise_level:.4f}")
    print(f"[cfg] num_pairs={args.num_pairs} frames={args.num_frames} iters={args.path_iters}")

    score = Score_Distillation(
        pipe, time_step=args.tau,
        grad_guidance_0=1, grad_guidance_1=1,
        grad_weight_type="uniform", grad_sample_type="ori_step",
    )
    annulus = AnnulusStats.load(args.annulus_stats)
    m_fn = annulus_m(annulus)
    print(f"[metric] μ_ε={annulus.mu:.4f} σ_ε={annulus.sigma:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"idx0": idx0, "idx1": idx1, "seed": args.seed,
                "num_frames": args.num_frames, "tau": args.tau,
                "latent_cache": str(args.latent_cache)},
               args.out_dir / "pair_index.pt")

    # ---- Run both configs ----
    for label, lambda_m in [("lambda1", 1.0), ("lambda0", 0.0)]:
        print(f"\n=== {label}: λ_m = {lambda_m} ===")
        run_config(label, lambda_m, pairs, latents, pipe, score, m_fn,
                   embed_empty, args, decode_kwargs, args.out_dir)

    # Free the SD pipeline before loading inception/lpips, in case GPU is tight.
    del pipe, score
    torch.cuda.empty_cache()

    # ---- Metrics ----
    metrics = {}
    if not args.skip_metrics:
        print("\n[metrics] loading LPIPS + FID networks...")
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.image.fid import FrechetInceptionDistance

        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=args.lpips_net, normalize=False
        ).to(args.device)
        lpips_net = lpips_metric.net  # underlying NoTrainLpips, returns per-pair distances

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(args.device)

        for label in ("lambda1", "lambda0"):
            print(f"\n--- metrics: {label} ---")
            cfg_dir = args.out_dir / label
            ppl_pdv = compute_ppl_pdv(cfg_dir, args.num_pairs, args.num_frames,
                                       args.device, lpips_net)
            fid_metric.reset()
            fid = compute_fid(cfg_dir, args.num_pairs, args.num_frames,
                              args.device, fid_metric)
            re = compute_reconstruction_error(cfg_dir, pairs, args.num_pairs,
                                              args.num_frames, args.orig_img_dir)
            entry = {"lambda_m": 1.0 if label == "lambda1" else 0.0,
                     **ppl_pdv, **fid, **re}
            metrics[label] = entry
            for k, v in entry.items():
                print(f"  {k}: {v}")
    else:
        print("[metrics] skipped (--skip-metrics)")

    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "annulus": {"mu": annulus.mu, "sigma": annulus.sigma},
        "metrics": metrics,
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary -> {args.out_dir / 'summary.json'}")

    n_grid = min(args.num_grid_pairs, args.num_pairs)
    if n_grid > 0:
        print(f"[viz] generating {n_grid} comparison grids...")
        make_comparison_grid(args.out_dir, n_grid, args.num_frames)
    print(f"[done] outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
