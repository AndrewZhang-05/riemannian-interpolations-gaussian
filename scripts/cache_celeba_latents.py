"""Cache DDIM-inverted CelebA-HQ latents at τ=600.

Reads CelebA-HQ images from disk, encodes through the SD 2.1-base VAE, runs deterministic
DDIM inversion with the empty prompt and guidance_scale=0 up to τ, and saves the resulting
(N, 4, 64, 64) latents as a single tensor.

Usage (single GPU, full 30k):
    python scripts/cache_celeba_latents.py
Smoke test (16 images):
    python scripts/cache_celeba_latents.py --num-images 16 --out data/celeba_hq/smoke.pt
Manual sharding across 4 GPUs (one shell each):
    CUDA_VISIBLE_DEVICES=0 python scripts/cache_celeba_latents.py --start-index 0     --num-images 7500 --out data/celeba_hq/shard_0.pt
    CUDA_VISIBLE_DEVICES=1 python scripts/cache_celeba_latents.py --start-index 7500  --num-images 7500 --out data/celeba_hq/shard_1.pt
    ...
"""
import argparse
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diffusion_pipeline import load_pipe

DEFAULT_IMG_DIR = Path("/scratch/azhang/data/celebahq/CelebAMask-HQ/CelebA-HQ-img")
DEFAULT_OUT_PATH = Path("data/celeba_hq/train_celebahq_sd21_tau600.pt")


class CelebaHQImages(Dataset):
    def __init__(self, img_dir: Path, indices, image_size=512):
        self.img_dir = img_dir
        self.indices = list(indices)
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.img_dir / f"{idx}.jpg").convert("RGB")
        return idx, self.tf(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    ap.add_argument("--num-images", type=int, default=30000)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau", type=int, default=600)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    args = ap.parse_args()

    pipe = load_pipe(args.device, num_inference_steps=args.num_inference_steps)
    embed_empty = pipe.prompt2embed("")

    # Choose noise_level so the deepest training timestep visited during inversion is >= args.tau.
    # Scheduler timesteps descend as [(N-1)*r, ..., r, 0] with r = num_train // num_inference;
    # latent_forward_inversion iterates the last K = int(N * noise_level) timesteps reversed,
    # so deepest timestep visited = r * (K - 1).
    step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    K = math.ceil(args.tau / step_ratio) + 1
    noise_level = K / pipe.scheduler.num_inference_steps
    deepest_t = step_ratio * (K - 1)
    print(f"τ requested = {args.tau} → K = {K} inversion steps, noise_level = {noise_level:.4f}, deepest τ visited = {deepest_t}")

    indices = list(range(args.start_index, args.start_index + args.num_images))
    dataset = CelebaHQImages(args.img_dir, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    out = torch.empty((args.num_images, 4, 64, 64), dtype=torch.float32)
    out_idx = 0
    for _, imgs in tqdm(loader, desc=f"DDIM-invert τ={args.tau}"):
        imgs = imgs.to(args.device, non_blocking=True)
        b = imgs.shape[0]
        with torch.no_grad():
            z0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs)["latent_dist"].mean
            embed = embed_empty.repeat(b, 1, 1)
            z_tau = pipe.latent_forward_inversion(
                z0, embed, noise_level=noise_level, guidance_scale=0, show_progress=False
            )
        out[out_idx:out_idx + b] = z_tau.cpu().float()
        out_idx += b

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Saved {out_idx} latents to {args.out}")
    print(f"  shape={tuple(out.shape)}, mean={out.mean():.4f}, std={out.std():.4f}")


if __name__ == "__main__":
    main()
