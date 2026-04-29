import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geopath_networks.unet import GeoPathUNet as UNetModel
from src.diffusion_pipeline import load_pipe
from src.metric import AnnulusStats, RiemannianMetric
from src.score import Score_Distillation


def main(args):
    with torch.no_grad():
        if args.dataset == 'celeba_hq':
            ## load the cached τ=600 SD-inverted latents (one tensor, (N, 4, 64, 64))
            latents = torch.load(args.latent_cache, map_location='cpu', weights_only=False)
            assert latents.dim() == 4 and latents.shape[1:] == (4, 64, 64), \
                f"unexpected cache shape {tuple(latents.shape)}"
            latents = latents.float()
            print(f"[data] loaded {len(latents)} latents from {args.latent_cache}")

            lt_size = tuple(latents.shape[1:])  # (4, 64, 64)

            args_curnet = {"dim": lt_size,
                           "num_channels": args.num_channels,
                           "num_res_blocks": 2,
                           "channel_mult": tuple(args.channel_mult),
                           "dropout": 0.0,
                           "attention_resolutions": args.attention_resolutions,
                           }
            curvature_net = UNetModel(
                **args_curnet
            ).to(args.device)

        else:
            raise NotImplementedError()

        ## build the score-Jacobian + Gaussian-annulus metric (G = G_{x_t} + G_ε).
        pipe = load_pipe(args.device, num_inference_steps=50)
        pipe.unet.eval()
        score = Score_Distillation(
            pipe,
            time_step=args.tau,
            grad_guidance_0=1, grad_guidance_1=1,
            grad_weight_type='uniform',
            grad_sample_type='ori_step',
        )
        embed_empty = pipe.prompt2embed("")  # (1, T, D)
        annulus = AnnulusStats.load(args.annulus_stats)
        print(f"[metric] μ_ε={annulus.mu:.4f} σ_ε={annulus.sigma:.4f}")

        if args.model_name is not None:
            path_to_save = os.path.join(args.save_root, args.dataset, args.model_name)
            print(path_to_save)
            os.makedirs(path_to_save, exist_ok=True)
            if args.wandb:
                import wandb
                wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                           config=vars(args), name=args.model_name)
            torch.save(args, path_to_save + '/param.config')

    metric = RiemannianMetric(
        score=score,
        annulus=annulus,
        embed_cond=embed_empty,
        jvp_backend=args.jvp_backend,
        fd_eps=args.fd_eps,
        chunk_size=args.jvp_chunk_size,
    )

    dt = torch.tensor(1.0 / (args.t_steps - 1))
    optimizer = torch.optim.Adam(curvature_net.parameters(), lr=args.lr)
    all_loss = []

    t_ = torch.linspace(0, 1, args.t_steps).view(1, args.t_steps, *[1 for _ in lt_size]).to(args.device).detach()
    pair_gen = torch.Generator(device='cpu').manual_seed(args.seed)
    step_times = []
    torch.cuda.synchronize()
    t_last = time.time()
    for it in range(args.nb_iteration):
        idx0 = torch.randint(low=0, high=latents.size(0), size=(args.batch_size,), generator=pair_gen)
        idx1 = torch.randint(low=0, high=latents.size(0), size=(args.batch_size,), generator=pair_gen)
        z0 = latents[idx0].unsqueeze(1).to(args.device, non_blocking=True)
        z1 = latents[idx1].unsqueeze(1).to(args.device, non_blocking=True)

        z0 = z0.repeat(1, args.t_steps, *[1 for _ in lt_size]).detach()
        z1 = z1.repeat(1, args.t_steps, *[1 for _ in lt_size]).detach()
        t = t_.repeat(z0.size(0), 1, *[1 for _ in lt_size])  # .expand_as(z0).detach()
        c_t = curvature_net(z0.reshape(-1, *lt_size), z1.reshape(-1, *lt_size), t.view(-1))
        z_t = (1 - t) * z0 + t * z1 + 2 * t * (1 - t) * c_t.view(args.batch_size, args.t_steps, *lt_size)
        z_t_dot = (z_t[:, 1:] - z_t[:, :-1]) / dt
        energy = metric.kinetic(z_t[:, :-1], z_t_dot)
        energy = energy.view(args.batch_size, args.t_steps - 1)
        kinetic_energy = (energy * dt).sum(dim=1).mean()
        loss = kinetic_energy
        loss.backward()
        all_param = torch.nn.utils.clip_grad_norm_(
            curvature_net.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf'),
        )
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())
        torch.cuda.synchronize()
        now = time.time()
        step_times.append(now - t_last)
        t_last = now

        with torch.no_grad():
            if (it + 1) % args.log_every == 0 or it == 0:
                warm = step_times[-min(args.log_every, len(step_times)):]
                print(
                    f"[{it+1:>6}/{args.nb_iteration}] kinetic={kinetic_energy.item():0.3f}  "
                    f"grad={all_param.item():0.3f}  step={sum(warm)/len(warm):.2f}s")
                if args.wandb and args.model_name is not None:
                    import wandb
                    wandb.log({"train/kinetic_energy": loss.item(),
                               "train/grad_norm": all_param.item(),
                               "train/step_time_s": step_times[-1]}, step=it + 1)

            if (it + 1) % args.save_every == 0 and args.model_name is not None:
                curvature_net.cpu()
                to_save = {"weight": curvature_net.state_dict(),
                           "type": type(curvature_net),
                           "args": args_curnet
                           }
                torch.save(to_save, path_to_save + f'/ep_{it+1}.model')
                curvature_net.to(args.device)

    if step_times:
        warm = step_times[max(1, len(step_times) // 5):]
        print(f"[summary] mean step (warm) = {sum(warm)/len(warm):.2f}s  "
              f"final loss = {all_loss[-1]:.4f}")

    if args.model_name is not None:
        curvature_net.cpu()
        to_save = {"weight": curvature_net.state_dict(),
                   "type": type(curvature_net),
                   "args": args_curnet,
                   "losses": all_loss,
                   "step_times": step_times,
                   }
        torch.save(to_save, path_to_save + f'/last.model')
        curvature_net.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CelebA-HQ interpolant training")

    ## dataset
    parser.add_argument('--dataset', type=str, choices=['celeba_hq'], default='celeba_hq')
    parser.add_argument('--latent_cache', type=Path,
                        default=REPO_ROOT / 'data/celeba_hq/train_celebahq_sd21_tau600.pt')
    parser.add_argument('--annulus_stats', type=Path,
                        default=REPO_ROOT / 'data/celeba_hq/annulus_stats_sd21_tau600.pt')
    parser.add_argument('--save_root', type=str, default=str(REPO_ROOT / 'runs'),
                        help='path to save the checkpoint')

    ## interpolation
    parser.add_argument('--t_steps', type=int, default=100)
    parser.add_argument('--nb_iteration', type=int, default=200_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    ## metric (score-Jacobian + Gaussian annulus)
    parser.add_argument('--tau', type=int, default=600,
                        help='training-timestep index where the score is evaluated')
    parser.add_argument('--jvp_backend', choices=['jvp', 'fd'], default='jvp',
                        help='`jvp`: torch.func.jvp under math SDPA (exact). '
                             '`fd`: finite-difference Jv with Flash SDPA (approx).')
    parser.add_argument('--fd_eps', type=float, default=1e-3)
    parser.add_argument('--jvp_chunk_size', type=int, default=12,
                        help='chunk B*(t_steps-1) JVPs to fit in GPU memory')

    ## device / training
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name of the model to save')

    ## interpolant network
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument('--attention_resolutions', type=str, default='16')

    ## logging
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10_000)
    parser.add_argument('--wandb', action='store_true', help='log to wandb')
    parser.add_argument('--wandb_project', type=str, default='celeba_interp')
    parser.add_argument('--wandb_entity', type=str, default=None)

    ## smoke preset
    parser.add_argument('--smoke', action='store_true',
                        help='100 iters, batch=2, t_steps=10 — for wallclock instrumentation')

    args = parser.parse_args()
    if args.smoke:
        args.nb_iteration = 100
        args.batch_size = 2
        args.t_steps = 10
        args.log_every = 10
        args.save_every = 10**9
        if args.model_name is None:
            args.model_name = f'smoke_{args.jvp_backend}'
    if args.model_name is None:
        args.model_name = f'celeba_{args.jvp_backend}'

    main(args)