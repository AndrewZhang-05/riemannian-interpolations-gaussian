# Project: Riemannian Interpolations in Gaussian Noise Space

## Goal

Define a Riemannian metric in diffusion noise space and use it to learn geodesic image interpolations via a UNet-parameterized interpolant.

The metric combines:
1. **Score-Jacobian term** $G_{x_t} = J_{x_t}^\top J_{x_t}$ (from "Be Tangential to Manifold", https://arxiv.org/abs/2510.05509). Computed via `torch.func.jvp` — never materialize $J$.
2. **Gaussian-annulus term** $G_\epsilon = \big(\frac{\|x\| - \mu_\epsilon}{\sigma_\epsilon}\big)^2$. $\mu_\epsilon, \sigma_\epsilon$ are estimated from 1000 dataset images noised to $\tau=600$.

Combined: $G = G_{x_t} + G_\epsilon$. Interpolant is trained with the Metric Flow Matching procedure (see `RiemannEBM/`).

## Canonical specs — read these first

- [ClaudeInstructions/algorithm.md](ClaudeInstructions/algorithm.md) — full math, training loop, score-function choice per dataset.
- [ClaudeInstructions/experiment_plan.md](ClaudeInstructions/experiment_plan.md) — datasets, baselines, scope decisions.
- [ClaudeInstructions/celeba_hq_spec.md](ClaudeInstructions/celeba_hq_spec.md) — implementation contract for the first full-scale experiment (CelebA-HQ).

These two files are the source of truth. If anything below conflicts with them, trust the ClaudeInstructions files.

## Reference repos in this tree

- [DiffMorpher/](DiffMorpher/) — LoRA-based interpolation baseline.
- [GeodesicDiffusion/](GeodesicDiffusion/) — source of the negative-prompt-corrected score for AFHQ/CelebA-HQ. See [GeodesicDiffusion/model/score.py](GeodesicDiffusion/model/score.py).
- [NoiseDiffusion/](NoiseDiffusion/) — training-free baseline.
- [RiemannEBM/](RiemannEBM/) — reference for the interpolant training loop and UNet architecture (originally for $4\times64\times64$ VAE latents). See [RiemannEBM/train_interpolant.py](RiemannEBM/train_interpolant.py).

These are pulled in as references — read them to understand patterns and reuse code, but our own code lives outside these directories.

## Score-function choice (per dataset)

| Dataset      | Score                              |
|--------------|------------------------------------|
| Two-Moons    | actual score                       |
| Rotated MNIST| actual score                       |
| AFHQ         | negative-prompt-corrected score    |
| CelebA-HQ    | negative-prompt-corrected score    |

## Conventions

- Operate in DDIM-inverted noise space at $\tau=600$ unless stated otherwise.
- Image-domain experiments interpolate in the $4\times64\times64$ SD v1.4 latent, not pixel space.
- JVPs over Jacobian materialization. Always.

## Working style for Andrew

- Andrew specs the math; I implement and verify against the spec.
- Default to small, runnable increments — train Two-Moons first, then MNIST, then image datasets.
- If algorithm.md or experiment_plan.md conflicts with this file, the ClaudeInstructions files win.
