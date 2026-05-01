# CelebA-HQ Experiment Spec

## Context

This is the first full-scale experiment for the project. We are testing whether the proposed metric $G = G_{x_t} + G_\epsilon$ (score-Jacobian + Gaussian-annulus) yields perceptually better image interpolations than existing baselines, when the geodesic is parameterized by a UNet $\varphi_{t,\eta}$ trained via the Metric Flow Matching procedure.

CelebA-HQ is a faces dataset where smoothness, identity preservation, and realism along the interpolation are all visually salient — making it a natural arena for comparing interpolation methods.

The math, training loop, and metric definition live in [algorithm.md](algorithm.md). The dataset / baseline scope decisions live in [experiment_plan.md](experiment_plan.md). This file pins down the *implementation contract* for CelebA-HQ specifically.

## Diffusion-model setup

| Item                | Value                                              | Source                              |
|---------------------|----------------------------------------------------|-------------------------------------|
| Backbone            | `stabilityai/stable-diffusion-2-1-base`            | Be Tangential (paper, Sec. exps).   |
| Sampler             | DDIM                                               |                                     |
| Train timesteps     | 1000 (default)                                     | scheduler default                   |
| Inference steps     | T = 50                                             | matches Be Tangential and `pipeline.py` |
| Latent shape        | 4×64×64                                            | SD VAE                              |
| Pixel shape         | 3×512×512                                          |                                     |
| Working noise level | $\tau = 600$ (training-timestep index)             | [algorithm.md](algorithm.md)        |

Reuse [GeodesicDiffusion/model/pipeline.py](../GeodesicDiffusion/model/pipeline.py) (`load_pipe`, `SimpleDiffusionPipeline`) as-is — its `model_id` already points at SD 2.1-base.

## Data

- **Source:** CelebA-HQ (30k images at 1024², downsampled to 512² to match the SD pipeline).
- **Pairing:** random pairs $(x_0, x_1)$ sampled uniformly from the train split each iteration.
- **Eval set:** a fixed held-out set of **5,000 pairs**, seed-determined and reused across all baselines for direct comparability. For each pair we render a length-10 subsampled interpolation (10 equally-spaced $t$ values).
- **Image preprocessing:** standard `[-1, 1]` rescaling expected by SD VAE (use the existing `load_image` helper in [GeodesicDiffusion/model/utils.py](../GeodesicDiffusion/model/utils.py)).
- **Caching:** to avoid re-running DDIM inversion every iteration, cache $(z_0^{(\tau=600)}, z_1^{(\tau=600)})$ to disk for at least the eval set, and ideally for a large training pool. ~30k × (4·64·64·4 bytes) ≈ 2 GB — fits.

## Encoding to noise space (DDIM inversion)

1. RGB image → SD VAE latent ($z \in \mathbb{R}^{4\times64\times64}$) via `pipe.img2latent(...)`.
2. Latent → $\tau = 600$ via deterministic DDIM inversion: [pipeline.py:`latent_forward_inversion`](../GeodesicDiffusion/model/pipeline.py#L92-L107).
3. Use empty prompt embedding (`pipe.prompt2embed("")`) and `guidance_scale = 0` (matches "negative-prompt SDS, empty conditional embed").
4. Translating $\tau = 600$ to the `noise_level` arg: with 50 inference steps over 1000 train timesteps, `noise_level = 600 / 1000 = 0.6`. (`get_t` in pipeline.py picks the timestep accordingly.)

Decoding back to pixel space at evaluation time uses [pipeline.py:`latent_backward`](../GeodesicDiffusion/model/pipeline.py#L79-L90) with the same empty prompt and `guidance_scale = 0`.

## Score function

Use the negative-prompt SDS direction as $s_\theta$, exactly as Be Tangential does.

- Module: [GeodesicDiffusion/model/score.py:`Score_Distillation`](../GeodesicDiffusion/model/score.py).
- Settings: `grad_guidance_0 = grad_guidance_1 = 1`, `grad_weight_type = 'uniform'`, `time_step = 600`, `grad_sample_type = 'ori_step'`, `embed_cond = pipe.prompt2embed("")` (empty), `embed_neg` left at the hard-coded default in [score.py:23-24](../GeodesicDiffusion/model/score.py#L23-L24).
- The `grad_compute(latent, embed_cond)` output is our $s_\theta(x_t, \tau=600)$.
- Determinism: with `grad_sample_type='ori_step'`, `grad_prepare` is the identity and no extra noise is sampled inside `grad_compute`, so $s_\theta$ is a deterministic function of $x_t$ — a hard requirement for JVP.

## Metric

### Score-Jacobian term $G_{x_t}$

$g_{x_t}(v, v) = \lVert J_{x_t} v \rVert^2$ where $J_{x_t} = \nabla_{x_t} s_\theta(x_t, \tau)$.

- Compute $J_{x_t} v$ via `torch.func.jvp(lambda x: score_module.grad_compute(x, embed_empty), (x_t,), (v,))`. **Never materialize $J$.**
- Squared L2 norm of the JVP gives $g_{x_t}(v, v)$.

### Gaussian-annulus term $G_\epsilon$

$G_\epsilon(x) = \big(\frac{\lVert x \rVert - \mu_\epsilon}{\sigma_\epsilon}\big)^2$, treated as a *scalar conformal multiplier* on the Euclidean inner product (so it contributes $G_\epsilon(x) \cdot \lVert v \rVert^2$ to the kinetic energy). This matches the structure of `ConformalMetric` in [RiemannEBM/utils/Riemannian_metric.py:59-74](../RiemannEBM/utils/Riemannian_metric.py#L59-L74).

**Calibration (one-time):**
1. Sample 1000 CelebA-HQ images uniformly from the train split (fixed seed).
2. Encode each via VAE → latent $z_0$.
3. DDIM-invert each latent to $\tau = 600$ using [pipeline.py:`latent_forward_inversion`](../GeodesicDiffusion/model/pipeline.py#L92-L107) with the empty prompt embedding and `guidance_scale = 0` — same procedure used to encode the training pairs into noise space, so $\mu_\epsilon, \sigma_\epsilon$ reflect the actual distribution the interpolant operates on.
4. Compute $\mu_\epsilon = \mathrm{mean}_i \lVert z_\tau^{(i)} \rVert_2$ and $\sigma_\epsilon = \mathrm{std}_i \lVert z_\tau^{(i)} \rVert_2$ over the 1000 samples.
5. Cache `(μ_ε, σ_ε)` to disk; only recompute if dataset/τ changes.

**Note on terminology:** [algorithm.md:21](algorithm.md#L21) phrases this as *"noise them to the τ=600 level"*, which colloquially means q-sample (`scheduler.add_noise(z_0, ε, t=600)`, drawing fresh Gaussian noise). We are deliberately using DDIM inversion instead because q-sample produces the *forward* marginal at $\tau=600$ (full Gaussian-annulus shell), whereas the interpolant only ever sees the more concentrated DDIM-inverted distribution. Calibrating on the latter makes $G_\epsilon$ a sharper penalty.

### Combined kinetic energy

$$g(x_t, v) = \lVert J_{x_t} v \rVert^2 + G_\epsilon(x_t) \cdot \lVert v \rVert^2$$

## Interpolant network $\varphi_{t,\eta}$

Use the user-supplied UNet under the project root:

- [unet_base.py](../unet_base.py) — guided-diffusion UNet (`UNetModel`, `UNetModelWrapper`). `zero_module` is preserved on the final conv and on each `ResBlock`'s output conv, so the network outputs ≈0 at initialization → $x_{t,\eta}$ starts as plain LERP and only deviates as training drives it.
- [geopath_networks/unet.py](../geopath_networks/unet.py) — `GeoPathUNet` channel-concats $(x_0, x_1)$ along the channel axis and forwards through `UNetModelWrapper(geopath_model=True, ...)`, which doubles `in_channels` accordingly.

**Constructor args** (mirror [`train_interpolant.py:64-70`](../RiemannEBM/train_interpolant.py#L64-L70), adjusted for our shape):

```python
GeoPathUNet(
    dim=(4, 64, 64),         # (C, H, W) of a single latent — wrapper doubles in_channels for x0||x1
    num_channels=64,         # base width; ablate to 128 if undersized
    num_res_blocks=2,
    channel_mult=(1, 2, 2, 2),  # capacity-trimmed at bottleneck; default for image_size=64 in unet_base.py:907 is (1,2,3,4)
    dropout=0.0,
    attention_resolutions="16", # attention at the 4x4 level (64/16)
)
```

`channel_mult` and `attention_resolutions` could be left at defaults; listing them explicitly to make the spec self-contained.

**Import-path TODO:** `geopath_networks/unet.py` imports `from mfm.networks.unet_base import UNetModelWrapper`, but `unet_base.py` is at the repo root. Either move `unet_base.py` under `mfm/networks/` or rewrite the import to `from unet_base import UNetModelWrapper` (or a similar package path once we create one). Resolve when scaffolding the project layout.

**Interpolant equation:** $x_{t,\eta} = (1-t)x_0 + tx_1 + 2t(1-t)\,\varphi_{t,\eta}(x_0, x_1)$ (factor of 2 follows [train_interpolant.py:176](../RiemannEBM/train_interpolant.py#L176); keeps the deviation magnitude ≤ ½ when $|\varphi| \leq 1$).

## Training loop

Modeled directly on [train_interpolant.py:160-189](../RiemannEBM/train_interpolant.py#L160-L189). Pseudocode:

```python
for it in range(nb_iter):
    # 1. Sample image pair, fetch cached τ=600 latents.
    z0, z1 = sample_pair_latents(batch_size)        # (B, 4, 64, 64) at τ=600

    # 2. Replicate over t-grid; t ∈ [0, 1] linspace with t_steps points.
    z0r = z0[:, None].expand(B, t_steps, 4, 64, 64)
    z1r = z1[:, None].expand_as(z0r)
    t   = torch.linspace(0, 1, t_steps).view(1, t_steps, 1, 1, 1).expand_as(z0r)

    # 3. Forward through interpolant.
    phi = geopath_unet(z0r.reshape(-1, 4, 64, 64),
                        z1r.reshape(-1, 4, 64, 64),
                        t.reshape(-1)).view(B, t_steps, 4, 64, 64)
    z_t = (1 - t) * z0r + t * z1r + 2 * t * (1 - t) * phi   # (B, t_steps, 4, 64, 64)

    # 4. Finite-difference time derivative (matches train_interpolant.py:178).
    dt = 1.0 / (t_steps - 1)
    z_t_dot = (z_t[:, 1:] - z_t[:, :-1]) / dt              # (B, t_steps - 1, 4, 64, 64)

    # 5. Per-step kinetic energy:
    #    e_i = ||J_{z_t_i} z_t_dot_i||^2 + G_ε(z_t_i) * ||z_t_dot_i||^2
    e = kinetic_energy(z_t[:, :-1], z_t_dot)                # (B, t_steps - 1)

    # 6. Path energy (Riemann sum) and backprop.
    loss = (e * dt).sum(dim=1).mean()
    loss.backward(); opt.step(); opt.zero_grad()
```

`kinetic_energy` flattens spatial dims and applies the metric defined above. It uses `torch.func.jvp` for the score-Jacobian term (one JVP call per `(z_t_i, z_t_dot_i)`).

**Hyperparameters (Andrew's specified starting point):**

| Param            | Value     | Notes                                                      |
|------------------|-----------|------------------------------------------------------------|
| `t_steps`        | 100       |                                                             |
| `batch_size`     | 12        |                                                             |
| `lr`             | 1e-4      | Adam                                                       |
| `nb_iteration`   | 200,000   |                                                             |
| optimizer        | Adam      |                                                            |
| AMP              | bf16/fp16 | the SD UNet runs in autocast inside `Score_Distillation`   |

These mirror Andrew's prior `--metric conf_ebm_logp` runs in RiemannEBM. Note that "conformal" there refers to a scalar metric multiplier (their EBM log-prob); ours has both the scalar component ($G_\epsilon$) **and** the non-conformal score-Jacobian term ($G_{x_t}$), but the empirical hyperparameter envelope should transfer.

**Per-step JVP cost:** $B \cdot (t_{\text{steps}}-1) = 12 \cdot 99 = 1188$ JVPs through the SD UNet per iteration. At 200k iterations this is the dominant cost — see compute plan for batching strategy.

## Compute plan

Hardware: 1 node × 4× 48 GB GPUs.

- **Phase 1 (cache):** VAE-encode + DDIM-invert all 30k CelebA-HQ images at $\tau=600$. Embarrassingly parallel — split across 4 GPUs.
- **Phase 2 (calibrate $G_\epsilon$):** 1000 images, single GPU, < 5 min.
- **Phase 3 (train interpolant):** 200k iterations × 1188 JVPs is heavy. First-pass strategy: single-GPU run with the JVP batched over the full $B \cdot (t_\text{steps}-1)$ axis, chunked to fit in 48 GB. Wallclock will be the deciding factor — instrument step time on a 100-iteration smoke test before committing. Fallbacks if too slow:
  - **DDP across all 4 GPUs** (each does $B/4$ pairs), straightforward since the loss is per-pair-mean.
  - **Stochastic $t$-subsampling:** pick $K \ll t_{\text{steps}}$ random indices per step.
  - Drop `batch_size` to 8 or use gradient checkpointing through the SD UNet.
- **Phase 4 (eval):** Single GPU per baseline, run in parallel.

## Evaluation

**Eval pairs:** the fixed held-out set of 5,000 pairs.

**For each method, output:** $K = 10$ equally-spaced interpolated images per pair, decoded back to pixel space.

**Metrics:**

1. **FID-of-midpoints** — pool all interior $\{x_t : t \in (0, 1)\}$ across all pairs and compute FID against CelebA-HQ. Measures realism along the path.
2. **Perceptual Path Length (PPL)** — $\sum_k \mathrm{LPIPS}(x_{t_k}, x_{t_{k+1}})$ averaged over pairs. Lower = smoother.
3. **Qualitative figures** — side-by-side panels for a few hand-picked pairs across all methods.

**Baselines:**

| Baseline             | Source / status                                                                   |
|----------------------|------------------------------------------------------------------------------------|
| LERP / SLERP         | Implement directly. Operate in $\tau=600$ DDIM-inverted latent space, decode same way as ours. |
| Be Tangential        | Implement using same metric infra minus our $G_\epsilon$ term — i.e., $G = G_{x_t}$ only. Direct ablation. |
| GeodesicDiffusion    | Use [GeodesicDiffusion/test_bvp.py](../GeodesicDiffusion/test_bvp.py) directly.   |
| NoiseDiffusion       | Adapt from [NoiseDiffusion/](../NoiseDiffusion/). Training-free; should drop in.   |
| DiffMorpher          | Use [DiffMorpher/main.py](../DiffMorpher/main.py). Requires a per-pair LoRA finetune step. |
| IMPUS                | TBD — not in this tree; will need to clone separately if we go for it.            |

The baselines that operate in noise space (LERP, SLERP, NoiseDiffusion, Be Tangential) should share our DDIM-inversion + decoding pipeline so the only varying factor is the interpolation method itself.

## Implementation milestones

In priority order, each a tangible deliverable:

1. **`scripts/cache_celeba_latents.py`** — VAE-encode + DDIM-invert the full CelebA-HQ train split at $\tau=600$, save to `data/celeba_hq/latents_tau600/{idx}.pt`. Verify by round-tripping a sample back to pixels.
2. **`scripts/calibrate_annulus.py`** — compute and save $(\mu_\epsilon, \sigma_\epsilon)$. Print sanity-check stats (e.g., mean/std should be roughly $\sqrt{D}$ given an isotropic-Gaussian assumption, with $D = 4\cdot64\cdot64$).
3. **`src/metric.py`** — implement `kinetic_energy(z_t, z_t_dot)` with the JVP score-Jacobian + annulus terms. Unit-test on synthetic inputs.
4. **`src/train_interpolant.py`** — adapt [RiemannEBM/train_interpolant.py](../RiemannEBM/train_interpolant.py) to our metric, dataset, and `GeoPathUNet`. Smoke-test for ~100 iterations with small `t_steps` to confirm the loss decreases and to instrument step time (for the Phase-3 wallclock decision).
5. **`scripts/eval_interpolation.py`** — given a method + eval set, output the $K$-frame trajectories and metric scores. Methods registered in a small registry.
6. **Baselines plumbing** — wire LERP, SLERP, Be Tangential (= ours minus $G_\epsilon$) through `scripts/eval_interpolation.py`. Add the others (DiffMorpher, NoiseDiffusion, GeodesicDiffusion) as separate adapters.

## Deferred items

- **Wallclock feasibility:** instrument step time on a 100-iter smoke test before committing to the full 200k iterations; cut paths are listed in the compute plan.
- **Import-path fix:** `geopath_networks/unet.py` imports from `mfm.networks.unet_base`, but [unet_base.py](../unet_base.py) is at repo root. Resolve at project-scaffolding time.
