**What baselines are we going to compare against?**

Let's search for baselines by looking at the papers that we're drawing ideas from.
Be Tangential To Manifold:

- LERP - Included
- SLERP - Included
- NAO - Not needed
- NoiseDiff - Included
- GeoDiff - Included
- FIM-based - Feels like I don't need to do, it was rejected from ICLR

Follow the energy, find the path
- No useful baselines

Probability Density Geodesics in Image Diffusion Latent Space
- NoiseDiff - Included
- AID - Not needed
- IMPUS - Included since review to Be Tangential To Manifold says it's fundamental
- DiffMorpher - Included since review to Be Tangential To Manifold says it's fundamental
- SmoothDiff - Not needed

How easy is it to setup the selected baselines

  

LERP/SLERP
- Easy

Diffmorpher - can be adapted as long as we have a diffusion model, just need a LoRA
- Requires finetuning a LoRA, but probably also easy

Isometric Representation Learning for Disentangled Latent Space of Diffusion Models
- Requires Retraining, not easy

Probability Density Geodesics in Image Diffusion Latent Space
- Should be easy

NoiseDiffusion: Correcting Noise for Image Interpolation with Diffusion Models beyond Spherical Linear Interpolation
- Training free, should be easy

Be Tangential To Manifold:
- Realized as the $\lambda_m = 0$ ablation of our method (same per-pair optimization, same code, same hyperparameters — only the metric drops the $G_\epsilon$ term). See [algorithm.md](algorithm.md) and [intermediate_optimization.md](intermediate_optimization.md).

**What datasets are we going to interpolate on?**

  

Lowest-End

Two-Moons 2D dataset
Just use normal score estimating since there's no conditioning

Need to train a diffusion model with $T = 50$ steps.

  

Compare against:
LERP/SLERP
Be Tangential To Manifold:
Follow the energy, find the path

Middle
Rotated MNIST.
Just use normal score estimating since there's no conditioning

Need to train a diffusion model with $T = 1000$ steps.

Compare against:
LERP/SLERP
Be Tangential To Manifold:
Follow the energy, find the path


Full-scale (Do all the interpolation in the $4\times64\times64$ latent)

CelebA-HQ $3 \times 512 \times 512$
Stable Diffusion v2.1-base diffusion model
- Evaluation protocol: see [intermediate_optimization_eval.md](intermediate_optimization_eval.md). 100 pairs (50 male + 50 female, LPIPS $<0.6$ between endpoints), $N-1=9$ interior frames per pair, metrics PPL / PDV / FID / RE.


AFHQ $3 \times 512 \times 512$
Stable Diffusion v2.1-base diffusion model

Compare against:
All baselines


**Method (across datasets).** Per-pair discrete path optimization in noise space, parameterized by the metric $G = G_{x_t} + \lambda_m\, G_\epsilon$. See [algorithm.md](algorithm.md) for the full procedure and [intermediate_optimization.md](intermediate_optimization.md) for implementation details. The Be-Tangential vs ours comparison is the $\lambda_m = 0$ vs $\lambda_m > 0$ ablation, run with [scripts/compare_annulus_ablation.py](../scripts/compare_annulus_ablation.py).