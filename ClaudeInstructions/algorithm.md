I want to define a new Riemannian metric for interpolation.

Let's flesh out this general algorithm a bit more. We're using the Jacobian of the score predictor in noise space. Formally, $J_{x_t} = \nabla_{x_t} s_\theta(x_t, t)$ is the Jacobian of the score function $s_\theta(\cdot, t)$ at $x_t$,

The noise space we're operating in for each experiment is specified in experiment_description.md. So, basically we encode the images into that noise space using DDIM inversion, and then decode the images into pixel space using the deterministic denoising process also in DDIM inversion.

Since we're using a different training approach than them, how do we appropriately adapt it?

We take $G_{x_{t}} = J_{x_{t}}^{\top}J_{x_{t}}$ from the paper Be Tangential to Manifold https://arxiv.org/abs/2510.05509. This means that $g_{x_{t}}(v,v) = v^{\top}J_{x_{t}}^{\top}J_{x_{t}}v = ||J_{x_{t}}v||.$ We can calculate $J_{x_{t}}v$ just by using torch.func.jvp.

  
For the score function we can either
- Use the actual score (we have to do this for interpolations on MNIST and TwoMoons)

- Use the score given by the negative prompt correction to produce the noise-free score distillation direction. This method is described in the GeodesicDiffusion Repo in GeodesicDiffusion/model/score.py.

Let's default to the second for the AFHQ/CelebA-HQ experiments because that's what was used in Be Tangential to Manifold. Let's use the first for the TwoMoons and MNIST dataset.

Now, we add a Gaussian annulus based correction term. The underlying assumption is that since we're operating in the $\tau = 600$ noise space, the data should roughly resemble an isotropic Gaussian. The Gaussian annulus theorem then gives us the result that the magnitude of most of the data should be the same.

So, to create this term, we take $1000$ images from the dataset, noise them to the $\tau=600$ level and calculate the the mean magnitude $\mu_{\epsilon}$ and the standard deviation of the magnitude $\sigma_{\epsilon}$ Then, we define a new loss $G_{\epsilon} = \left( \frac{||x|| - \mu_{\epsilon}}{\sigma_{\epsilon}} \right)^2.$


Our metric is going to be $G = G_{x_{t}} + G_{\epsilon}$, with an optional weight $\lambda_m$ on the annulus term that we can tune at optimization time:
$$g_{x}(v, v) = \|J_{x} v\|^2 + \lambda_m\, G_{\epsilon}(x)\, \|v\|^2.$$

Setting $\lambda_m = 0$ recovers the score-Jacobian-only metric (the Be Tangential ablation); $\lambda_m > 0$ is the full proposed metric.

## Computing the geodesic interpolation

Rather than amortizing across pairs with a learned interpolant network, we solve each pair independently via discrete path optimization in noise space. This bypasses the SGD-over-many-pairs loop entirely; full derivation and pseudocode are in [intermediate_optimization.md](intermediate_optimization.md).

Given endpoint images $x_0, x_1$:

1. **Invert.** Map both endpoints into the $\tau$ noise space via deterministic DDIM inversion to get $z_0, z_N$ (frozen throughout optimization).

2. **Discretize.** Take a discrete path $z_0, z_1, \dots, z_N$ with $N$ segments (default $N=10$, giving $9$ interior points). Initialize the interior points via SLERP between $z_0$ and $z_N$.

3. **Discrete energy.** With $\Delta u = 1/N$, approximate the continuous energy $E[\gamma] = \tfrac{1}{2}\int_0^1 g(\gamma, \dot\gamma)\,du$ by
$$\mathcal{L} = \frac{1}{2\Delta u} \sum_{i=0}^{N-1} \Big[\,\|s_\theta(z_{i+1}, \tau) - s_\theta(z_i, \tau)\|^2 \;+\; \lambda_m\, G_\epsilon(z_i)\, \|z_{i+1} - z_i\|^2\,\Big].$$
The score-Jacobian term $\|J_x \dot\gamma\|^2$ is approximated as the squared *finite difference of scores at adjacent path points* (which equals $\|d s_\theta/du\|^2$ by the chain rule). This avoids JVP computation entirely — only standard score evaluations and their normal autograd graphs are needed.

4. **Optimize.** Solve
$$\min_{z_1, \dots, z_{N-1}} \mathcal{L}(z_1, \dots, z_{N-1})$$
with Adam (lr $10^{-3}$, cosine decay to $10^{-4}$, $500$ iterations). Pairs are batched together for SD-UNet throughput.

5. **Decode.** DDIM-denoise each optimized $z_i^{\star}$ back to pixel space to get the interpolated images.

The ablation $\lambda_m = 0$ vs $\lambda_m = 1$ directly measures the contribution of the Gaussian-annulus term within the same optimization framework — same code, same hyperparameters, same pair set, only the metric differs. See [intermediate_optimization_eval.md](intermediate_optimization_eval.md) for the CelebA-HQ-specific evaluation protocol.