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


Our metric is just going to be $G = G_{x_{t}} + G_{\epsilon}.$
  

Now that we have a metric, we're going to parametrize the interpolation with a model, and learn it. This is the same approach described in the RiemannEBM repo. The original UNET model for the parametrization of the interpolation from with a model Metric Flow Matching https://arxiv.org/pdf/2405.14780 is designed for interpolations in the $64 \times 64 \times 4$ latent space of a VAE, so we'll replicate that approach here.


The general approach is. Set interpolant of form $$

x_{t, \eta} = (1-t)x_{0} + tx_{1} + t(1-t)\varphi_{t,\eta}(x_{0}, x_{1}).$$

Our objective is to learn $\eta$ such that $x_{t, \eta}$ approximates the geodesic $\gamma_{t}^*$ where $\gamma_{t}^*$ is the path minimizing $$

\gamma_{t}^* = \arg\min_{\gamma_{t}: \gamma_{0} = x_{0}, \gamma_{1}=x_{1}} \mathcal{E}_{g}(\gamma_{t}), \text{ where }\mathcal{E}_{g}(\gamma_{t}) := \mathbb{E}[\dot{\gamma}_{t}^{\top}\mathbf{G}(\gamma_{t}; \mathcal{D})\dot{\gamma_{t}}].$$

  

This gives us the following training loop.

1. Sample $(x_{0}, x_{1}) \sim q$ and $t \sim \mathcal{U}(0,1)$

2. $x_{t, \eta} = (1-t)x_{0} + tx_{1} + t(1-t)\varphi_{t,\eta}(x_{0}, x_{1})$

3. $\dot{x}_{t, \eta} = x_{1} - x_{0} + t(1-t)\dot{\varphi}_{t,\eta}(x_{0}, x_{1}) + (1-2t)\varphi_{t,\eta}(x_{0}, x_{1})$

4. $\ell(\eta) \leftarrow (\dot{x}_{t, \eta})^{\top} \mathbf{G}(x_{t,\eta}; \mathcal{D})\dot{x}_{t, \eta}$

5. Update $\eta$ using $\nabla_{n}\ell(\eta)$