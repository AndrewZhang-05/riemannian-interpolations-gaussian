
We have the metric $$
J_x^\top J_x + m(x)^2 I.$$We will try two versions of $m(x)$. The first is $m(x) = \frac{||x|| - \mu}{\sigma}$. I will outline the second later.

I want to use the metric

$$  
g_x(v,w)=\langle J_x v, J_x w\rangle + m(x)^2 \langle v,w\rangle,  
$$

with matrix form

$$  
G_x = J_x^\top J_x + m(x)^2 I,  
$$

 a paper-style interpolation procedure between two images (x_1) and (x_2) looks like this. This follows the same overall protocol as the paper: invert the endpoints into noise space, initialize a discrete path, optimize the intermediate points by minimizing a discretized energy, then denoise the optimized path back to image space.

---

# Step 1: Choose the diffusion model and the working timestep

Let (s_\theta(x,t)) be the score function (or equivalently the noise-prediction network up to a known scale).

Pick:

- a pretrained diffusion model,
    
- a total number of timesteps (T),
    
- a working interpolation timestep (\tau > 0).
    

Following the paper as closely as possible, you would use:

- Stable Diffusion v2.1-base,
    
- (T=50),
    
- interpolation at (\tau = 0.6T).
    

---

# Step 2: Map the two endpoint images into noise space

Start with two clean images:

$$  
x_1,; x_2.  
$$

Using **DDIM inversion**, map them into the noise space at time (\tau):

$$  
z_0 := \mathrm{DDIM\text{-}Fwd}(x_1,\tau), \qquad  
z_N := \mathrm{DDIM\text{-}Fwd}(x_2,\tau).  
$$

Here I’m writing the noisy latent path as

$$  
z_0, z_1, \dots, z_N  
$$

to keep it separate from the clean image notation.

So:

- (z_0) is the noisy version of (x_1),
    
- (z_N) is the noisy version of (x_2).
    

These two endpoints stay **fixed** throughout optimization.

---

# Step 3: Discretize the interpolation path

Choose the number of segments (N). Then the path is

$$  
z_0, z_1, z_2, \dots, z_{N-1}, z_N.  
$$

- (z_0) and (z_N) are fixed.
    
- (z_1,\dots,z_{N-1}) are trainable.
    

If you want to match the paper’s image interpolation setup, use

$$  
N-1=9  
$$

intermediate points.

Let

$$  
\Delta u = \frac{1}{N}.  
$$

---

# Step 4: Initialize the intermediate points with SLERP

Initialize the interior points by spherical linear interpolation between the noisy endpoints:

$$  
z_i^{(0)} = \mathrm{SLERP}(z_0,z_N,u_i),  
\qquad  
u_i = \frac{i}{N}, \quad i=1,\dots,N-1.  
$$

So your initial discrete path is:

$$  
z_0,; z_1^{(0)},; z_2^{(0)},; \dots,; z_{N-1}^{(0)},; z_N.  
$$

This is exactly the role SLERP plays in the paper: it gives the optimizer a reasonable starting path.

---

# Step 5: Write down the continuous energy for your metric

Your metric is

$$  
g_x(v,v)=|J_x v|^2 + m(x)^2|v|^2.  
$$

So the continuous Riemannian energy is

# $$  
E[\gamma]

\frac12\int_0^1  
\left(  
|J_{\gamma(u)}\gamma'(u)|^2  
+  
m(\gamma(u))^2|\gamma'(u)|^2  
\right),du.  
$$

Using the chain rule,

# $$  
J_{\gamma(u)}\gamma'(u)

\frac{d}{du}s_\theta(\gamma(u),\tau),  
$$

so equivalently,

# $$  
E[\gamma]

\frac12\int_0^1  
\left(  
\left|\frac{d}{du}s_\theta(\gamma(u),\tau)\right|^2  
+  
m(\gamma(u))^2|\gamma'(u)|^2  
\right),du.  
$$

---

# Step 6: Discretize the energy the same way the paper discretizes theirs

Approximate the path derivative and the score derivative on each segment by finite differences:

$$  
\gamma'(u_i)\approx \frac{z_{i+1}-z_i}{\Delta u},  
$$

and

$$  
\frac{d}{du}s_\theta(\gamma(u_i),\tau)  
\approx  
\frac{s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau)}{\Delta u}.  
$$

Because you said you want (m) evaluated at (z_i) rather than at the midpoint, the discrete energy becomes

# $$  
E_{\text{disc}}(z_1,\dots,z_{N-1})

\frac{1}{2\Delta u}  
\sum_{i=0}^{N-1}  
\left[  
|s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau)|^2  
+  
m(z_i)^2 |z_{i+1}-z_i|^2  
\right].  
$$

# $$  
\mathcal L(z_1,\dots,z_{N-1})

\sum_{i=0}^{N-1}  
\left[  
|s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau)|^2  
+  
m(z_i)^2 |z_{i+1}-z_i|^2  
\right].  
$$

This is the direct analogue of the paper’s objective, with your additional (m(z_i)^2|z_{i+1}-z_i|^2) term added.

---

# Step 7: Optimize only the intermediate points

Now solve

$$  
\min_{z_1,\dots,z_{N-1}} \mathcal L(z_1,\dots,z_{N-1}),  
$$

with (z_0) and (z_N) fixed.

This is exactly how the paper handles interpolation: the path endpoints are fixed, and only the interior points are updated to minimize the discrete energy.

The optimization loop is:

1. Form the full path (z_0,\dots,z_N).
    
2. Evaluate the score network (s_\theta(z_i,\tau)) at every path point.
    
3. Compute  
    $$  
    \Delta s_i = s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau).  
    $$
    
4. Compute  
    $$  
    \Delta z_i = z_{i+1}-z_i.  
    $$
    
5. Compute the loss  
    $$  
    \mathcal L = \sum_{i=0}^{N-1}  
    \left[  
    |\Delta s_i|^2 + m(z_i)^2 |\Delta z_i|^2  
    \right].  
    $$
    
6. Backpropagate through the score evaluations and through (m(z_i)).
    
7. Update (z_1,\dots,z_{N-1}) with Adam.
    

---

# Step 8: Use the same optimizer protocol as the paper

If you want to mimic the paper closely, use:

- **Adam**
    
- learning rate (10^{-3}),
    
- cosine decay to (10^{-4}),
    
- **500 optimization iterations**.
    

So in practice:

$$  
(z_1,\dots,z_{N-1})  
\leftarrow  
\mathrm{AdamStep}\left(  
\nabla_{z_1,\dots,z_{N-1}} \mathcal L  
\right)  
$$

repeated for 500 iterations.

---

# Step 9: Denoise the optimized noisy path back to clean images

After optimization, you have an optimized noisy path

$$  
z_0^\star,; z_1^\star,; \dots,; z_N^\star,  
$$

where

- (z_0^\star = z_0),
    
- (z_N^\star = z_N),
    
- (z_1^\star,\dots,z_{N-1}^\star) are the optimized intermediate points.
    

Now map each point back to clean image space using deterministic DDIM denoising:

$$  
\hat x_i = \mathrm{DDIM\text{-}Bwd}(z_i^\star,\tau),  
\qquad i=0,\dots,N.  
$$

Then

- (\hat x_0) reconstructs (x_1),
    
- (\hat x_N) reconstructs (x_2),
    
- (\hat x_1,\dots,\hat x_{N-1}) are your interpolated images.
    

---

# Step 10: Final output

Return the clean image sequence

$$  
\hat x_0,\hat x_1,\dots,\hat x_N.  
$$

That sequence is your interpolation between (x_1) and (x_2) under the metric

$$  
G_x = J_x^\top J_x + m(x)^2 I.  
$$

---

# Compact algorithm

Here is the full procedure in one block.

## Inputs

- endpoint images (x_1,x_2),
    
- pretrained score model (s_\theta),
    
- scalar field (m(x)),
    
- interpolation timestep (\tau),
    
- number of segments (N),
    
- optimization steps (K).
    

## Procedure

1. **Invert endpoints**  
    $$  
    z_0 = \mathrm{DDIM\text{-}Fwd}(x_1,\tau),\qquad  
    z_N = \mathrm{DDIM\text{-}Fwd}(x_2,\tau).  
    $$
    
2. **Initialize path**  
    $$  
    z_1,\dots,z_{N-1} \leftarrow \text{SLERP between } z_0 \text{ and } z_N.  
    $$
    
3. **Optimize**  
    For (k=1,\dots,K):
    
    - compute (s_\theta(z_i,\tau)) for all (i=0,\dots,N),
        
    - compute  
        $$  
        \mathcal L =  
        \sum_{i=0}^{N-1}  
        \left[  
        |s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau)|^2  
        +  
        m(z_i)^2|z_{i+1}-z_i|^2  
        \right],  
        $$
        
    - update (z_1,\dots,z_{N-1}) with Adam.
        
4. **Denoise optimized path**  
    $$  
    \hat x_i = \mathrm{DDIM\text{-}Bwd}(z_i,\tau),\qquad i=0,\dots,N.  
    $$
    
5. **Return**  
    $$  
    \hat x_0,\hat x_1,\dots,\hat x_N.  
    $$
    

---

# Intuition for what this does

The two terms play different roles:

1. **Jacobian / score term**  
    $$  
    |s_\theta(z_{i+1},\tau)-s_\theta(z_i,\tau)|^2  
    $$  
    encourages the path to move in directions where the score changes slowly, i.e. approximately tangential directions.
    
2. **(m(z_i)^2|z_{i+1}-z_i|^2) term**  
    penalizes large Euclidean motion, weighted by location.  
    So if (m(z_i)) is large in some region, the path avoids making big moves there.
    

So compared to the paper’s original metric, your version adds an isotropic spatial penalty on top of the tangency-aware one.
