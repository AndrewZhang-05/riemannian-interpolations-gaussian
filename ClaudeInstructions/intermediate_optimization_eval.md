For **CelebA-HQ (CA)**, the paper’s interpolation evaluation is set up like this.

They use **Stable Diffusion v2.1-base** with **(T=50)** diffusion timesteps. For each interpolation, they take two real CelebA-HQ endpoint images, invert them into the diffusion noise space using **DDIM inversion**, compute the interpolation path in noise space at **(\tau = 0.6T)**, and then denoise each interpolated noise point back to image space. They generate **(N-1=9)** intermediate/interpolated images per endpoint pair. 

For the **CelebA-HQ dataset construction**, they randomly sample:

[
50 \text{ male pairs} + 50 \text{ female pairs} = 100 \text{ total pairs}.
]

They only keep pairs whose endpoint images have **LPIPS < 0.6**, to ensure the two endpoints are semantically similar. The text prompts are:

[
\text{``a photo of a man''}
]

for male pairs, and

[
\text{``a photo of a woman''}
]

for female pairs. 

So for CelebA-HQ, the number of evaluation images is:

[
100 \text{ pairs} \times 2 \text{ endpoints} = 200 \text{ reference images},
]

and

[
100 \text{ pairs} \times 9 \text{ interpolated images per pair} = 900 \text{ interpolated images}.
]

The paper’s FID protocol compares the **900 interpolated images** against the **200 endpoint/reference images** using Inception-v3 features. 

They report four metrics for CelebA-HQ interpolation:

| Metric  | Meaning                                                                                                                                                                |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PPL** | Perceptual Path Length: sum of LPIPS distances between adjacent images; lower means a more direct perceptual transition.                                               |
| **PDV** | Perceptual Distance Variance: standard deviation of adjacent LPIPS distances; lower means the transition is more uniform.                                              |
| **FID** | Distributional distance between interpolated images and reference endpoint images using Inception-v3 features; lower is better.                                        |
| **RE**  | Reconstruction Error: MSE between the original endpoints and their reconstructed endpoints after DDIM inversion/denoising; lower means endpoints are preserved better. |