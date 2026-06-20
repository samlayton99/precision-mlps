# Exp08 -- randomness and noise in the least-squares readout (1D, tanh)

## Question

Follow-up to exp07. exp07 fit and evaluated every geometry on a uniform grid, which may favor uniform centers. exp08 tests randomness on an equal footing: it varies the geometry (center placement), the sample points (where we observe $(x, f(x))$ to solve the readout), and adds $y$-noise -- all as overdetermined fp64 least-squares fits. Two questions: (1) is reaching machine precision governed by the centers or by the sample points? (2) how does uniform-center recovery degrade under perturbation of $x$ (random sample positions) and $y$ (additive noise)?

## Setup

All conditions are overdetermined least squares: $W$ neurons $< N_\text{train}$ sample points $< N_\text{eval}$ eval points, with $N_\text{train} = 1031$ and $N_\text{eval} = 7919$ chosen prime so the uniform train and eval grids do not align (an aligned eval grid would sample near-zero error at the training points and hide the true error). Centers reuse exp07's sizing: for each base grid size $N$, `uniform_geometry(N)` (QI grid + halo) fixes the width $W$, the center span, and $h = 2/N$; random centers are $W$ uniform draws over the same span. Every condition uses $\gamma = \lambda/h$ at each swept $\lambda$, so only randomness/noise differ. Target `sine`; error is always measured against the clean target on the eval grid (recovery of the true function). Best over the $\lambda$ sweep, mean over 3 seeds.

Code: `experiments/exp08_sampling_and_noise/run.py` (config in `config.yaml`). Reuses `src/construction/center_geometry.py` (`uniform_centers`, `random_centers`), `src/construction/readout.py::solve_readout_with_bias`, and `default_halo` -- no new `src/` code.

- Base grid sizes $N \in \{32, 64, 128, 256\}$ (widths $W \in \{173, 205, 269, 461\}$).
- Bandwidth sweep $\lambda \in \{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00\}$.
- $N_\text{train} = 1031$, $N_\text{eval} = 7919$ (both prime), seeds $\{0,1,2\}$ for stochastic conditions.
- $y$-noise (figure 2): additive Gaussian, std $10^{-3}$.
- Figure 3 (sample-count scaling): uniform centers, fixed $N = 64$ and $\lambda = 0.25$; sample count swept over $\{256, 512, \ldots, 131072\}$ (powers of two, all $> W = 205$); $y$-noise std swept over $\{0, 10^{-8}, 10^{-7}, \ldots, 10^{-2}\}$. The augmented $\Phi$ (and its SVD) depends only on the sample count, so it is factored once per count and reused across noise levels and seeds.

## Figure 1 -- geometry x sampling (clean)

`error_vs_width_geometry.png`. Two panels (left relative $L_2$, right $L_\infty$), $x = N$ (log), $y = $ best-over-$\lambda$ error (log), four curves = {uniform, random} centers $\times$ {uniform, random} samples. Mean over seeds; shaded band = per-seed min--max.

Best eval $L_\infty$ (mean over seeds):

| condition | $N{=}32$ | $64$ | $128$ | $256$ |
|---|---|---|---|---|
| uniform ctr / uniform samp | 1.5e-13 | 5.5e-14 | 1.4e-13 | 6.2e-13 |
| uniform ctr / random samp | 4.4e-13 | 2.4e-12 | 1.3e-12 | 6.7e-12 |
| random ctr / uniform samp | 1.9e-10 | 1.9e-11 | 1.3e-11 | 2.2e-11 |
| random ctr / random samp | 3.2e-10 | 5.7e-11 | 2.5e-10 | 1.7e-10 |

Read it as: do the two uniform-center curves sit at the fp64 floor while the two random-center curves plateau above it, regardless of sample type.

## Figure 2 -- uniform centers, x/y perturbation

`error_vs_width_noise.png`. Same panel layout, all uniform centers, four curves = {uniform, random} samples $\times$ {clean $y$, noisy $y$ (std $10^{-3}$)}. Here random samples is the $x$-perturbation and noisy is the $y$-perturbation.

Best eval $L_\infty$ (mean over seeds):

| condition | $N{=}32$ | $64$ | $128$ | $256$ |
|---|---|---|---|---|
| uniform samp / clean | 1.5e-13 | 5.5e-14 | 1.4e-13 | 6.2e-13 |
| random samp / clean | 4.4e-13 | 2.4e-12 | 1.3e-12 | 6.7e-12 |
| uniform samp / noisy | 6.7e-4 | 8.6e-4 | 1.0e-3 | 1.4e-3 |
| random samp / noisy | 7.3e-4 | 8.5e-4 | 2.8e-3 | 6.6e-3 |

Read it as: clean curves at the floor, noisy curves at a much higher plateau near the noise magnitude; whether random sampling adds anything on top of the $y$-noise.

## Figure 3 -- sample-count scaling under y-noise

`error_vs_samples_noise.png`. Uniform centers, fixed $N = 64$ and $\lambda = 0.25$. Two panels (left relative $L_2$, right $L_\infty$), $x = $ number of uniform sample points (log), $y = $ error (log), one curve per $y$-noise level (clean plus $\sigma = 10^{-8} \ldots 10^{-2}$). Mean over seeds; shaded band = per-seed min--max. The question: with noise present, does adding data recover the clean floor?

Relative $L_2$ at three sample counts (mean over seeds):

| $\sigma$ | $n{=}256$ | $n{=}4096$ | $n{=}131072$ |
|---|---|---|---|
| clean | 1.8e-14 | 1.9e-14 | 1.8e-14 |
| 1e-8 | 7.5e-9 | 1.9e-9 | 3.6e-10 |
| 1e-6 | 7.5e-7 | 1.9e-7 | 3.6e-8 |
| 1e-4 | 7.5e-5 | 1.9e-5 | 3.6e-6 |
| 1e-2 | 7.5e-3 | 1.9e-3 | 3.6e-4 |

Read it as: the slope of each noisy line on the log-log axes (the rate at which more data buys accuracy), the constant vertical spacing between levels, and whether any noisy line bends down toward the clean floor.

## Conclusions

Plainly visible in the data:

- **Reaching machine precision is governed by the centers, not the sample points.** Both uniform-center curves sit at the fp64 floor ($\sim 10^{-13}$ eval $L_\infty$, $\sim 10^{-14}$ relative $L_2$) whether the samples are uniform or random; both random-center curves plateau $\sim 10^{-10}$--$10^{-11}$ whether the samples are uniform or random. This holds the exp07 finding and removes the sampling grid as the cause: random samples neither rescue random centers nor spoil uniform centers.
- **$x$-perturbation alone is essentially harmless.** With uniform centers and clean $y$, switching from uniform to random sample positions stays at the floor ($\sim 10^{-13}$--$10^{-12}$ eval $L_\infty$; the small rise at larger $N$ is within the band).
- **$y$-noise sets a hard recovery floor near the noise magnitude.** With $y$-noise std $10^{-3}$, the best eval error plateaus at $\sim 2$--$7 \times 10^{-4}$ relative $L_2$ and $\sim 7 \times 10^{-4}$--$10^{-3}$ eval $L_\infty$ -- roughly the noise level, $\sim 10$ orders above the clean floor -- and rises slowly with $N$.

From figure 3:

- **Figure 3 exposes a scaling law with data size, and it holds no surprises:** the noise error falls as $1/\sqrt{n}$, exactly as statistics predicts. Each noisy line falls along a straight log-log line of slope $\approx -0.48$ (i.e. $\propto n^{-1/2}$) across the full range $256 \to 131072$, with no plateau over the tested range.
- **The lines are evenly spaced by $\sigma$:** at fixed $n$ the relative $L_2$ error is $\approx 0.75\,\sigma\,(256/n)^{1/2}$ -- exactly proportional to the noise std (e.g. $7.5 \times 10^{-3}$, $7.5 \times 10^{-4}$, $\ldots$ at $n{=}256$ for $\sigma = 10^{-2}, 10^{-3}, \ldots$).
- **The clean line stays at the floor** ($\sim 1.8 \times 10^{-14}$ relative $L_2$) independent of sample count.
- **Recovering machine precision from finite noise is not blocked in principle, but is infeasible in practice:** because error $\propto \sigma\,n^{-1/2}$, driving $\sigma = 10^{-2}$ down to the clean floor would require $n \sim (\sigma/\text{floor})^2 \approx 10^{23}$ samples; each $100\times$ in data buys only $\sim 10\times$ in accuracy.
