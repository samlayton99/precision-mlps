# Results

Cross-experiment conclusions. Per-experiment detail (setup, figures, raw data) lives in each `results/<exp>/`.

## Status -- heading into the next phase

Experiments 00 through 0D characterize the QI construction and the least-squares readout on a fixed geometry. Once the geometry (centers and $\gamma$) is fixed in a reasonable regime, recovering the readout is a fast, fp64-accurate, convex linear solve, though an underdetermined one; the open difficulty is choosing the geometry, not computing the weights on it. The next phase (`experiments/exp12_geometry_ladder/`) moves to trained optimizers: relax the fixed geometry one constraint at a time and, at each rung, test first whether a machine-precision solution still exists (exact least-squares) and then whether an optimizer can reach it.

## Checkpoint 1 -- experiments 01-06

The QI construction provides an explicit, training-free set of weights, obtained by a Toeplitz solve for cardinal coefficients followed by a convolution; exp01 confirms it reaches its claimed precision (and that the halo, the choice of solver, and solving $\Phi$ directly rather than the normal equations all matter). exp02 confirms the error-versus-$\lambda$ tradeoff the theory predicts. Comparing the construction against simply solving for the readout by least squares on the same geometry (exp03) shows least squares is at least as accurate at equal precision, and substantially so in fp64. Asking whether the two reach the same solution (exp04): they agree as functions to $\sim 10^{-12}$ and their weights coincide once the null space of the feature matrix is removed, but that null space is large ($\sim 110$ dimensions), so the readout is underdetermined. Probing where the null space comes from by changing the activation (exp05), GELU's null space is larger still and grows linearly with width while tanh's stays roughly constant -- two different regimes. Finally, since least squares is the stronger method, exp06 checks its $\lambda$ tradeoff: it has a U-shape like QI, but where QI's optimum is fixed near $0.30$, the least-squares optimum moves -- rising with target frequency and falling with width.

### exp01 -- numerics sanity (`experiments/exp01_numerics_sanity/`)

- QI construction reaches its claimed precision: $\sim 5\times10^{-12}$ (fp64) and $\sim 2\times10^{-15}$ (mpmath) at $N=128$; the reported $L_\infty$ is stable across eval-grid density, so it is not an evaluation artifact.
- On fixed QI geometry, least-squares and truncated SVD recover the readout to $\sim 10^{-13}$ to $10^{-14}$; QR inflates the weights (max $|v| \sim 10^3$) and ridge / normal-equations solves are far worse ($\sim 10^{-4}$ to $10^{-6}$) because forming $\Phi^\top\Phi$ squares the condition number. Solving $\Phi$ directly is required.
- The halo is essential: using interior-only centers (no halo) collapses accuracy to $\sim 10^{-4}$ to $10^{-5}$.
- tanh evaluated in fp64 vs mpmath differs by $\sim 10^{-16}$; activation evaluation is not the bottleneck.

### exp02 -- lambda tradeoff (`results/exp02_lambda_tradeoff/`)

- The U-shaped error-vs-$\lambda$ curve is confirmed for both methods. QI's fp64 optimum is near $\lambda = 0.30$ ($\sim 5\times10^{-12}$ at $N=128$); its mpmath optimum is near $\lambda = 0.25$ ($\sim 2\times10^{-15}$).
- At equal arithmetic precision, the least-squares readout is at least as accurate as QI at every tested $\lambda$.
- Least-squares in fp64 reaches $\sim 10^{-13}$; the floor is fp64 cancellation, not the method. (The earlier reading of a single optimal $\lambda \approx 0.25$ for both methods is refined by exp06.)

### exp03 -- QI vs learned readout, four-way (`results/exp03_qi_vs_lstsq/`)

On identical QI geometry, at equal precision, least squares is more accurate than QI; the margin is largest in fp64. Eval $L_\infty$ on sine at $\lambda = 0.25$:

| $N$ | QI mpmath | QI fp64 | lstsq fp64 | lstsq mpmath |
|---|---|---|---|---|
| 32 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 64 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 96 | 2.7e-15 | 3.3e-11 | 2.8e-13 | 1.0e-15 |
| 128 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

QI fp64 is poor at $\lambda = 0.25$ because that is outside its fp64 regime ($\lambda \approx 0.30$); least squares is insensitive to this. The comparison is in eval $L_\infty$, a different norm from the least-squares training objective, so the result is empirical rather than definitional.

### exp04 -- coefficient closeness (`results/exp04_coeff_nullspace/exp04_results.md`)

- On a fixed geometry the QI and least-squares weight vectors differ substantially (largest per-coefficient deviation of order the typical coefficient or larger), but the entire difference lies in the $\sim 110$-dimensional null space of $[\Phi,\mathbf{1}]$ (at $\lambda=0.30$); the two functions agree to $\sim 10^{-12}$.
- The data-visible (row-space) coefficients coincide; the readout is therefore underdetermined -- the same function is represented by many weight vectors that differ only in the null space.

### exp05 -- activation conditioning, GELU vs tanh (`results/exp05_activation_conditioning/exp05_results.md`)

The two activations sit in different regimes (effective rank and null dimension of $[\Phi,\mathbf{1}]$ at $\lambda = 0.22$): tanh's null space is roughly constant in $N$ (rank $\approx N$), while GELU's grows linearly with $N$ (rank $\approx 0.43\,N$).

| $N$ | tanh rank | tanh null | GELU rank | GELU null |
|---|---|---|---|---|
| 32 | 44 | 150 | 26 | 168 |
| 64 | 76 | 150 | 41 | 185 |
| 96 | 106 | 152 | 56 | 202 |
| 128 | 138 | 152 | 71 | 219 |

- At a fixed geometry tanh reaches the fp64 floor ($\sim 10^{-13}$); GELU is one to three orders worse.
- Neither activation has a sharp $\lambda$ optimum for the least-squares fit; both are flat-bottomed until the high-$\lambda$ cancellation wall.

### exp06 -- optimal lambda vs frequency (`results/exp06_lambda_vs_frequency/exp06_results.md`)

Both methods show a U-shape, but their optima behave differently: QI's optimum is fixed near $\lambda = 0.30$ regardless of frequency, while the least-squares optimum rises with target frequency (and, separately, falls with width). Optimal $\lambda$ at $N = 128$:

| $\sin(k\pi x)$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}8$ | $k{=}16$ |
|---|---|---|---|---|---|
| QI | 0.30 | 0.32 | 0.30 | 0.29 | 0.28 |
| lstsq | 0.09 | 0.17 | 0.29 | 0.27 | 0.25 |

### Open questions

- Whether reaching machine precision as $N$ grows requires $\gamma$ proportional to $N$, or whether a fixed $\gamma$ matched to the target bandwidth suffices. exp06 shows least-squares fitting a fixed-frequency target with $\gamma$ roughly constant in $N$, which raises the possibility that $\gamma \propto N$ is a feature of the QI construction rather than a precision requirement; this is not established. A fixed-$\gamma$, sweep-$N$ test would settle it.
- Whether there is a precision scaling law: independent results suggest that on a log-log plot the error descends along a straight line as $N$ increases, shifting downward until it saturates at a precision floor. Its slope, dependence on activation and geometry, and floor location are uncharacterized.
- Why the GELU and tanh null spaces fall in different regimes (constant vs linearly growing). The mechanism is not established.
- Whether a standard optimizer can discover the geometry -- the central question for the geometry ladder.

### Experiment cost (rough)

All fp64 work is milliseconds per config at $N=128$: a least-squares solve $\sim 5$ ms, a QI construction $\sim 1$ ms ($\sim 0.4$ ms with cached coefficients), so whole fp64 sweeps (hundreds of configs) finish in tens of seconds. The only slow path is mpmath: a fresh QI Toeplitz solve $\sim 1$ minute (its coefficients are target-independent and cache to sub-second), and an mpmath least-squares SVD is tens of seconds per config.

## Checkpoint 2 -- experiments 07-08

Checkpoint 1 isolated the readout as a fast, accurate, but underdetermined linear solve on a fixed geometry, leaving the open question of which part of the problem actually controls precision. Experiments 07 and 08 answer that by varying, one at a time, the two ingredients of the linear solve -- where the tanh centers sit (the first-layer geometry) and where the target is sampled (the data fed to least squares) -- and then by perturbing the data with noise. The picture that emerges is clean: precision is controlled almost entirely by the first-layer geometry. Uniform center placement (the right geometry over the interval), together with a bandwidth $\lambda$ in the viable regime, is what lets least squares recover machine-precision coefficients; no other placement we tried does. The sampling, by contrast, barely matters -- least squares backs out the correct coefficients from uniform or random sample points alike, as long as the fit is overdetermined. Additive $y$-noise is the one thing that genuinely degrades recovery, and it does so in a fully predictable, statistical way.

### exp07 -- center-geometry comparison (`results/exp07_center_geometry/`)

Five center placements -- uniform, uniform-random, pseudo-clustered, a regular grid broken into geometric clusters, and centers extracted from a trained net -- are compared head-to-head with the same width, span, and $\gamma = \lambda/h$ at each $\lambda$, so only placement differs; each is solved by least squares with a $\lambda$ sweep on the same target (sine).

- Only uniform placement reaches machine precision ($\sim 10^{-13}$--$10^{-14}$, flat across width). Every non-uniform placement plateaus $\sim 10^{-11}$--$10^{-12}$ and does not descend toward the floor as width grows -- semiregular, well-interspersed random, mild geometric clustering, and trained-extracted centers all fail to recover the precision. The right geometry over the interval is required, not merely a well-spread one.
- The bandwidth keeps the U-shaped tradeoff of exp02/exp06: each reported error is the minimum of a $\lambda$ sweep, so reaching the floor needs both the right geometry and a viable $\lambda$.
- The plateau does not track $\mathrm{cond}(\Phi)$: uniform, random, and the regular-clustered placements all sit at $\mathrm{cond}(\Phi)\sim 10^{19}$--$10^{20}$, yet random's error is $2$--$3$ orders worse than uniform's. Feature-matrix conditioning, as measured here, is not the explanation for the gap.

### exp08 -- randomness and noise in the readout (`results/exp08_sampling_and_noise/`)

Overdetermined fp64 least squares (neurons $<$ sample points $<$ eval points, the latter two prime so the uniform grids do not align), varying center geometry, sample points, and $y$-noise on the same target.

- Precision is governed by the centers, not the sample points. With uniform centers the fit reaches the floor whether the sample points are uniform or random; with random centers it plateaus regardless of sampling. Perturbing $x$ alone (random sample positions on clean data) is essentially harmless -- least squares recovers the correct coefficients either way. This also retires the concern that exp07 favored uniform centers by always sampling on a uniform grid.
- Additive $y$-noise dominates: with noise std $10^{-3}$ the best achievable error plateaus near the noise magnitude ($\sim 10^{-4}$--$10^{-3}$), about ten orders above the clean floor.
- Adding data recovers precision along a predictable scaling law with no surprises. At fixed geometry and width, sweeping the sample count gives error $\propto \sigma\,n^{-1/2}$ (log-log slope $\approx -0.48$, lines evenly spaced by the noise std $\sigma$), exactly as statistics predicts. There is no plateau: precision can be bought back with data, but only at the $1/\sqrt{n}$ rate, so recovering many orders of magnitude is infeasible in practice (driving $\sigma=10^{-2}$ to the clean floor would need $n\sim 10^{23}$) -- it can be done, but only within the orders of magnitude of data one can afford.

### Open questions

- The causal mechanism of why uniform placement specifically reaches the floor is not pinned down. At the level of the theory, uniform spacing is what admits the quasi-interpolant representation -- the high-precision solution exists on that geometry and not (or not as accessibly) on the others -- but a sharper, quantitative account of why placements at comparable conditioning cannot recover it is still missing; $\mathrm{cond}(\Phi)$ is shown not to be the discriminating quantity.
- All of the above is fp64. An mpmath rerun would confirm whether the non-uniform plateaus and the noise floor are fp64-limited, but this is not a priority: the clean uniform result already sits at the fp64 floor established by the apples-to-apples mpmath-vs-fp64 comparison (exp03), so extended precision is expected to lower the clean floor without changing the geometry or noise conclusions.
