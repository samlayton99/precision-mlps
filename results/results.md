# Results

Cross-experiment conclusions. Per-experiment detail (setup, figures, raw data) lives in each `results/<exp>/`.

## Status -- heading into the next phase

Experiments 00 through 0D characterize the QI construction and the least-squares readout on a fixed geometry. Once the geometry (centers and $\gamma$) is fixed in a reasonable regime, recovering the readout is a fast, fp64-accurate, convex linear solve, though an underdetermined one; the open difficulty is choosing the geometry, not computing the weights on it. The next phase (`experiments/exp03_geometry_ladder/`) moves to trained optimizers: relax the fixed geometry one constraint at a time and, at each rung, test first whether a machine-precision solution still exists (exact least-squares) and then whether an optimizer can reach it.

## Checkpoint 1 -- experiments 00-0D

The QI construction provides an explicit, training-free set of weights, obtained by a Toeplitz solve for cardinal coefficients followed by a convolution; exp00 confirms it reaches its claimed precision (and that the halo, the choice of solver, and solving $\Phi$ directly rather than the normal equations all matter). exp01 confirms the error-versus-$\lambda$ tradeoff the theory predicts. Comparing the construction against simply solving for the readout by least squares on the same geometry (exp0A) shows least squares is at least as accurate at equal precision, and substantially so in fp64. Asking whether the two reach the same solution (exp0B): they agree as functions to $\sim 10^{-12}$ and their weights coincide once the null space of the feature matrix is removed, but that null space is large ($\sim 110$ dimensions), so the readout is underdetermined. Probing where the null space comes from by changing the activation (exp0C), GELU's null space is larger still and grows linearly with width while tanh's stays roughly constant -- two different regimes. Finally, since least squares is the stronger method, exp0D checks its $\lambda$ tradeoff: it has a U-shape like QI, but where QI's optimum is fixed near $0.30$, the least-squares optimum moves -- rising with target frequency and falling with width.

### exp00 -- numerics sanity (`experiments/exp00_sanity/`)

- QI construction reaches its claimed precision: $\sim 5\times10^{-12}$ (fp64) and $\sim 2\times10^{-15}$ (mpmath) at $N=128$; the reported $L_\infty$ is stable across eval-grid density, so it is not an evaluation artifact.
- On fixed QI geometry, least-squares and truncated SVD recover the readout to $\sim 10^{-13}$ to $10^{-14}$; QR inflates the weights (max $|v| \sim 10^3$) and ridge / normal-equations solves are far worse ($\sim 10^{-4}$ to $10^{-6}$) because forming $\Phi^\top\Phi$ squares the condition number. Solving $\Phi$ directly is required.
- The halo is essential: using interior-only centers (no halo) collapses accuracy to $\sim 10^{-4}$ to $10^{-5}$.
- tanh evaluated in fp64 vs mpmath differs by $\sim 10^{-16}$; activation evaluation is not the bottleneck.

### exp01 -- lambda tradeoff (`results/exp01_lambda_tradeoff/`)

- The U-shaped error-vs-$\lambda$ curve is confirmed for both methods. QI's fp64 optimum is near $\lambda = 0.30$ ($\sim 5\times10^{-12}$ at $N=128$); its mpmath optimum is near $\lambda = 0.25$ ($\sim 2\times10^{-15}$).
- At equal arithmetic precision, the least-squares readout is at least as accurate as QI at every tested $\lambda$.
- Least-squares in fp64 reaches $\sim 10^{-13}$; the floor is fp64 cancellation, not the method. (The earlier reading of a single optimal $\lambda \approx 0.25$ for both methods is refined by exp0D.)

### exp0A -- QI vs learned readout, four-way (`results/exp0A_QI_vs_learn/`)

On identical QI geometry, at equal precision, least squares is more accurate than QI; the margin is largest in fp64. Eval $L_\infty$ on sine at $\lambda = 0.25$:

| $N$ | QI mpmath | QI fp64 | lstsq fp64 | lstsq mpmath |
|---|---|---|---|---|
| 32 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 64 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 96 | 2.7e-15 | 3.3e-11 | 2.8e-13 | 1.0e-15 |
| 128 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

QI fp64 is poor at $\lambda = 0.25$ because that is outside its fp64 regime ($\lambda \approx 0.30$); least squares is insensitive to this. The comparison is in eval $L_\infty$, a different norm from the least-squares training objective, so the result is empirical rather than definitional.

### exp0B -- coefficient closeness (`results/exp0B_coeff_diff/exp0B_results.md`)

- On a fixed geometry the QI and least-squares weight vectors differ substantially (largest per-coefficient deviation of order the typical coefficient or larger), but the entire difference lies in the $\sim 110$-dimensional null space of $[\Phi,\mathbf{1}]$ (at $\lambda=0.30$); the two functions agree to $\sim 10^{-12}$.
- The data-visible (row-space) coefficients coincide; the readout is therefore underdetermined -- the same function is represented by many weight vectors that differ only in the null space.

### exp0C -- activation conditioning, GELU vs tanh (`results/exp0C_gelu_conditioning/exp0C_results.md`)

The two activations sit in different regimes (effective rank and null dimension of $[\Phi,\mathbf{1}]$ at $\lambda = 0.22$): tanh's null space is roughly constant in $N$ (rank $\approx N$), while GELU's grows linearly with $N$ (rank $\approx 0.43\,N$).

| $N$ | tanh rank | tanh null | GELU rank | GELU null |
|---|---|---|---|---|
| 32 | 44 | 150 | 26 | 168 |
| 64 | 76 | 150 | 41 | 185 |
| 96 | 106 | 152 | 56 | 202 |
| 128 | 138 | 152 | 71 | 219 |

- At a fixed geometry tanh reaches the fp64 floor ($\sim 10^{-13}$); GELU is one to three orders worse.
- Neither activation has a sharp $\lambda$ optimum for the least-squares fit; both are flat-bottomed until the high-$\lambda$ cancellation wall.

### exp0D -- optimal lambda vs frequency (`results/exp0D_lambda_frequency/exp0D_results.md`)

Both methods show a U-shape, but their optima behave differently: QI's optimum is fixed near $\lambda = 0.30$ regardless of frequency, while the least-squares optimum rises with target frequency (and, separately, falls with width). Optimal $\lambda$ at $N = 128$:

| $\sin(k\pi x)$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}8$ | $k{=}16$ |
|---|---|---|---|---|---|
| QI | 0.30 | 0.32 | 0.30 | 0.29 | 0.28 |
| lstsq | 0.09 | 0.17 | 0.29 | 0.27 | 0.25 |

### Open questions

- Whether reaching machine precision as $N$ grows requires $\gamma$ proportional to $N$, or whether a fixed $\gamma$ matched to the target bandwidth suffices. exp0D shows least-squares fitting a fixed-frequency target with $\gamma$ roughly constant in $N$, which raises the possibility that $\gamma \propto N$ is a feature of the QI construction rather than a precision requirement; this is not established. A fixed-$\gamma$, sweep-$N$ test would settle it.
- Whether there is a precision scaling law: independent results suggest that on a log-log plot the error descends along a straight line as $N$ increases, shifting downward until it saturates at a precision floor. Its slope, dependence on activation and geometry, and floor location are uncharacterized.
- Why the GELU and tanh null spaces fall in different regimes (constant vs linearly growing). The mechanism is not established.
- Whether a standard optimizer can discover the geometry -- the central question for the geometry ladder.

### Experiment cost (rough)

All fp64 work is milliseconds per config at $N=128$: a least-squares solve $\sim 5$ ms, a QI construction $\sim 1$ ms ($\sim 0.4$ ms with cached coefficients), so whole fp64 sweeps (hundreds of configs) finish in tens of seconds. The only slow path is mpmath: a fresh QI Toeplitz solve $\sim 1$ minute (its coefficients are target-independent and cache to sub-second), and an mpmath least-squares SVD is tens of seconds per config.
