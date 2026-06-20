# Exp10 -- 2D ridge/Radon regime: hexagonal tangent-line geometry

## Question

Extend the 1D "fixed geometry + least-squares readout" result (exp03/0B/exp12) to $\mathbb{R}^2 \to \mathbb{R}$. Each $\tanh$ neuron is a ridge (plane wave) whose transition line is pinned tangent to a circle at a hexagonally-packed grid point. With the geometry frozen, we build the feature matrix $\Phi$ and solve the readout by least squares (the method exp03/0B showed matches or beats the QI construction in fp64), then sweep the bandwidth and width to ask: does this structured, training-free geometry reach the fp64 floor on smooth 2D targets, and how does the error scale with width $N$?

## Construction

Network: $\hat f(x) = b_0 + \sum_{m=1}^{N} a_m \tanh(W_m^\top x + \beta_m)$, $x \in \mathbb{R}^2$. For a hex grid point $c_m$ with radius $r_m = \lVert c_m \rVert$ and outward unit normal $\hat n_m = c_m / r_m$, the first-layer row is $W_m = \gamma\,\hat n_m$ and the bias is $\beta_m = -\gamma\,r_m$. The preactivation is then $\gamma(\hat n_m^\top x - r_m) = \gamma\cdot\operatorname{signed\text{-}dist}(x, \ell_m)$ where $\ell_m = \{\hat n_m^\top x = r_m\}$ is the line tangent to the circle of radius $r_m$ at $c_m$. So each neuron is $\tanh(\gamma \cdot \text{distance to its tangent line})$ -- a smoothed step across that line. Every first-layer row has norm exactly $\gamma$; only the bias sets the offset. This is the 2D analog of the 1D center $x_m$: a signed scalar position becomes a (direction, offset) pair $(\hat n_m, r_m)$, a point in the Radon/line domain.

Code: `src/construction/hex_geometry.py` (`hex_pack`, `tangent_geometry`, `build_hex_geometry`, `build_phi_2d`), `src/data/targets2d.py`, `src/data/sampling2d.py`. The readout solve reuses `src/construction/readout.py::solve_readout_with_bias` unchanged. Verified by `tests/test_hex_geometry.py` (43 tests: exact-$N$ packing within radius, min-spacing compactness, $W_m^\top c_m + \beta_m = 0$, $W_m \parallel c_m$, $\lVert W_m \rVert = \gamma$, $\lambda \leftrightarrow \gamma$ round-trip, the $\Phi$ formula, and an in-span recovery control to $<10^{-9}$).

### Packing

Disk radius $R = 1 + \text{halo}$ (default halo $0.4$, so $R = 1.4$). Centered hexagonal rings: ring $k$ holds $6k$ points, so $K$ complete rings give the centered-hexagonal number $H_K = 1 + 3K(K{+}1)$ (1, 7, 19, 37, 61, ...). For arbitrary $N$, fill the largest $K$ complete rings with $H_K \le N$, then place the remaining $N - H_K$ points at uniform angles on the partial outer ring. The spacing $d$ is set so the outermost occupied ring lands at radius $R$, so the packing grows denser as $N$ increases. The $r = 0$ center point (undefined normal) takes an arbitrary fixed direction $(1,0)$ with $\beta = 0$. Targets are fit and evaluated on the unit disk (radius 1); neurons extend into the collar out to $R$.

### Bandwidth (dual parameterization)

With hex spacing $d = d(N)$ the dimensionless bandwidth is $\lambda = \gamma d$ (the 1D analog of $\lambda = \gamma h$). The builder accepts either $\lambda$ (default; $\gamma = \lambda/d$) or $\gamma$ ($\lambda = \gamma d$); the other is backed out from $d$. Fixing $\lambda$ makes $\gamma \propto \sqrt N$ grow automatically with width. The driver fixes $\lambda$ and sweeps it by default; set `SWEEP_VAR = "gamma"` to swap roles.

## What the code runs

`experiments/exp10_radon_hex2d/run.py` (config documented in `config.yaml`). All fp64, numpy.

- Targets (6): `sine2d` $=\sin(\pi x)\cos(\pi y)$, `sine2d_hi` $=\sin(4\pi x)\cos(4\pi y)$, `gauss_bump` $=e^{-(x^2+y^2)}$, `runge2d` $=1/(1+25(x^2+y^2))$, `mixed2d` $=e^{\sin(\pi x)+\cos(\pi y)}$, `planewave` $=\tanh(3(0.6x+0.8y))$.
- Widths $N$: $\{37, 61, 100, 127, 271, 547\}$ (centered-hex numbers plus two partial-ring $N$).
- Bandwidth sweep: $\lambda \in \{0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.60, 0.90, 1.40\}$.
- Train points: $8000$ area-uniform samples on the unit disk (`disk_uniform`, seed 0). Eval: a $160\times160$ Cartesian grid clipped to the unit disk ($\approx 19{,}856$ points; `disk_grid`).
- Per cell ($\text{target} \times N \times \lambda$) it builds $\Phi_\text{train}$, solves the readout by lstsq, and records: eval $L_\infty$ ($\max$ abs error on the eval grid), eval relative $L_2$ ($\lVert \text{resid}\rVert / \lVert y\rVert$ on the eval grid), the train residual norm, and conditioning diagnostics from the augmented $[\Phi, \mathbf 1]$ train matrix (cond, rank, null dim, stable rank with cutoff $10^{-13}\,s_{\max}$), plus $\max |v|$, $\gamma$, $\lambda$, $d$.
- $396$ cells, $\approx 52$ s total.

Metric note: eval $L_\infty$ on a finite grid is a lower bound on the true sup error -- the same convention as the 1D experiments.

## Results

Data: `results/exp10_radon_hex2d/phase1_data.json`. Figures in the same directory.

Best eval $L_\infty$ over the $\lambda$ grid, per target and width:

| target | $N{=}37$ | $61$ | $100$ | $127$ | $271$ | $547$ |
|---|---|---|---|---|---|---|
| sine2d | 5.2e-3 | 1.3e-4 | 8.9e-6 | 7.8e-7 | 1.2e-7 | 1.8e-8 |
| gauss_bump | 4.4e-5 | 2.9e-7 | 2.6e-9 | 3.4e-9 | 7.4e-11 | 1.4e-10 |
| mixed2d | 2.9e-1 | 9.0e-2 | 1.3e-2 | 4.3e-3 | 1.3e-4 | 4.5e-6 |
| planewave | 5.7e-2 | 1.7e-2 | 3.5e-3 | 1.7e-3 | 1.6e-4 | 1.0e-5 |
| sine2d_hi | 1.1e0 | 1.2e0 | 1.3e0 | 1.2e0 | 9.2e-2 | 8.4e-3 |
| runge2d | 3.6e-1 | 1.8e-1 | 1.2e-1 | 8.4e-2 | 2.5e-2 | 8.1e-3 |

Conditioning at the best-$L_\infty$ cell for `gauss_bump` (geometry-only; cond is target-independent): $N{=}100$ cond $3.3\times10^{12}$ (null dim 0), $N{=}271$ cond $1.4\times10^{16}$ (null dim 61), $N{=}547$ cond $3.6\times10^{17}$ (null dim 304). The best $\lambda$ drifts down with width ($0.30 \to 0.16 \to 0.12$).

### How to read the figures

**`error_vs_lambda.png`** -- 6 rows (targets) $\times$ 2 columns (relative $L_2$, $L_\infty$), one curve per width $N$, both axes log. For each target this traces error against the swept $\lambda$; the minimum of each curve is the best bandwidth at that width. Read it for the U-shape (high error at small $\lambda$ from ill-conditioning, high error at large $\lambda$ from aliasing) and for whether the minimum moves left as $N$ grows.

**`error_vs_width.png`** -- 2 panels (left relative $L_2$, right $L_\infty$), one curve per target, both axes log. Each point is the best-over-$\lambda$ error at that width. Read it as the scaling law: a descending line is exponential-in-width convergence; a flattening or upturn marks a floor.

**`conditioning.png`** -- 2 panels, one curve per width, log-$x$. Left: cond$(\Phi)$ vs $\lambda$. Right: null dim of $[\Phi, \mathbf 1]$ vs $\lambda$. cond is computed from the geometry only (target-independent), shown for the first target. Read it for how conditioning worsens toward small $\lambda$ and large $N$.

## Conclusions

Plainly visible in the data:

- The 2D hex tangent-line geometry with a plain fp64 lstsq readout produces a working approximation whose best-over-$\lambda$ error descends with width on the smooth targets: `gauss_bump` reaches $7.4\times10^{-11}$ (eval $L_\infty$, $N{=}271$) and `sine2d`, `mixed2d`, `planewave` descend monotonically across the tested widths. The qualitative 1D picture (exp03/0D) carries over to 2D.
- The bandwidth sweep is U-shaped in $\lambda$ with an optimum in the viable regime ($\approx 0.12$--$0.30$) that drifts downward as $N$ grows -- the same direction as the 1D least-squares optimum (exp06).
- cond$(\Phi)$ grows with $N$ and falls with $\lambda$; the null dimension of $[\Phi, \mathbf 1]$ grows with $N$ at fixed small $\lambda$. The readout is underdetermined in the same way as 1D (exp04), and the truncated lstsq still produces a low-error function.
- The smooth-target error has not reached machine $\varepsilon$ at the tested widths: `gauss_bump` bottoms at $N{=}271$ ($7.4\times10^{-11}$, cond $1.4\times10^{16}$) and is slightly worse at $N{=}547$ ($1.4\times10^{-10}$, cond $3.6\times10^{17}$). The non-monotonic tail coincides with cond$(\Phi)$ crossing $\sim 10^{16}$--$10^{17}$.
- `sine2d_hi` and `runge2d` remain far from the floor at all tested widths (high-frequency / steep targets, resolution-limited).

Proposed, not yet established (pending review / further runs):

- That the $N{=}547$ tail for `gauss_bump` is the fp64 cancellation floor (cond $\sim 10^{17}$) rather than a too-coarse $\lambda$ grid. An mpmath solve and a finer $\lambda$ grid near the optimum would separate these. The driver isolates the readout solve behind `solve_readout_with_bias`, so an extended-precision path is a localized change.
- That this regime can reach the fp64/eps floor in 2D the way the 1D construction does. Not demonstrated here; it would require pushing $N$ further and/or extended precision, and the resolution-limited targets need larger $N$.
