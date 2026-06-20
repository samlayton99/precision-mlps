# Exp07 -- 1D center-geometry comparison (least-squares readout, tanh)

## Question

In 1D, the uniform QI grid with a least-squares readout reaches machine precision (exp03: lstsq fp64 on sine $\sim 10^{-13}$). How much of that depends on the centers being placed *uniformly*? We compare four center geometries head-to-head, all solved by the same fp64 least-squares readout, all sharing the same $\gamma = \lambda/h$ at each swept $\lambda$, on the same target. Only the placement of the $\tanh$ centers differs, so any gap between geometries is attributable to placement alone.

## Geometries

Each places $W$ centers over a common span:

- **uniform** -- equispaced (the QI grid baseline; reaches $\sim 10^{-13}$ on sine).
- **random** -- uniform random over the span.
- **clustered** -- two-stage pseudo-clustered: a few uniform meta-centers, then Gaussian draws around them (overlapping, so there are no super-large gaps).
- **trained** -- centers $x_0 = -b/w$ extracted from a trained width-$W$ single-hidden-layer $\tanh$ net; everything else (the trained $\gamma$ and readout) is thrown away. Dead neurons ($|w|$ below $10^{-8}$) are parked at a span endpoint and all centers are clipped to the span.
- **reg_clustered** -- a regular grid broken into evenly-spaced clusters of `cluster_size` points (default $8$), each compressed toward its cluster center by a geometric `ratio` (default $0.75$). The $m$-th of a cluster's $n{-}1$ gaps is $s\cdot\text{ratio}^{\min(m,\,n-2-m)+1}$ with $s$ the regular spacing, so the innermost gaps are smallest and clusters tile with center spacing $n\,s$. $\text{ratio}=1$ is exactly uniform; smaller ratio clumps each cluster tighter and opens voids between clusters. Deterministic.

Code: `src/construction/center_geometry.py`, verified by `tests/test_center_geometry.py` (21 tests: exact count, sorted, within-span, equispacing of uniform, seed reproducibility of random, clustered coverage / no-large-gaps and clumpiness vs uniform, the $x_0 = -b/w$ extraction including dead-neuron handling, and for reg_clustered: the exact spec example $(n{=}7,\text{ratio}{=}0.5,s{=}1)$, count/span, $\text{ratio}{=}1\Rightarrow$ uniform, and more clumping at smaller ratio). The readout solve reuses `src/construction/readout.py::solve_readout_with_bias` unchanged.

## Apples-to-apples setup

For each base grid size $N$ the **uniform** QI geometry (grid + halo, halo sized at the reference $\lambda = 0.25$) fixes three things shared by all four geometries at that $N$: the total width $W = N + 2R + 1$, the center span $[\,x_{\min}, x_{\max}\,]$, and $h = 2/N$. The other three geometries place exactly $W$ centers over the same span, and every geometry uses $\gamma = \lambda/h$ at each swept $\lambda$. Thus at a given $(N, \lambda)$ the four runs differ only in where the centers sit.

The **trained** net is trained on the target over $[-1, 1]$ (Adam, $4000$ steps, lr $10^{-3}$); its inner-layer $(w, b)$ give the centers. Its centers therefore mostly land inside $[-1, 1]$, with the halo region of the span sparsely populated.

## What the code runs

`experiments/exp07_center_geometry/run.py` (config documented in `config.yaml`). All fp64.

- Target: `sine` $= \sin(\pi x)$ on $[-1, 1]$.
- Base grid sizes $N \in \{32, 64, 128, 256\}$, giving widths $W \in \{173, 205, 269, 461\}$.
- Bandwidth sweep: $\lambda \in \{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00\}$.
- Seeds: $\{0, 1, 2\}$ for the stochastic geometries (random, clustered, trained); uniform and reg_clustered are deterministic (one seed).
- reg_clustered parameters: `cluster_size` $= 8$, `cluster_ratio` $= 0.75$.
- Train: $1024$ equispaced points on $[-1, 1]$. Eval: $4096$ equispaced points on $[-1, 1]$.
- Per cell ($\text{geometry} \times N \times \text{seed} \times \lambda$) it builds $\Phi$, solves the readout by lstsq, and records eval $L_\infty$ ($\max$ abs error), eval relative $L_2$ ($\lVert \text{resid}\rVert / \lVert y\rVert$), and cond of the augmented $[\Phi, \mathbf 1]$ train matrix, plus $\gamma$, $h$, $\lambda$. Seed-0 center positions are also saved for the number-line figure.
- Total runtime $\approx 81$ s (the trained geometry's NN training dominates).

The readout solve is isolated in `solve_geometry()` and the precision is a single flag (`PRECISION`), so an extended-precision (mpmath) rerun is a localized change.

## Results

Data: `results/exp07_center_geometry/data.json`. Figure: `results/exp07_center_geometry/error_vs_width.png`.

Best error over the $\lambda$ sweep, **mean over seeds**:

Eval $L_\infty$:

| $W$ | uniform | random | clustered | trained | reg_clustered |
|---|---|---|---|---|---|
| 173 | 8.9e-14 | 1.9e-10 | 1.4e-10 | 1.0e-11 | 2.5e-11 |
| 205 | 4.0e-14 | 1.9e-11 | 5.6e-11 | 9.7e-12 | 3.4e-11 |
| 269 | 2.2e-13 | 1.3e-11 | 1.9e-11 | 7.5e-12 | 2.0e-12 |
| 461 | 2.8e-13 | 2.2e-11 | 1.2e-11 | 7.0e-9 | 1.0e-12 |

Eval relative $L_2$:

| $W$ | uniform | random | clustered | trained | reg_clustered |
|---|---|---|---|---|---|
| 173 | 2.3e-14 | 4.9e-11 | 3.6e-11 | 3.3e-12 | 8.5e-12 |
| 205 | 9.1e-15 | 5.9e-12 | 2.2e-11 | 2.2e-12 | 1.3e-11 |
| 269 | 2.4e-14 | 5.8e-12 | 1.1e-11 | 3.3e-12 | 1.9e-12 |
| 461 | 2.4e-14 | 4.8e-12 | 6.0e-12 | 2.5e-9 | 5.2e-13 |

cond of $[\Phi, \mathbf 1]$ at the best-$L_\infty$ cell (seed 0):

| $W$ | uniform | random | clustered | trained | reg_clustered |
|---|---|---|---|---|---|
| 173 | 6.7e19 | 1.8e19 | 1.3e33 | 8.0e30 | 1.9e19 |
| 205 | 5.5e19 | 1.7e20 | 1.4e29 | 5.4e34 | 3.7e19 |
| 269 | 7.3e20 | 4.5e19 | 3.7e33 | 2.9e25 | 2.0e20 |
| 461 | 9.4e19 | 1.7e20 | 1.3e33 | 2.4e36 | 2.1e20 |

Seed spread (illustrates the stochastic geometries' variance): trained at $W = 461$ ranges from $1.9 \times 10^{-11}$ to $2.1 \times 10^{-8}$ in eval $L_\infty$ across the three seeds (mean $7.0 \times 10^{-9}$); the mean is pulled up by one seed. The uniform optimum $\lambda$ drifts down with width ($0.25, 0.25, 0.15, 0.10$ for $N = 32, 64, 128, 256$), consistent with exp06.

### How to read the figures

**`error_vs_width.png`** -- two panels (left relative $L_2$, right $L_\infty$), $x = W$ (total centers, log), $y = $ best-over-$\lambda$ error (log), one curve per geometry. Each point is the mean over seeds; the shaded band spans the per-seed min--max (uniform and reg_clustered have no band -- one seed). Read it as: how low can each placement strategy get at each width, and does it improve as width grows. A curve sitting at $\sim 10^{-13}$ is at the fp64 floor; a curve flat at $\sim 10^{-11}$ is plateaued above it.

**`centers_numberline.png`** -- one panel per geometry ($2 \times 3$ grid, last cell blank), each stacking the seed-0 center positions for the four widths as ticks on number lines, least dense ($W = 173$) at top to most dense ($W = 461$) at bottom; the two dotted verticals mark the unit domain $[-1, 1]$. Note the span widens for smaller $N$ (the halo is sized in grid nodes, and $h = 2/N$ is larger), so the top rows are both wider and sparser. Read it to see the placement structure: uniform's even ticks, random's irregular gaps, reg_clustered's periodic clusters, and trained's concentration inside $[-1, 1]$ with sparse outliers in the halo.

**`conditioning.png`** -- a single panel, $x = W$ (log), $y = \mathrm{cond}(\Phi)$ (log, augmented matrix) at each geometry's best-$L_\infty$ cell (seed 0), one curve per geometry. A light-touch diagnostic only.

## Conclusions

Plainly visible in the data:

- **Only the uniform geometry reaches machine precision.** It sits at $\sim 10^{-13}$--$10^{-14}$ (both metrics) across all widths, roughly flat.
- **All four non-uniform geometries plateau above uniform.** random and clustered sit $\sim 10^{-11}$--$10^{-12}$; reg_clustered (mild clustering, ratio $0.75$) is similar at small width and reaches $\sim 10^{-12}$ ($1.0\times10^{-12}$ eval $L_\infty$) at $W = 461$, the best of the non-uniform geometries but still $\sim 4\times$ above uniform; trained is comparable to or slightly better than random/clustered at the smaller widths but is high-variance across seeds and degrades at $W = 461$ (mean $7 \times 10^{-9}$ eval $L_\infty$, one seed at $2 \times 10^{-8}$). None reach the uniform floor at any tested width.
- Because $W$, the span, and $\gamma$ are held identical across geometries at each $(N, \lambda)$, the gap is attributable to **center placement alone**: with everything else equal, the least-squares-to-machine-precision behavior is specific to uniform placement, not generic to the least-squares readout.
- **The plateau does not track $\mathrm{cond}(\Phi)$.** uniform, random, and reg_clustered all have $\mathrm{cond}(\Phi) \sim 10^{19}$--$10^{20}$ at their best cell, yet random's error is 2--3 orders worse than uniform's. clustered and trained have much larger and more erratic cond ($10^{25}$--$10^{36}$) but are not systematically worse than random. So feature-matrix conditioning, at least as measured by cond of the augmented train matrix, does not explain the accuracy gap.
