# Exp11 -- 2D ridge-geometry zoo: which geometry reaches precision?

**Status: draft -- pending Sam's review and sign-off on conclusions.**

## Question

Exp10 showed one 2D ridge geometry (hex-packed points, radial tangent lines) with a frozen-geometry fp64 least-squares readout descending with width but bottoming around $\sim 10^{-10}$ on the best smooth target. This experiment generalizes exp10 to a head-to-head zoo of SIX 2D ridge geometries on a four-target suite, all fp64 lstsq with $\lambda$ finetuned per cell, and asks: of these candidate geometries, which (if any) reaches the fp64 floor ($\sim 10^{-14}$) on a smooth 2D target, and how does each scale with width $N$ up to $8192$? Each neuron is a ridge $\tanh(\gamma(w_m^\top x - t_m))$ -- a smoothed step across the line at signed distance $t_m$ from the origin in unit direction $w_m$. A geometry is just a recipe for the $(w_m, t_m)$ pairs (the Radon/line view), so the comparison is purely "where do you place the ridges".

## Geometries

All six (`experiments/exp11_geometry_zoo_2d/geometries.py`) return `(dirs[M,2], offsets[M])`. Three are point-based (place disk points $c$, then take the radial tangent rule $w = c/\lVert c\rVert$, $t = \lVert c\rVert$); the last two place lines directly in $(\theta, t)$.

- `hex` (1) -- hex-packed disk points (`src/construction/hex_geometry.py::hex_pack`), radial tangent ridge per point. The exp10 geometry.
- `mode_radial` (2) -- exact-$N$ mode radial grid: $\sqrt{\cdot}$-spaced rings, DP-allocated angular counts (biased toward divisor-friendly counts), staggered ring phases, tangents.
- `sixn_rings` (4) -- exactly $6k$ points on ring $k$, $\sqrt{\cdot}$-spaced radii, all rings phase-aligned to the same axes (symmetry-first), tangents.
- `random_ridges` (5) -- random disk points (seed 0); each point emits 3 ridges at $0/60/120^\circ$ off a random per-point base orientation (directions decoupled from position).
- `radon_tensor` (6a) -- uniform tensor grid in the Radon domain: $J$ equispaced directions $\times$ $M$ uniform cell-centered signed offsets over $[-R, R]$, directions offset off the axes.
- `radon_interlaced` (6b) -- same tensor grid, but the offset grid is shifted by half a step on alternate directions (Natterer-style efficient/hex sampling).

Per-geometry halo (option B, node-count style). The point/tangent geometries (1, 2, 4, 5) take a modest collar $R = 1.6$ (`TANGENT_R`), empirically near-optimal. The Radon geometries (6a, 6b) are 1D-per-direction and take a larger linear (node-count) halo $R = 2.5$ (`RADON_R`), sized via `_jm` so the INTERIOR $[-1,1]$ offset resolution (and the direction count $J$) matches the tangent geometries' interior resolution ($\approx 0.63\sqrt{N}$), with the extra offsets forming the per-direction halo. The dimensionless bandwidth uses a COMMON reference spacing for every geometry at a given width, $\gamma = \lambda / h_{\mathrm{ref}}$ with $h_{\mathrm{ref}} = 2.8/\sqrt{N}$, so $\lambda$ is comparable across geometries (interior resolution is matched by construction). Note: because $J\cdot M$ only approximates $N$, the Radon geometries place slightly more than $N$ ridges (e.g. $260$ at $N{=}256$, $8208$ at $N{=}8192$); all others hit $N$ exactly.

## What the code runs

`experiments/exp11_geometry_zoo_2d/run.py` (main suite) plus `extra_analysis.py` (the $N{=}8192$ cells, the $\lambda$-sweep figure, and the N-sweep animation). All fp64, numpy.

- Targets (4): `gauss_bump` $=e^{-(x^2+y^2)}$ and `runge2d` $=1/(1+25(x^2+y^2))$ (radially symmetric); `sine2d` $=\sin(\pi x)\cos(\pi y)$ and `mixed2d` $=e^{\sin(\pi x)+\cos(\pi y)}$ (asymmetric). Fit and evaluated on the unit disk; ridges live out to radius $R$.
- Widths $N$: $\{64, 144, 256, 576, 1024, 2048, 4096\}$ from `run.py`, plus $8192$ appended by `extra_analysis.py --add8192`.
- $\lambda$ grid (main suite): $\{0.05, 0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.80, 1.20\}$; the best (min eval $L_\infty$) cell is kept per $(\text{geom}, N, \text{target})$. The $8192$ cells use a focused grid $\{0.10, 0.15, 0.22, 0.32, 0.45\}$ (optima cluster there). The $\lambda$-sweep figure and animation use a finer $14$-point grid.
- Train: $8000$ area-uniform points on the unit disk (`disk_uniform`, seed 0); the $8192$ cells raise this to $12000$ for overdetermination. Eval: a $120\times120$ Cartesian grid clipped to the unit disk ($\approx 11\text{k}$ points; `disk_grid`).
- Per cell it builds $\Phi_{\text{train}}$, augments with a bias column $[\Phi, \mathbf 1]$, solves by truncated SVD (cutoff $10^{-13}\,s_{\max}$), and records eval $L_\infty$ ($\max$ abs error on the eval grid), eval relative $L_2$ ($\lVert \text{resid}\rVert/\lVert y\rVert$), and the chosen $\lambda$, $\gamma$. The exact solved MLP ($f(x) = \text{bias} + \sum_m v_m \tanh(\gamma(w_m^\top x - t_m))$, plus dirs/offsets/radius) is saved per cell to `weights/{geom}_N{N}_{target}.npz`.
- $6$ geometries $\times$ $8$ widths $\times$ $4$ targets $= 192$ rows in `data.json`.

Metric note: eval $L_\infty$ on a finite grid is a lower bound on the true sup error -- the same convention as the 1D and exp10 experiments.

## Results

Data: `results/exp11_geometry_zoo_2d/data.json` (keys `config`, `rows`, `lambda_sweep`; each row has `geom`, `N`, `target`, `linf`, `rel_l2`, `lambda`, `gamma`, `n_ridges`, `weights_file`). Figures in the same directory.

Best eval $L_\infty$ over the $\lambda$ grid and over all six geometries, per target and width (the winning geometry in parentheses):

| target | $N{=}64$ | $144$ | $256$ | $576$ | $1024$ | $2048$ | $4096$ | $8192$ |
|---|---|---|---|---|---|---|---|---|
| gauss_bump | 1.5e-7 (hex) | 8.8e-11 (sixn) | 1.5e-12 (radon-il) | 5.8e-14 (radon-il) | 1.4e-14 (radon-t) | 1.1e-14 (radon-t) | 9.6e-14 (radon-t) | 5.9e-14 (radon-t) |
| runge2d | 1.6e-1 (sixn) | 6.8e-2 (random) | 1.9e-2 (random) | 2.1e-3 (random) | 7.8e-4 (random) | 6.8e-5 (random) | 3.0e-6 (random) | 7.9e-9 (random) |
| sine2d | 9.7e-5 (hex) | 9.7e-8 (sixn) | 2.0e-11 (radon-t) | 2.2e-10 (sixn) | 2.9e-12 (radon-t) | 3.9e-12 (radon-t) | 1.5e-11 (radon-il) | 7.5e-12 (sixn) |
| mixed2d | 7.4e-2 (sixn) | 1.5e-3 (hex) | 2.2e-5 (random) | 1.0e-7 (random) | 2.5e-9 (random) | 5.2e-10 (random) | 2.4e-10 (radon-il) | 2.9e-11 (radon-t) |

At $N{=}8192$, all six geometries (eval $L_\infty$): `gauss_bump` -- radon-tensor 5.9e-14, radon-interlaced 7.1e-14, the four point geometries 3.6e-12 to 7.6e-12 (best non-Radon: mode_radial 3.6e-12); `sine2d` -- sixn 7.5e-12, radon-il 8.4e-12, radon-t 8.4e-12, the rest 1.3e-11 to 3.7e-11; `mixed2d` -- radon-t/radon-il both 2.9e-11, sixn 5.4e-11, the rest 7.6e-11 to 1.7e-10; `runge2d` -- random_ridges 7.9e-9, radon-t 2.6e-8, mode_radial/sixn $\sim 1$--$2\text{e-}6$, hex 9.7e-5.

### How to read the figures

`geometry_suite.png` (the deliverable) -- a $4\times2$ grid: 4 targets (rows: `gauss_bump`, `runge2d`, `sine2d`, `mixed2d`) $\times$ 2 metrics (cols: relative $L_2$, $L_\infty$), one colored line per geometry, both axes log, $x = $ width $N$. Each point is the best-over-$\lambda$ error at that width. Read each panel for which geometry is lowest and for the scaling slope: a steadily descending line is fast (exponential-in-width) convergence; a flattening marks a floor. The `gauss_bump` row is where the two Radon lines (red/orange) separate from the four point geometries and reach the bottom of the plot near $\sim 6\text{e-}14$; the `runge2d` row stays high for every geometry; the `mixed2d` row descends steadily for all.

`error_vs_lambda.png` -- same $4\times2$ layout, but at fixed $N{=}1024$ with one line per geometry over a finer 14-point $\lambda$ grid (both axes log, $x = \lambda = \gamma\,h_{\mathrm{ref}}$). Read it for the U-shape: error rises at small $\lambda$ (ill-conditioning) and at large $\lambda$ (aliasing), with a minimum in between. It locates each geometry's best $\lambda$ at this width and shows how flat/sharp the optimum is (the Radon lines are notably flatter and lower on `gauss_bump` and `mixed2d`).

`error_vs_lambda_anim.html` / `.mp4` / `.gif` -- the same error-vs-$\lambda$ figure animated as $N$ sweeps $\{64 \to 4096\}$, one frame per width, with y-limits fixed across frames so the descent is visible. The `.html` is the pausable interactive player (play/pause/step/loop); the `.mp4`/`.gif` are the quick-look versions. Read it for how each geometry's U-shape minimum drops and drifts as width grows.

`targets_3d.png` -- 3D surface plots of the four targets over the unit disk (crimson ring = disk boundary, dots = training samples). Read it for shape/difficulty: `gauss_bump` and `mixed2d` are smooth; `runge2d` is a sharp central spike; `sine2d` is a smooth oscillation.

`geometry_viz_sample.png` -- a $2\times3$ snapshot of all six ridge geometries at $N{=}256$ (points/lines on the disk, fit boundary in black, halo boundary outer). Read it to see the spatial layout: hex/mode-radial/sixn rings vs random scatter vs the concentric Radon offset circles.

`geometry_viz.ipynb` -- interactive notebook with two views: `plot_geometries(N)` (the exact ridge geometry for all six at any width) and `plot_error_heatmap(geometries, functions, N)` (a pointwise $|f - f_{\text{true}}|$ heatmap on the disk for the saved best-$\lambda$ MLP, optionally with a 3D surface), so you can see WHERE each geometry interpolates well vs poorly.

## Conclusions

Plainly visible in the data (`data.json`, $N{=}8192$ unless noted):

- Of the six geometries, only `gauss_bump` under the Radon geometry reaches the fp64 floor: radon-tensor $5.9\times10^{-14}$ and radon-interlaced $7.1\times10^{-14}$ at $N{=}8192$ (best non-Radon geometry is mode_radial at $3.6\times10^{-12}$, $\sim 60\times$ worse). On `gauss_bump` the Radon geometries already hit $\sim 10^{-14}$ by $N{=}576$--$1024$ and stay there.
- The Radon geometries are best or tied for best at the largest width ($N{=}8192$) on 3 of the 4 targets: `gauss_bump` (radon-tensor), `sine2d` (radon-t/radon-il $\approx 8.4\times10^{-12}$, within a factor of the sixn winner $7.5\times10^{-12}$), and `mixed2d` (radon-t/radon-il both $2.9\times10^{-11}$, the lowest). On `runge2d` the Radon geometries are not best.
- `runge2d` stays resolution-limited at every width: the best geometry (random_ridges) only reaches $7.9\times10^{-9}$ at $N{=}8192$ and is far from the floor for all six at all widths (the central spike target).
- At small $N$ the Radon geometries underperform the point geometries: at $N{=}64$ the best overall is `hex`/`sixn` (e.g. `gauss_bump` hex $1.5\times10^{-7}$ vs radon-tensor $2.4\times10^{-5}$; `sine2d` Radon $\approx 3.1\times10^{-1}$, essentially failing), consistent with too few directions $J$ at small $N$. The Radon advantage appears only once $N$ is large enough.
- `random_ridges` is the best geometry on the two hardest non-Gaussian targets across most widths: it wins every `runge2d` width from $N{=}144$ up, and wins `mixed2d` from $N{=}256$ through $N{=}2048$ (radon-t edges it at $8192$).

Flagged (not independent evidence / not established here): the eval $L_\infty$ values are finite-grid lower bounds, so absolute floors are only approximate. The non-monotonic `gauss_bump` Radon tail ($1.1\text{e-}14$ at $N{=}2048 \to 9.6\text{e-}14$ at $N{=}4096 \to 5.9\text{e-}14$ at $N{=}8192$) sits at the fp64 floor and may be conditioning/grid noise rather than a real trend; this writeup does not record conditioning diagnostics (none are in `data.json`), so whether the floor is fp64 cancellation is not separated here. Whether the Radon geometry generalizes the 1D machine-$\varepsilon$ result to 2D beyond `gauss_bump`, and whether extended precision would push the other targets to the floor, is not demonstrated.
