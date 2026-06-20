# Exp02 -- Lambda tradeoff (U-shaped error vs $\lambda$)

**Status: draft -- pending Sam's review and sign-off on conclusions.**

## Question

Does the error of the QI construction (and of the least-squares readout on the same geometry) trace the U-shaped curve in $\lambda = \gamma h$ that QI theory predicts -- aliasing at large $\lambda$, ill-conditioning at small $\lambda$ -- and where does the optimum sit for each method and each arithmetic precision?

Code: `experiments/exp02_lambda_tradeoff/` (`run.py`, `config.yaml`, `plot_consolidated.py`). Run the fp64 sweep with `python3 experiments/exp02_lambda_tradeoff/run.py`; the mpmath and fine sweeps with `--mpmath`, `--fine`, `--fine-fp64`; the consolidated figure with `python3 experiments/exp02_lambda_tradeoff/plot_consolidated.py`.

## What the code runs

All sweeps loop over four targets (`sine`, `runge`, `exp`, `sine_mixture`) and, for each $(N, \lambda)$, build a QI construction and a least-squares readout on the same geometry, then evaluate both on a grid of $N_{\text{eval}} = 2048$ equispaced points in $[-1, 1]$. Construction uses $K_c = 160$ cardinal coefficients and `halo = default_halo(N, lambda_star=lam)`; the geometry is centers at $-1 + n h$ with $h = 2/N$ and a single shared $\gamma = \lambda / h$. The least-squares readout is solved on a training grid of $\max(512, 4N)$ equispaced points via `solve_readout_with_bias(..., method="lstsq")` (the direct $\Phi$ solve, not normal equations), with a fitted bias. Metrics per config: $L_\infty = \max_x |\hat f(x) - f(x)|$ over the eval grid, and relative $L_2 = \lVert \hat f - f\rVert_2 / \lVert f \rVert_2$, reported separately for QI (`qi_linf`, `qi_rel_l2`) and least squares (`lstsq_linf`, `lstsq_rel_l2`).

The sweep variants, each writing its own data file:

- **fp64 coarse** (`data_fp64.json`, shipped as `data.json`): widths $N \in \{16, 32, 64, 128, 256\}$; $\lambda \in \{0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0\}$; QI in fp64. Spans both walls of the U (240 configs).
- **mpmath** (`data_mpmath.json`): widths $N \in \{128, 256, 512\}$; $\lambda \in \{0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40\}$; QI in mpmath (`mp_dps=50`), least squares in fp64 (84 configs).
- **fine fp64** (`data_fine_fp64.json`): widths $N \in \{16, 32, 64, 128\}$; $\lambda \in \{0.220, 0.225, \dots, 0.280\}$ (step $0.005$); QI and least squares both in fp64 -- a fine zoom on the viable bottom of the U (208 configs).
- **fine** (`data_fine.json`): same widths and fine $\lambda$ grid; QI in mpmath, least squares in fp64 (208 configs).

`plot_consolidated.py` merges `data.json`, `data_fine_fp64.json`, `data_mpmath.json`, `data_fine.json`, and the four-way `exp03_qi_vs_lstsq/data.json` into `all_data.json` (704 entries: 396 fp64/fp64, 260 mpmath/fp64, 48 mpmath/mpmath), excluding $N \in \{256, 512\}$, then draws the 3x4 consolidated figure.

## Results

Data files live in `results/exp02_lambda_tradeoff/`: `data.json` (fp64 coarse), `data_mpmath.json` (mpmath QI), `data_fine_fp64.json` and `data_fine.json` (fine zooms), and the merged `all_data.json`.

**QI fp64 optimum (`data.json`).** For the well-resolved targets the QI fp64 $L_\infty$ minimizes at $\lambda = 0.30$: `sine` reaches $9.2\times10^{-12}$ ($N{=}32$), $5.4\times10^{-12}$ ($N{=}64$), $5.5\times10^{-12}$ ($N{=}128$), $2.8\times10^{-12}$ ($N{=}256$); `exp` reaches $4.9\times10^{-13}$ ($N{=}64$) to $1.2\times10^{-12}$ ($N{=}128$). Across `sine`+`exp` the argmin is $\lambda=0.30$ in 9 of 10 (target,$N$) cells. `runge` and `sine_mixture` need width to resolve at all (at $N{=}16$ they sit at $O(1)$ error and only enter the QI regime by $N{=}128$, e.g. `sine_mixture` $2.2\times10^{-11}$ at $\lambda=0.30$), so their best-$\lambda$ drifts from $0.15$--$0.25$ at small $N$ up to $0.30$ once resolved.

**QI mpmath optimum (`data_mpmath.json`).** With mpmath arithmetic the QI $L_\infty$ minimizes at $\lambda = 0.25$ for all 12 (target,$N$) cells, reaching machine epsilon: `sine` $2.3\times10^{-15}$ ($N{=}128$), $1.7\times10^{-15}$ ($N{=}256$); `exp` $6.2\times10^{-15}$ / $8.9\times10^{-16}$; `runge` $2.4\times10^{-15}$ / $1.4\times10^{-15}$; `sine_mixture` $9.5\times10^{-14}$ ($N{=}128$) falling to $2.7\times10^{-15}$ ($N{=}512$). So the optimum shifts from $\lambda\approx0.30$ in fp64 to $\lambda\approx0.25$ in mpmath, and the floor drops from $\sim10^{-12}$ to $\sim10^{-15}$.

**Least-squares readout (`data.json`, `data_fine_fp64.json`).** On the same geometry in fp64, least squares reaches $\sim10^{-13}$ on the well-resolved targets: `sine` best $L_\infty$ $1.6\times10^{-13}$ ($N{=}32$), $6.6\times10^{-14}$ ($N{=}64$); `exp` $8.8\times10^{-14}$ ($N{=}32$); relative $L_2$ bottoms even lower, $\sim10^{-14}$--$10^{-15}$ (e.g. `runge` $4.8\times10^{-15}$ at $N{=}128$, `exp` $9.2\times10^{-15}$ at $N{=}64$). The least-squares argmin $\lambda$ is not pinned: it scatters across $0.10$--$0.30$ rather than sitting at a single value (consistent with the wider flat bottom characterized in exp06). The fine fp64 zoom over $\lambda\in[0.22,0.28]$ confirms the floor: `runge` $7.3\times10^{-15}$, `sine` $5.5\times10^{-14}$, `exp` $8.2\times10^{-14}$ at their best $N$.

### Figures

**`lambda_tradeoff_linf_fp64.png` / `lambda_tradeoff_linf.png`, `lambda_tradeoff_rel_l2_fp64.png` / `lambda_tradeoff_rel_l2.png` (fp64 coarse).** A 2x2 grid, one panel per target, plotting $L_\infty$ (resp. relative $L_2$) on a log $y$-axis against $\lambda$ on the $x$-axis; one color per width $N$, dashed-circle = QI, solid-square = least squares. *How to read:* each curve is a U -- error rises on the left (small $\lambda$: ill-conditioning) and on the right (large $\lambda \ge 0.5$: aliasing/fp64 cancellation), with a minimum in between. QI dashed curves bottom near $\lambda\approx0.30$; least-squares solid curves sit at or below the QI curves through the viable band and have a flatter, wider trough. `runge` and `sine_mixture` panels show the high-error small-$N$ curves that only descend into the well once $N$ is large enough to resolve the target.

**`lambda_tradeoff_linf_mpmath.png`, `lambda_tradeoff_rel_l2_mpmath.png` (mpmath QI).** Same 2x2 layout for $N \in \{128, 256, 512\}$ over $\lambda\in[0.10,0.40]$. *How to read:* the QI dashed curves now bottom near $\lambda\approx0.25$ and reach $\sim10^{-15}$ (machine epsilon) -- the trough is both shifted left and roughly three decades lower than the fp64 version. The least-squares solid curves are still fp64 (floor $\sim10^{-13}$), so they sit above the mpmath QI curves in this figure.

**`lambda_tradeoff_linf_fine_fp64.png`, `lambda_tradeoff_rel_l2_fine_fp64.png`, `lambda_tradeoff_linf_fine.png`, `lambda_tradeoff_rel_l2_fine.png` (fine zooms).** Same 2x2 layout, $\lambda$ restricted to $[0.22, 0.28]$ at step $0.005$. *How to read:* a magnified view of the bottom of the U. The `*_fine.png` pair has QI in mpmath (curves at $\sim10^{-15}$); the `*_fine_fp64.png` pair has QI in fp64 (curves at $\sim10^{-12}$); least squares is fp64 in both ($\sim10^{-13}$), letting you read off how flat the trough is in the immediate neighborhood of the optimum.

**`consolidated_linf.png` (3x4, built by `plot_consolidated.py` from `all_data.json`).** Rows = precision combination (top fp64 QI / fp64 lstsq, middle mpmath QI / fp64 lstsq, bottom mpmath QI / mpmath lstsq); columns = the four targets; $x$-axis $\lambda$ clipped to $[0.1, 0.5]$, log $y$-axis $L_\infty$; one color per $N\in\{16,32,64,96,128\}$, dashed = QI, solid = least squares. *How to read:* compare rows to see the precision floor drop -- the top row bottoms near $\sim10^{-12}$--$10^{-13}$, the bottom row near $\sim10^{-15}$. Within a panel, the dashed (QI) trough sits near $\lambda\approx0.30$ in the fp64 row and near $\lambda\approx0.25$ in the mpmath rows; the solid (lstsq) trough is wider and flatter. Note that the lstsq curves in rows 1-2 are fp64 and in row 3 are mpmath, so only row 3 shows both methods at extended precision.

## Conclusions

*Proposed, pending Sam's review -- not yet approved. Only statements plainly visible in the data are listed.*

- The error-vs-$\lambda$ curve is U-shaped for both the QI construction and the least-squares readout: error rises at small $\lambda$ and again at large $\lambda$ ($\ge 0.5$), with an interior minimum.
- The QI fp64 optimum is near $\lambda = 0.30$ (argmin at $0.30$ for 9 of 10 well-resolved `sine`/`exp` cells), with best $L_\infty \sim 5\times10^{-12}$ at $N{=}128$ on `sine`.
- The QI mpmath optimum is near $\lambda = 0.25$ (argmin at $0.25$ for all 12 cells), with best $L_\infty \sim 10^{-15}$ (machine epsilon) -- both the optimum location and the floor differ from the fp64 path.
- The least-squares readout on the same fp64 geometry reaches $\sim 10^{-13}$ in $L_\infty$ (and $\sim10^{-14}$--$10^{-15}$ in relative $L_2$); its optimal $\lambda$ is not pinned to a single value the way QI's is.
