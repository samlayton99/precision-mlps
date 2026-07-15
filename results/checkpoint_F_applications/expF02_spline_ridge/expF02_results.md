# expF02 -- KAN-style B-spline ridges: floor and adaptive knots

**Status: draft.**

## TL;DR

- **The precision floor needs a spectral univariate family.** Swapping tanh for
  a cubic B-spline bump (same Radon ridge geometry, same one-shot lstsq) drops
  the floor from $3\times10^{-14}$ to $\sim2\times10^{-4}$ at $W=2304$ on both
  Part-A problems — algebraic ($\sim h^4$, C$^2$ family) vs spectral
  convergence, ten orders apart. Quintic/higher splines would only improve the
  exponent, not restore the fp64 floor.
- **Adaptive knot insertion does not beat the rough-Darcy stall.** On darcy_421
  instance 0 ($\sigma=0$, width-matched at 2304): dense tanh 7.49e-2 (reproduces
  the continuous-mlps 7.2e-2 within re-sampling noise), dense spline 7.65e-2,
  adaptive spline 7.3--8.8e-2 over rounds 0--2 — then a conditioning collapse
  (rel $L_2\to1$) at rounds 3--4 when inserted knots cluster and the per-neuron
  $\gamma=\lambda/(0.875\,\text{gap})$ blows up.
- Together with the diffuse (not interface-concentrated) error maps from the
  continuous-mlps sweep, the stall is **not offset-resolution-limited**: adding
  1D resolution along ridge directions where residual mass sits does not help.
  The bottleneck is representing a rough-coefficient solution in ANY smooth
  ridge basis at this width, or the non-divergence-form operator itself
  (escalation: FOSLS first-order system, or genuinely 2D-localized bumps).

## Question

Does a compact-support (KAN-style) spline univariate family keep the expF01
precision floor, and does its locality — cashed in as residual-guided knot
insertion with per-neuron $\gamma$ — beat the 7.2e-2 rough-Darcy stall that
presmoothing ($\sigma=4\to$ 2.8e-3) only sidesteps?

## Experiment design

Model as expF01 (Radon tensor ridges + poly$\le3$ + one min-norm lstsq,
`rcond=1e-13`), generalized: `family(order, Z)` and per-neuron $\gamma_m$
(`ridge_core.py`). Cubic B-spline bump: support $[-2,2]$, C$^2$, closed-form
derivatives to order 3 (order-3 piecewise constant). Part A: poisson and
smooth-coefficient Darcy control ($a=3+e^{\sin\pi x\sin\pi y}$,
$u^*=\sin\pi x\sin\pi y+\tfrac12\sin2\pi x\sin\pi y$, both FD-verified),
$W\in\{144,256,576,1024,2304\}$, $\lambda\in\{0.2,0.25,0.3\}$, best-of-$\lambda$,
tanh vs bspline. Part B: darcy_421 instance 0 via cubic-spline coefficient
surrogate ($\sigma=0$, cell-centered grid), $-a\Delta u-\nabla a\cdot\nabla u = 1/4$
on $[-1,1]^2$, $u=0$ boundary; baselines dense tanh/bspline at $W=2304$;
adaptive: start $W=1024$ uniform, 4 rounds $\times$ 320 knots inserted at the
highest-|residual|-mass Radon bins per direction (proportional allocation),
per-neuron $\gamma$ from local gaps; rel $L_2$ against the dataset reference
(421-grid, stride 3).

**Code & data.** `experiments/expF02_spline_ridge/` (`ridge_core.py`,
`problems.py`, `darcy_data.py`, `adaptive.py`, `run.py`). Data: `data.json`
(gitignored, regenerate with `run.py` / `run.py --adaptive`, ~25 min CPU).
Figures: `error_vs_width.png`, `adaptive_rounds.png`.

## Results

Part A, best-of-$\lambda$ rel $L_2$:

| W | poisson tanh | poisson bspline | darcy_ctrl tanh | darcy_ctrl bspline |
|---|---|---|---|---|
| 144 | 6.9e-04 | 3.9e-02 | 6.6e-04 | 3.1e-02 |
| 256 | 1.7e-06 | 7.5e-03 | 1.6e-06 | 6.6e-03 |
| 576 | 1.5e-10 | 6.2e-04 | 1.4e-10 | 5.2e-04 |
| 1024 | 3.5e-13 | 4.0e-04 | 4.6e-13 | 4.8e-04 |
| 2304 | **3.5e-14** | 2.2e-04 | **3.0e-14** | 2.5e-04 |

tanh: spectral straight to the floor (reproduces expF01/the continuous-mlps
darcy control). bspline: clean algebraic decay that flattens near $10^{-4}$ —
the C$^2$ family's approximation order, not conditioning (the $\lambda$ sweep
moves cells by $<$1 order).

Part B (rough darcy_421 instance 0, $\sigma=0$):

| method | knots | rel $L_2$ |
|---|---|---|
| dense tanh | 2304 | 7.49e-2 |
| dense bspline | 2304 | 7.65e-2 |
| adaptive bspline r0 | 1024 | 8.00e-2 |
| adaptive bspline r1 | 1344 | 7.32e-2 |
| adaptive bspline r2 | 1664 | 8.82e-2 |
| adaptive bspline r3 | 1984 | 9.98e-1 |
| adaptive bspline r4 | 2304 | 9.92e-1 |

Two families, one adaptive scheme, one number: everything sits at the same
$\sim$7.5e-2 stall until the adaptive geometry destroys itself. The knot
clustering that was supposed to add resolution instead produces near-duplicate
neurons with huge $\gamma$ (gap $\to$ 0), i.e. an ill-conditioned dictionary —
and even before the collapse there is no gain, consistent with the residual
mass being spread across all directions rather than concentrated at a few
under-resolved offsets.

## Conclusions

1. Locality (KAN-style splines) costs the precision floor: use tanh whenever
   fp64-grade accuracy is the point.
2. The rough-Darcy stall survives family choice and offset adaptivity at
   matched width. Remaining suspects: the non-divergence-form operator on a
   spline surrogate of a rough $a$ (FOSLS escalation), or the need for
   genuinely 2D-localized basis elements rather than 1D ridge refinements.
3. If adaptivity is revisited: cap per-neuron $\gamma$ (e.g. min-gap floor) or
   deduplicate knots before solving — the round-3 collapse is avoidable
   engineering, though the flat rounds 0--2 suggest the ceiling is real.
