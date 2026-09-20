# expG03 -- Extrapolation & data-poor generalization

**Status:** drafted (construction-only first pass, single seed). 27-cell sweep,
seconds of CPU.

## TL;DR

- On the trained region the uniform-$\gamma$ construction stays at the **fp64
  floor** ($10^{-15}$--$10^{-13}$) in every protocol -- precision is untouched
  by hold-outs.
- At the precision-optimal $\lambda=0.25$ the held-out error is **~$10^{-1}$
  regardless of target or protocol**: the single sharp length scale fills a
  held-out region with a near-linear ramp and cannot reconstruct held-out
  structure (basis figures confirm no neuron carries the curvature).
- Lowering $\lambda$ (wider kernels) exposes a **target-dependent tradeoff**:
  for **analytic** targets (sine, exp) extrapolation improves by 2--3 orders
  (down to ~$10^{-4}$), confirming Sam's "low $\lambda$ fixes the blowup"; for
  the **non-analytic Runge peak** it is *catastrophic* -- held-out error rises
  to $10^{4}$--$10^{5}$ and the readout norm $\lVert v\rVert$ explodes to
  ~$10^{6}$, and small $\lambda$ even pulls the *trained* region off the floor
  ($4.6\times10^{-6}$).
- So generalization is **not** free from precision here: the geometry that hits
  the fp64 floor is the worst extrapolator, and the low-$\lambda$ fix only works
  when the held-out function is smooth enough to be analytically continued.

## Question

expG01 (interactive) showed an *interior* hold-out fills with a linear ramp.
This batch experiment asks the head-to-head, extrapolation version: with the
fixed uniform-$\gamma$ construction, how large is the held-out error when the
data is one-sided or absent, how does it move with $\lambda$, and what do the
per-neuron basis contributions reveal about the mechanism?

## Experiment design

- **Solver.** Reuses expG01's geometry (`-1 + arange(-halo, N+halo+1)·h`,
  $h=2/N$, $N=128$, `default_halo(N, 0.25)`, per-center $\gamma=\lambda/h$) and
  `src.construction` (`build_phi` + `solve_readout_with_bias(method="svd")`,
  rcond $10^{-13}$), fp64. **$N$ sets the geometry; the training-sample count is
  a separate knob** (>~ effective DOF so the trained region reaches the floor).
- **Protocols** (`protocols.py`): `edge_holdout` (train $[-1,0.5]$, held-out
  $(0.5,1]$; 300 train pts), `beyond_domain` (train $[-1,1]$, held-out
  $[-1-\!0.3,-1)\cup(1,1+\!0.3]$ past the last neuron; 400 train pts),
  `sparse_half` (dense $[-1,0]$ + 3 pts on $(0,1]$; held-out $(0,1]$,
  data-poor not data-free).
- **Sweep**: $\lambda\in\{0.25,0.10,0.05\}$ x targets {`sin(2*pi*x)`,
  `1/(1+25x^2)`, `exp(x)`} x 3 protocols = 27 cells. Metrics: rel $L_2$ /
  $L_\infty$ over entire / unmasked / held-out, plus readout norm
  $\lVert v\rVert$.
- **Reproduce**: `uv run --extra dev python experiments/expG03_extrapolation/run.py`
  (`--smoke`, `--plot`). Figures + `data.json` under this directory.

## Results

**Held-out rel $L_2$** (unmasked is at the fp64 floor $10^{-15}$--$10^{-13}$
everywhere except the Runge/small-$\lambda$ cells, noted below):

| protocol | target | $\lambda=0.25$ | $0.10$ | $0.05$ |
|---|---|---|---|---|
| edge_holdout | sine | 3.1e-1 | 3.4e-1 | **9.6e-3** |
| edge_holdout | exp | 2.2e-1 | 6.1e-2 | **4.1e-3** |
| edge_holdout | runge | 1.8e-1 | 3.1e+1 | **3.3e+5** |
| beyond_domain | sine | 6.6e-2 | 2.4e-2 | **3.3e-4** |
| beyond_domain | exp | 9.3e-2 | 5.5e-3 | **1.4e-4** |
| beyond_domain | runge | 1.9e-2 | 8.5e+0 | **1.9e+4** |
| sparse_half | sine | 3.5e-1 | 3.9e-1 | 8.2e-2 |
| sparse_half | exp | 1.2e-1 | 7.8e-2 | 2.0e-2 |
| sparse_half | runge | 1.1e-2 | 1.5e+0 | **5.8e+3** |

- **Precision is preserved on the trained region** in all smooth-target cells
  ($\lVert v\rVert \sim 0.1$--$1.7$). The Runge peak at small $\lambda$ is the
  exception: `edge_holdout/runge` at $\lambda=0.05$ has unmasked rel $L_2$
  $4.6\times10^{-6}$ and $\lVert v\rVert = 9.6\times10^{5}$ -- the ill-condition
  contaminates even the fit region.
- **$\lambda=0.25$ (precision-optimal) is the worst extrapolator**: held-out
  error clusters at $10^{-1}$ for every target. The basis figures
  (`basis_*_lam0.25.png`) show the fit matching the target on the trained side,
  then ramping linearly across the shaded held-out band with the target curving
  away; the individual weighted ridges $c_k\phi_k$ are all small and none tracks
  the held-out curvature -- the min-norm equal-steps mechanism from expG01.
- **Analytic targets reward small $\lambda$**: sine/exp under `beyond_domain`
  reach $10^{-4}$ at $\lambda=0.05$ (wide kernels analytically continue a smooth
  function past the last neuron). See `summary_held_vs_lambda.png` -- the
  smooth-target curves slope down to the left, the Runge curves shoot up.
- **`beyond_domain` = true extrapolation past the last center**: the fit decays
  toward the polynomial/bias tail outside $[-1,1]$; the basis figure's fit line
  continues as a gentle ramp with no support beyond the halo.

## Conclusions

1. **Hold-outs do not cost precision** on the trained region -- the construction
   is a faithful interpolant of the data it sees. The generalization question is
   entirely about the *unseen* region.
2. **The precision-optimal geometry generalizes worst.** A single sharp scale
   ($\lambda=0.25$) reconstructs a held-out region as a linear ramp
   (~$10^{-1}$), independent of protocol or target. This is the expG01 interior
   result, now quantified for one-sided / beyond-domain / data-poor hold-outs.
3. **The low-$\lambda$ fix is conditional on smoothness.** Widening the kernels
   recovers extrapolation for analytic targets (2--3 orders) but is
   catastrophic for a non-analytic peak, where it amplifies the held-out
   reconstruction and blows up $\lVert v\rVert$ to ~$10^{6}$. Precision vs
   generalization is a genuine tradeoff, and the knob's sign depends on the
   target.
4. **Basis contributions localize the failure**: the held-out structure is not
   representable from one-sided data with this geometry, so the sum defaults to
   the min-norm ramp rather than any neuron doing the work.

## Open questions

- **Cascade multi-band geometry** (the deferred arm): do soft/wide bands
  alongside the sharp grid recover Runge extrapolation without the
  coefficient-norm blowup? This is the natural next experiment.
- **Adam-trained baseline** head-to-head on the same held-out grids.
- **Regularization (rcond / ridge)** as a stabilizer for the small-$\lambda$
  Runge blowup, trading a little precision for a bounded $\lVert v\rVert$.
- **Where does the analytic/non-analytic boundary bite?** A frequency or
  peak-sharpness sweep between sine and Runge to map when low-$\lambda$
  extrapolation flips from helpful to harmful.
