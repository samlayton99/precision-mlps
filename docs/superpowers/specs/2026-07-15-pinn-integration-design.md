# PINN integration: spline ridges, Newton-lstsq, and the PINN finisher

**Date:** 2026-07-15 · **Status:** approved · **Home:** this repo (precision-mlps), checkpoint F

## Motivation

expF01 established that a frozen ridge dictionary + one collocation lstsq hits
the fp64 floor on linear ODEs/PDEs with zero training. The Darcy transfer work
(continuous-mlps `experiments/precision_pde/`, July 14) confirmed the recipe on
2D smooth-coefficient Darcy (rel-L2 3e-14 at W=2304) but stalls at 7.2e-2 on
rough coefficients (presmoothing σ=4 reaches 2.8e-3, changing the problem).

The structural observation driving this work: the expF01 model is a one-layer
KAN with frozen inner weights, and freezing is what makes the PINN residual
minimization a *linear* problem. Three experiments cash this out as PINN
infrastructure:

1. **expF02** — the KAN move proper: B-spline univariate family (locality →
   adaptive knots), attacking the rough-Darcy stall without presmoothing.
2. **expF03** — nonlinear PDEs via Newton-lstsq: each Newton step is one
   linear solve in the frozen basis; "PINN training" without gradient descent.
3. **expF04** — the finisher: any trained PINN + a few Newton-lstsq polish
   steps in the ridge basis ≈ machine-precision PINN.

## expF02_spline_ridge — KAN-style B-spline ridge basis

**Question:** does replacing tanh with a compact-support spline family keep the
precision floor, and does knot adaptivity beat the rough-Darcy stall?

**Basis.** u(p) = Σ c_m B(γ(w_m·p − t_m)) + poly≤3(p), where B is the cubic
B-spline bump (support [−2, 2]). Same Radon tensor geometry (√W directions ×
√W offsets), same γ = λ/h_ref scaling, same stacked min-norm lstsq
(rcond 1e-13). Derivatives of B up to order 3 are closed-form piecewise
polynomials (drop-in analogue of `psi()`; B ∈ C², so order-3 rows are piecewise
constant with jumps — acceptable for collocation, noted as a caveat).

**Part A (validation).** 2D steady problems from the expF01 zoo + the
manufactured smooth-Darcy control (vendored from continuous-mlps
`darcy_problems.py`). W ∈ {144, 256, 576, 1024, 2304}, λ swept on the expF01
schedule. **Pass:** spline floor within ~1 order of tanh at matched W.

**Part B (adaptive knots on rough Darcy).** darcy_421 instance 0, σ=0 (no
presmoothing), npz path via `--darcy-path` (default: the continuous-mlps data
dir). Loop (≤4 rounds): solve → residual on a fine grid → project |residual|
mass onto each ridge direction (Radon binning) → insert new offsets per
direction proportional to projected mass → re-solve. Total width budget matched
to the dense tanh baseline (W=2304) so the comparison is fair.
**Success:** ≥1 order below the 7.2e-2 raw stall. **Stretch:** ≤2.8e-3 (beats
presmoothing without touching the coefficient).

**Outputs** (`results/checkpoint_F_applications/expF02_spline_ridge/`):
`error_vs_width.png` (spline vs tanh, per problem), `adaptive_rounds.png`
(rel-L2 + knot histogram per round), residual/error heatmaps, `data.json`.

## expF03_newton_burgers — nonlinear PDEs via Newton-lstsq

**Question:** does the one-solve precision story survive nonlinearity when the
outer loop is Newton and each step is a linear collocation solve?

**Problem.** Steady viscous Burgers on [−1,1]²:
u·∇u = ν Δu + f, ν ∈ {0.1, 0.01}, manufactured Taylor-Green exact solution
u = −cos(πx) sin(πy), v = sin(πx) cos(πy) (period 2, matches [−1,1]²);
f = u·∇u − ν Δu in closed form in problems.py, FD-verified at random points
(the expF01/darcy convention). Dirichlet BCs from the exact solution on all
four edges.

**Solver.** Two ridge expansions (c_u, c_v), tanh family (spline optional
later; keep one new variable per experiment). Newton step at iterate (u_k, v_k):

    ν Δδu − (u_k ∂x + v_k ∂y) δu − (∂x u_k) δu − (∂y u_k) δv = r_u(u_k, v_k)
    ν Δδv − (u_k ∂x + v_k ∂y) δv − (∂x v_k) δu − (∂y v_k) δv = r_v(u_k, v_k)

assembled as a 2×2 block lstsq from `rows_2d` with callable coefficients
(iterate fields evaluated via `eval_model`); BC rows enforce δ = exact − u_k on
the boundary. Damped Newton: backtracking line search on the stacked residual
norm (halve step, ≤8 halvings); full steps expected at ν=0.1. Init at zero.
Stop when residual norm stalls (ratio > 0.5 for 2 iterations) or 12 iterations.

**Metrics:** rel-L2(u) + residual norm vs Newton iteration (expect quadratic
convergence to a plateau near the linear-solve floor, ~1e-12), floor vs W,
both ν. **Pass:** ν=0.1 reaches ≤1e-10 rel-L2; ν=0.01 documented wherever it
lands (harder, sharper layers).

**Outputs:** `newton_convergence.png`, `error_vs_width.png`, error heatmaps,
`data.json`.

## expF04_pinn_finisher — lstsq polish for trained PINNs

**Question:** can a few Newton-lstsq steps in the frozen ridge basis take an
ordinary trained PINN from its optimization plateau to solver-grade precision?

**Baseline PINN.** torch tanh-MLP, 4 hidden layers × 64 units, (x,y) → (u,v),
trained with Adam (lr 1e-3, cosine decay, ~50k steps, resampled collocation
batches) on the expF03 Burgers problem (ν=0.1), loss = PDE residual MSE + BC
MSE. Train to plateau; seeded; save checkpoint + loss/rel-L2 curves. Expected
plateau ~1e-3 rel-L2 (whatever it is, it is recorded, not tuned toward).

**Finisher.** Freeze the PINN. Run the expF03 Newton loop warm-started at
u_0 = PINN output: linearization coefficients are callables evaluating the
frozen PINN and its first derivatives (torch autograd) at collocation points;
the ridge basis carries the correction δ; total solution u = PINN + Σ δ_i.
2–3 polish steps. The PINN is evaluated per Newton iterate only through the
residual and coefficient fields — after step 1 the iterate is PINN + ridge
correction, so residual evaluation composes both (eval chunked, numpy/torch
boundary kept at collocation arrays).

**Metrics:** rel-L2 before / after each polish step; wall-clock: PINN training
time vs total polish time. **Pass:** ≥4 orders improvement, landing within ~1
order of the expF03 floor at the same W. **Report line:** "N minutes of Adam +
M seconds of lstsq = machine-precision PINN."

**Outputs:** `finisher_convergence.png` (one curve: Adam plateau then polish
steps), `data.json`, PINN checkpoint under the results dir.

## Shared conventions

- Repo format: self-contained `experiments/expF0N_*/run.py` (+ `problems.py`);
  expF04 imports expF03's problems via the expF01 sys.path pattern.
- `--smoke` (reduced grids) and `--plot` (regenerate figures from data.json)
  flags on every run.py; incremental writes to data.json.
- Tests in `tests/`: FD verification of spline derivative rows (orders 0–3,
  interior points away from knots), one Newton smoke test (tiny W, few iters,
  asserts monotone residual decrease), PINN-finisher smoke (2 Adam steps + 1
  polish step end-to-end shape check). Marked `slow` where >10s.
- All CPU (torch CPU fine at 4×64); commits styled `expF0N: ...`.
- Build order: expF02 → expF03 → expF04 (F04 depends on F03's problems/solver).

## Risks / escalations

- **Spline conditioning:** compact support can under-cover the collar; if
  Part A floors degrade >1 order vs tanh, widen the bump (quintic spline) before
  concluding.
- **Adaptive knots don't close the gap:** if <1 order gained, the rough-Darcy
  bottleneck isn't offset resolution — record the negative and stop (FOSLS
  remains the continuous-mlps escalation; the diffuse error maps already cast
  doubt).
- **ν=0.01 Newton stalls:** acceptable — report with damping diagnostics;
  continuation in ν (solve 0.1 → warm-start 0.05 → 0.01) is the pre-planned
  escalation, one extra loop.
- **PINN trains too well/too poorly:** the finisher claim only needs a plateau
  strictly above the solver floor; no tuning toward a target.
