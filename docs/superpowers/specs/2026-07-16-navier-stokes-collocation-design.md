# expF09 -- Navier-Stokes via frozen-geometry collocation (design)

**Date:** 2026-07-16
**Checkpoint:** F (applications)
**Status:** approved (Stage A detailed; B/C/D roadmapped), pending implementation plan
**Builds on:** expF01 (linear collocation core), expF06 (frozen-basis Newton), expF08 (vendored core pattern)

## Motivation

The frozen-geometry + one-lstsq recipe reaches the fp64 floor on linear scalar
PDEs (expF01, expF08 Darcy operator: 3e-14) and ~1e-7 on nonlinear scalar PDEs
via Newton (expF06 Burgers). The natural next target is the flagship
operator-learning PDE: incompressible Navier-Stokes. expF06 already solves a
coupled two-field velocity system `F_u = u u_x + v u_y - nu lap u - f` by frozen-
basis Newton -- that is NS momentum minus pressure and incompressibility, using
only 2nd-order derivatives. This program adds the missing pieces (pressure,
continuity, time) in stages, each with an *exact* solution so the precision
claim stays verifiable.

Formulation: **primitive velocity-pressure `(u, v, p)`** (decided). Reuses
expF06's momentum Newton, needs only <=2nd derivatives (best conditioning for
precision), extends to 3D, and can report vorticity for the Stage-D benchmark.
Streamfunction (4th-order, gamma^4 roundoff risk) and vorticity-streamfunction
(ill-posed wall vorticity BC) were rejected.

## Program shape

Four stages, each adding exactly one capability. **Stages A-C live on a box
`[-1,1]^2` with Dirichlet BCs and the existing tanh/Radon ridges**; periodicity
(Fourier ridges) is needed only for the Stage-D benchmark and is deferred.

| Stage | Problem | New machinery | Exact verifier | Target |
|---|---|---|---|---|
| **A** | Stokes (linear) | `solve_system` + pressure gauge | manufactured div-free (streamfunction) | fp64 floor |
| **B** | Steady NS | Newton on the system (expF06 pattern) | Kovasznay flow (exact steady NS) | ~1e-7 |
| **C** | Unsteady NS | time as a coordinate + IC block | Taylor-Green (Dirichlet+IC from exact; no periodicity) | ~1e-7 space-time |
| **D** | FNO NS benchmark | Fourier ridges + periodic torus + data | dataset + trained FNO | beat/match FNO |

**This spec details Stage A + the shared core.** B/C/D are roadmapped here and
each gets its own detailed spec/plan when its gate opens.

**Decision gates:** A must reach the fp64 floor before B; B must converge on
Kovasznay (~1e-7, quadratic) before C; C must verify against Taylor-Green before
D. A stage that stalls is a valid finding and is written up rather than forced.

## Shared infrastructure (built in Stage A, in `core_system.py`)

Vendors the scalar primitives it needs from the expF01/expF08 core (`radon_
geometry`, `rows_2d`, `psi`, `MONO_2D`, `boundary_points_square`, `RCOND`) so
expF09 is self-contained (the expF08 pattern), then adds:

- **`solve_system(fields, equations, W, lam, seed)`** -- the multi-field
  generalization of `solve_square`. Every field in `fields` (e.g.
  `["u","v","p"]`) gets the *same* frozen ridge geometry + degree-3 poly tail;
  the unknown coefficient vector is the concatenation over fields (length
  `n_fields * (W + len(MONO_2D))`). Each `equation` is a collocation constraint

  ```
  {points: [n,2],
   blocks: {field_name: [((ax,ay), coeff), ...]},   # operator on that field
   rhs: callable(P)->[n] or scalar}
  ```

  meaning `sum_field blocks[field] applied to field = rhs` at `points`. Assembly:
  for each equation build a row block that places each field's operator-rows
  (`rows_2d(points, ..., blocks[field])`) in that field's column slot and zeros
  elsewhere; stack all equations, scale each block to O(1), and solve one
  min-norm `lstsq` (rcond 1e-13). Returns a model with the geometry and a
  per-field coefficient slice.
- **`eval_field(model, field, P, terms=...)`** -- evaluate a field (or a
  derivative operator of it) at points, reusing `rows_2d` on that field's slice.
- **Pressure gauge** -- pressure is determined up to a constant; add one
  equation `{points: [p_ref], blocks: {"p":[((0,0),1.0)]}, rhs: p_ref_value}`
  fixing `p` at a reference point.

Later stages extend this core without changing its interface: **Newton** (B)
calls `solve_system` on the linearized residual each iterate; **time** (C) adds
a `t` column to `points` and an IC equation block; **Fourier ridges** (D) swap
the feature builder for the periodic domain.

## Stage A -- Stokes (detailed)

**Equations on `[-1,1]^2`:** `-nu lap u + p_x = f_u`, `-nu lap v + p_y = f_v`,
`u_x + v_y = 0`, with `nu = 1` (a scalar; Stokes is linear in it). As
`solve_system` equations:
- momentum-x: `blocks={"u":[((2,0),-nu),((0,2),-nu)], "p":[((1,0),1.0)]}`, `rhs=f_u`
- momentum-y: `blocks={"v":[((2,0),-nu),((0,2),-nu)], "p":[((0,1),1.0)]}`, `rhs=f_v`
- continuity: `blocks={"u":[((1,0),1.0)], "v":[((0,1),1.0)]}`, `rhs=0`
- BC-u / BC-v: `blocks={"u":[((0,0),1.0)]}` / `{"v":[((0,0),1.0)]}` on
  `boundary_points_square`, `rhs=u*/v*`
- gauge: fix `p` at the origin to `p*(0,0)`.

**Manufactured solution** (divergence-free by construction from a
streamfunction `psi = sin(pi x) sin(pi y)`):
`u* = psi_y = pi sin(pi x) cos(pi y)`, `v* = -psi_x = -pi cos(pi x) sin(pi y)`
(so `u*_x + v*_y = 0` exactly), and `p* = cos(pi x) cos(pi y)`. Forcing
`f_u = -nu lap u* + p*_x`, `f_v = -nu lap v* + p*_y`, all hand-coded and
FD-verified (the expF08 `verify_control` pattern). Error is vs the exact
`(u*, v*, p*)` on a fresh grid.

**Metrics:** rel L2 / Linf of `(u_hat, v_hat)` vs `(u*, v*)`; `p_hat` vs `p*`
(after matching the gauge constant); the pointwise **divergence residual**
`|u_hat_x + v_hat_y|` at fresh interior points (the incompressibility check);
the momentum PDE residual. Swept over `W` and `lambda` like expF08.

**Success:** velocity rel L2 and divergence residual at the fp64 floor
(`<= 1e-10`, expected `~1e-13`) at the largest clean `W`, matching the Darcy
control's behaviour -- Stokes is linear, so nothing should stop it reaching the
floor.

## Stages B/C/D -- roadmap (own specs later)

- **B -- Steady NS (Kovasznay).** Add convection `(u.grad)u`; damped Newton on
  the `(u,v,p)` system, each iterate one `solve_system` on the linearized
  residual (expF06 template). Verifier: **Kovasznay flow**, an exact analytic
  steady 2D NS solution on a box (`u = 1 - e^{lam x} cos(2 pi y)`, etc., with
  `lam` a function of Re) -- Dirichlet BCs from the exact field. Target ~1e-7
  (nonlinear floor); Reynolds/nu continuation if a cold start diverges (expF06
  showed nu=0.01 needs it).
- **C -- Unsteady NS (Taylor-Green).** Add `t` as a collocation coordinate over
  a space-time box `[-1,1]^2 x [0,T]` plus an initial-condition equation block;
  no time-stepping. Verifier: **Taylor-Green vortex**, exact unsteady NS
  (`u = cos x sin y e^{-2 nu t}`, ...). Impose exact Dirichlet BCs on the
  spatial box faces and the exact IC -- **no periodicity needed**, so the tanh
  ridges still apply. Target ~1e-7 on the full space-time field. This is the
  "predict the trajectory to high precision" result.
- **D -- FNO NS benchmark.** The periodic torus, forced vorticity form,
  nu in {1e-3, 1e-4}. Needs **Fourier (sin/cos) ridge features** (the one real
  solver change) and the benchmark data. Solve per-instance / short-horizon;
  compare to the dataset reference and to a trained FNO. The hard case
  (convection-dominated low-nu may stall -- a valid finding).

## File structure (Stage A)

- `experiments/expF09_navier_stokes/__init__.py`
- `experiments/expF09_navier_stokes/core_system.py` -- vendored scalar
  primitives + `solve_system` / `eval_field` / gauge.
- `experiments/expF09_navier_stokes/stokes.py` -- Stage-A manufactured solution,
  equation assembly, `verify_stokes` (FD checks).
- `experiments/expF09_navier_stokes/run_stokes.py` -- W/lambda sweep, `data.json`,
  `--smoke`/`--plot`.
- `tests/test_expF09_stokes.py`
- `results/checkpoint_F_applications/expF09_navier_stokes/` -- Stage-A results +
  a program `README.md` tracking the stage roadmap. `data.json` gitignored;
  figures + writeup tracked.

## Tests (Stage A, `tests/test_expF09_stokes.py`)

1. **Single-field equivalence:** `solve_system` on a one-field Poisson
   (`-lap u = f`, Dirichlet) matches the scalar `solve_square` result to fp
   precision -- the multi-field assembly reduces correctly.
2. **FD-verify forcing:** `verify_stokes` -- every hand-coded `f_u`, `f_v`,
   `lap u*`, `p*_x` matches central differences (tol 2e-4).
3. **Divergence-free target:** `u*_x + v*_y == 0` analytically at random points.
4. **Stokes reaches the floor:** velocity rel L2 `< 1e-10` and divergence
   residual `< 1e-9` at `W` large enough (e.g. 1600) on the manufactured case.

Run under `uv run --extra dev pytest`.

## Non-goals (deferred)

- Periodic BCs / Fourier ridges (Stage D only).
- 3D.
- Turbulent / very-low-viscosity regimes beyond what Newton+continuation reach.
- Inf-sup/LBB stability theory for the collocation velocity-pressure pair -- the
  min-norm SVD truncation is relied on empirically (as elsewhere in the repo);
  if the pressure is unstable, a pressure-space coarsening is the escalation.

## Reproduce (Stage A)

```
uv run --extra dev python experiments/expF09_navier_stokes/run_stokes.py [--smoke] [--plot]
```
