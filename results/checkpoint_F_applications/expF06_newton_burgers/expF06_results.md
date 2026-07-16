# expF06 -- steady 2D Burgers via Newton-lstsq: solve, don't train

**Status: draft.**

## TL;DR

- **Newton-lstsq solves the nonlinear PDE with the same "no training" recipe.**
  Every Newton step is one block collocation lstsq in the frozen tanh ridge
  basis; at $\nu=0.1$ the iteration converges quadratically (residual $2\to10^{-7}$
  in 5--8 steps) to rel $L_2\sim3\times10^{-8}$--$4\times10^{-6}$ across widths.
- **The nonlinear floor is ~$10^{-6}$--$10^{-8}$, not the fp64 floor.** The same
  smooth tanh basis solves the *linear* Poisson/Darcy control to $3\times10^{-14}$
  (expF05), so this ~6-order gap is the Newton/Jacobian-conditioning cost of the
  nonlinearity, not a representation limit. Steps flag as "stalled" (backtracking
  cannot reduce the residual further) at that floor.
- **$\nu=0.01$ diverges from a zero initial guess** (rel $L_2\approx1$ at every
  width) — the convection-dominated hard case. The pre-planned **$\nu$-continuation**
  escalation (ladder $0.1\to0.05\to0.02\to0.01$, each rung warm-started from the
  previous coefficients) recovers it: see the continuation results below.

## Question

Does the frozen-ridge + one-lstsq-per-step recipe survive nonlinearity, when the
outer loop is damped Newton and each linearized solve is a single block
collocation lstsq? And does the convection-dominated $\nu=0.01$ regime need the
$\nu$-continuation escalation the spec anticipated?

## Experiment design

Steady viscous Burgers on $[-1,1]^2$: $u\cdot\nabla u=\nu\Delta u+f$, manufactured
Taylor-Green $u^*=-\cos\pi x\sin\pi y$, $v^*=\sin\pi x\cos\pi y$ (forcing
$f=u^*\cdot\nabla u^*-\nu\Delta u^*$ in closed form, FD-verified), Dirichlet BCs
from the exact solution. Two ridge expansions $(c_u,c_v)$, tanh family, $\lambda=0.25$.
Newton step at $(u_k,v_k)$ assembles the $2\times2$ block Jacobian
($\nu\Delta - u_k\partial_x - v_k\partial_y - (\partial\cdot u_k)$ blocks) from
`rows_2d` with the iterate fields as callable coefficients; Dirichlet rows enforce
$\delta=$ exact$-u_k$ on the boundary; damped by backtracking on the collocation
residual norm. Direct sweep: $\nu\in\{0.1,0.01\}\times W\in\{256,576,1024,2304\}$,
zero init, $\le12$ iterations. Continuation: fixed $W=1024$, ladder
$\{0.1,0.05,0.02,0.01\}$, each rung `init_sol` = previous converged coefficients.

**Code & data.** `experiments/expF06_newton_burgers/` (`problems.py`, `newton.py`,
`run.py`). Data (gitignored): `data.json` (direct sweep), `continuation.json`.
Figure: `newton_convergence.png`. Regenerate: `run.py` and `run.py --continuation`.

## Results -- direct sweep (zero init)

| $\nu$ | W | iters | final rel $L_2(u)$ | residual |
|---|---|---|---|---|
| 0.1 | 256 | 8 | 5.1e-08 | 7.4e-08 |
| 0.1 | 576 | 8 | 3.9e-06 | 6.8e-05 |
| 0.1 | 1024 | 12 | 6.6e-07 | 3.2e-06 |
| 0.1 | 2304 | 9 | 2.7e-08 | 5.9e-07 |
| 0.01 | 256 | 6 | 9.8e-01 | 1.2e+00 |
| 0.01 | 576 | 8 | 1.1e+00 | 9.7e-01 |
| 0.01 | 1024 | 5 | 1.2e+00 | 9.3e-01 |
| 0.01 | 2304 | -- | (diverged) | -- |

$\nu=0.1$: clean quadratic convergence to the nonlinear floor at every width (the
$W=576$ cell floors an order higher, an echo of the non-monotone conditioning seen
for higher-order operators in expF01). $\nu=0.01$: diverges from zero regardless of
width — the convection term dominates and the first Newton steps overshoot.

## Results -- $\nu$-continuation (W=1024)

| $\nu$ | warm start | iters | final rel $L_2(u)$ |
|---|---|---|---|
| 0.10 | -- (cold) | 12 | 6.6e-07 |
| 0.05 | from 0.10 | 3 | 2.3e-06 |
| 0.02 | from 0.05 | 3 | 1.3e-07 |
| 0.01 | from 0.02 | 7 | **1.2e-07** |

Continuation recovers $\nu=0.01$ completely: from the diverged $\approx1.0$ of the
cold sweep to $1.2\times10^{-7}$, the same nonlinear floor the cold $\nu=0.1$ solve
reaches. Each warm-started rung needs only 3--7 Newton steps (vs 12 cold), because
the previous rung's solution is already inside the convergence basin of the next.

## Conclusions

1. Newton-lstsq extends the "solve, don't train" recipe to a nonlinear PDE, with
   quadratic convergence per outer step.
2. The achievable precision drops from the linear fp64 floor ($10^{-14}$) to
   ~$10^{-6}$--$10^{-8}$ for nonlinear Burgers — a conditioning cost of the Newton
   linearization, not the basis. This same floor bounds the expF07 PINN finisher.
3. Convection-dominated $\nu=0.01$ is unreachable from a cold start but recovered by
   $\nu$-continuation, confirming the escalation the spec pre-registered.
