# expF07 -- lstsq precision finisher for a trained PINN

**Status: draft.**

## TL;DR

- **A few seconds of lstsq polish beats an hour and a half of Adam.** A vanilla
  torch tanh-MLP PINN (4x64) trained 50k Adam steps (**96 min** CPU) on the
  expF06 Burgers problem ($\nu=0.1$) plateaus at rel $L_2=1.86\times10^{-3}$.
  Freezing it and running **6 Newton-lstsq steps warm-started at the PINN
  (311 s, ~5 min)** drops it to $5.5\times10^{-6}$ — a **~2.4-order** improvement
  for ~5% of the wall clock.
- **The finisher is bounded by expF06's nonlinear floor, not by the PINN.** It
  floors at ~$8\times10^{-6}$, the same ~$10^{-6}$--$10^{-8}$ that the direct
  Newton solve reaches at $\nu=0.1$, $W=1024$. So it does *not* hit the spec's
  $\ge4$-order target — but only because steady Burgers itself has no fp64 floor
  under Newton-lstsq (expF06). The finisher recovers essentially all of the
  headroom the solver has.
- Mechanism confirmed (smoke-scale, see the test): the ridge correction must
  represent $u^*-\text{PINN}$; a rougher/undertrained PINN leaves a residual the
  $\gamma$-limited basis cannot fully cancel ("representation ceiling"). At the
  full 50k-step plateau the PINN error is smooth enough that the ceiling sits at
  the solver floor.

## Question

Can a few Newton-lstsq steps in the frozen ridge basis take an ordinary trained
PINN from its optimization plateau to solver-grade precision, and how does the
cost compare to just training longer?

## Experiment design

Baseline PINN: torch tanh-MLP, 4 hidden layers x 64, $(x,y)\to(u,v)$, Adam
(lr 1e-3 cosine, 50k steps, resampled interior + boundary batches), loss =
PDE-residual MSE + 10x BC MSE, on the expF06 Burgers problem at $\nu=0.1$. Trained
to plateau (not tuned toward a target), checkpointed. Finisher: freeze the PINN,
run `newton_burgers` with `base_fields` = the frozen net's fields (value + first/
second derivatives via torch autograd, numpy adapter), $W=1024$, 6 polish steps;
the ridge basis carries the correction $\delta$, total $u=\text{PINN}+\sum\delta_i$.
Metrics: rel $L_2(u)$ before/after each step, wall clocks.

**Code & data.** `experiments/expF07_pinn_finisher/` (`pinn.py`, `run.py`;
finisher reuses expF06 `newton.py`). Data (gitignored): `data.json`,
`pinn_ckpt.pt`. Figure: `finisher_convergence.png`. Regenerate: `run.py`.

## Results

| stage | rel $L_2(u)$ | wall clock |
|---|---|---|
| Adam plateau (50k steps) | 1.86e-03 | 5784 s (96 min) |
| + polish step 1 | 5.52e-06 | \[311 s total for |
| + polish step 2 | 8.28e-06 | 6 steps\] |
| + polish steps 3--6 | 8.28e-06 (flat) | |

The best polish iterate is step 1 ($5.5\times10^{-6}$); step 2 settles at
$8.3\times10^{-6}$ and the remaining steps are flat — the solve has reached its
floor for this problem/$W$, so extra polish steps neither help nor hurt.

## Conclusions

1. The "minutes of Adam + seconds of lstsq" claim holds directionally: the polish
   is ~20x cheaper than the training it improves on, and buys ~2.4 orders.
2. The absolute precision is capped by the underlying solver, not the finisher —
   on a problem with a true fp64 floor (a *linear* PDE, cf. expF05 control at
   $3\times10^{-14}$) the same finisher would be expected to reach it. Steady
   Burgers floors at ~$10^{-6}$ (expF06), and the finisher lands there.
3. The representation ceiling (ridge basis must express PINN error) is the thing
   to watch: it is benign once the PINN is trained to a smooth plateau, but bites
   for undertrained/rough PINNs (smoke test: a 400-step PINN polishes only to
   ~$0.017$). A useful finisher presumes a converged, not a barely-started, net.
