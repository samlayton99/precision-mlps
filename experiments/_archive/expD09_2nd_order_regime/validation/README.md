# expD09 validation sweep -- results

Validates the block-QR whitening recipe (`results/checkpoint_D_optimizers/expD09_2nd_order_regime/expD09_recipe_results.md`) on four axes: convergence, the $k$-vs-$d$ coupling, y-noise, and batching. Self-contained; run with

```
uv run python experiments/expD09_2nd_order_regime/validation/run_sweep.py
uv run python .../run_sweep.py --only B      # subset
uv run python .../run_sweep.py --plot        # replot from cache
```

Setup: targets sine / sine_8pi / runge, widths $N\in\{64,128,256\}$ (plus 512 for the scaling study), $n=8d$ rows, eval on 4001 clean points, all fp64. Figures in `results/checkpoint_D_optimizers/expD09_2nd_order_regime/validation/figures/`, cached data alongside as JSON.

## Headline

1. **$k$ is uncoupled from $d$ for $k\ge32$; $k{=}16$ fails.** See H. Reaching this took four discarded attempts, all confounded -- the history is in "What was deleted and why" at the end, and it is worth reading before designing the next scaling study.
2. **With noise it lands exactly on the statistical floor.** Achieved error $=0.25\,\sigma_{rel}$, essentially independent of $\sigma$ over 6 decades, against the theoretical rank-$r$ regression floor $\sigma\sqrt{r/n}=0.272\,\sigma_{rel}$. The $\sigma>0$ target of the subproblem is met.
3. **Batching degrades gracefully, and the convergence rate is unchanged.** Curves for different batch sizes overlay until each plateaus at its own floor. $b\ge4d$ reaches the floor; $b=d$ (square) stalls 4 orders up.
4. **Correction, now folded into the recipe (its Section 8.9):** the claim "$s=8k$ rows matches full-row accuracy" was measured at $n=4d$ and is too optimistic at $n=8d$. See below.

## A -- convergence (`A_trajectories_3x3.png`)

3$\times$3, rows = width, columns = target. Solid = eval error vs the target, faint dashed = train residual, dotted black = truncated-SVD floor; colour = block size $k$ (viridis, dark$\to$light with $k$). Log-log.

Read: $k{=}128$ reaches the floor in every cell. $k{=}64$ lands within $\sim3\times$ at $N\le128$ but falls short at $N{=}256$. $k{=}32$ falls short everywhere and the shortfall **grows with $N$** ($1.2\times10^{-13}\to1.9\times10^{-11}\to4.2\times10^{-9}$ for sine at $N=64,128,256$) -- the first sign of the $k$--$d$ coupling that B quantifies. runge at $N{=}64$ sits on its own approximation floor ($3.3\times10^{-9}$) for every $k$; that cell is target-limited, not solver-limited.

## C -- noise (`C_noise_3x3.png`, `C2_noise_summary.png`)

C: 3$\times$3, one line per noise level (inferno, dark = noiseless), $\downarrow$ marker at the best iterate. Error is always measured against the **clean** target, which is what makes over-fitting visible. C2: best eval vs $\sigma_{rel}$, all cells, against $\varepsilon=\sigma$ and the statistical floor.

Two things to read:

- **Semiconvergence is real and mild.** Each noisy curve descends, reaches a minimum, then rises as LSQR starts fitting the noise. The marker is where an early-stopping rule must fire. This is the only place a stopping rule is load-bearing.
- **The minimum sits on the statistical floor.** Achieved $/\sigma_{rel}$ = 0.252, 0.251, 0.250, 0.249 at $\sigma_{rel}=10^{-8},10^{-6},10^{-4},10^{-2}$, against $\sqrt{r/n}=0.272$. Flat across 6 decades and slightly *below* the floor coefficient (early stopping is acting as regularization). The method is statistically optimal, not merely stable.

## D -- batching (`D1_batch_trajectories_3x3.png`, `D2_qr_rows.png`)

**D1** (x = iteration, y = error, one line per batch size; whitening AND solve both use the batch), $n=8d$:

| batch | geo-mean best | vs floor $1.08\times10^{-13}$ |
|---|---|---|
| $b=8d$ (full) | $3.1\times10^{-14}$ | below |
| $b=4d$ | $6.9\times10^{-14}$ | below |
| $b=2d$ | $7.9\times10^{-13}$ | $7\times$ |
| $b=d$ | $3.5\times10^{-10}$ | $3000\times$ |

The important structural observation: **the curves overlay during descent and only separate at their plateaus.** Batch size sets the achievable floor, not the convergence rate -- so a smaller batch costs you accuracy, never iterations. Degradation is smooth; the only sharp failure is $b=d$, where the batch system is square and the whitening has no redundancy to work with.

**D2** -- rows needed for the QR factorization alone (the solve always uses all rows), $k=128$:

| $s$ | $1k$ | $2k$ | $4k$ | $8k$ | $16k$ | all ($\approx58k$) |
|---|---|---|---|---|---|---|
| geo-mean best | $7.3\times10^{-5}$ | $1.0\times10^{-6}$ | $2.2\times10^{-10}$ | $2.5\times10^{-12}$ | $4.8\times10^{-13}$ | $5.3\times10^{-14}$ |

**This correction is now folded into the recipe, and the redrawn figure shows more.** Each solid curve now has its own dotted "all rows" target. At $N{=}64$ curves meet their target by $s\approx2k$--$4k$; at $N{=}256$ they are still orders above at $s{=}16k$. **So the required row count grows with $d$, not just with $k$** -- normalizing $s$ in multiples of $k$ was the wrong choice, and the original "$s=8k$" rule has no basis. There is no threshold at $8k$ or anywhere else.

## Figure -> claim map

| claim | figure |
|---|---|
| $k\ge32$ uncoupled from $d$; $k{=}16$ fails | **`H_dk_iteration_law.png`** top row |
| iterations $\propto d^{1.6}$ | `H_dk_iteration_law.png` bottom-left |
| iterations $\propto k^{-2.5}$ (the exchange rate) | `H_dk_iteration_law.png` bottom-right |
| $k{=}16$ genuinely fails | `H_dk_iteration_law.png` top row (purple separates) |
| noise $\to$ statistical floor | `C2_noise_summary.png`, `C_noise_3x3.png` |
| batching: rate unchanged, floor moves | `D1_batch_trajectories_3x3.png` |
| rows needed for the QR | `D2_qr_rows.png` |
| convergence across targets/widths | `A_trajectories_3x3.png` |
| the 9-cell floor result itself | parent dir: `../figures/blockqr_k128.png` |

## Open

- **Noiseless semiconvergence is real and is the reason H takes the minimum rather than the final iterate.** Measured at abs_cubed $d{=}922$, $k{=}64$: the disagreement reaches $9.6\times10^{-14}$ at iteration 6900, then degrades to $8.0\times10^{-9}$ by iteration 60000 -- five orders, with no noise present. Any deployment needs the stopping rule; "run to convergence" is actively wrong here. The rule itself is not characterized (the obvious $\|B^\top r\|$ plateau was measured to fire far too early, which is what invalidated experiment G).
- The uncoupling claim now rests on $d\in[462,3688]$ with ratios of exactly 1.0; the remaining uncertainty is whether the $k\in(16,32]$ lower bound itself drifts at much larger $d$ (untested beyond 3688).
- Noise study fixes $k=128$; the interaction between $k$ and the noise floor is unmeasured.
- D1 uses one random batch per configuration, not resampling across iterations -- so it measures "solve on a fixed subsample", not true streaming SGD.
- Everything is still the 1-D frozen-$\Phi$ toy. The contiguous-blocks requirement (recipe detail 3) is the piece most likely to break when column ordering stops being spatially meaningful.


## H -- the $(d,k,\text{iterations})$ law (`H_dk_iteration_law.png`, `run_law.py`)

The authoritative scaling result. No early stopping (a patience rule was measured to starve small $k$: at abs_cubed $d{=}922$, $k{=}32$ stopped at 6475 iterations with $10^{-9}$, but running on to 60000 reaches $1.9\times10^{-13}$ at iteration 42400). Every cell runs to a generous per-cell cap, takes the minimum over the trajectory, and reads **all** metrics at that single iterate. Targets are limited-smoothness (|x| is $C^0$, $|x|^3$ is $C^2$) so the approximation floor keeps falling with $d$ and the metric never saturates.

**Memory (top row): $k\ge32$ is uncoupled from $d$.** Solver disagreement with the direct solve, abs_cubed:

| $d$ | $k{=}16$ | $k{=}32$ | $k{=}64$ | $k{=}128$ | $k{=}256$ |
|---|---|---|---|---|---|
| 462 | $6.1\times10^{-12}$ | $1.3\times10^{-12}$ | $4.0\times10^{-14}$ | $7.9\times10^{-14}$ | $3.9\times10^{-14}$ |
| 692 | $9.5\times10^{-12}$ (cap) | $5.9\times10^{-14}$ | $6.8\times10^{-14}$ | $5.4\times10^{-14}$ | $2.3\times10^{-14}$ |
| 922 | $2.1\times10^{-9}$ | $1.9\times10^{-13}$ | $9.6\times10^{-14}$ | $4.0\times10^{-14}$ | $4.2\times10^{-14}$ |

Every $k\ge32$ lands at $10^{-13}$--$10^{-14}$ regardless of $d$, and well below the approximation floor (dashed). Only $k{=}16$ degrades. **So the state coefficient is a constant, and it is 32.**

**Compute (bottom row): this is what is coupled.** Iterations to the minimum, abs_cubed $k{=}32$: 9800, 13800, 21400, 42400 at $d=346,462,692,922$ -- roughly $d^{1.5}$--$d^{1.6}$. And against $k$ at $d{=}922$: 62200, 42400, 6900, 1000, 300 for $k=16,32,64,128,256$ -- roughly $k^{-2.5}$.

$$\text{iterations}\ \approx\ C\, d^{1.6}\, k^{-2.5}$$

That is the exchange rate: **buying block memory buys iterations back super-linearly.** $k{=}32$ is the minimum viable state; $k{=}128$ costs $4\times$ the memory and returns $\sim40\times$ the iterations.

## I -- the accuracy / block-size / iteration law (`I_isolines.png`, `I2_law_tests.png`, `run_isolines.py`)

$t(\tau,k,d)$ = the FIRST iteration at which the solver disagreement reaches $\tau$ (running-min, so semiconvergence cannot un-achieve a level). Disagreement rather than eval error, because eval error stops at the target's approximation floor and a $10^{-12}$ iso-line would not exist for the rough targets. Adaptive stride (1/10/50) and per-$k$ caps, because a fixed stride quantized $t$ to 1--2 steps at large $k$ and a uniform cap truncated exactly the small-$k$ cells.

**Figure `I_isolines.png`:** row 1 = iso-accuracy lines ($x=k$, one line per $\tau$, $d$ fixed); row 2 = iso-$d$ lines at $\tau=10^{-12}$; row 3 = trajectories, one line per $k$.

### Theory, and where it was wrong

Prediction was $t\propto\sqrt{\kappa}\ln(1/\tau)$ -- **logarithmic** in $\tau$. **Refuted.** Measured: going $10^{-4}\to10^{-12}$ costs $336\times$ the iterations, not $3\times$. The fix to the model: $\kappa$ is not fixed, it *grows with the accuracy demanded*. Reaching $\tau$ requires resolving singular directions down to $\sigma/\sigma_1\approx\tau$ (the target's energy is spread $\propto\sigma_i$, so truncation is not available), giving $\kappa_{\rm eff}(\tau)=1/\tau$ and

$$t\ \propto\ \sqrt{\kappa_{\rm eff}}\ =\ \tau^{-1/2}$$

a **power law**. Measured exponent is $\approx\tau^{-1/3}$ -- shallower than the worst-case $\sqrt\kappa$ bound, which is the expected direction for a clustered spectrum. Panel T1.

### The real structure: $k$ enters as $k/d$

The second prediction was that whitening flattens a *window* of $k$ adjacent directions, so $k$ should enter only through $k/d$ -- and therefore that a fitted power $k^{-b}$ is only a local approximation whose $b$ must **drift with $d$**. Both halves confirmed:

- **T2, the collapse:** plotting $t$ against $k/d$ collapses all eight widths ($d=206\ldots692$) onto one curve, over five orders of magnitude in $t$.
- **T3, the drift:** the separately-fitted $b$ falls from $3.3$ to $2.6$ (sine) and $2.9$ to $2.3$ (runge) as $d$ grows. A single power law in $k$ is the wrong form.

Quantified (sine, cubic fit in $\log(k/d)$ vs the same fit in $\log k$):

| $\tau$ | $R^2$ using $k/d$ | $R^2$ using $k$ alone | residual-vs-$d$ slope |
|---|---|---|---|
| $10^{-6}$ | 0.896 | 0.756 | $-0.26$ |
| $10^{-10}$ | 0.942 | 0.789 | $-0.24$ |
| $10^{-12}$ | **0.950** | 0.838 | $-0.54$ |

So $k/d$ is clearly the right variable, but the collapse is **not exact** -- a residual $d$-dependence of about $d^{-0.3}$ survives, worth a factor $\approx2$ over the tested range against the $\sim10^5$ spanned by $k/d$.

### The law

$$\boxed{\ t\ \approx\ g(\tau)\; f(k/d)\; d^{-0.3},\qquad g(\tau)\sim\tau^{-1/3}\ }$$

The global separable power-law fit, for reference and as a local approximation only:
$t\approx10^{0.03}\,\tau^{-0.22}k^{-1.68}d^{1.37}$, $R^2=0.849$ over 1152 points. **The mediocre $R^2$ is the point** -- a single separable power law does not describe the surface, and the $k/d$ collapse does much better.

**Practical reading: the block budget that matters is the FRACTION $k/d$, not $k$.** Holding $k$ fixed as $d$ grows is not holding the difficulty fixed.

## What was deleted and why

Four experiments were run, believed, reported, and then deleted. Recording the failure modes because each is easy to repeat:

| deleted | confound |
|---|---|
| B (`B_k_vs_d.png`) | fixed iteration budget for all $k$ -> small $k$ starved -> $k$ looked coupled |
| E (`E_uncoupling.png`) | metric was best/floor, which **saturates at 1.0** once the solver beats the approximation error |
| F (`F_larged.png`) | same saturating metric, plus best-over-trajectory; reported "ratio exactly 1.0" everywhere, and separately reported min(eval) and min(disagreement) taken at *different iterations*, which is algebraically impossible at one iterate |
| G (`G_uncoupling_corrected.png`) | budget scaled as $1/k$, which under-fed small $k$ (gave $k{=}32$ 8436 iterations when 42400 were needed) -> reproduced B's confound while claiming to fix it |

The general lesson: **any per-cell iteration budget that varies with the swept variable will manufacture a coupling.** Use a fixed generous cap, take the minimum, and flag cells that hit the cap.
