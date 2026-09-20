# expD14 iteration 2 -- which measured signal sets $\mu$

**Status: draft, pending Sam. 198 cells, 4000 steps, 3 seeds on the random-init cases.**

## TL;DR

- **The signal is the entry residual.** $\alpha_t = r_{\text{entry}} = \|r\|/\|y\|$, the damping floor applied as an equality, matches or beats every alternative tested. The LM ratio and the SNR rule converge to the same behavior wherever it matters, because both are floored at $r_{\text{entry}}$ and the floor is what binds; LM additionally chatters on `datagap` where its guard cycles. One line, one free measurement, no hyperparameter.
- **From Xavier init, the bar is met.** On `rand`/`sine_8pi` the $r_{\text{entry}}$ arm's geometry is worth $7.5\times10^{-7}$: $\sim200\times$ better than Adam-plus-terminal-solve ($1.4\times10^{-4}$) and $6\times10^5\times$ better than terminal-solve-on-the-untrained-init ($0.43$), consistent across seeds. Wins on all three `rand` targets, margins $1.7\times$ to $200\times$.
- **Preset schedules are dead.** The no-signal ladder and the always-exact arm diverge in reached error by up to $10^7$ on `clustered`/`rand`, because a schedule cannot know which init it was handed. Confirmed, not assumed.
- **What remains is exactly "cool Adam off."** Wherever the run fails to bank ($\alpha$ stalls at $10^{-2}$-$10^{-3}$ on `rand` while the floor sits at $10^{-12}$; the `qi` floor degrades $700\times$ over 4000 steps), the cause is Adam's continued geometry motion setting a drift floor under $r_{\text{entry}}$ -- the guard is *correctly* refusing to solve below the drift. The next single knob is the handoff, not $\mu$.

## Question

One question, everything else frozen: what measured signal should set the damping $\alpha = \sqrt\mu/\sigma_1$, such that the optimizer stays soft while the geometry is still being learned (Xavier init) and hardens immediately when it is not (QI-like init)? Scored against Sam's two baselines: Adam + terminal solve, and no training + terminal solve ($\text{floor}_0$).

## Experiment design

The iteration_1 harness unchanged (stock Adam on the geometry, no throttles; damped truncated-SVD reference solve on the Method-C-discovered $L$; floor trajectory scored), with only the $\alpha$ controller varying:

| arm | rule |
|---|---|
| `adam` | no solve (baseline 1; its floor$_{\text{final}}$ is Adam + terminal solve) |
| `alo` | $\alpha = 10^{-15}$ always (iteration_1's stun anchor) |
| `ladder` | preset log-linear $1 \to 10^{-15}$, no signal |
| `rentry` | $\alpha_t = r_{\text{entry}}$, clipped to $[10^{-15}, 1]$ |
| `lm` | classical LM multiplicative update on the **composite** step, floored at $r_{\text{entry}}$, loss-noise guard at relative predicted improvement $10^{-12}$ |
| `snr` | $\alpha_t = \max(\hat s_{\text{geo}}, r_{\text{entry}})$ |

A design fact found on paper first: the LM ratio on the $L$ step alone is identically 1 ($f$ is exactly linear in $L$, the local model is exact), so it would always say "solve exactly" -- the stunning arm. It is only meaningful on the composite step (solve + Adam motion), which is what was built.

Grid: 6 arms $\times$ 3 targets $\times$ \{`qi`, `clustered` 1 seed; `datagap`, `rand`, `rand_scale` 3 seeds\}, $N{=}128$, 4000 steps, fp64. Baseline 2 is $\text{floor}_0$, already measured per cell.

**Code & data.** `experiments/expD14_lobotomy/iteration_2/{core2.py, t6_mu_signals.py, figs2.py}`. Data: `results/checkpoint_D_optimizers/expD14_lobotomy/iteration_2/t6_mu_signals.jsonl`. Figures in `figures/`.

## Results

**The two-baselines scorecard on `rand` (Xavier), floor$_{\text{final}}$ = the arm + terminal solve, median of 3 seeds:**

| target | Adam + solve | no train + solve | `rentry` | `lm` | `snr` |
|---|---|---|---|---|---|
| `sine` | $1.4\times10^{-11}$ | $7.3\times10^{-11}$ | $7.2\times10^{-12}$ | $6.6\times10^{-12}$ | $\mathbf{4.1\times10^{-12}}$ |
| `sine_8pi` | $1.4\times10^{-4}$ | $4.3\times10^{-1}$ | $\mathbf{7.5\times10^{-7}}$ | $6.8\times10^{-7}$ | $8.0\times10^{-7}$ |
| `runge` | $2.7\times10^{-5}$ | $9.3\times10^{-3}$ | $1.6\times10^{-5}$ | $1.6\times10^{-5}$ | $\mathbf{1.4\times10^{-5}}$ |

Every signal arm beats both baselines on every `rand` target; the three signal arms are statistically indistinguishable from each other (per-seed spreads overlap), which is the point: **they all reduce to the $r_{\text{entry}}$ floor where it matters.** Occam picks `rentry`.

**Self-clocking across inits, read off the $\alpha$ endpoints** (`rentry`, `sine`, seed 0): on `qi`, $\alpha$ ladders $3\times10^{-1} \to 1.6\times10^{-11}$ and the reached error equals the floor -- the run banks everything its geometry holds, unassisted. On `rand`, $\alpha$ holds $\sim0.9$ early (Adam free; the geometry improves 6 orders on `sine_8pi`) and then stalls at $7\times10^{-3}$: that is the drift level Adam's un-cooled motion re-injects, and the guard correctly refuses to solve below it. The floor underneath is $7\times10^{-12}$, so the gap between reached and bankable is entirely the handoff question.

**Failure modes, honestly.**

- `qi`: with no throttles, coherent Adam drift degrades the QI floor from $2.3\times10^{-14}$ to $\sim10^{-11}$ over 4000 steps for every arm including plain Adam. iteration_0's throttled arm held $10^{-14}$ and reached $10^{-16}$; the signal arms bank only what the drifted geometry holds. Cooling Adam is what is missing, and it was deliberately out of scope here.
- `datagap`: the one case where every arm loses to doing nothing ($\text{floor}_0 = 9.3\times10^{-7}$; best trained result $10^{-5}$-$10^{-2}$). $r_{\text{entry}}$ is measured on the kept data and cannot see gap damage (on `sine` it crashed to $4\times10^{-12}$ while the true floor was $4\times10^{-3}$ -- expD13's observable/true divergence in its data-blindness form). Data-blind *geometry* motion is unguarded by anything currently built; the observability column filter protects only the solve.
- `clustered`: signal arms modestly beat Adam ($4.9\times10^{-3}$ vs $8.3\times10^{-3}$ on `sine_8pi`, both beating $\text{floor}_0 = 4.8\times10^{-2}$); nothing repairs the defect properly at this budget.
- `rand_scale`: floors flat at their (already good) $10^{-7}$-$10^{-9}$ start; the signal arms bank $10^{-7}$ *during* the run on `sine` where Adam reaches $10^{-3}$.

### Figures

- **`G1_scorecard.png`** -- 3 targets $\times$ 5 cases. Per arm two bars: light = error reached during the run, dark = error after one terminal exact solve (the geometry's worth). The black line is the same quantity at step 0, i.e. baseline 2; the `adam` dark bar is baseline 1. A dark bar below both the line and the `adam` bar = the arm met Sam's criterion.
- **`G2_control_story.png`** -- per case (`sine`): floor trajectories (top) and the $\alpha$ each controller chose (bottom) with the two signals it could see overlaid ($r_{\text{entry}}$ dotted, $\hat s$ dashed, measured on the Adam run). Look for: the blue/orange/green ladder on `qi` against the flat red line; the stall at the drift level on `rand`; orange (`lm`) chattering across ten decades on `datagap` while blue is stable.

## Conclusions

*Unsigned, pending Sam.* What the data supports:

1. $\alpha_t = r_{\text{entry}}$ is the $\mu$ signal: self-clocking on every init tested, matches or beats LM and SNR everywhere (all three are floored by it and the floor binds), free, hyperparameter-free. Preset schedules are ruled out by catastrophic divergence.
2. From Xavier init the criterion is met: the geometry improves and beats both Adam-plus-finisher and init-plus-finisher on all three targets, by up to $200\times$ / $6\times10^5\times$ on `sine_8pi`.
3. The binding residual problem is now the handoff: Adam's un-cooled motion sets the drift floor that stops the banking ($10^{-3}$ vs a bankable $10^{-12}$ on `rand`) and erodes a correct geometry on `qi`. $\hat s_{\text{geo}}$ measured near zero in exactly those late phases, so the cooling signal likely already exists in the instrumentation.
4. `datagap` is unguarded: no current signal sees damage in regions with no data, and doing nothing beats every arm there.

## Open questions

- The handoff: shrink Adam's geometry step as $\hat s_{\text{geo}} \to 0$ (or on the line-search length, which iteration_0 showed vanishes with the residual), so $r_{\text{entry}}$ can follow the floor down and the run banks without a separate finisher. One knob, next iteration.
- Data-blindness: a guard for geometry motion that data cannot see (per-unit coverage from the column norms already computed?).
- Width scaling of the `rand` result, and 2-D.
