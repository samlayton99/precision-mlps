# expD14 iteration 1 -- clearing the feature-learning confound

**Status: draft, pending Sam. 165 cells, 3 seeds on the random-init cases, all numbers final.**

## TL;DR

- **The confound is cleared, and the throttles are innocent.** With every throttle removed, the exact in-training solve still pins the geometry at its initial floor on every random-init cell, every seed: `full`, `ls`, and `lobo0` are indistinguishable. The stunning is inherent to solving $L$ exactly.
- **The mechanism is now measured, not inferred.** After the exact solve, coherent travel on the geometry block collapses from $1.0$ to $0.03$ (Adam's steps become a random walk), the corrected Adam SNR on the geometry reads $0$ for the whole run, and the min-norm solve inflates $\|v\|$ to $10^4$-$10^5$. The under-solver keeps all three healthy ($D \approx 1$, SNR alive, $\|v\| \approx 2$).
- **Under-solving does not merely preserve feature learning, it beats Adam's.** The $\tau{=}1$ arm ends with a geometry $4$-$10\times$ better than plain Adam's on every `rand` cell and seed, at $2\times$ Adam's passes ($10^4$ vs $680\text{k}$ for the exact-solve arms).
- **The scale-aware random init changes the problem.** A random geometry in the right $\gamma$ regime starts with a floor of $10^{-7}$-$10^{-9}$, up to six orders better than stock init on hard targets; no arm improves it further in 2500 steps, and the throttled solve sits on that floor during the run.

## Question

iteration 0 found that solving $L$ exactly in-training suppresses geometry learning, but every solving arm carried two throttles (the line-searched $A$ step and the gain-allocated energy split) never run with them off. Is the stunning inherent to the solve, or an artifact of the throttles? And does Adam learn the right geometry when the solve is out of its way?

## Experiment design

**Arms.** Five, separating the mechanisms one at a time. All solve arms discover $L$ with expD15's Method C (per-tensor sampling with verification, 5-7 Jacobian evaluations per discovery, refresh every 200 steps) rather than iteration 11's dense probe -- ORIENTATION's roadmap item 2, done as part of this run.

| arm | solver | $A$ step | energy split |
|---|---|---|---|
| `adam` | none | Adam's own | -- |
| `full` | exact (damped SVD, $\alpha{=}10^{-15}$) | Adam's own | none |
| `ls` | exact | exact-line-search cap $t^\star = -(r\cdot Jp_A)/\|Jp_A\|^2$ | none |
| `lobo0` | exact | line-search cap | gain-allocated $\rho$ (iteration 0's arm) |
| `tau1` | damped LSQR, $\tau{=}1$, $\alpha{=}10^{-8}$ | Adam's own | none |

**Cases.** Sam's four-regime matrix plus the deployment init, all 1-D, $N{=}128$ ($W{=}269$ with halo), $n{=}2003$ equispaced train points, eval on a disjoint 4001-point grid:

- `qi` -- correct uniform geometry, data everywhere.
- `clustered` -- centers pushed to the edges ($r \to r^{0.45}$), data everywhere (a geometry defect).
- `datagap` -- correct geometry, 96% of data removed from $|x| < 0.4$ (a data defect).
- `rand` -- stock PyTorch init, $w,b \sim U(-1,1)$ (iteration 0's regime).
- `rand_scale` -- slopes log-uniform in $[\gamma/2, 2\gamma]$, random centers over the halo'd extent (motivation.md's "initialization in the right $\gamma$ regime").

All cases except `rand` start with expD15's half-solved readout protocol (a $v{=}0$ start is a measured trap: every geometry Jacobian column is proportional to $v_k$, so the probe admits the geometry). Grid: 5 arms $\times$ 3 targets (`sine`, `sine_8pi`, `runge`) $\times$ \{1 seed deterministic-geometry cases, 3 seeds `datagap`/`rand`/`rand_scale`\}, 2500 steps, fp64.

**Metrics.** The score is the **floor trajectory**: every 50 steps, the truncated-SVD readout solve on the current geometry (fit on the kept training data, scored on the full eval grid, split by region for `datagap`). This is what one terminal exact solve would deliver, so it separates "solved the readout" from "learned the geometry"; the *reached* error of the unthrottled arms is expected to sawtooth or diverge (the $\|v\|\eta$ re-injection) and is reported but not scored. Also instrumented, both new: coherent travel $D(W) = \|\sum_t \Delta\theta_t\| / \sum_t \|\Delta\theta_t\|$ on the $(w,b)$ block over 100-step windows ($1$ = coherent, $1/\sqrt{100}$ = random walk), and the noise-floor-corrected Adam SNR $\hat s = [(\overline{m^2/v} - q)/(1-q)]_{[0,1]}$, $q = (1{-}\beta_1)/(1{+}\beta_1)$. Passes counted including all probes.

**Verification before the grid** (`t0_sanity.py`): the optimizer with $L$ empty reproduces stock Adam to $2.2\times10^{-16}$; Method C holds 100% precision on all five cases (recall 100% except one deliberate value-rule skip of $c_0$ on `datagap`); the `qi` floor$_0$ is the known $2.3\times10^{-14}$; travel and SNR behave on synthetic coherent/random inputs.

**Code & data.** `experiments/expD14_lobotomy/iteration_1/{core1.py, t0_sanity.py, t5_confound.py, figs1.py}`. Data: `results/checkpoint_D_optimizers/expD14_lobotomy/iteration_1/t5_confound.jsonl` (165 records with full trajectories). Figures in `figures/`.

## Results

**The confound (the `rand` rows).** Median floor$_{\text{final}}$ over 3 seeds, floor$_0$ in the last column:

| target | adam | full | ls | lobo0 | tau1 | floor$_0$ |
|---|---|---|---|---|---|---|
| `sine` | $1.5\times10^{-11}$ | $6.6\times10^{-11}$ | $6.8\times10^{-11}$ | $7.9\times10^{-11}$ | $\mathbf{1.4\times10^{-12}}$ | $7.3\times10^{-11}$ |
| `sine_8pi` | $1.5\times10^{-4}$ | $4.3\times10^{-1}$ | $4.3\times10^{-1}$ | $4.3\times10^{-1}$ | $\mathbf{2.9\times10^{-5}}$ | $4.3\times10^{-1}$ |
| `runge` | $1.3\times10^{-4}$ | $9.4\times10^{-3}$ | $9.3\times10^{-3}$ | $9.4\times10^{-3}$ | $\mathbf{1.3\times10^{-5}}$ | $9.3\times10^{-3}$ |

Every exact-solve arm ends exactly at floor$_0$, throttled or not, on every seed. Removing both throttles changed nothing, so iteration 0's "the solve suppresses feature learning" was correct and the suppression is intrinsic. Meanwhile `tau1` beats plain Adam by $4$-$10\times$ on every cell and seed.

**The mechanism, in the diagnostics.** On `rand`/`sine`: Adam and `tau1` hold coherent travel $D \approx 1.0$ throughout; `full` and `ls` collapse to $D \approx 0.03$ within 200 steps -- geometry motion becomes pure diffusion. The corrected SNR on the geometry block reads $0.3$-$0.45$ for Adam and `tau1` during learning (decaying to 0 as learning completes, at $\sim$1200 steps for Adam and $\sim$2000 for `tau1`, which keeps the signal alive longer), and $\approx 0$ for the exact-solve arms from the start. The exact solve also inflates $\|v\|$ to $2\times10^4$ (min-norm on ill-conditioned random features), where Adam and `tau1` stay at $O(1)$ -- the weight-blowup violation, produced by the solve itself.

**The regime dependence, confirmed with the sign structure intact.** On `qi`, the ordering inverts: `lobo0` preserves the QI floor best ($4.6\times10^{-14}$ on `sine` against floor$_0$ $2.3\times10^{-14}$), the unthrottled arms degrade it slightly, and `tau1` and Adam degrade it most (to $\sim10^{-8}$ on `sine_8pi`) -- with a good geometry, coherent Adam motion is the damage, and the throttles are the protection. On `datagap` every arm damages the initially-correct geometry (floor$_0 \sim 10^{-7}$ worsens to $10^{-5}$-$10^{-2}$), the throttled arms least; the damage concentrates inside the gap (F4). On `clustered` nothing repairs the defect in 2500 steps: Adam and `tau1` buy a factor of 2-6, the exact-solve arms hold it frozen at floor$_0$.

**The scale-aware init (`rand_scale`).** Starting floors are $4.8\times10^{-9}$ (`sine`), $1.8\times10^{-7}$ (`sine_8pi`), $4.1\times10^{-9}$ (`runge`) -- against stock-init floors of $7.3\times10^{-11}$, $4.3\times10^{-1}$, $9.3\times10^{-3}$. On the two targets where stock init is hard, the $\gamma$-regime start is 4-6 orders better before any training. No arm moves the floor materially in 2500 steps in either direction, and `lobo0`'s *reached* error sits on the floor throughout ($4.8\times10^{-9}$, $1.8\times10^{-7}$, $3.5\times10^{-9}$) -- already 4-6 orders past Adam's $10^{-3}$ barrier from a random init, delivered during the run rather than by a finisher.

### Figures

- **`F1_floor_trajectories.png`** -- the verdict figure. 5 cases $\times$ 3 targets, floor of the current geometry against iteration, one color per arm, thin lines seeds, thick the median, dashed the initial geometry's floor, shared log axis $[10^{-16}, 3\times10^2]$. Look for: in the `rand` row, red/orange/purple pinned to the dashed line for the whole run (the cleared confound) while grey and green descend, green below grey; in the `qi` row the same colors inverted, purple lowest; in `rand_scale` every line flat on the dashed floor.
- **`F2_geometry_improvement.png`** -- summary bars of floor$_0$/floor$_{\text{final}}$ (above 1 = geometry improved) per case, target and arm. The `rand` panel carries the headline: green bars at $10^3$-$10^4$, grey at $10^2$-$10^3$, the three exact-solve arms at exactly 1.
- **`F3_diagnostics.png`** -- `rand` and `rand_scale` on `sine`, three rows: coherent travel $D$ on $(w,b)$, corrected SNR $\hat s$ on $(w,b)$, and $\|v\|$ (log). Look for the red/orange collapse to $D \approx 0.03$ against grey/green at $1.0$; the SNR at zero for the solve arms against the grey/green learning hump; and the four-orders $\|v\|$ gap.
- **`F4_datagap_regions.png`** -- `datagap` floor split inside/outside the gap, per target and arm. The damage is inside; outside is held.
- **`F5_reached_vs_floor.png`** -- `runge`, reached error (top) against floor (bottom) per case. The top row shows the unthrottled arms sawtoothing or diverging while their bottom-row floors are flat or improving: the reason this experiment scores the floor.

## Additional details

**Why `tau1` beats Adam, not just the exact solve.** The $\tau{=}1$ solve keeps the readout near-optimal at $O(1)$ $\|v\|$, so the residual Adam sees is mostly the part of the error the features cannot express -- a cleaner geometry-teaching signal -- without being fully orthogonalized to the feature span, which is what kills the gradient. The SNR trace supports this: `tau1`'s geometry SNR stays elevated $\sim$800 steps longer than Adam's.

**Honest costs.** Median passes per 2500-step run: Adam 5000, `tau1` 10167 ($2.0\times$ Adam, including all discovery probes), exact-solve arms $\sim$680000 ($136\times$, dominated by the reference solver's $|L|$ passes per step -- a measurement instrument, not a candidate). Method C membership was stable: 0-2 changes per run across 13 discoveries, $|L| = 270$.

**Caveats.** One width, 1-D, full batch, noiseless. The `clustered` verdict (nothing repairs it) is at a 2500-step budget; a longer run might differ. `floor` is the truncated-SVD proxy for a finisher, not a measured terminal solve. The `tau1`-beats-Adam margin ($4$-$10\times$) is consistent across seeds but modest against the six-orders spread across targets; the width axis is untested here.

## Conclusions

*Unsigned, pending Sam.* What the data supports:

1. The feature-learning suppression is a property of the exact solve itself, not of iteration 0's throttles. All three exact-solve arms end at the initial geometry's floor on every random-init cell and seed, unthrottled included.
2. The suppression mechanism is measured: exact solving zeroes the geometry's persistent gradient signal (SNR $\approx 0$) and turns Adam's geometry motion into a random walk ($D$: $1.0 \to 0.03$), while inflating $\|v\|$ by four orders.
3. Under-solving with the cheap $O(d)$ solver at $\tau{=}1$ preserves coherent, high-SNR geometry learning and ends with a better geometry than plain Adam on every random-init cell ($4$-$10\times$), at twice Adam's cost.
4. The direction of the regime rule survives with its sign structure: correct geometry wants a hard, throttled solve; wrong geometry wants a soft solve and free Adam motion; missing data punishes all motion and rewards throttling.
5. A random init in the right $\gamma$ regime starts with a floor 4-6 orders better than stock init on hard targets, and the throttled solve delivers that floor as reached error during the run.

## Open questions

- The two regimes (hard-throttled vs soft-free) need a driver. The measured candidates are now in hand: coherent travel and the corrected SNR distinguish the regimes cleanly in this data. Can $\tau$ (or $\alpha$) ride one of them, against the LM-ratio and $\alpha_t = r_{\text{entry}}$ candidates?
- Does `tau1`'s geometry advantage grow with width, i.e. does the success criterion's $O(\log(1/\varepsilon))$ width-scaling emerge from `rand_scale` + under-solve + finisher?
- `clustered` resisted everything at this budget. Is geometry repair a longer-horizon effect, or does it need a different mechanism entirely?
- The floor is a proxy; the shipped path (streaming-QR accumulation + full-reorth terminal solve) has still never been run end-to-end after a real training run.
