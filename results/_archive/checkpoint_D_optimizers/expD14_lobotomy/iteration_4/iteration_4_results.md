# expD14 iteration 4 -- the held-out signal, the latch, and the full suite

**Status: draft, pending Sam. 198 grid cells, a 132-cell suite, and seven probe-scale experiments. Floors are held-out-VALIDATED throughout (see "the finisher defect").**

## TL;DR

- **The optimizer is a two-mode design with the mode measured once, at init.** The held-out signal $s = \|r_{ho} + J_{ho}d^\star\|/\|y_{ho}\|$ (the error an exact solve would leave, refreshed by one amortized solve-event per 200 steps) separates every case measured by five orders ($s_0 \ge 0.2$ or $\le 2\times10^{-6}$, one boundary exception below). **Learn mode** ($s_0$ large): stock Adam, soft solve at $\alpha = r_{\text{entry}}$. **Preserve mode** ($s_0$ small): geometry gated by $\min(1, s/s_{\text{ref}})$, solve at $\alpha = s$.
- **It banks machine epsilon on `qi` in 1-D and 2-D.** 1-D: $2.5\times10^{-16}$, $1.5\times10^{-15}$, $1.3\times10^{-16}$ reached in-run, floors intact. 2-D (expE01's radon geometry at $N{=}576$, the construction that actually dominates): **reached $1.5\times10^{-14}$ in-run with the $1.6\times10^{-14}$ floor held**, while Adam degrades the same init to $3\times10^{-11}$ and reaches $6.8\times10^{-3}$.
- **It learns where learning is the task.** `rand`/`sine_8pi`: floor $0.43 \to 6.5\times10^{-7}$ (the uncooled winner, recovered); `datagap`: floors held or improved where Adam loses 3-5 orders; `rand_scale`: banks $10^{-8}$ in-run. On the dl tasks it is within $1.2$-$2.6\times$ of Adam's train MSE with test parity on parkinsons -- passes the litmus, does not win.
- **Two known defects, stated plainly.** (1) `rand`/`runge` sits exactly at the mode threshold ($s_0 \approx 9\times10^{-3}$ vs $s_{\text{ref}} = 10^{-2}$), classifies as preserve, and freezes $300\times$ short of learn mode; a threshold at $10^{-3}$ classifies every measured cell correctly but would be post-hoc tuning until confirmed on fresh cases. (2) dl parity gap: in learn mode the damped solve on the last layer displaces Adam's own adaptive steps there and contributes less on noisy data.

## The design, and how it was reached

Seven probe-scale experiments (400-12000 steps, single cells, each one knob, each checked against `docs/REQUIREMENTS.md` before running) drove four design revisions inside this iteration:

| probe | question | verdict |
|---|---|---|
| relative signal (`fgen`) | cool by inexpressible *fraction*? | dead: deadlock, and converged noise reads 100% inexpressible |
| lagged damped signal (`fabsd`) | deployable one-step-lag form? | dead: cannot separate "not solved yet" from "not expressible"; self-stalls at $\alpha \approx 0.1$ |
| absolute signal (`fabs`/`fabsc`) | cool $\propto s$? | qi/datagap superb; **kills learning** -- cooling proportional to remaining error is harmonic, not geometric, convergence (12000-step run: `rand` plateaus at $3.7\times10^{-3}$ forever) |
| the gate | $\text{cool} = \min(1, s/s_{\text{ref}})$? | fixes the harmonic bug at probe scale, but a *deep solve re-stuns learning even uncooled* (latch probe: $1.5\times10^{-3}$ vs $6.7\times10^{-7}$) -- the solve depth must also stay soft while learning |
| $s_{\text{ref}}$ sensitivity | is the one constant finicky? | robust across two decades (requirement 4) |
| fp32 | precision-agnostic? | yes with `finfo(dtype)`-derived constants: drives to fp32's floor ($3.9\times10^{-7} \approx 3\varepsilon_{fp32}$). Caveat: the floor instrument itself stayed fp64 |
| minibatch $b{=}256$ | the batching litmus | passes: floors intact, in-run depth at the $1/\sqrt b$ agreement scale, terminal solve recovers |

The surviving composition (`latch`) is exactly the two previously-measured winners, selected by the init-time signal: learn mode is iteration_2's uncooled $r_{\text{entry}}$ arm; preserve mode is the gate. No new state, no new passes, one threshold.

**The finisher defect, found by Sam's reading of the loss-curve figure.** The truncated min-norm terminal solve was *worse* than the in-run state in 26 of 132 cells: it overfits the fit rows on `datagap` (one cell: $5\times10^{-4}$ reached $\to 0.79$ "finished"), and a single solve is cancellation-limited at $10^{-14}$ where in-run every-step refinement reaches $10^{-16}$. The shipped finisher and the floor metric are now a **held-out-validated damped solve** (ladder of $\alpha$, keep the hold-out winner); all floors in this document use it.

**The 2-D benchmark defect, found by Sam's reading of the signal figure.** The previous 2-D cases (core15's small-$N$ Radon variant) sit below expE01's crossover where the Radon construction beats random ridges, so all signals read identically and nothing discriminated. The suite now uses expE01's `build_radon_tensor` at $N{=}576$ (floor$_0 = 1.6\times10^{-14}$ vs random's $2.1\times10^{-12}$), where the benchmark means something.

## Results

**The grid (validated floors, medians over seeds), latch | Adam-then-finisher | do-nothing-finisher:**

| case / target | latch floor | latch reached | Adam+fin | floor$_0$ |
|---|---|---|---|---|
| `qi`/`sine_8pi` | $3.0\times10^{-14}$ | $\mathbf{1.5\times10^{-15}}$ | $2.6\times10^{-8}$ | $3.2\times10^{-14}$ |
| `datagap`/`sine_8pi` | $\mathbf{3.9\times10^{-6}}$ | $7.8\times10^{-6}$ | $6.7\times10^{-1}$ | $8.6\times10^{-6}$ |
| `rand`/`sine_8pi` | $\mathbf{6.5\times10^{-7}}$ | $9.7\times10^{-1}$ | $6.7\times10^{-7}$ | $4.3\times10^{-1}$ |
| `rand`/`runge` | $5.3\times10^{-3}$ (defect 1) | $1.9\times10^{-1}$ | $\mathbf{1.6\times10^{-5}}$ | $9.4\times10^{-3}$ |
| `rand_scale`/`sine` | $4.8\times10^{-9}$ | $2.0\times10^{-8}$ | $4.7\times10^{-9}$ | $4.8\times10^{-9}$ |
| 2-D radon $N{=}576$ | $1.7\times10^{-14}$ | $\mathbf{1.5\times10^{-14}}$ | $3.0\times10^{-11}$ | $1.6\times10^{-14}$ |

Full 15-row 1-D table plus `clustered` (latch = Adam there, both $\sim$$5\times10^{-3}$, mode correctly "learn") in the JSONL.

**Scaling (S1, `sine_8pi`).** `qi`: $2.4\times10^{-9} \to 3.7\times10^{-14}$ over $N{=}32..256$, riding the theoretical floor -- from QI init, so it is the preservation-and-banking law, not the success criterion. `datagap` inverts with width above $N{=}64$ (the problem's own floor rises: more halo centers over the same starved gap). `rand` learns at every width ($10^{-5}$-$10^{-7}$), far above the criterion.

**The suite figures** (`suite/figures/`): S1 scaling 2$\times$2, S2 six-target loss curves 2$\times$3, S3/S4 signal traces 1-D and 2-D (the 2-D panels now discriminate: preserve-mode flat-frozen traces on radon, learn-mode on the core15 cases), S5 the four dl tasks, and 12 parameter-movement gifs (4 scenarios $\times$ latch/gate/adam; the `datagap` latch-vs-adam pair shows the gap neurons held vs scattered).

**Code & data.** `experiments/expD14_lobotomy/iteration_4/{core4.py, t8_fgen.py, suite/}`; data `t8_fgen.jsonl` (198 cells, arms `fgen`/`none`/`fabs`/`fabsc`/`gate`/`latch`), `suite/suite.jsonl` (132 cells), `suite/suite_v1_unvalidated_floors.jsonl` (superseded, kept for the finisher-defect audit trail).

## Conclusions

*Unsigned, pending Sam.*

1. One measured scalar, read once at init, selects between two measured-winner modes, and the composition reproduces both: machine epsilon banked in-run on the right geometry in 1-D and 2-D, and full-speed feature learning on stock-random init. Both dials (solve depth AND geometry motion) must stay soft during learning; each alone re-stuns.
2. The finisher must validate its damping on held-out data; raw min-norm refits are wrong in 20% of cells, worst exactly where data is scarce.
3. The mode threshold has one measured boundary failure (`rand`/`runge`) and the dl parity gap is real ($1.2$-$2.6\times$); neither is hidden by the headline cells.

## Open questions

- The mode threshold: $10^{-3}$ classifies every measured cell correctly -- confirm on held-out cases (new targets/seeds) rather than adopting post-hoc. Or replace the hard threshold with hysteresis ($s_0 < s_{\text{ref}}/10$ for preserve).
- dl parity: let Adam keep stepping $L$ in learn mode (`adam_on_L`, measured neutral-to-positive in iteration_0) so the damped solve supplements rather than displaces.
- Below-$s_{\text{ref}}$ geometry refinement: no mover exists that improves an already-good geometry (Adam only damages it; P3 shows `rand_scale` floors falling only $\sim N^{-2.4}$ with width). This is what stands between the current state and the success criterion from admissible init.
- `datagap`'s width inversion; the 2-D learn-mode cases at expE01 scale; noise-referenced $s$ for noisy regimes.
