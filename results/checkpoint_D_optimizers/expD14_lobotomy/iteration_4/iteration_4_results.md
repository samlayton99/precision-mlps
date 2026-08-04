# expD14 iteration 4 -- the held-out signal, the assembled optimizer, and the full suite

**Status: draft, pending Sam. 132 grid cells plus an 84-cell suite (scaling, six targets, gifs, 2-D, 2-layer, real data).**

## TL;DR

- **One measured scalar now drives the whole optimizer.** $s = \|r_{ho} + J_{ho}d^\star\|/\|y_{ho}\|$ -- the held-out error the exact solve would leave -- sets the damping ($\alpha = s$) and cools Adam's geometry step (factor $s$). Refreshing it from one amortized solve-event per 200 steps (`fabsc`) is indistinguishable from computing it every step, which is what makes it requirements-compatible.
- **Where it wins, it wins completely.** `qi`: machine epsilon reached in-run ($2.4\times10^{-16}$, $1.5\times10^{-15}$, $1.4\times10^{-16}$) with floors intact, and the width-scaling law has the success-criterion shape ($2.4\times10^{-9} \to 1.6\times10^{-13} \to 3.1\times10^{-14}$ over $N{=}32..128$, riding the theoretical floor, while Adam+finisher sits 5 orders above). `datagap`: best result ever recorded, at or above the do-nothing optimum at every width, while Adam's damage *grows* with width. `rand_scale`: banks $10^{-8}$-$10^{-6}$ in-run where Adam reaches $10^{-3}$.
- **Where it loses, it loses honestly: the cooling taxes bulk feature learning.** On stock-random hard targets the uncooled arm's geometry is still far better ($6.7\times10^{-7}$ vs $4.9\times10^{-2}$ on `sine_8pi` at 4000 steps -- `fabsc` improves the floor only $9\times$); in 2-D and on the multilayer real-data tasks Adam's floors are $2$-$4\times$ better. It is never unstable and always passes the dl_test litmus (within $2$-$3\times$ of Adam's MSE), but it does not win there.
- **Two variants were killed at the probe stage**, and the deaths are informative: the relative-fraction signal (`fgen`) deadlocks (`rand`/`sine`: "features fine" + "error too high to solve" = nothing ever moves) and misreads converged noise as 100% inexpressible, releasing Adam onto finished geometry; the one-step-lag damped-direction signal (`fabsd`) cannot distinguish "solve not deep yet" from "geometry needed" and self-stalls at $\alpha \approx 0.1$.

## The rule, and the gate

At each solve-event (every 200 steps; the spec's amortized cadence): compute the exact truncated solve $d^\star$ on the fit rows, evaluate its leftover on a held-out 10% split of the training data, $s = \|r_{ho} + J_{ho}d^\star\|/\|y_{ho}\|$. Until the next event: $\alpha_t = s$ (solve down to the error that generalizes, no deeper) and $\Delta_A \mathrel{*}= s$ (move the geometry in proportion to the error that is real). The hold-out is drawn from the *kept* training distribution, which is the load-bearing subtlety on `datagap`.

Section-8 checklist: no loss comparisons ($s$ is a residual level, linear in $r$); its floor reading is honest ($10^{-14}$ on noise, where the relative version reads 1.0); no estimator-zero test; battle-tested analog (validation-driven control); state $O(1)$ beyond Adam; per-step cost unchanged, one solve-event per 200 steps amortized exactly as expD10's tier model budgets. The reference instrument computes $d^\star$ from a materialized SVD; in deployment the solve-event uses the tier-3 machinery (streaming QR / subsampled exact solve). That composition is built into the cost model but not yet run end-to-end.

## Grid results (t8, same protocol as t6/t7)

| case / target | fabsc floor | fabsc reached | uncooled floor | floor$_0$ |
|---|---|---|---|---|
| `qi`/`sine_8pi` | $3.1\times10^{-14}$ | $\mathbf{1.5\times10^{-15}}$ | $2.6\times10^{-8}$ | $3.2\times10^{-14}$ |
| `datagap`/`sine_8pi` | $\mathbf{5.5\times10^{-6}}$ | $1.0\times10^{-5}$ | $6.7\times10^{-1}$ | $8.7\times10^{-6}$ |
| `rand`/`sine_8pi` | $4.9\times10^{-2}$ | $9.8\times10^{-1}$ | $\mathbf{6.7\times10^{-7}}$ | $4.3\times10^{-1}$ |
| `rand_scale`/`sine_8pi` | $1.8\times10^{-7}$ | $3.9\times10^{-6}$ | $2.1\times10^{-7}$ | $1.8\times10^{-7}$ |

(15 case-target rows in the JSONL; `fabs`, the every-step twin, matches `fabsc` within seed noise everywhere -- the cadence approximation costs nothing.)

## The suite

**Scaling (S1, `sine_8pi`, $N \in \{32..256\}$).** The `qi` panel is the program's success-criterion shape: the optimizer's post-finisher floor falls $2.4\times10^{-9} \to 3.1\times10^{-14}$ and its in-run reached error goes *below* the floor to $10^{-15}$-$10^{-16}$ at $N \ge 128$, while Adam+finisher plateaus at $10^{-6}$-$10^{-8}$. The `datagap` panel shows the problem's own floor rising with width (more halo centers over the same starved gap) -- the optimizer tracks the do-nothing optimum at every width while Adam's damage grows past $10^{0}$ at $N{=}256$. `rand` is Adam's panel: the cooled optimizer's floor is flat at $4\times10^{-2}$ against Adam's $10^{-4}$.

**Six targets (S2, stock-random init).** The reached/floor curves show the same split: competitive on `sine`/`exp`, behind Adam on the hard oscillatory targets.

**Signals (S3 1-D, S4 2-D).** The four scenarios on one function each, showing $\alpha$, $s$, $\hat s$, and the error: the ladder self-clocking on `qi`, holding at the noise level on `datagap`, and sitting high (soft) on `rand`. The 2-D panels include $\mu = (\alpha\sigma_1)^2$.

**Parameter movement (8 gifs).** Per scenario, optimizer vs Adam side by side: the model forming against the target (left) and every neuron's (center, bandwidth) trajectory colored red-to-blue by initial center (right). The `datagap` pair is the argument in miniature: Adam's gap neurons scatter, the optimizer's stay put.

**2-D (four ridge cases).** The optimizer banks its floor in-run (reached $8.7\times10^{-3}$ vs Adam's $1.8\times10^{-2}$) but Adam learns 2-D geometry $2$-$4\times$ better ($1.9\times10^{-3}$ vs $6.6\times10^{-3}$ post-finisher on `qi`-2D). 2000 steps, $N{=}48$: small-scale, and the cooling tax dominates.

**Multilayer (dl2 + the three real tasks).** Method C finds the last layer everywhere (33 and 65 members). Final test MSE -- airfoil: Adam $1.1\times10^{-2}$, optimizer $2.2\times10^{-2}$; parkinsons: $5.4\times10^{-2}$ vs $5.4\times10^{-2}$ (tie, optimizer marginally better); bike: $6.5\times10^{-3}$ vs $9.0\times10^{-3}$. Passes the litmus (stable, sustained, competitive order), does not win. On noisy real data $s$ floors at the irreducible noise level, permanently half-cooling Adam -- a measured, explainable cost.

**Code & data.** `experiments/expD14_lobotomy/iteration_4/{core4.py, t8_fgen.py, suite/}`. Data: `results/.../iteration_4/t8_fgen.jsonl`, `results/.../iteration_4/suite/suite.jsonl`. Figures and gifs under `results/.../iteration_4/{figures, suite/figures}`.

## Conclusions

*Unsigned, pending Sam.*

1. The optimizer is assembled and every control input is a free measurement: Method C (what), $\alpha = s$ (how hard), $\Delta_A \cdot s$ (the handoff), one terminal solve (the banking). On the regimes the program was founded on -- correct or near-correct geometry, data-starved regions, right-scale random init -- it is strictly better than Adam-plus-finisher, often by many orders, and it exhibits the success criterion's width-scaling shape from QI init.
2. Its single measured weakness is a learning-speed tax wherever bulk nonlinear feature learning is the whole task (stock-random init, 2-D at small scale, deep real-data nets). The uncooled variant wins there and remains available; the tension between safety and learning speed is now a one-parameter tradeoff on a measured signal rather than a design mystery.
3. The cadence form makes it requirements-compatible; the end-to-end composition with the deployable tier-3 solve-event machinery is the remaining engineering step.

## Open questions

- Can a softer cooling law (e.g. $\sqrt{s}$, or a floor on the cooling factor) recover stock-random learning speed without giving back `qi`/`datagap`? One-knob, falsifiable.
- `datagap`'s floor worsens with width (halo centers over a starved gap) -- is that intrinsic to the geometry or fixable by coverage-aware center placement?
- Width scaling from `rand_scale` init (the deployment path) was not in the scaling grid and is the natural route to the success criterion without constructive init.
- The real-data tasks floor $s$ at the noise level; should the cooling factor be $s$ relative to the *achievable* (current-floor) error rather than absolute, in noisy regimes?
