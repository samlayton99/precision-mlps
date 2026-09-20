# expD18 -- the win-condition experiment: QI optimizer x QI-inspired init on real tabular regression

**Status: draft -- pending Sam's review.**

## TL;DR

- **In-run, the two optimizers are the same line.** qiopt's pre-finisher error matches Adam's final in every cell (e.g. airfoil/std 5.23e-1 vs 5.23e-1); the periodic in-run solves added nothing measurable on these tasks.
- **The entire 35/36-cell final-error margin (4-20%) is the terminal solve.** Adam's own post-train probe -- i.e. plain Adam plus one validated lstsq refit at the end -- matches qiopt's final everywhere (sarcos 0.181 vs 0.182; beijing 0.154 vs 0.154) and is slightly *better* on bike_sharing and parkinsons. So the honest claim is "Adam + one terminal refit beats plain Adam" (expD02 win #2, now on tabular), **not** "the lobotomized optimizer beats Adam." The distinctive in-run mechanism is safe but earned nothing here, at 1.48x Adam's passes.
- **The 250-step probes make the parity graphic**: the would-be terminal solve agrees between the two arms at every probe step (within 1% on the well-sized tasks), sits below the live trajectories throughout training, and on the two smallest tasks reveals geometry overfitting (the would-be solve worsens after ~1000 steps on airfoil) that the raw curves hide.
- **The solve did not stun geometry learning.** The feared mechanism did not appear: the QI optimizer's post-train geometry probe matches Adam's within 0.4% (median; worse by >5% in only 3/36 cells). The damping floor $\alpha = r_{\rm entry}$ kept every in-run solve soft ($\alpha$ 0.16-0.91, median 0.29) because the entry residual is noise-dominated on real data.
- **The high-d QI-inspired init does not carry the geometry side by itself**: its untrained probe loses to the geometry Adam finds from standard init on all 6 tasks (PROGRAM_FRAMING §7.4). Its value is as a starting point: post-train geometry from QI init beats post-train from standard init on 4/6 tasks.
- Cost: 1.48x Adam's passes per step (the line-search JVP) plus 20 offline SVD solve events and one validated finisher per run.

## Question

On real tabular regression with the high-dimensional QI-inspired init, does the lstsq-lobotomized optimizer beat plain Adam, and does it hold parity on standard init? This is the one unmeasured cell of PROGRAM_FRAMING's win condition (§5, §7.3).

## Experiment design

The full 2x2 of PROGRAM_FRAMING §3 per task: {Adam, QI optimizer} x {standard init, QI-inspired init}, 3 seeds, single-hidden-layer tanh MLP (the expF04 d=1 control showed the QI-init advantage is tanh-bound), width 256, fp64, CPU.

- **Tasks (6):** superconductivity (81-d), sarcos (21-d), airfoil (5-d), parkinsons (19-d), bike_sharing (12-d), beijing_pm25 (13-d) from the expF04/all20 cache (`data/cache_all20/`), min-max normalized to $[-1,1]$ (targets too), train capped at 30k rows, the tasks' own train/test splits. Every 10th train row is held out of all gradients and solves for validation of probe/finisher damping; fits use the remaining 90%.
- **Inits:** `std` = PyTorch default. `qi` = expF04's best variant (`scaled_psqrt`, the all20 winner): $\sqrt{N}=16$ ridge bundles, each bundle one random unit direction $u_m$ with a 1-D QI sweep along it, per-ridge $\gamma_m=\lambda^*/h_m$ ($\lambda^*=0.25$, $h_m = 2A_m/16$ from the direction's own projection range $A_m$), centers sampled from the data projections, $b_k=-\gamma c_k$. Readout at default init in all arms.
- **Optimizers.** Both arms share one manual-Adam code path (lr $10^{-3}$, batch 128, 4000 steps), so the only difference is the solve machinery. The **QI optimizer** adds: (i) every 200 steps, a damped correction solve of the readout on the fit rows, $d=\arg\min\|Ad+r\|^2+\mu\|d\|^2$ with $\mu=(\alpha\sigma_1)^2$ and the damping floor $\alpha=\|r\|/\|y\|$ at entry; (ii) after the first solve, the exact-line-search cap $t^*=-(r\cdot Jp)/\|Jp\|^2$ clipped to $[0,1]$ on Adam's geometry step (one closed-form JVP per batch); (iii) one validated terminal finisher: full damped solve with the damping chosen on the held-out rows, the in-run readout competing as a candidate (the raw min-norm refit was worse than the in-run state in 26/132 expD14 cells). $L$ is hard-set to the readout; expD15's exact-zero rule would return exactly this set on this architecture, so discovery is bypassed, not violated.
- **Scoring (PROGRAM_FRAMING §4.3).** Two lstsq probes per cell, solved on the fit rows with held-out-validated damping and **scored on the test set**: pre-train (raw init geometry) and post-train (readout thrown away, geometry as trained). From these: geometry score (post vs pre), readout score (own final vs post), worth-running-at-all (pre vs own final). Metric throughout: test rel $L_2=\|\hat y-y\|_2/\|y\|_2$.
- **Cost accounting:** Adam 2 passes/step; QI optimizer 2 + 1 JVP/step after the first solve $\approx$ 3/step (measured 11,820 vs 8,000 per 4000 steps, ratio 1.48), plus 20 SVD solve events and 1 finisher of offline $O(n d_L)$/SVD work per run.

**Code & data.** `experiments/expD18_qi_opt_tabular/` (`run.py`, `qi_opt.py`, `config.yaml`); gate test `tests/test_expD18_qi_opt_tabular.py` (validated solve reaches $3.7\times10^{-14}$ on a frozen halo'd 1-D geometry; the halo-less expF04 ridge init floors near $3\times10^{-7}$ by geometry, documented). Data `results/checkpoint_D_optimizers/expD18_qi_opt_tabular/data/rows.jsonl` (72 cells; rerun with the 250-step would-be-terminal-solve probes in each row's `probes` field, 276 s wall on 8 workers). Figures `figures/expD18_curves.png`, `figures/expD18_probes.png`.

## Results

Median final test rel $L_2$ over seeds (own final readout):

| task | adam/std | qiopt/std | adam/qi | qiopt/qi |
|---|---:|---:|---:|---:|
| superconductivity | 2.29e-1 | 2.21e-1 | 2.31e-1 | **2.17e-1** |
| sarcos | 2.10e-1 | 1.84e-1 | 1.92e-1 | **1.71e-1** |
| airfoil | 5.22e-1 | **4.19e-1** | 4.95e-1 | 4.38e-1 |
| parkinsons | 7.87e-1 | 7.85e-1 | 7.47e-1 | **7.02e-1** |
| bike_sharing | 3.04e-1 | 2.94e-1 | 2.91e-1 | **2.78e-1** |
| beijing_pm25 | 1.70e-1 | **1.55e-1** | 1.67e-1 | **1.55e-1** |

- **qiopt's final beats Adam's final in 35/36 same-init cells, but the fair baseline is Adam + terminal refit, and that baseline is not beaten.** Adam's post-train probe equals qiopt's final within noise on every task (and beats it slightly on bike_sharing and parkinsons). The in-run curves of the two optimizers are nearly identical (curves figure; qiopt pre-finisher vs Adam final agree per cell), so the whole margin is the finisher harvesting Adam's unclaimed readout gap (e.g. sarcos/qi: 1.92e-1 own vs 1.71e-1 probe). The finisher validated to a damped solve in 65/72 runs ($\alpha$ between $10^{-12}$ and $10^{-4}$) and to the exact truncated solve in the rest; it never chose "keep the in-run readout."
- **Why the in-run solve is inert here:** the damping floor sets $\alpha = r_{\rm entry}$, and on these noise-dominated tasks $r_{\rm entry}$ stays $O(10^{-1})$ all run, so every mid-run solve is damped to roughly the accuracy Adam already has. The rule that prevents the stun also neuters the mid-run solve in this regime; the value concentrates entirely in the terminal solve.
- **No stun.** qiopt's post-train probe is within 0.4% of Adam's at the median (max 13% worse, 3/36 cells above 5%): with the damping floor riding the noise-dominated entry residual (in-run solve $\alpha$ median 0.29), the periodic solve left feature learning intact. The 1-D/2-D stun result does not carry to this regime at these damping levels.
- **Geometry scores.** From qi init the trained geometry improves on its own pre-probe substantially (sarcos 3.71e-1 $\to$ 1.71e-1); from std init the improvement is small, and on airfoil and parkinsons Adam's training *worsens* the std-init geometry slightly. Post-train geometry from qi init beats std init on 4/6 tasks (airfoil the exception, beijing a tie).
- **Worth-running-at-all:** training beats init+solve everywhere except airfoil (1,203 rows), where adam/std's own final (5.22e-1) is worse than its own pre-probe (3.88e-1) -- overfitting the smallest task; the QI optimizer's validated finisher recovers most of it.
- **The §7.4 number:** the untrained qi-init probe loses to Adam-from-std's post-train probe on all 6 tasks (sarcos 3.71e-1 vs 1.83e-1; parkinsons nearly tied, 8.21e-1 vs 8.20e-1). On tabular data the high-d init is not, by itself, a better geometry than what Adam finds; the information-theory suspicion of PROGRAM_FRAMING §2/§3 is confirmed with a number for this suite.

### The would-be-terminal-solve probes (added on rerun)

Every 250 steps (plus step 0 and the final step), both arms run a **non-invasive terminal-solve probe**: snapshot the geometry, run the exact validated-finisher procedure (damping chosen on held-out rows, in-run readout competing), score on test, discard. The live run is never touched. What the probes show:

- **Adam + terminal solve = QI optimizer, at every point in training, not just the end.** On superconductivity, sarcos, and beijing the two probe trajectories agree within 0.3-0.8% at every probe step; bike_sharing within 2-4%; airfoil and parkinsons (the two smallest, noisiest tasks) wiggle up to 9% with no consistent sign. Nowhere do the qiopt probes sit systematically below the Adam probes -- there is no in-run geometry benefit from the solve machinery. Where a consistent tiny gap exists (bike_sharing: 0.286 vs 0.294 at step 4000) it favors *plain Adam's* geometry.
- **The would-be solve beats the live trajectory throughout training.** The probe markers sit below the trajectory lines from step 250 onward (largest early: sarcos probe 0.188 at step 1000 while the trajectories are still at 0.21-0.23); the gap narrows as Adam converges. At any stopping point, geometry + validated solve is ahead of the raw readout.
- **Probe evolution tracks geometry health.** On the four well-sized tasks the probe improves monotonically under both arms (the geometry keeps getting better). On airfoil the probe *worsens* after ~1000 steps (0.378 at its best, 0.437 at the end) and on parkinsons/std it worsens from step 0 (0.763 to 0.820): geometry overfitting on the small tasks, visible only through the probe. On those tasks an early-stopped geometry + solve would beat the terminal solve.

### Figures

- **`figures/expD18_curves.png`** -- 2x3 panels (tasks), test rel $L_2$ vs iteration, log y shared across panels; color = optimizer (blue Adam, red QI optimizer), linestyle = init (dashed std, solid qi), median over 3 seeds; x markers (qi init) and + markers (std init) = the would-be validated terminal solve at that step, colored by optimizer. What to see: the four lines converge to near-coincidence in-run, and the blue and red probe markers overprint each other everywhere -- Adam + terminal solve equals the QI optimizer at every probe step. On airfoil the markers drift *up* after step ~1000 (geometry overfitting).
- **`figures/expD18_probes.png`** -- 2x3 panels, one column per arm: open circle = pre-train probe, filled circle = post-train probe, x = own final readout (median over seeds, log y shared). What to see: the pre$\to$post drop (geometry score, largest on the qi-init arms, e.g. sarcos), the x floating above the filled circle on the Adam arms (the unclaimed readout gap) and sitting on it for the QI optimizer, and airfoil's adam/std x far above even its pre-probe.

## Additional details

- The probes are validated (damping chosen on the held-out tenth, scored on test), per the expD14 iteration-4 lesson that the raw min-norm refit overfits; on these noisy tasks the raw refit would score the geometry optimistically.
- The gate test pins the two regimes apart: the same solve core reaches $3.7\times10^{-14}$ on a halo'd frozen 1-D geometry, while the halo-less expF04 ridge init floors near $3\times10^{-7}$ at d=1 -- the init family's geometry limit, not a solver defect. Machine epsilon was never reachable on the tabular tasks themselves (inherent noise, rel $L_2$ floors 0.15-0.8).
- Single width (256) and one solve cadence (200) throughout; neither was tuned.

## Conclusions

*Pending Sam.* On this six-task tabular suite the QI optimizer's in-run behavior is indistinguishable from Adam's, and its entire final-error advantage is the terminal validated solve, which plain Adam plus one refit matches. What the experiment establishes: the terminal refit is worth 4-20% on noise-floored tabular tasks; the in-run solve machinery is safe (no stun, geometry probe within 0.4% of Adam's) but inert in this regime and costs 1.48x passes; and the untrained high-d QI-inspired init is not a better geometry than Adam finds from scratch, though starting from it yields an equal-or-better final geometry on 4/6 tasks.

## Open questions

- Does the no-stun result hold when the solve goes deep (low-noise or larger-data tasks where $r_{\rm entry}$ is small)? Here every in-run solve was soft by construction.
- The 2-layer version of this grid (the QI-inspired init exists for layer 2 via `qi_ridge_init_layer_`; unexplored per PROGRAM_FRAMING §1).
- Whether a coverage-aware high-d init (centers where the data is, per PROGRAM_FRAMING §2) closes the §7.4 gap that the current projection-sampled init does not.
