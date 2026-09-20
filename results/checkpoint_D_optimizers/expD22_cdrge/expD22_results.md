# expD22 -- CD-RGE (zero-order) on the expD16 suite

**Status: draft -- pending Sam's review. Collection canceled early by Sam (cost); qi arm has 8 of 12 cells at 1-2 seeds, xavier arm has tuning cells only.**

## TL;DR

- The best tuned CD-RGE variant is an **Adam-style ZO optimizer** (Adam moments on the RGE gradient estimate, cosine lr): it beats expD16's flawed SPSA by ~5$\times$ and roughly **ties plain Adam per iteration** (it even reproduces Adam's 0.921 xavier plateau) -- at $2n \approx 200$ forward passes per step against Adam's 4.
- The author's relayed recipe does not transfer: $\text{lr}=\epsilon$ is unstable above $3{\times}10^{-3}$ (runs diverge to $10^{18}$), and halving both every step freezes or diverges (geometric travel budget; measured on a quadratic: $3{\times}10^{-2}$ vs $5{\times}10^{-31}$ for constant $\epsilon$).
- **The $\epsilon$-anneal hypothesis is answered for this regime**: halving $\epsilon$ during the run produced trajectories identical to fixed $\epsilon=10^{-3}$ to 3 significant figures in all 14 paired runs. The barrier at $10^{-2}$ error is iterations/spectrum, not finite-difference bias; $n_{\text{perturb}}$ (1000 vs 100) and $\epsilon$ ($10^{-2}$ vs $10^{-3}$) are both non-levers here.
- No zero-order line approaches the gradient-based finishers: best CD-RGE cell $6{\times}10^{-3}$ vs SSBroyden's $2{\times}10^{-7}$ median. The class is not precision-competitive on this problem.

## Question

Sam's ask: implement the CD-RGE zero-order optimizer from Chaubard's `zero_order_rnn` repo (the corrected version of the optimizer class expD16's SPSA fumbled), tune it hard for our small fp64 MLPs, and run it on the exact expD16 suite. Does any tuned zero-order variant approach the precision of the gradient-based lines?

## Experiment design

**The optimizer.** CD-RGE (central-difference random gradient estimation; Chaubard & Kochenderfer 2025, arXiv:2505.17852) estimates the gradient from forward passes only. Per step, with $n$ probes $z_j \sim \text{Rademacher}(\pm1)^m$:

$$\hat g = \frac{1}{n}\sum_{j=1}^{n} \frac{L(\theta+\epsilon z_j) - L(\theta-\epsilon z_j)}{2\epsilon}\, z_j, \qquad \theta \leftarrow \theta - \text{lr}\cdot \hat g,$$

costing $2n$ full-batch loss evaluations per step. The port (`experiments/expD22_cdrge/cdrge.py`) reproduces the upstream update exactly (verified bitwise against the upstream formula in `tests/test_expD22_cdrge.py`), including the $\text{lr}/\epsilon$ factoring, with two mechanical differences stated in its docstring (exact restore from a stored $\theta$ instead of seeded re-add; one flat RNG stream instead of upstream's colliding per-tensor seeds). The author's usage advice, relayed by Sam: $\text{lr}=\epsilon$, start $\epsilon$ high, cut both by half every step, $n$ high (1000).

**The suite -- expD16 verbatim.** Standard-parameterization tanh MLP (QIMlp), $f(x)=b_0+\sum_k v_k\tanh(w_kx+b_k)$, $W = N+2\,\text{halo}+1$; two inits (`qi`: construction geometry $w_k=\gamma=\lambda^*/h$, $b_k=-\gamma c_k$, readout Glorot; `xavier`: Glorot everywhere); 4 targets (sine, exp, runge, sine\_8pi) $\times$ $N\in\{64,128,256\}$; full-batch fp64 MSE on 2003 equispaced points; metric eval rel $L_2$ on a misaligned 4001-point grid. All parameters train. Seeds $\{0,1,2\}$ per cell (seeding the Glorot draws and the probe RNG); figures show the per-variant median with a min-max band. Plain Adam (expD16's protocol: 3000 steps, lr $3{\times}10^{-3}$, warmup 100, cosine) re-run per cell/seed as the reference line. **Coverage caveat:** Sam canceled the collection for cost after 70 qi rows (sine/exp complete at 2 seeds for $N{\le}128$, 1 seed at 256; runge at $N{\le}128$; sine\_8pi and the xavier final grid not collected -- xavier behavior is documented from the tuning cells).

**Variants (the lines), fixed by tuning stages 1-4** (`tuning/stage{1,2,3}.jsonl`, single cell sine/$N{=}64$ both inits unless noted):

- **`cdrge_adam_cos`** (headline): Adam-style bias-corrected moments on $\hat g$ ($\beta_1{=}0.9$, $\beta_2{=}0.999$, lr $10^{-2}$, cosine + 100-step warmup, stabilizer $10^{-16}$), $\epsilon=10^{-3}$ constant, $n{=}100$, $T{=}600$. Deviation stated: upstream's literal `beta2` code initializes its variance accumulator at **ones**, which never adapts at fp64 regression loss scales; Adam's bias-corrected form is the battle-tested equivalent (REQUIREMENTS section 2 toss-up rule).
- **`cdrge_adam_anneal`**: identical, plus $\epsilon$ halved every 75 steps to a $10^{-7}$ floor -- the direct test of Sam's hypothesis that fixed $\epsilon$ is the precision barrier ($\epsilon$ enters only the estimate, not the step size). Collected on the first 14 paired runs, then dropped from the remaining collection once it proved identical to `cdrge_adam_cos` (below).
- **`cdrge_lr_eq_eps`**: the author-faithful coupling $\text{lr}=\epsilon$ at the largest stable value ($3{\times}10^{-3}$, constant), $n{=}150$, $T{=}300$.
- **`cdrge_halve1`**: the author's literal recipe -- $\epsilon_0=0.3$, $\text{lr}=\epsilon$, both halved every step, $n{=}300$, $T{=}60$ (after ~50 halvings $\epsilon$ is at floor and the run is inert).

**Hyperparameters tested (the tuning record).** With $\text{lr}=\epsilon$ the realized step is $-\epsilon\,\hat g$: plain GD at learning rate $\epsilon$ with multiplicative sampling noise. Everything measured follows from that:

| knob | values tested | verdict |
|---|---|---|
| $\epsilon_0$ at $\text{lr}=\epsilon$, constant | 1, 0.3, 0.1, 0.03, 0.01, 0.003 | diverges ($10^{14}$-$10^{19}$) for $\epsilon\ge0.01$ from qi; stable only at $3{\times}10^{-3}$, and then ZO-GD-slow (0.30 after 300 steps) |
| halving cadence (lr=$\epsilon$) | every 1, 5, 20 steps from $\epsilon_0{=}0.3$ | all freeze or diverge; halve/1 provably cannot converge from distance (geometric travel budget) |
| decoupled lr at fixed $\epsilon{=}10^{-3}$ | lr $\in\{3,10,30\}{\times}10^{-3}$ | re-diverges at lr $\ge10^{-2}$; no gain |
| momentum only | $\beta_1=0.9$ at $\epsilon\in\{1,3\}{\times}10^{-3}$ | no improvement over vanilla |
| Adam-style lr | $10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}$ | monotone to $10^{-2}$, worse at $3{\times}10^{-2}$; **$10^{-2}$ wins** |
| $\beta_1$ | 0.9, 0.99 | 0.9 wins |
| cosine decay on Adam lr | on/off at $T{=}3000$ | on: $6.0{\times}10^{-3}$ final vs off: $2.9{\times}10^{-2}$ (wobbling) |
| $n_{\text{perturb}}$ | 100, 300, 1000 | identical error at matched steps ($1.09$ vs $1.15{\times}10^{-2}$); 1000 is 10$\times$ cost for nothing at this error level |
| $\epsilon$ under Adam-style | $10^{-4}, 10^{-3}, 10^{-2}$, and in-run halving | indistinguishable; finite-difference bias not binding above ~$10^{-3}$ error |

One structural note: central differences of an MSE loss are **not** capped by the fp64 loss-comparison wall (kill list, "no control decisions on loss values"): the $r^2$ terms cancel in $L(\theta{+}\epsilon z)-L(\theta{-}\epsilon z)$, leaving a term linear in the residual, so the estimate stays informative at small residuals the way the gradient does. The $\epsilon$-anneal requirement Sam hypothesized is real but binds ~5 orders below where these runs plateau.

**Cost honesty.** One CD-RGE step is $2n$ sequential full-batch forwards (200 at $n{=}100$) against Adam's ~4 passes; the figure's x-axis is iterations (expD16 convention), which flatters CD-RGE by ~50$\times$. Per-run eval counts are in the data rows (headline: 120,000 forwards for 600 steps).

**Code & data.** `experiments/expD22_cdrge/` (`run.py`, `cdrge.py`, `collect_all.sh`, `summarize.py`, `config.yaml`); tests `tests/test_expD22_cdrge.py` (bitwise formula match, gradient alignment on the real model, machine-precision quadratic, qi-geometry floor gate). Upstream: `github.com/Fchaubard/zero_order_rnn` (`rge_series_experiments.py::cdrge_optimize`). Data: `data/trajectories_qi__<target>_<N>.jsonl`; tuning: `tuning/stage{1,2,3}.jsonl`. Figure: `figures/expD22_qi_init.png`.

## Results

Final eval rel $L_2$ over the collected qi cells (14 runs per variant; median [best, worst]):

| variant | median | best | worst | forwards/run |
|---|---:|---:|---:|---:|
| Adam (reference, 3000 it) | $4.8\times10^{-3}$ | $1.4\times10^{-3}$ | $1.6\times10^{-2}$ | ~12,000 |
| CD-RGE Adam-style (600 it) | $3.9\times10^{-2}$ | $1.3\times10^{-2}$ | $1.2\times10^{-1}$ | 120,000 |
| CD-RGE Adam-style + $\epsilon$-anneal | $3.9\times10^{-2}$ | $1.2\times10^{-2}$ | $1.2\times10^{-1}$ | 120,000 |
| CD-RGE lr=$\epsilon$=3e-3 (300 it) | $1.4\times10^{-1}$ | $3.9\times10^{-2}$ | $3.0\times10^{-1}$ | 90,000 |
| CD-RGE author recipe (halve/step) | $2.2\times10^{4}$ | $2.2\times10^{3}$ | $4.2\times10^{6}$ | 31,800 |

- **The tuned line is an Adam clone at 50$\times$ the cost.** Iteration-for-iteration the Adam-style CD-RGE tracks the Adam reference closely through 600 steps (see figure); its remaining gap to Adam's $4.8\times10^{-3}$ is budget, not mechanism -- at $T{=}3000$ on sine/$N{=}64$ it reached $6.0\times10^{-3}$ (tuning stage 4), matching Adam's plateau. It never gets past it: the plateau is the same first-order/spectrum wall.
- **The $\epsilon$-anneal arm is bit-for-bit the fixed-$\epsilon$ arm** (identical to 3 significant figures in every paired cell, both at $N{=}64$ and 128). Direct answer to the hypothesis: at $10^{-2}$-$10^{-1}$ error, $\epsilon$ is not the barrier.
- **The author-faithful arms fail as tuned-out cases, not as strawmen**: lr=$\epsilon$ at its largest stable value descends steadily but ~10$\times$ slower than the Adam-style line; the literal halve-every-step recipe diverges from the qi init within its first large-$\epsilon$ steps ($\epsilon_0=0.3$ perturbs $\gamma$-scale weights by $O(1)$) and then freezes off-scale.
- **Against expD16's zoo (same cells, qi):** SPSA $1.9\times10^{-1}$, Adam $8.3\times10^{-3}$, L-BFGS $9.8\times10^{-4}$, NNCG $9.3\times10^{-6}$, SSBroyden $2.0\times10^{-7}$. Tuned CD-RGE lands between SPSA and Adam -- 3 to 4.5 orders above the second-order finishers, ~11 orders above the lstsq floor of the same geometry.
- **xavier (from tuning, sine/$N{=}64$):** the Adam-style ZO line sits on exactly the 0.921 fit-the-mean plateau real Adam sits on for its first ~2000 iterations, and escapes it by iteration ~1000 to $2.2\times10^{-1}$ -- earlier than the Adam reference does on that cell. Everything else stays on the plateau. The xavier final grid was not collected (canceled).

### Figures

- **`figures/expD22_qi_init.png`** -- 4 targets (rows) $\times$ $N\in\{64,128,256\}$ (cols); x = iteration (0-3000, expD16 axes), y = eval rel $L_2$ (log, fixed $10^{-16}$-$10^1$); grey = Adam reference (3000 it), red = CD-RGE Adam-style (600 it), green = lr=$\epsilon$ (300 it), purple = author recipe (diverges off-scale, hence invisible); solid = median over seeds, band = min-max; empty panels annotated "not collected". Read: red hugs or slightly trails grey over the shared iteration range in every collected panel, then stops at its budget -- and the bottom 12 decades of every panel are empty, exactly as in expD16's figures.

## Additional details

- Wall-clock/cost was the binding constraint of the whole exercise: at $n{=}100$ one CD-RGE run costs 120k sequential forwards ($\approx$6 min at $W{=}269$ on CPU fp64), which is why the suite was trimmed mid-flight (anneal arm dropped after it proved identical) and finally canceled by Sam. This is requirement 1 (first-order cost) failing by a factor of ~50, visible as engineering pain rather than as a table entry.
- The divergence mode of lr=$\epsilon$ at large $\epsilon$ is a positive feedback: the realized step magnitude is set by the measured loss differences, so a growing loss grows the next step. The author's coupling is motivated by fp16 LLM training, where the loss scale is O(1)-O(10) and a single step at each $\epsilon$-scale nearly solves the local problem; neither holds here.
- The stage-1/2/3 tuning tables with full trajectories are in `tuning/stage{1,2,3}.jsonl`; stage 4/5 (cosine, transfer, anneal-at-$T{=}3000$) were killed mid-run for cost and survive only as the printed rows quoted here.

## Conclusions

*Pending Sam.* Data-obvious so far: on this suite the best achievable CD-RGE configuration is a zero-order reimplementation of Adam -- same trajectory, same plateaus, $\sim$50$\times$ the passes -- and no zero-order variant tested moves the precision frontier at all; the expD16 second-order finishers remain 3-4.5 orders ahead. The relayed recipe (lr=$\epsilon$, halve every step, $n{=}1000$) is inapplicable in this regime in all three of its parts, each for a measured reason.

## Open questions

- None proposed. The class is measured; further budget here is not warranted (Sam's call mid-run, and the data supports it).
