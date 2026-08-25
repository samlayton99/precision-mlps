# expD07 -- least-squares optimizer suite (synthetic + QI phis)

**Status:** suite live -- optimizer testing in progress; conclusions held until the roster is done.

## TL;DR

- Fixed benchmark of 132 least-squares problems -- 60 synthetic (controlled spectrum / alignment / rank / scale) + 72 QI-geometry $[\Phi,\mathbf 1]$ systems -- each stored with its fp64 SVD floor. Adding an optimizer = one entry in `optimizers.py` + one subsection below; the cache reruns only new (problem, optimizer) pairs.
- SVD of the real $[\Phi,\mathbf 1]$: exponentially decaying spectrum from index 0 with a rank cliff at $\approx 0.6m$ (rank $\approx N+$const), target energy $\propto\sigma_i$ in every direction. Each extra digit costs a decade of spectrum depth; rel $L_2\le10^{-13}$ needs directions at $\sigma/\sigma_1\sim10^{-11}$--$10^{-14}$.
- First-order baselines stall $\sim$8--11 orders above the phi eval floors at 20k steps, in quantitative agreement with the truncation analysis.

## Question

Which optimizers can actually solve the lstsq readout, and what matrix property blocks the ones that can't? This isolates expD01's 4th barrier to the pure linear subproblem $\min_v\|Av-y\|^2$, with a synthetic half that decomposes the difficulty into controlled axes.

## Benchmark setup

Every problem is $L(v)=\|Av-y\|_2^2/R$, solved from $v_0=0$. Universal unit: **relative $L_2$** on a fixed $[10^{-16},1]$ scale, lower = better everywhere. All grading is **best-over-trajectory** (min across checkpoints, running min in trajectory plots): early stopping is free, so an optimizer is judged by the best point it ever visits, not where it ends.

- **Synthetic (60):** $A=U\,\mathrm{diag}(s)\,V^T$ with orthonormal random $U,V$; $y=c\,Av^*$ (+ optional off-range component). For a quadratic, optimizer behavior is fully determined by the spectrum $s$ and the alignment of $v^*$ with the singular directions -- these are the axes:
  - spectra: log-linear at $\kappa\in\{10^2,10^8,10^{14}\}$; `phispec` = exponential decay to $10^{-10}$ over the first $0.6m$ indices then a hard rank cliff (the measured $[\Phi,\mathbf 1]$ shape);
  - alignment: isotropic vs head (top 20% of directions) vs tail (bottom 20%);
  - rank-deficient ($0.6m$) with $y$ in-range (consistent) vs off-range (inconsistent, irreducible residual);
  - scale probes: $y$ scaled by $10^{\pm3}$ (catches $\epsilon$-floor failures in adaptive methods).
  - 10 cells $\times\ N\in\{64,128,256\}\times R/N\in\{1,4\}$. Metric: train rel $L_2=\|Av-y\|/\|y\|$; floor $=\sqrt{L^*/L_0}$ from a direct fp64 lstsq.
- **Phis (72):** repo-standard QI system: uniform centers + halo (`default_halo`, $\lambda^*=0.25$), $\gamma=\lambda^*/h$, $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, augmented $[\Phi,\mathbf 1]$; rows $=$ ratio $\times$ cols, ratio $\in\{0.5,1,2,4\}$; 6 targets (sine, sine_8pi, runge, sine_mixture, exp, abs_cubed) $\times\ N\in\{64,128,256\}$. Metric: eval rel $L_2$ on 4001 equispaced points (eval $L_\infty$ recorded too); floor = truncated-SVD solve at rcond $=10^{-15}$ ($\sim10^{-13}$--$10^{-14}$ for smooth targets at $N\ge128$).

## Experiment design

- Full-batch fp64, deterministic from $v_0=0$ (no seed axis), budget 20{,}000 grad evals, metrics at $\sim$60 log-spaced checkpoints. Budgets count grad evals, not iterations, so multi-eval methods (line searches) are charged fairly.
- **Caching:** `runs.jsonl` is keyed (problem, optimizer) -- new configs run only their missing pairs; floors live in the manifest so baseline changes never trigger reruns. Figures regenerate live as runs land.
- **Suite verification** (`tests/test_expD07_suite.py`): generators hit requested $\kappa$/rank/alignment; an oracle SVD solve scores $>12$ digits through the metric plumbing; phi floors reproduce construction precision; `phispec` matches the measured $[\Phi,\mathbf 1]$ shape.

**Code & data:** `experiments/expD07_lstsq_optim_suite/` (`build_problems.py`, `optimizers.py`, `run.py`, `plotting.py`, `analyze_phi_regime.py`); problems + manifest in `data/expD07_problems/`; runs + figures in `results/checkpoint_D_optimizers/expD07_lstsq_optim_suite/`.

## Optimizers

One compact subsection per tested config; the registry id (in backticks) is the key in `optimizers.py`, `runs.jsonl`, and every figure.

### SGD -- `sgd`

Plain full-batch gradient descent, lr $=1/L$ with $L=2\sigma_{\max}^2/R$ set per problem (half the $2/L$ stability limit; $\sigma_{\max}$ from the manifest). The raw first-order reference: direction $i$ converges on the $(\sigma_1/\sigma_i)^2$-step timescale, so its error curve should follow the truncation curve at $\tau\sim t^{-1/2}$.

### SGD + momentum -- `sgd_mom09`

Heavy ball, $\beta=0.9$, lr $=1/L$. The classical accelerated baseline: effective condition-number dependence improves from $\kappa$ to $\sim\sqrt{\kappa}$, so it should reach $\sim$2$\times$ the digits of plain SGD on deep spectra.

### Adam -- `adam_lr1e-3`

PyTorch defaults ($\beta_1{=}0.9$, $\beta_2{=}0.999$, $\epsilon{=}10^{-8}$), absolute lr $=10^{-3}$. The default deep-learning choice and the optimizer the training experiments (expD01/expD02) actually use; per-coordinate scaling means no lr rule is needed, but $\epsilon$ imposes an update floor probed by the scale cells. Theory predicts non-convergence: batch-mode Adam on quadratics has period-2 limit cycles for all hyperparameters (Bock & Weiss), so its curve should plateau, not descend.

### Adam eps-coupled (Impl A) -- `adam_epscoupled`

Stock Adam with $\epsilon=10^{-3}$ and lr $=\epsilon(1+\beta_1)/L$ (per problem), enforcing lr$/\epsilon<2(1+\beta_1)/L$: once $\sqrt{\hat v_t}\ll\epsilon$ the update is a stable *preconditioned heavy-ball* step with a frozen denominator, so linear convergence should be inherited. Tests the hypothesis that Adam's failure is exactly the non-vanishing step through the $\epsilon$-floor.

### AMSGrad (Impl B) -- `amsgrad_lr1e-3`

torch Adam with `amsgrad=True` (max-accumulated $\hat v_t$, which freezes as gradients shrink), lr $=10^{-3}$, $\epsilon=10^{-8}$. The other route to a frozen preconditioner $P_\infty\succ0$ (Reddi et al.): if H holds, its curve should be a straight line where stock Adam plateaus, and its frozen diagonal should retain adaptivity on scale-skewed problems.

### Nesterov -- `nesterov`

Accelerated gradient with the convex schedule $\beta_t=(t-1)/(t+2)$, lr $=1/L$. The right accelerated method when the smallest effective eigenvalue is $\approx0$ (our exponential spectra): $f\sim t^{-2}$ vs GD's $t^{-1}$, i.e. rel $L_2\sim t^{-1}$ vs $t^{-1/2}$ -- double the digits of SGD at equal budget, still polynomial.

### GD + Barzilai-Borwein -- `gd_bb`

Gradient descent with the BB1 step $\alpha_t=s^Ts/s^Ty$ (safeguarded to $1/L$ when $s^Ty\le0$). Famously competitive with CG on quadratics at one gradient per step, R-linear but nonmonotone; probes how far a *step-size rule alone* (no memory, no preconditioner) can get.

### CGLS -- `cgls`

Conjugate gradient on the normal equations $A^TA v=A^Ty$, one iteration $\approx$ one grad eval (two matvecs). The Krylov-*optimal* method for quadratics: exact convergence in $\le\mathrm{rank}$ iterations in exact arithmetic, superlinear on exponentially decaying spectra. This is the "matched to the problem" reference every gradient method is measured against; its fp64 stall level also calibrates what iterative methods can achieve at $\kappa\to10^{14}$.

### L-BFGS -- `lbfgs`

torch L-BFGS, strong-Wolfe line search, history 100, budget counted in closure calls. The quasi-Newton standard of the PINN machine-precision literature (the paper's own Adam$\to$SSBroyden pipeline is this family); on a quadratic, history $\ge$ rank makes it Newton-like, so it should go floor-deep wherever line searches survive the conditioning.

### SGD + momentum 0.99 / 0.999 -- `sgd_mom099`, `sgd_mom0999`

Heavy ball at $\beta\in\{0.99, 0.999\}$, lr $=1/L$. Round-2 said momentum is the best *standard* DL lever; heavy-ball theory says $\beta_{\mathrm{opt}}=1-O(1/\sqrt\kappa)$, so on our deep spectra $\beta=0.9$ is far from optimal. The cheapest possible upgrade: one hyperparameter, zero extra memory.

### NLCG Polak-Ribiere -- `nlcg_pr`

Nonlinear conjugate gradient, PR+ with descent restarts, exact line search on the quadratic; charged 2 grad evals/iteration (in DL the line search is a $\sim$1-extra-eval secant). O(1) memory (previous gradient + direction), no preconditioner, fully architecture-agnostic. On a quadratic with exact line search NLCG *is* CG -- this tests whether Krylov-depth survives in deployable form.

### BB + frozen RMS diagonal -- `bb_rms`

The two round-2 lessons combined: 100-step warmup accumulates an AMSGrad-style $\max g^2$ diagonal, then FREEZES $D=\sqrt{v_{\max}}$ (Impl-B mechanism, one Adam-like accumulator) and runs BB in the $D$-metric ($x_+ = x - \alpha\, g/D$, $\alpha = s^TDs/s^Ty$). Diagonal handles axis-aligned scale, BB's scalar step handles the spectrum; frozen $P\succ0$ keeps the convergence theory.

### LSQR -- `lsqr`

Golub-Kahan bidiagonalization (Paige-Saunders): solves $\min\|Av-y\|$ WITHOUT forming normal equations, 1 eval/iteration. Not a DL optimizer -- it is the numerically stable Krylov ceiling: if LSQR beats CGLS on the phis, CGLS's stall was fp64 normal-equation squaring ($\kappa^2$), not an information limit, and the phi floors are iteratively reachable.

### NLCG-PR + restart/refresh -- `nlcg_restart`

`nlcg_pr` with the direction reset AND the residual recomputed exactly every 250 iterations. Tests the iterative-refinement route to fixing fp64 recurrence drift (rank $\approx277\ll$ 20k iterations, so the round-3 Krylov stall must be roundoff).

### CGLS + refinement restarts -- `cgls_refine`

CGLS restarted on the exactly recomputed residual every 300 iterations (classic iterative refinement). Same hypothesis as `nlcg_restart`, applied to the stronger base method.

### CGLS reorthogonalized -- `cgls_reortho`

CGLS with FULL reorthogonalization of the $A^T$-residuals against the stored Krylov basis (memory $O(\mathrm{rank}\cdot n)$, charged 2 evals/iter). Not DL-deployable -- the ceiling reference that isolates WHICH fp64 mechanism causes the stall: if reorthogonalization reaches the floor where restarts don't, the mechanism is basis orthogonality loss, not residual drift.

## Results

Leaderboard (geo-mean **best-over-trajectory** relative $L_2$ over the whole half-suite; grows as configs are added):

| optimizer | synthetic (train) | phis (eval) |
|---|---|---|
| `cgls_reortho` | $1.6\times10^{-13}$ | $4.1\times10^{-12}$ |
| `lsqr` | $2.2\times10^{-11}$ | $1.8\times10^{-8}$ |
| `cgls` | $5.7\times10^{-11}$ | $2.5\times10^{-8}$ |
| `cgls_refine` | $1.9\times10^{-10}$ | $1.0\times10^{-6}$ |
| `nlcg_pr` | $2.3\times10^{-10}$ | $2.0\times10^{-7}$ |
| `nlcg_restart` | $4.2\times10^{-10}$ | $2.8\times10^{-6}$ |
| `gd_bb` | $6.3\times10^{-9}$ | $8.0\times10^{-5}$ |
| `sgd_mom099` | $2.1\times10^{-8}$ | $4.7\times10^{-4}$ |
| `bb_rms` | $1.9\times10^{-7}$ | $7.3\times10^{-4}$ |
| `sgd_mom09` | $6.1\times10^{-7}$ | $2.1\times10^{-3}$ |
| `lbfgs` | $1.6\times10^{-5}$ | $1.8\times10^{-4}$ |
| `nesterov` | $2.3\times10^{-5}$ | $1.0\times10^{-4}$ |
| `sgd_mom0999` | $3.8\times10^{-4}$ | $2.3\times10^{-4}$ |
| `amsgrad_lr1e-3` | $8.0\times10^{-4}$ | $1.5\times10^{-3}$ |
| `sgd` | $9.0\times10^{-4}$ | $8.7\times10^{-3}$ |
| `adam_epscoupled` | $2.6\times10^{-3}$ | $2.0\times10^{-2}$ |
| `adam_lr1e-3` | $4.3\times10^{-3}$ | $1.0\times10^{-3}$ |

Round-4 signal: **`cgls_reortho` reaches the floors** -- $10^{-13}$ synthetic, $4\times10^{-12}$ phi eval, i.e. an *iterative* method matches the direct SVD solve once the Krylov basis is kept orthogonal (it converges in $\le$ rank iterations and stops). The restart variants went *backwards* (`cgls_refine` $10^{-6}$ vs plain CGLS $2.5\times10^{-8}$ on phis): restarting discards the Krylov subspace, which costs more than the drift it fixes. Verdict: the last 5 orders are bought by BASIS ORTHOGONALITY (memory $O(\mathrm{rank}\cdot n)$), not by residual refreshing -- the precision/memory tradeoff is now explicit.

Round-3 signal: LSQR only *marginally* beats CGLS on the phis ($1.8$ vs $2.5\times10^{-8}$) -- the $10^{-8}$ iterative stall is NOT normal-equation squaring; at 20k evals it is the spectrum itself (plus fp64 orthogonality loss), so closing the last 5 orders to the SVD floor needs restarts/reorthogonalization or more budget, not a better formulation. NLCG-PR delivers Krylov-class depth in deployable form: $2.3\times10^{-10}$ / $2.0\times10^{-7}$ while paying 2 evals/iteration -- within one order of LSQR on phis with O(1) memory and no linear-solver structure. Cranking momentum works exactly as heavy-ball theory says: $\beta$ 0.9 $\to$ 0.99 buys $\sim$1.5 synthetic orders and 4.5$\times$ on phis, but $\beta=0.999$ overshoots at lr $=1/L$ (synthetic degrades; best-grading rescues phis). `bb_rms` LOST to plain BB -- the frozen diagonal is dead weight on rotated/isotropic problems and its conservative warmup+fallback slows the BB core; diagonal adaptation only ever pays on axis-aligned scale, which the suite isolates in the scale cells.

Round-2 signal: CGLS dominates -- it cliff-dives to the $\sim10^{-15}$ floor in $\lesssim$ rank iterations on well-conditioned / head-aligned / consistent-rank-deficient / scale cells, and is the only method past $10^{-7}$ on any phi (best $10^{-9.3}$ on runge; still 4--6 orders above the floors -- fp64 normal equations square the conditioning). Its *final* iterate diverges post-convergence on ill-conditioned cells ($10^{+11}$), which is why grading is best-over-trajectory. GD+BB is the surprise: a plain gradient method with the BB step tracks CGLS to the floor on four cells and beats every torch optimizer overall. The Adam-fix signatures: stock Adam's running-min flattens (limit cycle, as predicted); AMSGrad shows the frozen-preconditioner dive to $10^{-15.4}$ on `scale_tiny` but crawls on `scale_huge` (frozen $\hat v$ cuts the step both ways); $\epsilon$-coupled Adam descends steadily without plateau but its safe lr ($\propto\epsilon/L$) makes it budget-limited -- confirms H qualitatively, loses on speed. L-BFGS underwhelms: floor-ish only on `scale_huge`, plateaus at $10^{-3}$--$10^{-6}$ elsewhere (line-search stalls on deep spectra).

### Figures

- `ranking.png` -- two barh panels (synthetic train | phi eval), geo-mean rel $L_2$ per optimizer, best at top, lstsq floor dashed; fixed $[10^{-16},1]$ axis.
- `heatmap.png` -- two stacked panels (synthetic / phis), median $\log_{10}$ rel $L_2$ at $N=256$; rows sorted best$\to$worst with the SVD floor as the top row; one fixed $-16..0$ color scale.
- `profiles.png` -- fraction of problems driven below rel $L_2\in\{10^{-6},10^{-12}\}$ vs grad-eval budget; the dominating curve is the better optimizer (depth and speed in one plot).
- `trajectories.png` -- median rel-$L_2$ trajectory per problem class (4$\times$4 grid), lstsq floor dashed, fixed scale.
- `phi_regime.png` -- left: normalized singular spectra of $[\Phi,\mathbf 1]$ vs the synthetic cells; right: truncation curve = best achievable rel $L_2$ keeping directions down to $\sigma/\sigma_1=\tau$ (for GD at lr $=1/L$, direction $i$ converges on the $(\sigma_1/\sigma_i)^2$-step timescale, so this is the idealized error-vs-time curve).
- `budget_scaling.png` -- 200k-eval runs on three phi problems: NLCG/CGLS/LSQR vs CGLS with windowed reorthogonalization $k\in\{64,256,\mathrm{full}\}$; SVD floor dashed, 20k suite budget dotted. The memory-vs-depth threshold plot.

## Additional details

The truncation analysis quantifies which synthetic regime the phis occupy: the $[\Phi,\mathbf 1]$ spectrum decays exponentially from the first index (no flat head) with a rank cliff at $\approx0.6m$ (rank $\approx N+$const, expA04's bounded tanh null space), and the smooth-target $y$ carries energy $\propto\sigma_i$ in every kept direction -- the isotropic-alignment signature (the optimal readout $\approx$ scaled derivative samples is a smooth vector with a flat expansion in the right singular basis). Consequences, per the table printed by `analyze_phi_regime.py`: rel $L_2\le10^{-6}$ needs directions at $\tau\sim10^{-6}$ ($\sim10^{11}$ GD steps); $\le10^{-13}$ needs $\tau\sim10^{-11}$--$10^{-14}$ ($10^{20}$--$10^{27}$ steps). The observed 20k-step results match the idealized curve quantitatively ($\tau\sim1/\sqrt{20000}\Rightarrow\sim10^{-2.5}$). The `phispec_iso` cell was updated to this measured shape (previous flat-head version purged from the cache).

### Budget vs memory (`budget_scaling.py`)

10$\times$ budget (200k evals, three phi problems at $N{=}256$, $r{=}4$) buys the plain Krylov methods only $\sim$1 order ($\mathrm{err}\sim t^{-1}$-ish: NLCG $5\times10^{-7}\to3\times10^{-8}$, CGLS $10^{-7}\to6\times10^{-9}$; extrapolating, $10^{-13}$ would need $\sim10^{9}$ evals) -- budget is NOT the path to the floor. Windowed reorthogonalization has a sharp threshold at the effective rank ($\approx277$ here): $k{=}64$ is *worse* than no window (partial projection breaks CG's implicit conjugacy without restoring orthogonality), while $k{=}256$ reaches the SVD floor **within the ordinary 20k budget** ($5.6\times10^{-14}$ sine, $5.6\times10^{-16}$ runge, $1.5\times10^{-13}$ mixture; full-basis even lands below the truncated-SVD reference). Verdict: the floor costs orthogonalization memory $k\approx\mathrm{rank}$ -- for a readout of width $W$ that is a $\mathrm{rank}\times(W{+}1)$ basis ($\sim$1 MB at $N{=}256$), i.e. L-BFGS-class and entirely practical at network scale, though not at generic large-model scale.

### DL sanity test (`dl_test/`)

`experiments/expD07_lstsq_optim_suite/dl_test/run.py`: fp64 full-batch MLPs ($d\to64\to64\to1$, tanh) on three real regression tasks (airfoil, parkinsons, bike_sharing from the expF04 cache), budget 3000 function/grad evals, grading best-over-trajectory. The DEPLOYABLE suite roster runs STANDALONE from one shared init (suite ids kept): `adam`, `amsgrad`, `sgd_mom09`, `sgd_mom099`, `nesterov`, `gd_bb`, `nlcg_pr`, plus two two-stage designs, `adam_nlcgD` (Adam warmup $\to$ NLCG in the frozen Adam-diagonal metric) and **`padam_hb`** (Adam warmup 30% $\to$ freeze $D=\sqrt{\hat v}+10^{-8}$ from Adam's second moment $\to$ preconditioned heavy ball $x_+ = x - \mathrm{lr}\,g/D + \beta(x-x_-)$, lr $=(1+\beta)/(2\hat L)$ from a decayed-max Rayleigh estimate $\hat L\leftarrow\max(0.99\hat L,\ s^Ty/s^TDs)$, trust-window step rejection). SGD-family lrs use $1/\hat L$ probed at init (finite-difference curvature along $g$) -- the DL analogue of the suite's $1/L$ rule; every stage carries a revert-to-last-good safeguard (without it, heavy ball and Nesterov silently diverge -- observed, fixed, load-bearing).

Results (all numbers are train/test **MSE** -- the native DL metric; this figure has no rel-$L_2$ floors, so units are internally consistent): **`padam_hb` is best or within noise of best on all three tasks** (parkinsons train $1.4\times10^{-2}$ vs Adam's $3.2\times10^{-2}$, test $2.3\times10^{-2}$ vs $4.2\times10^{-2}$; bike train $5.2\times10^{-3}$ vs $6.7\times10^{-3}$); `sgd_mom099` wins airfoil ($2.9\times10^{-3}$) but collapses on parkinsons/bike (probe-lr sensitivity of fixed-lr methods); Adam/AMSGrad are the solid middle; the line-search methods (`nlcg_pr`, and `gd_bb`'s nonmonotone Armijo) trail on the nonconvex transient. The two-stage curves coincide with Adam until the stage switch by construction. Figure: `dl_test/dl_curves.png` (one line per optimizer from eval 1; earlier versions of this figure shared one warmup across all configs -- the apparent universal kink at $10^3$ was that artifact, now gone).

## Conclusions

(intentionally blank until the optimizer roster is tested)
