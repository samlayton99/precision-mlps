# batching_test -- the two step-2 solvers, and the battery that proves they batch

**Status: plan, pending Sam's approval. No Adam anywhere in this document** -- geometry drift, where it appears, is synthetic (a stand-in schedule), because step 2 is the solver, not the lobotomy.

Notation is the RECIPE's: $\Phi\in\mathbb R^{n\times d}$ the (Jacobian) feature matrix of the certified block, $w\in\mathbb R^d$ the unknowns, $r = y-\Phi w$ the residual, $k$ the block size (fixed, 64--128), $b$ the per-step batch rows, $\varepsilon$ the working dtype's unit roundoff. A **pass** = one $O(nd)$ traversal. Everything below is native-dtype (precision-agnostic); every tolerance is written as a multiple of $\varepsilon$.

---

# Part 1 -- the methods

Step 2 settles on **two solvers, one inner engine**. Both are LSMR; they differ only in the preconditioner and in when they run. Both are *steering-type*: the preconditioner only steers the iteration, correctness always comes from residuals of the current $\Phi$ on current rows. (This is the property that makes batching and staleness survivable at all -- see Part 2, T1.)

## Type A -- the streaming solver (LSMR + $k$-constant block-diagonal preconditioner)

**Operating contract** (the SUBPROBLEM access model, verbatim in spirit): each step $t$ draws a fresh row batch $S_t$ of size $b$ (any $b$, including $b<d$ -- no constraint on the user); the solver sees $\Phi_{S_t}$ and $y_{S_t}$ only; memory $O(d)$; compute per step $\approx$ one forward-backward pass; $w_0$ is handed to it.

**State.**
- Block factors: for each block $C_j$ of a fixed partition of the $d$ columns into $\lceil d/k\rceil$ blocks, a $k\times k$ factor of that block's (approximate) Gram. Total $d\cdot k$ floats.
- Momentum / Krylov scratch: $O(d)$ vectors.

**Blocking.** Blocks from `cluster_blocks` (agglomerative on $1-|\mathrm{corr}|$ from a row sample) when a sample is available; contiguous otherwise. At Type A's accuracy ceiling the ordering is measured to matter little (random $\approx$ contiguous at the $\varepsilon\kappa$ floor), so this is a mild optimization, not a requirement.

**Factor maintenance -- two variants; experiment A2 decides.**

- **A-EMA (default candidate).** Per step, for the (rotating) next block $j_t = t \bmod \lceil d/k\rceil$:
  $$G_{C} \leftarrow \beta\,G_{C} + (1-\beta)\,\tfrac{n}{b}\,\Phi_{S_t,C}^\top \Phi_{S_t,C},\qquad M_C^{-1/2} = V(\Lambda+\lambda_{\rm damp} I)^{-1/2}V^\top \ \text{from eigh}(G_C).$$
  Cost $O(bk^2)$ for the Gram update (a fraction of a pass at $k\le128$) $+\ O(k^3)$ for the one refreshed factor. Works at any $b$, including $b\ll k$: the EMA accumulates what one batch cannot see. The damping $\lambda_{\rm damp}$ is load-bearing twice over: it caps the factor's condition number against the Gram-squaring loss (Part 2, T4) and against EMA sampling noise. Default $\lambda_{\rm damp} = \max(10\,\varepsilon\,\mathrm{tr}(G_C)/k,\ \hat\sigma^2)$ with $\hat\sigma$ the running residual-floor estimate.
- **A-rotQR (the higher-ceiling candidate).** Per block, keep a row buffer of the last $m\approx 2k$ batch rows restricted to $C$; refresh that block's factor by pivoted QR of the buffer (never a Gram -- Part 2, T4), one block per step. Cost $O(mk^2)$ per step. Ceiling should match the measured full-batch block-Jacobi line ($\sim10^{-9}$); the price is the buffers ($O(mk)$ per block $= O(md)$ total -- still $O(d)$ with $m$ fixed) and slower refresh at tiny $b$.

**Per-step update -- two variants; experiment A1 decides.**

- **A-i ($\tau$-sweep LSMR).** Each step: run $\tau\in[1,3]$ LSMR iterations from scratch on $\min_\delta\|\Phi_{S_t}M^{-1/2}\delta - r_{S_t}\|$ with $r_{S_t}=y_{S_t}-\Phi_{S_t}w$ computed fresh (never carried -- Part 2, T7), then $w \leftarrow w + M^{-1/2}\delta$. Krylov state is discarded every step (it cannot survive a changed operator); the known restart-floor hazard (Part 2, T6) does not directly apply because every restart sees a fresh row sample, but whether the accumulated process converges below the single-batch floor is exactly what A1 measures.
- **A-ii (preconditioned heavy ball).** No Krylov at all: $g_t = M^{-1/2}\big(M^{-1/2}\big)^\top \Phi_{S_t}^\top r_{S_t}$ (two block-triangular applications, $O(dk)$), then $v_t = \mu v_{t-1} + g_t$, $w \leftarrow w + \eta_{\rm lr} v_t$. Structurally the most drift- and noise-tolerant form (its state is Adam-class); expected to trade a constant factor of iterations for that robustness.

**Stopping.** Type A does not stop; it runs as long as it is invoked. Its *reporting* observable is the preconditioned gradient norm $\|M^{-\top}\Phi_S^\top r_S\|$ smoothed by EMA.

**Ceiling (known, by design).** $\kappa(\Phi M^{-1})$ floors near $10^{9}$ for block-diagonal $M$ on these systems, so Type A saturates at $\sim10^{-7}$--$10^{-9}$ rel $L_2$ ($\sigma=0$, fp64), or at the batch statistical floor under noise. It is not, and is not meant to be, the machine-precision engine.

## Type B -- the finale solver (SPIR: sketch preconditioner + LSMR + iterative refinement)

**Operating contract.** Invoked once, on a *frozen* $\Phi$ (the trigger that freezes it is step-3 business; here it is given). Batches keep arriving; Type B first **accumulates** their rows -- legitimate because frozen $\Phi$ makes consecutive batches i.i.d. rows of one fixed matrix -- and then solves to the floor. The user's batch size is never constrained; the buffer does the work.

**Procedure, exactly:**

1. **Accumulate.** Buffer rows $(\Phi_{S_t}, y_{S_t})$ until $n_{acc} \ge 4d$ (default; B1 sweeps it). Memory: the buffer is $n_{acc}\times d$ -- transient, CPU-parkable, freed at the end.
2. **Sketch.** Draw $S\in\mathbb R^{s\times n_{acc}}$, $s = 2d$, Gaussian $/\sqrt s$ (SRHT at scale). Compute $SA$ where $A$ is the buffered matrix; SVD $SA = U\Sigma V^\top$; keep $\sigma_i > 10^2\varepsilon\,\sigma_1$; preconditioner $P = V_{:r}\Sigma_{:r}^{-1}$ ($d\times r$, transient).
3. **Refine.** $w \leftarrow w_0$ (warm -- Type A's output). For rounds $j=1,2,\dots$ (default cap 8):
   $$r_j = y_{acc} - A w \ \ \text{(fresh, never carried)};\qquad \delta_j = \mathrm{LSMR}(AP,\ r_j;\ \texttt{atol=btol}=5\varepsilon,\ \texttt{conlim}=0,\ \texttt{maxiter}=200);\qquad w \leftarrow w + P\delta_j.$$
   Inner budget 200 always suffices because $\kappa(AP)\approx2$--$5$ (this is why refinement restarts are safe here and nowhere else -- Part 2, T6).
4. **Stop** on the round-level observable: quit when $\|A^\top r_j\|$ fails to improve by $0.5\times$ (matches oracle on every measured cell), or under noise when it reaches the floor estimate.
5. **Guard.** Track $w_{\rm best}$ by the observable; if the final observable is not better than at entry, **return $w_0$ unchanged**. Worst case is a no-op.

**Why SPIR and not block-QR whitening here:** structure-blind (works on ridges/unstructured, which the certified blocks of real networks will be), stopping solved, refinement = native warm start, and its one cost -- the $d\times r$ transient -- is paid once, offline-grade. Block-QR remains an internal option when memory is tight *and* an internal structure probe passes; it is not part of the default formula.

**Floors (the acceptance contract for the whole battery).** $\sigma=0$: machine epsilon of the working dtype (fp64: rel $L_2\lesssim10^{-14}$ against the cell's SVD floor). $\sigma>0$: the statistical floor $\approx0.27\,\sigma\sqrt{r/n_{\rm eff}}$-scaled, i.e. $\sigma\sqrt{r/n_{acc}}$ for Type B, and the corresponding batch-limited floor for Type A.

---

# Part 2 -- lessons that make batching work (the accumulated tricks)

- **T1. Steering vs. baked-in preconditioners.** A preconditioner that only steers (block-Jacobi, sketch+IR) degrades gracefully when stale or batch-estimated: cost appears as iterations. A factorization baked into the solution map (whitening) is poisoned by staleness ($10^{-8}$ drift = 8 orders, measured). Both step-2 types are steering-type *on purpose*; do not "optimize" either into a baked-in form.
- **T2. Never carry a residual across batches.** A residual is indexed by rows; resampling invalidates it (measured crash + measured accuracy loss). Recompute $r$ fresh from the current rows, every step and every refinement round.
- **T3. Row budget for solve-to-floor is $4d$, hard.** On a fixed row set: $b=4d$ at floor, $b=2d$ fails under noise ($10^3\times$, measured) unless rescued by noise-scaled truncation, $b=d$ unrescuable. Type B satisfies this by accumulation; Type A never solves-to-floor so it is exempt. Never expose this as a user constraint.
- **T4. Gram formation loses half the digits; damping is the honest patch.** Whitening from eigh$(\Phi_C^\top\Phi_C)$: $1.6\times10^{-8}$ vs QR's $1.7\times10^{-15}$ (measured). For a *preconditioner at the $10^{-9}$ ceiling* a damped Gram is acceptable (that is A-EMA's compromise, priced at ~1--2 orders of ceiling); for anything aiming below $10^{-10}$, factor the block itself (QR/SVD of rows, A-rotQR / Type B's sketch-SVD).
- **T5. Rank-deficient blocks: pivot-and-drop with a *relative* threshold, and beware `np.where` evaluating both branches** (NaN poisoning on dead eigenvalues -- measured, cost a debugging round). Drop tolerance $10^2\varepsilon$ of the block's leading diagonal; under noise raise it to the residual floor $\hat\tau$ (this single knob rescued $b=2d$, measured).
- **T6. Restarted short-inner refinement has a hard floor** (directions needing Krylov degree > inner never converge; $3\times10^{-11}$ stall, measured). Safe only when the preconditioned operator's spectral degree fits the inner budget -- true for SPIR ($\kappa\approx4$), false for whitened or unpreconditioned operators. Type A's per-step restarts are a *different* regime (fresh rows each restart) and are measured by A1, not assumed.
- **T7. Fong-Saunders stopping with $\varepsilon$-scaled tolerances is the stopping rule** (`atol=btol` $\approx5\varepsilon$, `conlim=0`): stopped $\approx$ oracle on every cell including the semiconvergence stress case, *and* it self-regularizes under noise (iterations fall as $\sigma$ rises -- early stopping as regularization, measured). At fp32, carry the recurrence scalars in fp64 (scipy's internal condition estimate overflows otherwise; harmless but noisy).
- **T8. Keep-best by an observable + revert guard.** Semiconvergence and null-space drift announce themselves in observables ($\|B^\top r\|$ plateau, $\|z\|$ growth) thousands of iterations before eval damage (measured). Track best-by-observable; on exit, if not better than entry, return the entry point.
- **T9. Statistical floors are the success criterion under noise, and they are quantitative:** achieved $=0.25$--$0.27\,\sigma\sqrt{r/n}$ across six decades (measured). An experiment that doesn't state its predicted floor before running will fool itself (see T10).
- **T10. Methodology traps that manufactured false conclusions before** (all measured): per-cell budgets that vary with the swept variable; ratio-to-floor metrics that saturate; best-over-trajectory reported per-metric; fixed recording strides near the measurement scale; $\kappa$ computed from a formed $A^\top A$ (6 orders wrong). Fixed generous caps, cap-hit flags, non-saturating disagreement metrics, $\kappa$ from $A$ directly.
- **T11. Batch-order correlation is an untested hazard**: every batching measurement so far used shuffled/uniform rows. An unshuffled loader feeds correlated batches; A-EMA's Gram could bias. Tested by A4; until then, the method assumes shuffled batches and must say so.
- **T12. Warm starts inherit null components** ($\|w\|$ inflates $5$--$20\times$ over min-norm, measured); cold solves are near min-norm. Type B warm-starts from Type A's output by design; B3's guard covers the pathological case.

---

# Part 3 -- the experiment battery

All on the expD09 problem set (3 targets $\times$ $N\in\{64,128,256\}$, plus abs_cubed/$N{=}512$ as the stress cell), fp64 primary + one fp32 arm, seeds $\ge3$ where noise or sampling enters. Every experiment states its pass bar *before* running (T9), uses fixed generous caps with cap-hit flags (T10), and reports trajectories, not endpoints. Metrics: eval rel $L_2$ on the clean grid; disagreement with the direct solve where floors saturate; predicted floor lines drawn on every noise plot.

## Arm A -- Type A under batching

- **A1 -- update-rule bake-off and streaming ceiling.** Grid: $\{$A-i ($\tau{=}1,3$), A-ii$\}$ $\times$ $b\in\{d/4,d/2,d,2d,4d,8d\}$ $\times$ $\{$A-EMA, A-rotQR$\}$, frozen $\Phi$, $\sigma=0$, fresh rows each step, run to a fixed generous step cap. *Measures:* ceiling and steps-to-$\{10^{-4},10^{-6},\text{ceiling}\}$. *Pass:* ceiling within $10\times$ of the full-batch $1.1\times10^{-9}$ at $b\ge2d$; $\le10^{-6}$ at $b=d/4$; degradation monotone and smooth in $b$ (no cliff).
- **A2 -- factor-maintenance dynamics.** For the A1 winner: $\beta\in\{0.9,0.99,0.999\}$ $\times$ $\lambda_{\rm damp}$ grid $\times$ $b$ grid; track $\kappa(\Phi M^{-1})$ vs step as the EMA warms, and the warmup transient's cost. *Pass:* $\kappa$ within $10\times$ of full-batch block-Jacobi after $\le 2\lceil d/k\rceil$ steps; no $\lambda$ cliff (plateau, T-style).
- **A3 -- drift tolerance (synthetic).** Geometry perturbed by $\eta$ per step (expD08 protocol), $\eta\in\{10^{-8},\dots,10^{-3}\}$, factors maintained normally. *Measures:* the tracking floor vs $\eta$. *Pass:* floor $\propto\eta$ within $3\times$ over four decades (the coupling-law shape); no divergence at any $\eta$; recovery to the $\sigma{=}0$ ceiling when $\eta\to0$ mid-run.
- **A4 -- batch-order robustness (the unshuffled-loader trap, T11).** Sorted-by-$x$ batches vs shuffled, same budgets. *Pass:* $\le10\times$ ceiling loss sorted vs shuffled for the shipped variant; document the loss for both variants.
- **A5 -- noise $\times$ batch.** $\sigma\in\{10^{-6},10^{-4},10^{-2}\}$ $\times$ $b$ grid. *Measures:* where the achieved floor lands between $\sigma\sqrt{r/b}$ (one batch) and $\sigma\sqrt{r/n}$ (all rows) -- this locates Type A's effective sample size under EMA. *Pass:* monotone in both axes; never above $3\times$ the single-batch prediction; no blowup anywhere.
- **A6 -- fp32 arm.** A1's winning config at fp32 with $\varepsilon$-scaled knobs. *Pass:* lands within $10\times$ of the fp32-appropriate ceiling ($\sim10^{-6}$), same qualitative batch behavior.

## Arm B -- Type B under batching

- **B1 -- buffer size.** $n_{acc}/d\in\{1,2,4,8\}$, buffer filled from batches of size $b\in\{d/4,d\}$ (buffer built small-batch on purpose). *Measures:* $\kappa(AP)$ and final accuracy vs $n_{acc}$. *Pass:* floor at $n_{acc}\ge4d$ regardless of the $b$ that filled it; graceful below; $b$-independence confirmed (T3 satisfied invisibly).
- **B2 -- refinement row-source (the open design question).** Preconditioner from the buffer; refinement rounds on (i) the same buffer, (ii) a fresh batch per round, (iii) a growing buffer. *Measures:* floor reached, rounds needed, per-variant. *Pass:* at least one variant reaches the cell floor with observable stopping; ship the simplest passing variant. (ii) passing would make Type B fully batch-native -- worth knowing either way.
- **B3 -- the guard under a bad freeze.** Invoke Type B while residual drift $\eta\in\{0,10^{-8},10^{-5},10^{-3}\}$ is still present (a premature trigger). *Pass:* returned weights never worse than entry at any $\eta$ (the revert fires); at $\eta=0$ full floor as normal.
- **B4 -- noise + stopping vs the accumulated floor.** $\sigma\in\{10^{-6},10^{-4},10^{-2}\}$, $n_{acc}$ growing during the run. *Pass:* achieved $\le1.5\times$ the predicted $\sigma\sqrt{r/n_{acc}}$; stopped $\approx$ oracle (no semiconvergence into the noise, T7/T8).
- **B5 -- accumulation $\equiv$ big batch (plumbing).** Accumulated frozen-$\Phi$ rows vs one draw of equal size. *Pass:* statistically indistinguishable results (this is an identity; the test catches plumbing bugs, not math).
- **B6 -- warm vs cold entry.** $w_0$ from a converged Type A run vs $w_0=0$. *Measures:* rounds saved, final $\|w\|$ (T12). *Pass:* warm never worse in accuracy; report the norm inflation for the step-3 record.

## End-to-end (step-2 only, no Adam)

- **E2E -- the handoff.** On each cell: synthetic drift schedule $\eta_t$ annealing $10^{-3}\to0$ (the Adam stand-in); Type A runs throughout at $b\in\{d/2, 2d\}$; when the drift gauge quiets, freeze, hand $w$ to Type B, accumulate, solve, guard. *Pass:* Type A tracks the coupling-law floor while $\eta_t>0$; Type B lands each cell on its measured floor (machine eps for $\sigma=0$ cells, statistical floor for noisy arms); the guard never returns worse than its entry; **no configuration knob differs across cells.**

**Order of execution:** A1 first (it selects the Type A variant everything else uses), then A2/A3 in parallel with B1/B2, then the remainder, E2E last. Estimated compute: a few hours total on the existing rigs; every experiment writes JSONL + figures per repo conventions.
