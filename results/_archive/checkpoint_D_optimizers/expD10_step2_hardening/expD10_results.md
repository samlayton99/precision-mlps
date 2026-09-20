# expD10 -- hardening block-QR into the step-2 part

**Status: draft-pending-Sam.**

## TL;DR

- The expD09 recipe's four open failures are all closed by engineering, not new research: **stopping** = the textbook Fong-Saunders rule (the campaign had it disabled); **ordering** = spectral seriation from a 256-row Gram sample; **memory** = the whitened operator applied implicitly in compensated (double-word) arithmetic, so the $n\times d$ copy of $B$ is never formed; **staleness** = re-whiten at each solve event ($k$ passes, amortized).
- With observable stopping only (no oracle anywhere): 9-cell geo-mean $9.0\times10^{-14}$, below the SVD floor $1.08\times10^{-13}$; every noise level within $1.15\times$ of the statistical floor $0.272\sigma$ across six decades; batching at floor for $b\ge4d$.
- The pipeline is **precision-agnostic**: identical code at fp32 with $\varepsilon$-scaled tolerances lands on the fp32 floor (37--108 iterations). Nothing in the method knows its dtype.
- Two new negatives: restarted refinement with a small inner budget has a hard floor (Krylov degree truncation), and **ordering matters for block-Jacobi preconditioning too** once dd removes the fp64 stall (contiguous $10^{-16}$ vs random $10^{-5}$ at 400 iterations) -- the earlier "random $\approx$ contiguous" correction was an artifact of measuring at the fp64 stall.

## Question

Can the expD09 block-QR recipe be hardened into a complete step-2 part -- iterative least-squares, $O(d)$ state, one pass per iteration, batch-robust, noise-floor-reaching, order-blind, observably stopped, precision-agnostic -- with engineering only?

## Experiment design

Problems, floors, and noise protocol are expD09-validation's (`common_val.make_problem`): frozen $A=[\Phi,\mathbf 1]$, $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, uniform centers + halo at $\lambda^\*=0.25$; 9 cells = {sine, sine_8pi, runge} $\times$ $N\in\{64,128,256\}$; $n=8d$ rows; eval rel $L_2$ on 4001 clean points. The stress cell is abs_cubed $N{=}512$ ($d=922$), the recipe's semiconvergence worst case.

**The finalist ("hardened block-QR"), assembled from the arms below.** Stage 1: seriation if column order is unknown -- normalize an $s$-row sample's columns, build $W=|\mathrm{corr}|$, order by the Fiedler vector of the normalized Laplacian ($v = D^{-1/2}u_2$); then contiguous pivoted Householder QR blocks of $k{=}128$ in that order, drop below $\texttt{rcond}=10^{-13}$ (scaled as $\sim10^2\varepsilon$ of the dtype). Stage 2: LSMR on the whitened operator with **Fong-Saunders stopping** `atol=btol=1e-15` ($\sim5\varepsilon$), `conlim=0`; warm start via the residual ($\min_z\|Bz-(y-Aw_0)\|$, $w=w_0+$unwhiten$(z)$). Stage 3: unwhiten once. Memory mode A materializes $B$ (batch-sized); mode B applies $R^{-1}$ implicitly with the matvec computed in compensated double-word arithmetic of the native dtype (Dekker/Knuth `two_sum`/`two_prod`; dtype-generic, pure tensor ops), fp64 in/out.

- **E4 stopping:** instrumented LSMR on the stress cell recording eval, $\|B^\top r\|$, $\|z\|$ per 200 iterations; then Fong-Saunders vs an oracle grid (min over re-runs at 25--800 iterations) on all 9 cells; the $k{=}32$ slow-tail regime as an out-of-spec probe.
- **E1 ordering:** destroy order with a random permutation, recover with seriation from $s=256$ sample rows (sensitivity: $s=64$, similarity powers 1 and 4); compare true / permuted-naive / seriated through the full solve.
- **E2 memory:** materialized $B$ vs implicit-fp64 (control, recipe detail 2) vs implicit-compensated, on 3 cells.
- **E3 order-blindness of dd+block-Jacobi:** full-dd LSQR, $k{=}128$ block-Jacobi from contiguous / random / seriated blocks, 400 iterations. (Also reproduces the round-6 result and pins its config: $k{=}128$, not 64.)
- **E5 floors:** noise $\sigma_{rel}\in\{0,10^{-8},10^{-6},10^{-4},10^{-2}\}$ $\times$ 9 cells; batching = whiten AND solve on the same random $b\in\{8d,4d,2d,d\}$ rows. All under Fong-Saunders stopping.
- **E6 warm/anytime:** $w_0$ = SVD solution + mid-spectrum row-space perturbation at eval $10^{-3}$ (adversarial: the error sits in one hard direction); pass budgets 3--300.
- **Precision-agnostic arm:** the identical pipeline in float32 end to end (`rcond=1e-6`, `atol=btol=1e-7`) vs a float32 truncated-SVD floor.

**Code & data.** `experiments/expD10_step2_hardening/` (`core10.py`, `run_floors.py`, `run_evidence.py`); tests `tests/test_expD10.py` (6, all passing). Data: `floors.json`, `warm.json`, `evidence.json`. Figures: `figures/F1_stopping.png`, `F2_ordering.png`, `F3_memory_precision.png`, `F4_floors.png`.

## Results

- **Stopping (E4).** Fong-Saunders `atol=btol=1e-15` stops the stress cell at iteration 4011 exactly on its floor ($3.69\times10^{-10}$) -- the case that previously burned 60k iterations and drifted five orders. The observables tell the whole story: eval reaches the floor by $\sim$3200, $\|B^\top r\|$ goes flat by $\sim$4200, and the null-drift explosion announces itself in $\|z\|$ ($36\to4\times10^{6}$) thousands of iterations before eval degrades. On the 9 cells: stopped geo-mean $9.0\times10^{-14}$ vs oracle $5.6\times10^{-14}$ (worst cell $7.4\times$), both below the SVD floor. Out-of-spec $k{=}32$ stops $13\times$ off oracle at $2.8\times10^{-12}$: spec the part at $k\ge64$.
- **Ordering (E1).** Permuted columns cost five orders ($7.7\times10^{-9}$ geo-mean, the random-blocks arm). Seriation from 256 sample rows recovers $3.1\times10^{-13}$, within $3.4\times$ of the true order ($9.0\times10^{-14}$); the residue is paid in iterations (up to $\sim$4000 vs $\sim$250), not accuracy. Robust to $s=64$ and to the similarity power.
- **Memory (E2).** Implicit-compensated matches materialized $B$ (oracle parity, e.g. runge $N{=}256$: $6.8\times10^{-15}$ vs $1.0\times10^{-14}$); the fp64-implicit control fails at $10^{-6}$--$10^{-7}$ as the recipe measured. Consequence: the only data-sized array the method needs is $\Phi$ itself, which training already stores (for the certified block, $\Phi$ is the stored Jacobian/activations).
- **dd + block-Jacobi ordering (E3).** At $k{=}128$: contiguous reaches $10^{-16}$ by 400 iterations on every tested cell; random blocks sit at $10^{-5}$--$10^{-7}$; seriated recovers to $10^{-9}$--$10^{-12}$ and is still descending. The prior "random $\approx$ contiguous for preconditioning" held only at the fp64 stall, where both arms floor at $\varepsilon\kappa$; removing the stall (dd) exposes that contiguous blocking clusters the spectrum (13 vs 83 distinct singular-value clusters at $N{=}128$) and Krylov speed lives on clustering.
- **Noise + batching (E5).** Every cell at every noise level lands at $1.06$--$1.15\times$ the statistical floor $0.272\sigma$, and iterations *fall* under noise (55--70 at $\sigma=10^{-2}$) -- the stopping rule performs early-stopping regularization on its own. Batching: $b{=}4d$ at floor on all cells; $b{=}2d$ mostly at floor; $b{=}d$ degrades smoothly to $10^{-8}$--$10^{-10}$.
- **Warm/anytime (E6).** From eval $10^{-3}$, warm solves reach $5\times10^{-14}$--$2\times10^{-13}$ in cold-like iteration counts (adversarial perturbation direction; no speedup, no loss). Anytime curve (sine $N{=}256$, warm): 3 passes $\to8\times10^{-5}$, 30 $\to10^{-6}$, 100 $\to7\times10^{-9}$, 300 $\to8\times10^{-14}$.
- **Precision-agnostic.** fp32 pipeline lands $5\times10^{-7}$--$3\times10^{-6}$ against fp32 SVD floors $2.5\times10^{-7}$--$8\times10^{-6}$, in 37--108 iterations. Only the two tolerances scale with $\varepsilon$.

### Figures

- **`F1_stopping.png`** -- left: stress-cell trajectories (eval, $\|B^\top r\|$, $\|z\|$ vs iteration; dashed line = where Fong-Saunders stops). Look for: both observables move thousands of iterations before eval degrades. Right: per-cell stopped (squares) vs oracle (circles) vs floor (dashes).
- **`F2_ordering.png`** -- left: fp64 whitening, eval per cell for true/permuted/seriated order (one line each). Look for: seriated hugging the true line, permuted five orders up. Right: same three arms for dd+block-Jacobi at 400 iterations.
- **`F3_memory_precision.png`** -- left: bars per cell, materialized vs implicit-fp64 vs implicit-compensated. Look for: the fp64 control alone failing. Right: fp64 and fp32 pipelines each against their own dtype floor.
- **`F4_floors.png`** -- noise (all cells vs the $0.272\sigma$ line), batching (eval vs $b/d$), anytime (eval vs pass budget).

## Additional details

- **New negative: restarted refinement has a hard floor.** LSMR restarted on fresh residuals with inner budget $m$ cannot converge directions whose Krylov polynomial needs degree $>m$: measured $3\times10^{-11}$ stall (inner=100, $N{=}256$) vs $1.7\times10^{-14}$ for a single 342-iteration run. This is why SPIR tolerates restarts (its preconditioner gives $\kappa\approx4$, so 200 inner iterations always fully converge) and the whitened operator must not be restarted short. Goes in the do-not-retry table.
- **Staleness policy, per Sam:** whitening is part of the solve event, not cached state -- re-factor at each solve ($k$ passes, amortized over the solve interval); geometry stationarity improves through training, making solve events cheaper to justify, not harder.
- Warm solves inherit $w_0$'s null components: $\|w\|$ lands $5$--$20\times$ above min-norm. Coupling-law exposure ($\|v\|\eta$) to watch in step 3; a cold solve at the final event restores min-norm-ish output.
- The compensated ("dd") arithmetic is Dekker/Knuth error-free transformations: 5--10 native adds/muls per compensated op, expressible as tensor ops in any IEEE dtype (bf16 pairs $\approx$ 16 mantissa bits, fp32 $\approx$ 48, fp64 $\approx$ 106). It is a localized implementation detail of mode B, not a precision regime change.
- scipy `lsmr` at fp32 with `conlim=0` emits harmless overflow warnings from its internal condition estimate; a production implementation should carry the recurrence scalars in fp64 (they are $O(1)$ memory).

## Hardening battery (H1-H6 + scaling twins)

Run after the main results as the final gate before step 3. Data: `hardening.json`, `hardening_extra.json`; figure `figures/F5_scaling_twins.png`.

- **H1, family generality (24 cells: 6 targets $\times$ $N\in\{64..512\}$): all pass.** At/below the SVD floor, or -- for targets whose own approximation floor is high -- *exactly on that floor at every width* (abs_cubed: $5.3\times10^{-7}\to3.7\times10^{-10}$ tracking its floor as $N$ grows; no overfit past it).
- **H2, scale:** floor held at $d=1844$ (sine $N{=}1024$: $8.0\times10^{-14}$ vs $6.6\times10^{-14}$). Iterations grow superlinearly on the band structure: $62\to284\to564\to1930$ over $d=270\to1844$ at fixed $k{=}128$.
- **H3 + d-series, spectral twins (random matrices with the QI system's spectrum, rank cliff, and $\propto\sigma_i$ target energy):**
  - *Structure-free* (Haar singular vectors): fails, and the failure **grows with $d$** -- $7\times10^{-11}$ ($d{=}462$) $\to$ $2.7\times10^{-8}$ ($922$) $\to$ $1.8\times10^{-6}$ ($1844$) at a 20k-iteration cap, against floors of $\sim5\times10^{-15}$. SPIR ($O(dr)$) reaches $1.3\times10^{-15}$ at every $d$. This is the round-4 information argument made concrete: no blocking exists, so whitening removes only a $k/d$ fraction of the conditioning.
  - *Hidden-cluster* (correlated within clusters, independent across, columns permuted): with **agglomerative cluster-matched blocks** the floor holds at every $d$ with nearly flat cost -- $9.9\times10^{-15}$ (38 it), $9.7\times10^{-15}$ (42 it), $1.4\times10^{-14}$ (**131 it at $d{=}1844$**). Fiedler+stride blocking *fails* at $d{=}1844$ ($6\times10^{-7}$): stride boundaries split clusters.
  - **Blocking upgrade shipped:** `cluster_blocks` (average-linkage agglomerative on $1-|\mathrm{corr}|$ from an $s$-row sample, capped at $k$) replaces Fiedler seriation everywhere -- it also *beats* seriation on the permuted band case ($5.4\times10^{-14}$ in 1068 it vs $2.9\times10^{-13}$ in 2404). A gap-cut stride variant was tried and does not work.
  - **Applicability gate shipped:** `kappa_gate` whitens the sample itself with the candidate blocks and reads $\kappa(B_s)$ -- QI band $\sim10^{9\text{-}10}$ (works), matched clusters $\sim10^0$ (instant), structure-free $\sim10^{13}$ (will not reach the floor). Three-orders separation, decided before touching the full data. The earlier $\rho$-mass detector does NOT separate these cases and is withdrawn.
- **H4, noise $\times$ batching jointly:** $b{=}4d$ lands at $\approx1.1\times$ the batch statistical floor $\sigma\sqrt{r/b}$ (one cell $6.6\times$). **$b{=}2d$ blows up under noise** on two cells (up to $6\times10^{-2}$, $10^3\times$ above floor) -- the $\sigma{=}0$ finding "2d mostly fine" does not carry to noisy data. Spec: $b\ge4d$ hard.
- **H5, knob sweeps** (rcond $\times$ atol, $3\times3$): plateaus, no cliffs; total spread one order; rcond $=10^{-15}$ marginally best and flattest.
- **H6, cap honesty:** the seriated arm converged at 2109 iterations, unchanged at a $4\times$ cap -- "premium in iterations, not accuracy" survives.
- **Loose end pinned:** sine $N{=}512$ converges to $2.1\times10^{-13}$ at any budget, $6\times$ above its SVD floor -- a genuine, mild, cell-specific gap (not stopping; $N{=}1024$ is back at $1.2\times$).
- **Implicit-compensated mode parity BREAKS at $d{=}1844$:** measured $1.16\times10^{-11}$ against materialized $8.0\times10^{-14}$ ($150\times$). Diagnosis: the implicit operator is $A_{\rm kept}R^{-1} = Q + ER^{-1}$ with a *fixed, consistent* $\varepsilon\kappa(R)\approx10^{-6}$ deviation from $Q$ -- present at every $d$ (measured $\sim3\times10^{-6}$ per call at both $d{=}270$ and $1844$), benign at small $d$ (parity held to $d\approx500$), damaging at $d{=}1844$, plausibly by splitting the exactly-degenerate singular-value clusters that Krylov speed depends on. Oracle-vs-stopping attribution pending (dd runs at this size are slow). **Guidance: materialized $B$ is the validated mode; implicit-compensated is validated only to $d\approx500$ and experimental beyond.**

## Conclusions

Pending Sam. Proposed: the hardened block-QR part meets the step-2 contract **when the column-correlation structure is clusterable** -- $O(dk)$ state with $k$ fixed, one pass per iteration, floor-reaching under an observable stopping rule at $d$ up to 1844, statistical-floor-reaching under noise, batch-robust at $b\ge4d$ (hard under noise), order-recovered by agglomerative blocking, precision-agnostic, with the $n\times d$ whitened copy eliminable at full accuracy, and with a pre-solve sample gate (`kappa_gate`) that detects the structure-free regime where no $O(d)$ method can reach the floor.

## Open questions

- Blocking on a genuinely 2-D geometry (expE01 zoo): `cluster_blocks` is embedding-free (unlike Fiedler) and should transfer, but measure before trusting.
- The implicit mode's large-$d$ gap: attribute (stopping vs attainable floor), and test whether carrying the QR itself in compensated arithmetic removes the $\varepsilon\kappa(R)$ term.
- $\Phi$ as a Jacobian block (not last-layer activations): same algebra, unmeasured spectrum.
- The seriated arm's iteration premium ($4$--$15\times$): block-boundary alignment against the halo/live transition is unexplored and probably cheap to fix.
- bf16 arm (needs a PyTorch implementation; numpy has no bf16).
