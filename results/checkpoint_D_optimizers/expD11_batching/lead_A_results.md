# lead_A -- an independent attack on the 7-requirement solver

**Status: draft-pending-Sam.** Independent line, run without steering. Nothing here modifies `core11.py` / `run_frozen.py` / expD09 / expD10.

## TL;DR

- **No configuration met all 7 requirements.** The blocker is requirement 2 ($O(d\cdot k)$, $k$ fixed), and it is now blocked by *two independent* measured walls that are both $\Theta(d\cdot r)$, not one.
- **New best result in the repo at the $O(d\cdot r)$ tier, and it needs no setup at all:** plain LSQR with one-sided **full reorthogonalization**, no preconditioner, no sketch, no factorization, reaches **geo-mean $1.7\times10^{-15}$ at an observable stopping rule across 8 cells** -- below every cell's truncated-SVD floor (geo-mean $6.6\times10^{-15}$), in $\approx r$ iterations. It does this on the two cases that previously required SPIR: 2-D radon ridges and the structure-free twin. Requirement 3 is satisfied (no cubic setup); requirement 2 is not (state $=r\cdot d$).
- **The negative is now a clean one-parameter law.** With a fixed reorthogonalization window $c$, attainable accuracy is a function of $c/r$ *alone* -- identical $c/r$ gives identical error across a factor-8 range in $d$ -- with a knee at $c/r\approx0.8$. So $c$ must scale with $r\propto d$. Fixed $c=256$ on the structure-free twin degrades $6\times10^{-16}\to1.4\times10^{-13}\to1.5\times10^{-8}\to3.7\times10^{-6}$ as $d$ goes $256\to2048$.
- Two escape hatches tested and dead: **multi-level (butterfly) block whitening** does not move $\kappa$ at all on unstructured or 2-D ridge systems ($1.0\times10^{15}$ at every $k\le128$, $L\le4$, 3 permutation families, up to $512d$ of state); and the window law is **not** a precision artifact -- double-double (eps $10^{-32}$) buys a uniform $7$--$10\times$ and does not change the law's shape.
- Requirement 7 (drift) is *easier* than believed for this family: with no preconditioner, attainable error $\approx3$--$6\eta$, slope 1 over six decades -- the operator-accuracy limit, with no $\eta^{1.73}$ staleness amplification. Requirement 5 (literal: fresh batch per matvec) **fails completely** -- a Krylov recurrence cannot survive a changing operator; residual stays $O(1)$ at every $b<n$.

## Question

Is there a solver for $\min_w\|\Phi w-y\|_2$ that simultaneously satisfies all seven requirements, or can that be ruled out? Approach chosen from first principles rather than continuing the preconditioner search: first establish *which resource* is actually binding, then attack that resource.

## Experiment design

The object is $A\in\mathbb R^{n\times d}$ with $\kappa\approx10^{14}$--$10^{15}$, numerical rank $r$ ($0.6d$ to $1.0d$), spectrum exponentially graded with no gap, target energy $\propto\sigma_i$. Metric throughout is eval rel $L_2=\|A_{\rm ev}\hat w-y_{\rm ev}\|/\|y_{\rm ev}\|$ on the held-out grid; the reference is `prob["floor_pool"]`, the truncated-SVD solve on the whole pool. Every $\kappa$ below is taken from the SVD of the matrix itself, never from a formed Gram.

**Step 0 -- locate the binding resource.** Three measurements, cheapest first.

- *Precision arms.* One LSQR code path run at three precisions: all-fp64; **mixed** (fp64 matvecs, double-double recurrence vectors and scalars); all-**dd** (Dekker double-double, $\epsilon\approx10^{-32}$, exactly 2 words per number so still $O(d\cdot k)$ with $k=2$). If the fp64 stall were a recurrence-rounding effect, `mixed` would fix it; if it were a matvec-rounding effect, only `dd` would.
- *Krylov-optimal rate.* The best residual attainable in $\mathcal K_m(A^\top A, A^\top y)$: build the basis with twice-repeated Gram-Schmidt and solve the projected LS exactly at every $m$. This is an exact-arithmetic emulation and upper-bounds what *any* Krylov method can do, so it separates "rate problem" from "precision problem".
- *Fixed-window reorthogonalization.* LSQR where the new Golub-Kahan vector $v_{j+1}$ is orthogonalized (two passes) against a stored set of $c$ earlier $v$'s. One-sided only, so state is $c\cdot d$ floats with $c$ fixed -- exactly requirement 2's budget. Window contents swept: `recent` (sliding), `first` (frozen on the first $c$, i.e. the dominant directions), `both` (half and half). $c=\infty$ means full reorthogonalization, state $=(\text{iterations})\cdot d$.

**Step 1 -- multi-level (butterfly) block whitening, Gate 1 only.** The one preconditioner family with the right parameter count that the campaign had not tested. $L$ levels; at each level the columns are permuted by $P_\ell$, partitioned into blocks of $k$, and each block replaced by the $Q$ of its pivoted Householder QR (so blocks are orthonormal *by construction* -- never by applying an inverse factor to data, per round 6). Composite whitener $M=D_1P_1\cdots D_LP_L$; state $L\cdot d\cdot k$; setup $L\cdot(d/k)$ QRs of $n\times k$ blocks $=O(Lndk)$, sub-cubic. Permutations: `stride` (perfect shuffle, the butterfly pattern), `random`, `offset` ($k/2$ shift). Rank-deficient blocks pivot-and-drop at `rcond` $=10^{-13}$. Gate 1 asks one thing: does $\kappa$ of the whitened matrix fall as levels are added?

**Step 2 -- the frontier.** Best eval vs $c\in\{32,64,128,256,\infty\}$ across `qi1d` $N\in\{64,128,256,512\}$, `qi2d` (radon_tensor, $\lambda=0.12$) $N\in\{144,324,576,1024\}$, `twin_unstruct` $d\in\{256,512,1024,2048\}$. Iteration cap fixed at $3r+80$ for every arm so the budget never varies with the swept variable.

**Step 3 -- the hybrid.** Block-QR whitening at $k=128$ (state $kd$) *composed with* a window of $c$ (state $cd$): the only configuration in this study whose total state is $(k+c)d$ with both constants fixed.

**Step 4 -- requirements 5, 6, 7 and stopping.** Drift: a fresh relative perturbation $A\odot(1+\eta g)$ per matvec, $\eta\in\{0,10^{-14},\dots,10^{-3}\}$. Batching: a fresh random row batch of $b$ rows per matvec, $b=d/\{1,2,4,8,16,64\}$, against both `floor_pool` and `floor_1batch(b)`. Stopping: Fong-Saunders style, first iterate with LSQR's $\bar\phi\le5\epsilon\|y\|$, reported next to the oracle best on all 8 cells. Noise: `noise_rel` $\in\{10^{-6},10^{-4},10^{-2}\}$.

**Code & data.** `experiments/expD11_batching/lead_A/`: `ml.py` (multi-level whitening + $\kappa$ helpers), `wlsqr.py` (windowed-reorth LSQR), `ddsolve.py` + `run_ddwin.py` (precision arms, dd windowed reorth), `run_gate1.py`, `run_frontier.py`, `run_hybrid.py`, `run_batch_drift.py`, `run_stop.py`, `figs.py`. Data: `gate1.jsonl`, `frontier.jsonl`, `hybrid.jsonl`, `batch_drift.jsonl`, `ddwin.jsonl`, `stop.jsonl` (+ `.log` transcripts). Figures: `results/checkpoint_D_optimizers/expD11_batching/figures/leadA_F{1,2,3}*.png`. Problems and floors come from `core11.py` unchanged; dd arithmetic is imported from `expD09_2nd_order_regime/dd.py`.

## Results

**The binding resource is orthogonality, not conditioning and not precision.** The Krylov-optimal curve is exponential and front-loaded from iteration 1 -- about $10^{-1/12}$ per iteration on `qi1d` $N{=}128$ and $10^{-1/20}$ on `twin_unstruct` $d{=}512$ -- so $\Theta(r)$ iterations suffice in exact arithmetic and there is no rate problem to precondition away. Plain LSQR gets nowhere near that curve; the gap is entirely loss of orthogonality. Raising precision does not close it: on `qi1d` $N{=}128$ the mixed arm (fp64 matvecs, dd recurrence) reproduces fp64 to three digits ($2.46\times10^{-6}$ vs $2.46\times10^{-6}$ at $c=16$), so the recurrence arithmetic is irrelevant and it is the matvec's $\epsilon$-level backward error that drives the loss; going all-dd, i.e. 16 extra digits, buys only $7$--$10\times$ at every window size.

**Restoring orthogonality fully solves the problem, at $O(d\cdot r)$ state and zero setup.** One-sided fully reorthogonalized LSQR, unpreconditioned, at the observable stopping rule:

| cell | $d$ | $r$ | SVD floor | stopped | @it | oracle | @it |
|---|---|---|---|---|---|---|---|
| qi1d $N{=}128$ | 270 | 144 | $1.46\times10^{-14}$ | $1.32\times10^{-15}$ | 142 | $4.7\times10^{-16}$ | 159 |
| qi1d $N{=}256$ | 462 | 274 | $1.61\times10^{-14}$ | $1.48\times10^{-15}$ | 332 | $1.5\times10^{-15}$ | 271 |
| qi1d $N{=}512$ | 922 | 527 | $3.52\times10^{-14}$ | $2.44\times10^{-15}$ | 585 | $2.4\times10^{-15}$ | 525 |
| qi2d radon | 571 | 252 | $3.83\times10^{-15}$ | $1.37\times10^{-15}$ | 198 | $3.9\times10^{-16}$ | 274 |
| qi2d radon | 1021 | 378 | $1.66\times10^{-15}$ | $1.19\times10^{-15}$ | 308 | $1.7\times10^{-16}$ | 374 |
| twin-unstruct | 512 | 307 | $3.13\times10^{-15}$ | $1.52\times10^{-15}$ | 309 | $1.5\times10^{-15}$ | 309 |
| twin-unstruct | 1024 | 614 | $3.85\times10^{-15}$ | $1.62\times10^{-15}$ | 611 | $1.6\times10^{-15}$ | 617 |
| twin-cluster | 1024 | 1024 | $5.58\times10^{-15}$ | $3.72\times10^{-15}$ | 1085 | $3.6\times10^{-15}$ | 974 |

Geo-mean stopped $1.70\times10^{-15}$ against floors of $6.59\times10^{-15}$; stopped/oracle geo-mean $1.7\times$. Iterations land at $0.9$--$1.2\,r$ on every cell -- the "one direction per iteration" rate, exactly. This matches SPIR's accuracy while removing SPIR's sketch and its $O(d^3)$ SVD, and it clears the two cases block-QR whitening could not do (2-D radon at practical $k$; structure-free twin). Honest cost caveat: reorthogonalization is $O(dr^2)$ flops and the matvecs are $O(ndr)$, so measured wall-clock scales as $d^{3.1}$ (`twinU`, `qi1d`) and $d^{2.4}$ (`qi2d`, where $r/d$ falls). There is no cubic *setup* -- the method is anytime and front-loaded -- but total flops to machine precision are not below a factorization's.

**The fixed-window frontier collapses onto $c/r$.** Best eval, iteration cap $3r+80$:

| family | $d$ | $r$ | floor | $c{=}32$ | $c{=}64$ | $c{=}128$ | $c{=}256$ | full |
|---|---|---|---|---|---|---|---|---|
| qi1d | 270 | 144 | $1.5\times10^{-14}$ | $3.6\times10^{-7}$ | $9.4\times10^{-9}$ | $2.3\times10^{-15}$ | $5.0\times10^{-16}$ | $5.0\times10^{-16}$ |
| qi1d | 462 | 274 | $1.6\times10^{-14}$ | $1.7\times10^{-6}$ | $3.5\times10^{-7}$ | $7.9\times10^{-9}$ | $1.1\times10^{-15}$ | $1.5\times10^{-15}$ |
| qi1d | 922 | 527 | $3.5\times10^{-14}$ | $2.3\times10^{-6}$ | $9.8\times10^{-7}$ | $1.8\times10^{-7}$ | $3.3\times10^{-9}$ | $2.4\times10^{-15}$ |
| twinU | 256 | 153 | $2.3\times10^{-15}$ | $9.6\times10^{-7}$ | $4.7\times10^{-9}$ | $4.8\times10^{-14}$ | $6.3\times10^{-16}$ | $6.3\times10^{-16}$ |
| twinU | 512 | 307 | $3.1\times10^{-15}$ | $3.8\times10^{-5}$ | $3.4\times10^{-6}$ | $1.7\times10^{-8}$ | $1.4\times10^{-13}$ | $1.5\times10^{-15}$ |
| twinU | 1024 | 614 | $3.9\times10^{-15}$ | $9.1\times10^{-5}$ | $4.1\times10^{-5}$ | $3.6\times10^{-6}$ | $1.5\times10^{-8}$ | $1.6\times10^{-15}$ |
| twinU | 2048 | 1235 | $6.1\times10^{-15}$ | $1.3\times10^{-4}$ | $1.0\times10^{-4}$ | $4.0\times10^{-5}$ | $3.7\times10^{-6}$ | $1.7\times10^{-15}$ |

Read the twin block along constant $c/r$: $0.83\to4.8\times10^{-14}$ and $1.4\times10^{-13}$; $0.42\to1.7\times10^{-8}$ and $1.5\times10^{-8}$; $0.21\to3.6\times10^{-6}$ and $3.7\times10^{-6}$. Equal ratio, equal error, across a factor 4 in $d$. `qi1d` behaves the same ($c/r\approx0.48\to7.9\times10^{-9}$ and $3.3\times10^{-9}$). Window *contents* shift the curve but not its shape: freezing the window on the first $c$ vectors (the dominant directions) beats the sliding window by $10$--$40\times$ at every $c$ -- `twinU` 512 at $c{=}64$ goes $3.4\times10^{-6}\to1.6\times10^{-7}$ -- worth roughly a factor 2 in $c$, not a change of exponent.

**Multi-level whitening: Gate 1 fails, decisively, on everything without block structure.** `twin_unstruct` $d{=}512$: $\kappa=1.0\times10^{14}$ initially, and $9.5$--$10.0\times10^{14}$ for *every* one of the 36 combinations ($k\in\{32,64,128\}$, $L\in\{1..4\}$, three permutation families), including at $512d$ of state. `qi2d` $d{=}571$: $9.5\times10^{14}\to8.7$--$10.0\times10^{14}$, same story. Only `qi1d` moves ($5.2\times10^{14}\to3.8\times10^{11}$ at `offset`, $k{=}64$, $L{=}2$, $62d$), and the one configuration that reaches $\kappa=O(10)$ needs $k{=}128$ at $d{=}270$, i.e. $k/d\approx0.5$ -- a full QR wearing a costume. Per the discipline, this was stopped at Gate 1 and no solver was built on it.

**The hybrid helps only where whitening already helped, and buys speed rather than accuracy.** On `qi1d` $N{=}256$, block-QR at $k{=}128$ alone reaches $1.75\times10^{-14}$ in 490 iterations; adding a $c{=}64$ window reaches $1.73\times10^{-14}$ in **60** iterations, at $192d$ fixed state -- an $8\times$ iteration cut for free, and the most front-loaded configuration measured. At $N{=}512$ the same pair gives $2.09\times10^{-13}$ ($6\times$ above the floor) and on `qi2d` $d{=}571$ it gives $1.4\times10^{-9}$, where the whitening-free full-reorth run gives $2.7\times10^{-16}$. So the hybrid inherits block-QR's scope boundary exactly.

**Requirements 5 and 7.** Drift: attainable error $=3.0$--$3.5\eta$ (`qi1d`) and $6.4$--$6.8\eta$ (`twinU`), slope 1 across $\eta=10^{-14}$ to $10^{-3}$, saturating at the $\sigma{=}0$ floor when $\eta\lesssim10^{-15}$. That is the accuracy of the operator itself, which no method can beat, and it carries no staleness amplification because there is no preconditioner to be stale. Batching, literal reading: total failure. At every $b<n$ the residual never leaves $O(1)$ ($7\times10^{-1}$ to $1.0$), worse than `floor_1batch` at every $b$, and a growing-pool variant does not help. Noise: the oracle iterate lands at $0.53$--$0.80\times$ `floor_pool` at $\sigma_{\rm rel}\in\{10^{-6},10^{-4},10^{-2}\}$, i.e. on the cell's statistical floor, but the $\epsilon$-scaled Fong-Saunders test never fires under noise and needs a $\sigma$-scaled tolerance (the same fix as expD10's F2).

### Figures

- **`leadA_F1_window_frontier.png`** -- two panels, both log-log, best eval rel $L_2$ on $y$ with the truncated-SVD floor region shaded. Left: $x=c/r$; colours are families (blue `qi1d`, green `qi2d`, red `twinU`), marker shape is $d$ within a family, stars are the full-reorth runs plotted at the state they actually consumed. Look for the curves at different $d$ lying on top of each other, and for the knee at the dotted line $c/r=0.8$. Right: the identical data against absolute $c$ -- the curves fan out by $d$, which is the whole negative in one picture.
- **`leadA_F2_trajectories.png`** -- three panels (`qi1d` $N{=}128$, `twin_unstruct` $d{=}512$, `qi2d` radon $d{=}571$), eval rel $L_2$ vs LSQR iteration. Grey = no reorthogonalization, blues = windows $c=32,64,128,256$ darkening with $c$, red = full reorthogonalization; dashed horizontal = SVD floor, dotted vertical = iteration $=r$. Look for the red line being a straight exponential from iteration 1 that hits the floor precisely at $r$ (front-loading, requirement 6), and for each blue line peeling off the red one and flattening at its own $c/r$-determined level.
- **`leadA_F3_batch_drift.png`** -- left: attainable error vs per-matvec perturbation $\eta$, both cells, with a slope-1 guide and the $\sigma{=}0$ floors dotted; look for slope 1 with a small constant and no divergence anywhere. Right: attainable error vs rows-per-matvec $b$, solid = stochastic LSQR, dashed = the single-batch SVD floor; look for the solid lines pinned at $O(1)$ for every $b<n$ and *above* the dashed reference -- the failure is qualitative, not a tuning matter.

## Additional details

**Why two walls and not one.** Reaching $10^{-14}$ requires resolving $\sim r$ singular directions (target energy $\propto\sigma_i$, so nothing can be truncated -- given). There are only two ways to get them: collapse the spectral range with a preconditioner, or let Krylov resolve them one per iteration. Route 1 needs a whitener whose singular vectors match $V$; specifying a generic (Haar) $V$ takes $\Theta(d\cdot r)$ numbers, and a product of $L$ block-diagonal factors carries $L\cdot d\cdot k$, which is a measure-zero subfamily -- the Gate 1 table is that dimension count showing up as a flat line. Route 2 already runs at the optimal rate but needs orthogonality over the whole $\Theta(r)$-iteration history, and the frontier says the history cannot be compressed below $\approx0.8r$ vectors. Both routes price out at $\Theta(d\cdot r)$, which is why the $10^{-8}$-ish ceiling at fixed state has been so stubborn.

**A structural consequence worth stating.** Requirements 1 and 4 plus "$\Theta(r)$ directions are needed" force total work $\Theta(n d r)$, which is $\Theta(d^3)$ whenever $r\propto d$. So *no* iterative method can be asymptotically cheaper than a factorization at this tolerance on this spectrum. Requirement 3 as written forbids a cubic *setup*, and the reorth-LSQR route does satisfy that -- it is anytime, front-loaded, and never factorizes $A$ -- but the "if you can afford cubic, use the pseudoinverse" argument does bite the total, and the honest defence of the iterative route is front-loading and drift tolerance, not flops.

**It beats the pseudoinverse it is supposed to be a cheap substitute for.** Against LAPACK `gelsd` (`np.linalg.lstsq`, a direct SVD solve) on the same pool: `twin_unstruct` $d{=}512$ eval $1.52\times10^{-15}$ vs $8.37\times10^{-13}$, train residual $6.8\times10^{-14}$ vs $3.4\times10^{-11}$; `qi2d` $d{=}571$ eval $1.37\times10^{-15}$ vs $5.01\times10^{-14}$. Output norm is $1.1$--$1.2\times$ the min-norm solution's, so there is no null-space inflation. This is the expected ordering -- LSQR is backward stable in $A$ while a direct SVD solve pays $\epsilon\kappa$ in the parameters -- but it is worth recording, because it means requirement 3's "just apply the pseudoinverse" fallback is $2$ orders *worse* than the iterative route at this conditioning, not equivalent to it.

**Things I checked to avoid the listed traps.** Every $\kappa$ comes from `np.linalg.svd` of the matrix itself. No custom adjoint is used anywhere (`A.T` explicitly), so there is no adjoint to verify. Iteration caps are tied to $r$ by a fixed multiplier in every arm of every sweep. Best-over-trajectory and the observable stopping choice are reported side by side for the headline method. `rcond` for the block drops was swept only at $10^{-13}$; the whitening arms were killed at Gate 1 so this was not pursued further.

**Not tested (honest residue).** Thick-restart / GCRO-DR style recycling with an explicitly harmonic-Ritz coarse space (the frontier's `first`-window arm is a crude proxy and shifts the curve by $\approx2\times$ in $c$; a proper recycling scheme could plausibly do better than that but would have to beat a factor $\sim4$ in $c$ to change the verdict). Storing the reorth basis in reduced precision (a constant-factor memory win at best -- `mixed` says the basis needs the matvec's accuracy, not more). fp32/bf16 end-to-end. Real Jacobian blocks from a training run.

## Conclusions

Pending Sam's review. What the data plainly shows:

1. Unpreconditioned LSQR with one-sided full reorthogonalization reaches or beats the truncated-SVD floor on all 8 cells tested, at an observable stopping rule, in $0.9$--$1.2\,r$ iterations, with no setup step and no preconditioner -- including the 2-D radon and structure-free cases that previously required SPIR. Its state is $r\cdot d$.
2. With a fixed reorthogonalization window $c$, attainable accuracy is a function of $c/r$ alone, with a knee near $c/r\approx0.8$; therefore fixed $c$ degrades as $d$ grows, and requirement 2 is not met by this family.
3. Multi-level/butterfly block whitening does not reduce $\kappa$ at all on unstructured or 2-D ridge systems, at any block size, level count, or permutation tested, up to $512d$ of state.
4. The window requirement is not a rounding artifact: double-double arithmetic changes it by $7$--$10\times$ uniformly and not in shape.
5. Requirement 7 is met, and at the operator-accuracy limit ($3$--$6\eta$). Requirement 5 in its literal per-matvec form is incompatible with any Krylov recurrence.
6. No configuration met all 7 requirements.

## Open questions

- Does a proper recycling scheme (harmonic-Ritz / GCRO-DR, state $c\cdot d$) beat the `first`-window proxy by more than the $\approx4\times$ in $c$ that would be needed to change the verdict?
- Do real trained-network Jacobian blocks sit on the `twin_cluster` side (where block structure exists) or the `twin_unstruct` side? Everything in this note bifurcates on that, and it is measurable directly.
- Requirement 5 asks for something no Krylov method can do. Is the accumulate-then-solve reading (already legitimate for a frozen $\Phi$) the intended contract, or is a genuinely stochastic solver required -- because those are different problems with different answers.
