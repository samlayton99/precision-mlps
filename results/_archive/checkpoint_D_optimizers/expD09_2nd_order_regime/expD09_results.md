# expD09 -- the second-order regime: can block coordinate descent solve the readout?

> **SUPERSEDED (rounds 1--2 only).** This documents the block-coordinate-descent
> investigation, which was a dead end. The method that actually works, and every
> conclusion that survived, is in `expD09_recipe_results.md` (mirror of
> `expD09_recipe_results.md`). Several kappa numbers in the
> text below were later found to be measured wrongly -- see the Methodology
> warnings section of the recipe. Kept for the negative results only.

**Status:** rounds 1--2 complete (baselines, full-batch exact BCD, preconditioned Krylov); conclusions unsigned pending Sam. **Contains two corrections to numbers reported to Sam mid-round -- see "Corrections" below.**

## TL;DR

- Exact block-coordinate descent on the frozen $[\Phi,\mathbf 1]$ system stalls **8--11 orders above the SVD floor** on all 9 cells, at every block size and for both blockings. It is converging, but the measured per-sweep contraction is $\rho\approx0.997$--$0.9995$, i.e. $\sim10^4$ more sweeps to the floor. BCD as a standalone solver is not the answer.
- **Round 2 reaches the floor on all 9 cells.** A sketch-preconditioned CGLS (Blendenpik/LSRN) hits $4\times10^{-16}$--$3\times10^{-15}$ eval rel $L_2$ in $\sim$50 sweeps, at or *below* the truncated-SVD floor everywhere, with $\kappa(AM^{-1})=2.5$--$4.6$ and $\|a\|=O(1)$.
- The barrier is conditioning, and the number that matters is $\kappa$, not block structure. Block-Jacobi preconditioning buys $\sim$5 orders ($7.3\times10^{14}\to1.1\times10^{9}$) and lands exactly at the predicted $\epsilon\kappa\approx10^{-7}$ stall. Only a preconditioner that knows the **row space** gets to $\kappa=O(1)$, and that costs $O(d\cdot\mathrm{rank})$ state.
- Block *selection* is not a lever: random and contiguous blocking are within a factor of 3 of each other, with random slightly ahead. Diagonal preconditioning is nearly a no-op because every $\tanh$ column has the same norm (spread $1.6\%$).
- The naive sub-solve (repo-standard rcond $10^{-15}$) drives $\|a\|$ to $\sim10^{12}$, essentially all of it in the null space. Past $\|a\|\sim10^{10}$ the residual $\Phi a-y$ can no longer be formed in fp64 without catastrophic cancellation, so the *measurement* goes noise-limited too.

## Corrections

Two numbers reported to Sam during round 1 were wrong, both from the same root cause, and both are corrected throughout this document.

1. **The $\kappa_{\rm eff}$ table was computed from an explicitly formed $\Phi^\top\Phi$.** In fp64 the small eigenvalues of that Gram are floored at $\epsilon\|\Phi\|^2\approx1.2\times10^{-10}$, against a true $\sigma_r^2\approx1.0\times10^{-24}$ -- so every "$\kappa_{\rm eff}$" was really the noise floor of the Gram, not a condition number. Recomputed as singular values of $\Phi M^{-1}$ from $\Phi$ directly: unpreconditioned $3.4\times10^{8}\to7.3\times10^{14}$; block-Jacobi contiguous $k{=}64$, $7.4\to3.9\times10^{9}$.
2. **"Contiguous blocks beat random by 4 orders" was wrong, and the sign is reversed.** Corrected: random $k{=}64$ gives $1.1\times10^{9}$ against contiguous $3.9\times10^{9}$. The conditioning is *not* local in the column index in the way the Schwarz reading predicted; block selection buys essentially nothing. The pasted spec's prediction was closer to correct than the one in this document's first draft.

A third, smaller error: a first PCG probe used a tiny-ridge Cholesky on rank-deficient halo blocks, which capped it at $10^{-4}$ and briefly suggested conditioning was not binding. With truncated per-block pseudo-inverses it reaches $10^{-8}$, and conditioning *is* binding. `plotting.render_spectrum` now computes from $\Phi$ directly and carries a comment naming the trap.

## Question

Can block coordinate descent -- exact second-order solves on $k\times k$ blocks, at $O(d+k^2)$ memory -- drive the frozen-$\Phi$ least-squares readout to machine epsilon, where first-order methods stall 8--11 orders short (expD01, expD07)? And is the ill-conditioning of $[\Phi,\mathbf 1]$ local in the column index (so contiguous blocks of neighbouring neuron centers capture it) or global (so no block scheme can)?

## Experiment design

The object is the **frozen** QI feature system: nothing about the geometry is learned, only the linear readout is solved.

$$\Phi_{ik}=\tanh\!\big(\gamma\,(x_i-c_k)\big),\qquad A=[\,\Phi\;\;\mathbf 1\,]\in\mathbb R^{n\times d},\qquad \min_a \tfrac1n\|Aa-y\|_2^2,\quad a_0=0 .$$

Repo-standard geometry throughout: uniform centers plus halo, $c_k=-1+kh$ over $k\in[-\text{halo},N+\text{halo}]$ with `default_halo`, $h=2/N$, $\gamma=\lambda^\*/h$ at $\lambda^\*=0.25$.

- **Grid (9 cells).** Targets sine (low frequency), sine_8pi (high frequency), runge (peaked) $\times$ widths $N\in\{64,128,256\}$, giving $d=206,270,462$ columns at numerical rank $80,144,273$. All three targets have deep floors, so a stall is the solver's fault -- the one exception is runge at $N{=}64$, whose own floor is $3.3\times10^{-9}$ (the known small-$N$ runge gap), and its panel is read accordingly.
- **Rows.** $n=4d$ equispaced on $[-1,1]$; eval on $N_{\rm ev}=4001$ equispaced points, misaligned with every train grid.
- **Metrics, every sweep.** train rel $L_2=\|Aa-y\|/\|y\|$ (the objective actually minimized), eval rel $L_2=\|A_{\rm ev}a-y_{\rm ev}\|/\|y_{\rm ev}\|$, eval $L_\infty$, $\|a\|$, and the null-space drift $\|P_{\rm null}a\|$ with $P_{\rm null}$ from the train SVD.
- **Floor.** Truncated-SVD min-norm solve at rcond $10^{-15}$, scored on both train and eval. Geo-mean eval floor over the 9 cells: $8.7\times10^{-14}$.
- **Cost axis.** One **sweep** $=$ one pass over $\Phi$: for BCD a full cycle through all $\lceil d/k\rceil$ blocks, for GD/CGLS one iteration. This makes the $x$ axis cost-comparable in matvec passes across methods. It is deliberately *not* flop-comparable -- an exact block solve is $O(nk^2)$, so a $k{=}64$ sweep is $386\times$ a matvec in flops. Every record carries `flops_per_sweep` and each figure title states it. Budget 400 sweeps, which exceeds the rank of every cell.

**Solvers (round 2).** `pcg_bjac_k64`: CGLS preconditioned by block-Jacobi, each block inverted by *truncated pseudo-inverse* (a ridge instead turns the rank-deficient halo blocks into a $1/\mu$ amplifier). Persistent state $d\cdot k$. `cgls_sketch`: Gaussian sketch $S$ with $s=2d$ rows, $S\Phi=U\Sigma V^\top$, preconditioner $M^{-1}=V_r\Sigma_r^{-1}$ keeping only sketch-resolved directions, then CGLS on $B=\Phi M^{-1}$. Random-projection theory gives $\kappa(B)=O(1)$ independent of $\kappa(\Phi)$; the rank truncation on the sketch is what supplies the row-space knowledge no $O(d)$-state preconditioner has. Built once against the frozen $\Phi$; persistent state $d\cdot\mathrm{rank}$.

**Solvers (round 1).** Baselines: GD at lr $=1/L$, $L=2\sigma_{\max}^2/n$; CGLS. Exact BCD: blocks $\in\{$contiguous, random$\}$ $\times$ $k\in\{32,64,128\}$, where contiguous means consecutive runs in *center order* (a 1D Schwarz domain decomposition) and random means a fresh random partition every sweep. Each block step is the min-norm solution of $\min_\delta\|A_{:,C}\delta+r\|$ by truncated SVD -- never normal equations. Two ablations: sub-solve rcond $10^{-8}$ (damped) versus the repo-standard $10^{-15}$, and carried-versus-refreshed residual (expD08 lesson 2).

**Preconditioner diagnostic.** For $P\in\{I,\ \mathrm{diag}(G),\ \text{block-Jacobi contiguous},\ \text{block-Jacobi random}\}$ built from $G=\Phi^\top\Phi$ in the **original column coordinates** (with $P$'s eigenvalues floored at $10^{-14}\max_k G_{kk}$, so unobservable columns get no boost), the reported quantity is $\kappa=\sigma_1/\sigma_r$ of $\Phi P^{-1/2}$, computed by SVD of $\Phi P^{-1/2}$ **directly**. It must not be computed as eigenvalues of $P^{-1/2}GP^{-1/2}$: forming $G$ in fp64 floors its small eigenvalues at $\epsilon\|\Phi\|^2$, which is 14 orders above $\sigma_r^2$, and every resulting number is the Gram's noise floor rather than a condition number. That is correction 1 above.

**Code & data:** `experiments/expD09_2nd_order_regime/` (`common.py`, `build_problems.py`, `solvers.py`, `run.py`, `plotting.py`); problems + manifest in `data/expD09_problems/`; runs (`runs.jsonl`) and figures in `results/checkpoint_D_optimizers/expD09_2nd_order_regime/`; verification in `tests/test_expD09.py` (7 tests, all passing).

## Results

**BCD stalls, uniformly.** Geo-mean best eval rel $L_2$ over the 9 cells, against a geo-mean floor of $8.7\times10^{-14}$:

| solver | geo-mean best eval | median $\rho$ (per sweep) | median sweeps still needed | max $\|a\|$ |
|---|---|---|---|---|
| `bcd_rand_k128` | $1.1\times10^{-5}$ | 0.9986 | $1.6\times10^{4}$ | $2.1\times10^{10}$ |
| `bcd_contig_k128` | $2.7\times10^{-4}$ | 0.9968 | $6.8\times10^{3}$ | $5.0\times10^{9}$ |
| `bcd_rand_k64` | $2.8\times10^{-4}$ | 0.9983 | $1.4\times10^{4}$ | $2.2\times10^{12}$ |
| `bcd_contig_k64_damped` | $2.8\times10^{-4}$ | 0.9961 | $6.5\times10^{3}$ | $1.1\times10^{4}$ |
| `bcd_contig_k64` | $7.1\times10^{-4}$ | 0.9972 | $8.3\times10^{3}$ | $4.2\times10^{12}$ |
| `bcd_contig_k32` | $1.4\times10^{-3}$ | 0.9968 | $7.4\times10^{3}$ | $4.3\times10^{12}$ |
| `cgls` (baseline) | $2.4\times10^{-6}$ | 0.9939 | $3.1\times10^{3}$ | $3.3\times10^{1}$ |
| `gd` (baseline) | $1.6\times10^{-1}$ | 0.9973 | $1.1\times10^{4}$ | $5.5\times10^{-1}$ |

Every BCD arm is beaten by plain CGLS, which costs $2$ matvecs a sweep against BCD's $386$. The block methods are paying roughly two orders of magnitude more flops per sweep to lose. Nothing in the roster is within 8 orders of the floor.

The stall is a rate problem, not stagnation: BCD on a convex quadratic converges linearly, and the tail ratio $\rho$ fixes the remaining cost exactly. At the measured $\rho$, every cell needs $\sim10^{4}$ more sweeps. Block size buys almost nothing across $k\in\{32,64,128\}$, and random blocking is not systematically worse than contiguous -- at $k=128$ it is better.

**The diagnostic says block structure is not the lever.** Corrected numbers at $N{=}256$, sine, $\kappa=\sigma_1/\sigma_r$ of $\Phi P^{-1/2}$:

| $P$ | storage | $\kappa$ |
|---|---|---|
| none | $0$ | $7.25\times10^{14}$ |
| diagonal | $d$ | $3.36\times10^{13}$ |
| block-Jacobi contiguous, $k=64$ | $dk$ | $3.86\times10^{9}$ |
| block-Jacobi random, $k=64$ | $dk$ | $\mathbf{1.13\times10^{9}}$ |
| sketch, $s=2d$ | $d\cdot\mathrm{rank}$ | $\mathbf{4.59}$ |

Block preconditioning is worth $\sim$5 orders and random is marginally *better* than contiguous -- block selection is not the lever the Schwarz reading predicted. Diagonal is nearly a no-op, and that mechanism is clean: every $\tanh$ column has essentially the same norm (column norms span $42.31$ to $42.99$, a $1.6\%$ spread), so diagonal scaling is close to a multiple of the identity. Only the sketch, which is the one preconditioner here that knows the row space, reaches $\kappa=O(1)$.

The $\epsilon\kappa$ rule then predicts every measured stall: $\kappa\approx10^{9}$ gives $10^{-7}$ (measured $10^{-7}$--$10^{-8}$ for `pcg_bjac_k64`), and $\kappa\approx5$ gives machine precision (measured $10^{-15}$ for `cgls_sketch`). Conditioning, not block structure and not residual precision, is what sets the floor a method can reach.

**The undamped sub-solve is numerically unsafe.** At the repo-standard rcond $10^{-15}$ each block inverts its *own* near-null directions. The increments cancel within the block, so the residual never objects, but $a$ random-walks to $\|a\|\sim10^{12}$ with more than $99.9\%$ of that in the null space. Past $\|a\|\sim10^{10}$ the residual $Aa-y$ cannot be formed in fp64 without catastrophic cancellation ($\epsilon\|A\|\|a\|\sim10^{-1}$), so the recorded curve itself goes noise-limited -- visible as the ragged $N{=}64$/$N{=}128$ sine panels. The damped arm holds $\|a\|<10^{4}$ at the same error and is the trustworthy measurement.

The residual-handling ablation is a wash on error and slightly worse on rate for the carried arm ($\rho=0.9995$), consistent with drift accumulating over 400 unrefreshed sweeps.

**Round 2 -- the floor is reached.** Best eval rel $L_2$ against the SVD floor, all 9 cells:

| cell | floor | `cgls_sketch` | sweeps | `pcg_bjac_k64` |
|---|---|---|---|---|
| sine $N{=}64$ | $2.5\times10^{-15}$ | $\mathbf{4.3\times10^{-16}}$ | 42 | $2.0\times10^{-8}$ |
| sine $N{=}128$ | $3.4\times10^{-14}$ | $\mathbf{4.2\times10^{-16}}$ | 47 | $2.0\times10^{-8}$ |
| sine $N{=}256$ | $1.2\times10^{-14}$ | $\mathbf{2.1\times10^{-15}}$ | 66 | $1.6\times10^{-7}$ |
| sine_8pi $N{=}64$ | $1.4\times10^{-13}$ | $\mathbf{1.3\times10^{-13}}$ | 40 | $1.1\times10^{-7}$ |
| sine_8pi $N{=}128$ | $4.1\times10^{-14}$ | $\mathbf{1.7\times10^{-15}}$ | 192 | $1.5\times10^{-7}$ |
| sine_8pi $N{=}256$ | $3.6\times10^{-14}$ | $\mathbf{1.9\times10^{-15}}$ | 62 | $3.6\times10^{-8}$ |
| runge $N{=}64$ | $3.3\times10^{-9}$ | $\mathbf{3.3\times10^{-9}}$ | 48 | $1.1\times10^{-6}$ |
| runge $N{=}128$ | $2.3\times10^{-14}$ | $\mathbf{7.6\times10^{-16}}$ | 46 | $1.5\times10^{-15}$ (best) |
| runge $N{=}256$ | $1.8\times10^{-14}$ | $\mathbf{1.5\times10^{-15}}$ | 78 | $3.3\times10^{-9}$ |

`cgls_sketch` is at or below the floor in every cell, typically an order *below* it (its rank-revealing truncation on the sketch differs slightly from rcond $10^{-15}$ on $\Phi$). It cliff-dives in $\sim$50 sweeps rather than crawling. `pcg_bjac_k64` sits at $10^{-7}$--$10^{-8}$, matching $\epsilon\kappa$ at the measured $\kappa\approx10^{9}$ -- the block preconditioner is doing exactly as much as its condition number allows and no more.

**Cost, stated plainly.** `cgls_sketch` needs $d\cdot\mathrm{rank}$ persistent state, which on this toy is $O(d^2)$ because the halo makes $\mathrm{rank}\approx0.6d$. That is the same memory class as expD07's `cgls_reortho`, reached by a different route, and it is **outside** the $O(d)$ budget. Nothing measured here reaches the floor inside $O(d)$ state: the frontier across expD07 and expD09 is now three independent methods (truncated SVD, reorthogonalized CGLS, sketch-preconditioned CGLS) all landing at $O(d\cdot\mathrm{rank})$, and everything at $O(d)$ stalling at $10^{-6}$ or worse. Per-*step* cost is one pass plus an $O(d\cdot\mathrm{rank})$ apply, and the sketch is built once against the frozen $\Phi$.

**Stopping is required, not optional.** Three $N{=}128$ panels diverge after $\sim$250 sweeps once the residual is at $\epsilon$ -- the post-convergence CG instability expD07 also recorded. Grading is best-over-trajectory, which handles it here; a deployed version needs the residual-norm stopping rule (halt when $\|B^\top r\|\le10^{-14}\sigma_1\|r\|$), which fired at 61--100 iterations in the standalone probe.

### Figures

- `cgls_sketch.png` -- the round-2 result in the same 3$\times$3 layout: a straight-line dive through 15 orders in $\sim$50 sweeps, flattening *below* the dotted floor. The three $N{=}128$ panels that climb back out after sweep $\sim$250 are the post-convergence instability above.
- `pcg_bjac_k64.png` -- the same layout for block-Jacobi preconditioning: a fast dive that flattens 6--7 orders above the floor, which is the $\epsilon\kappa$ prediction.
- `<solver_id>.png` (13 of them, one per method) -- the 3$\times$3 requested: rows $N\in\{64,128,256\}$, columns the three targets. Solid blue is train rel $L_2$, dashed red is eval rel $L_2$, dotted black is the SVD eval floor and dotted grey the SVD train floor. Fixed log axis $[10^{-16},10]$ in every panel so methods are comparable at a glance. Look for: the curves flattening by sweep $\sim$50 with 8--11 empty decades below them, and train sitting exactly on top of eval.
- `compare_eval.png` / `compare_train.png` -- all 11 methods overlaid on the same 3$\times$3. The one line that separates from the pack is CGLS.
- `block_spectrum.png` -- the corrected diagnostic: singular spectrum of $\Phi M^{-1}$ (computed from $\Phi$, **not** from a formed Gram) vs index, one curve per $P$, grey dash-dot at the numerical rank. Read the height at the rank line -- that is $1/\kappa$. All four curves fall off; the block-preconditioned ones sit $\sim$5 orders higher than unpreconditioned, and contiguous and random are close together rather than 4 orders apart.
- `contraction.png` -- sweeps still needed to reach the floor at the measured tail contraction rate, vs block size, contiguous against random, with $10^3$ sweeps marked. Everything sits at $10^{3.5}$--$10^{4.5}$. The single $10^{12}$ spike (sine, $N{=}64$, contiguous $k{=}64$) is the fp64-noise-limited case above, where $\rho$ is not estimable.
- `null_drift.png` -- $\|P_{\rm null}a\|$ per sweep, all methods. The undamped arms climb to $10^{12}$; the damped arm and the baselines stay bounded.

## Additional details

**Why train and eval coincide.** The two curves are indistinguishable in every panel, which was not the expectation going in. The reason is measurable: the numerical null space of the train system leaks into the eval system at the *same* $\sim10^{-15}$ relative level, so null drift of magnitude $M$ costs only $\sim M\times10^{-15}$ of eval error. At $\|P_{\rm null}a\|\sim10^{12}$ that is $\sim10^{-3}$ -- the same order as the residual stall, and therefore invisible against it rather than absent. The pasted spec's argument that column masking mandates a ridge is structurally correct here, but on this matrix the drift is not what is costing the orders.

**Practicality gate, stated against the measured frontier.** The gate asks for state in Adam's class ($2d$). Measured here: everything at $O(d)$ state stalls at $10^{-6}$ or worse; the two methods that reach the floor need $O(d\cdot\mathrm{rank})$. Block-Jacobi at $O(dk)$ sits in between and lands at $\epsilon\kappa\approx10^{-7}$. So the gate is currently in tension with the target, and the tension is quantitative rather than a matter of finding a better algorithm in the same class: reaching $\kappa=O(1)$ requires knowing the row space, and the row space of this system is $\mathrm{rank}\approx0.6d$ dimensional. The most promising way out is not a cleverer $O(d)$ preconditioner but a smaller rank -- see the halo question under Open questions.

**Verification** (`tests/test_expD09.py`, 7 passing): geometry matches `default_halo`/$\gamma=\lambda^\*/h$ exactly; the SVD floor reaches construction precision on the smooth cells; scoring the stored SVD solution through the metric plumbing reproduces the manifest floors; **BCD with one block containing every column reproduces the direct SVD solve in a single sweep** (the sub-solve calibration -- if this drifts, every number above is meaningless); exact BCD is monotone in the train objective while the iterate stays bounded; and the null-drift gain is confirmed predictive.

One diagnostic bug was found and fixed mid-round: the first version of `render_spectrum` built $P$ from the row-space Gram, which makes "diagonal" the exact inverse in the singular basis and reported that diagonal preconditioning collapses the spectrum. It does the opposite. The figure and the numbers above are from the corrected column-coordinate version.

## Conclusions

(intentionally blank until Sam has reviewed the numbers)

## Open questions

- **Can the floor be reached inside $O(d)$ state at all?** Three independent floor-reaching methods now all cost $O(d\cdot\mathrm{rank})$, and everything at $O(d)$ stalls at $10^{-6}$. Whether that is a real barrier or a gap in the roster is the central open question, and it should be attacked directly rather than by adding more $O(d\cdot\mathrm{rank})$ methods.
- **Is $\mathrm{rank}\approx0.6d$ an artifact of the halo?** The sketch preconditioner's state is $d\cdot\mathrm{rank}$; if the halo columns (which are saturated and numerically constant) were excluded or reparameterized, the rank -- and therefore the state -- could drop sharply. This is the cheapest available route to shrinking the cost and has not been tried.
- **Does the sketch survive row subsampling and label noise?** The sketch is built once against a frozen, noiseless $\Phi$. Under streaming rows it has to be rebuilt or updated, and the statistical floor $\sigma/\sqrt n$ enters.
- **Does any of this transfer when $\Phi$ is not frozen?** Everything here holds the geometry fixed. The expD08 coupling law says a moving geometry re-injects $\|v\|\eta$ of error per base step, so a floor-reaching readout solver is necessary but not sufficient.
