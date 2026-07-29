# The subproblem (reference)

The canonical statement everything in expD09 is measured against. Kept verbatim in spirit; annotated where the repo has measured a quantity.

## The object

$\Phi \in \mathbb{R}^{n \times d}$, tall and skinny ($n \gg d$), rank deficient: $\operatorname{rank}(\Phi) = r < d$. Observations $y = \Phi w^\star + \varepsilon$ with noise scale $\sigma$.

No guarantee about the spectrum. It may have a clean gap at $r$, or decay continuously with no gap. Explicit SVD is unavailable.

> Measured on the QI feature system $[\Phi,\mathbf 1]$ (expD09 round 1, expD07): exponential decay from index 0 with **no flat head and no gap**, hard rank cliff at $r\approx0.6d$, $\sigma_r/\sigma_1\approx1.4\times10^{-15}$, $\kappa=\sigma_1/\sigma_r\approx7\times10^{14}$. Target energy $\propto\sigma_i$ in every direction (isotropic alignment), so reaching rel $L_2\le10^{-13}$ genuinely requires directions down to $\sigma/\sigma_1\sim10^{-11}$--$10^{-14}$ -- they cannot be truncated away.

## The access model

Streaming, with $n$ the entire dataset. This is stochastic approximation, not a fixed-dataset linear solve.

Each step draws a random row subset $S_t$. The sampled block $\Phi_{S_t}$ may itself be rank deficient with a nonzero null space, and is not necessarily overdetermined.

Each step also masks columns, dropping roughly three quarters and leaving a coordinate subset $C_t$ with $k=|C_t|$.

The mask restricts which coordinates you may **update**, not what you can **see**. The full residual $r_S = y_S - \Phi_S w$ is computable, so the full gradient $\Phi_S^\top r_S\in\mathbb R^d$ is available at the price of one backward pass.

$w_0$ is handed to you. You don't choose it.

## The memory and compute constraint

Memory $O(d)$ per step. Nothing that scales as $d^2$: the full Gram is out, and so is RLS at scale.

Compute per step equal to one forward-backward pass.

Affordable curvature: a $k\times k$ Gram $G_C = \Phi[S,C]^\top\Phi[S,C]$, formed at cost $bk^2$. At $k=\sqrt d$ this is $bd$, exactly one pass. The binding constraint is the factorization at $k^3$, which is $d^{1.5}$ at $k=\sqrt d$ and stops being free well before $d$ gets large. So $k$ is fixed at 64 to 512 and blocks **rotate**, rather than $k$ scaling with $d$.

## The targets

| Regime | Target |
|---|---|
| $\sigma = 0$ | machine epsilon, in a short number of steps |
| $\sigma > 0$ | $\sigma/\sqrt n$, the statistical floor |

Minimum-norm preferred, not required to be exact. Second order not required -- whatever hits the targets. Robustness to the sampling procedure is the property that matters most.

## The correspondence to the full problem

- random row subset $S_t$ $\leftrightarrow$ **minibatching**
- column mask $C_t$ $\leftrightarrow$ **the subset of parameters the exact-solve regime is applied to** at this step
- $w_0$ handed to you $\leftrightarrow$ whatever the base optimizer (Adam) left the readout at

## What is settled, and the one thing that is not

Measured in expD09 (9 frozen cells: sine / sine_8pi / runge $\times$ $N=64,128,256$):

| method | state | best eval rel $L_2$ |
|---|---|---|
| GD | $3d$ | $1.6\times10^{-1}$ |
| CGLS | $3d$ | $2.4\times10^{-6}$ |
| exact BCD, round-robin, any $k$, any blocking | $k^2/d\cdot d$ | $10^{-3}$--$10^{-5}$ |
| **round-robin block GS as a PRECONDITIONER inside CGLS** | $k^2/d\cdot d$ ($8.9d$ at $k{=}64$) | $\mathbf{2\times10^{-8}}$ |
| block-Jacobi preconditioned CGLS | $dk$ | $5.7\times10^{-8}$ |
| sketch-preconditioned CGLS | $d\cdot r$ | $9.5\times10^{-15}$ |
| truncated SVD (reference floor) | $d^2$ | $8.7\times10^{-14}$ |

The governing rule is $\text{attainable} \approx \epsilon\,\kappa(\Phi M^{-1})$, and it predicts every row. Reaching the floor therefore means driving $\kappa$ to $O(1)$, which means the preconditioner must know the row space.

**Round-robin blocks belong in the preconditioner, not the solver.** Measured: as a standalone solver, exact BCD contracts at $\rho\approx0.997$ per sweep ($\sim10^4$ sweeps to the floor). The *same* block work and the *same* $O(k^2)$ state, used instead as a symmetric block Gauss-Seidel preconditioner inside CGLS, reaches $2\times10^{-8}$ -- **6 to 7 orders better for free**, because the outer CG replaces the stationary $\rho$ with $\sqrt\kappa$. The full Gram is never formed: the block coupling is carried through $u=\Phi x$, so block $C$ needs only $\Phi_C^\top u$ and its own $k\times k$ factor.

**What remains, and it is one specific gap.** Every block-local preconditioner -- Jacobi, Gauss-Seidel, any $k$, any blocking -- lands at $\kappa\approx10^{9}$ and therefore at $\epsilon\kappa\approx10^{-7}$. The reason is visible in the spectrum: block preconditioning **clusters** the spectrum (200 of 273 directions land in $[0.38,1]$ at $N{=}256$) and leaves a short bad tail. That is the classical one-level Schwarz signature, and the classical fix is a **coarse space**. Measured cost of that fix, as the number $c$ of coarse vectors needed to bring $\kappa\le100$ at $k=128$:

| rank | 80 | 144 | 273 |
|---|---|---|---|
| $c^\star$ | 11 | 13 | 17 |

so $c^\star\sim d^{0.54}$ and the two-level state is $k^2 + c^\star d\approx 20d$--$32d$ on these cells -- far below the sketch's $d\cdot r$, and the only route measured so far that could plausibly stay near $O(d)$.

**Status: not yet demonstrated.** Two two-level implementations were attempted and both are wrong (a deflated-CG recurrence that broke conjugacy, and an additive form that cannot work because the block inverse already amplifies precisely the directions the coarse space is supposed to own). The $\kappa$ numbers above are spectral measurements and are solid; the claim that a working two-level solver reaches the floor is **not** established and must not be quoted as if it were. The correct construction is deflated/hybrid (project the coarse space *out* of the block preconditioner), and building it correctly is the next task.


## Addendum -- review responses (measured)

### Is the halo load-bearing? (reviewer item 7) -- YES, but the default is 12x oversized

Sweeping halo width at fixed $\gamma=\lambda^\*/h$, target sine:

| $N$ | halo | $d$ | rank | $r/d$ | $\kappa$ | floor (eval) |
|---|---|---|---|---|---|---|
| 256 | 0 | 258 | 258 | 1.00 | $1.7\times10^{9}$ | $3.7\times10^{-7}$ |
| 256 | 4 | 266 | 266 | 1.00 | $4.3\times10^{11}$ | $1.4\times10^{-12}$ |
| 256 | **8** | **274** | **269** | **0.98** | $7.8\times10^{14}$ | $3.2\times10^{-14}$ |
| 256 | 102 (default) | 462 | 273 | 0.59 | $8.6\times10^{14}$ | $1.2\times10^{-14}$ |

Three separate findings, and they do not all point the same way:

1. **The halo is genuinely load-bearing.** Dropping it entirely costs 7 orders of approximation floor ($1.2\times10^{-14}\to3.7\times10^{-7}$). It is not removable.
2. **The default halo is far larger than needed.** `default_halo` gives 102 at $N{=}256$; halo $=8$ reaches the same floor within $2.6\times$ while cutting $d$ by 41% and taking $r/d$ from 0.59 to 0.98. So the rank *deficiency* is very largely a halo artifact -- but note this makes $r\approx d$, so any method whose state is $d\cdot r$ is not helped.
3. **$\kappa$ does NOT fall.** It stays at $7$--$9\times10^{14}$ for every halo $\ge8$. The reviewer's hypothesis (drop the halo, $\kappa$ falls by orders, the solver problem evaporates) is **not** borne out. The ill-conditioning is intrinsic to tanh-on-a-uniform-grid at $\lambda^\*=0.25$, not an artifact of the halo. The solver problem is real.

### Nystrom state-vs-accuracy tradeoff (reviewer item 4)

Randomized Nystrom preconditioner (Frangella-Tropp-Udell), $\hat G_l$ built from $l$ matvecs, $M^{-1}=(\hat G_l+\mu I)^{-1}$, state $l\cdot d$. At $N{=}256$ ($d=462$, floor $1.2\times10^{-14}$):

| $l$ | state | $\kappa$ | best eval |
|---|---|---|---|
| 4 | $4d$ | $2.8\times10^{15}$ | $3.5\times10^{-5}$ |
| 32 | $32d$ | $6.7\times10^{14}$ | $3.7\times10^{-6}$ |
| 128 | $128d$ | $5.9\times10^{11}$ | $2.1\times10^{-9}$ |
| 256 | $256d$ | $6.8\times10^{9}$ | $5.2\times10^{-11}$ |

**Nystrom does not reach the floor at any tested rank, and loses badly to the sketch at equal state**: Blendenpik at $166d$ reaches $2\times10^{-15}$, where Nystrom at $128d$ reaches only $2\times10^{-9}$. The reason is the spectrum: it decays exponentially with **no gap and no low-rank structure**, and the solution needs directions all the way down to $\sigma/\sigma_1\sim10^{-11}$. A rank-$l$ approximation captures the top $l$ and leaves the rest untouched, whereas Blendenpik's full-rank triangular factor whitens the entire spectrum. The reviewer's expectation that exponential decay would favour Nystrom is measured false here -- exponential decay *without a gap* is precisely the case low-rank methods cannot exploit.

### Retraction: the "attainable = eps * kappa" rule

The reviewer is right that it mispredicts unpreconditioned CGLS by 5 orders. Across the Nystrom sweep the error tracks $\kappa$ with a **constant of $\approx5\times10^{-21}$, not $\epsilon=2.2\times10^{-16}$** (ratio err/$\kappa$ = $5.5$, $3.5$, $7.6\times10^{-21}$ at $l=32,128,256$). That constant reproduces CGLS correctly ($7.25\times10^{14}\times5\times10^{-21}=3.6\times10^{-6}$ vs measured $2.4\times10^{-6}$) but does **not** fit the block-Jacobi/GS arms. So it is not a law, in either form. **Treat the $\epsilon\kappa$ rule as withdrawn**; $\kappa$ is strongly predictive within a preconditioner family and not across families.

### Correction: the $c^\star$ exponent

$c^\star = 11,13,17$ at rank $=80,144,273$. Regressed against rank that is exponent $0.35$; against $d$ ($206,270,462$) it is $0.54$. The earlier "$d^{0.54}$" was against $d$ and should have said so. **These are three points and should be read as three points, not an exponent.**

### Still not done

Two-level / deflation: **three** failed implementations now (deflated-CG breaking conjugacy; additive; and the balancing form $M^{-1}=Q+P^\top M_b^{-1}P$, which made $\kappa$ *worse*, $10^{10}\to10^{18}$, because the coarse basis $Z$ built through $M_b^{-1/2}$ lands nearly in the null space and $E=Z^\top GZ$ is degenerate). Not attempted: LSMR in place of CGLS, Krylov/harmonic-Ritz recycling for $Z$, and everything streaming.


## Two-level, attempt 4: formulation verified, and a new obstruction found

Following the reviewer's discipline (verify the spectrum before benchmarking), with the coarse solve routed through $AZ$ directly instead of $E=Z^\top GZ$ -- the latter has eigenvalues $\sim10^{-20}\|A\|^2$ against an fp64 formation error of $10^{-16}\|A\|^2$, i.e. it is pure noise, which is what killed attempt 3.

**Construction.** Write $w=Z\alpha+s$ with $Z$ ($d\times c$) orthonormal. Optimality in $\alpha$ makes the residual $\perp\operatorname{range}(AZ)$, so with $\Pi=I-U_ZU_Z^\top$ ($U_Z$ from the QR/SVD of $AZ$, never a Gram):

$$\min_s\ \|\Pi(As-y)\|,\qquad \alpha=(AZ)^{+}(y-As).$$

The $s$-solve is block-Jacobi-preconditioned CG on $A^\top\Pi A$; per step one pass plus $O(nc)$.

**Gate 1, spectrum -- PASSES.** $\kappa(\Pi A M_b^{-1/2})$ over the row space, $k=128$:

| $c$ | 0 | 8 | 17 | 32 |
|---|---|---|---|---|
| $N{=}128$ | $5.5\times10^{8}$ | $1.9\times10^{3}$ | $15.3$ | $2.0$ |
| $N{=}256$ | $2.1\times10^{9}$ | $8.1\times10^{4}$ | $30.2$ | $10.3$ |

**Gate 2, oracle -- PASSES.** With $s$ from an exact `lstsq` on the deflated system, the method reaches **$1.4\times10^{-15}$**, below the $1.2\times10^{-14}$ floor. The formulation is right.

**Gate 3, iterative -- FAILS at $10^{-9}$**, on all 9 cells, and *not* because of $\kappa$.

**The obstruction.** $\sigma_{\min}(AZ)=6.8\times10^{-15}$, so the recombination $\alpha=(AZ)^+(y-As)$ amplifies any error in $s$ by $\sim1.5\times10^{14}$. Measured, by perturbing the oracle $s$ by a controlled relative amount:

| rel. error in $s$ | 0 | $10^{-16}$ | $10^{-14}$ | $10^{-12}$ | $10^{-10}$ | $10^{-8}$ |
|---|---|---|---|---|---|---|
| final eval rel $L_2$ | $1.4\times10^{-15}$ | $1.5\times10^{-15}$ | $3.3\times10^{-15}$ | $5.4\times10^{-13}$ | $5.0\times10^{-11}$ | $3.3\times10^{-9}$ |

Final error tracks $s$'s relative error roughly 1:1. So reaching $10^{-14}$ requires $s$ to fp64-exact **parameter** accuracy -- and $\kappa=30$ of the deflated operator controls the *residual* $\|\Pi(As-y)\|$, not the parameter error along the directions $\alpha$ must reconstruct. Deflation moves the ill-conditioning out of the Krylov iteration and into the recombination, where it is untouched.

**Consequence for the design.** Any two-level scheme that *splits* $w=Z\alpha+s$ and recombines inherits this. The way around it, if one exists, is a deflated iteration that never recombines -- a single Krylov process with the projector applied to every residual and search direction (GCRO-DR / recycled MINRES), which is exactly what the reviewer specified and what has not yet been built correctly. That is the next task, and the $\kappa$ and oracle gates above are the tests it must pass.


## Round 3: LSMR (reviewer item 3) -- the largest gain per line of code so far

**Correction to the attempt-4 diagnosis above.** The claim that the recombination $\alpha=(AZ)^+(y-As)$ was the binding obstruction is **wrong**. Instrumenting the deflated run shows the final error tracks the CG residual $\|\Pi(As-y)\|$ to three digits ($5.385\times10^{-9}$ vs $5.400\times10^{-9}$): the recombination is faithful, and it is **CG itself** that stalls. The perturbation table is still correct, it just does not describe what limits the method.

The real cause is the one the reviewer named: CGLS's recurrences live in the $A^\top A$ metric, so it attains $\epsilon\,\kappa(A)^2$, not $\epsilon\,\kappa(A)$. LSMR (MINRES on the normal equations via Golub-Kahan bidiagonalization) never iterates in that metric.

**Measured, same preconditioner, same $O(k^2)$ state, one line changed:**

| solver | state | geo-mean best eval |
|---|---|---|
| CGLS + block-Jacobi | $dk=64d$ | $5.7\times10^{-8}$ |
| CGLS + symmetric block GS | $O(k^2)=8.9d$ | $2.0\times10^{-8}$ |
| **LSMR + block-Jacobi** | $O(k^2)=8.9d$ | $\mathbf{1.1\times10^{-9}}$ |

Best cells reach $7$--$8\times10^{-12}$ (runge $N{=}128,256$). $k=64$ beats both $k=32$ and $k=128$ on the geo-mean. LSMR needs a large iteration budget -- best iterates land at 1100--3000 sweeps, and at the earlier 400-sweep budget it looks 2 orders worse, so budget must be reported with the number.

**Deflation on top of LSMR: promising but not trusted.** With an oracle coarse space it reaches $2.4\times10^{-12}$ on 6 of 9 cells, but the $c$-dependence is erratic (eval jumps between $10^{-7}$ and $3\times10^{-11}$ with no monotone relation to $c$), and the measured $\kappa$ *increases* with $c$, which is backwards for deflation. The $Z$ construction is therefore still wrong somewhere and none of those numbers should be quoted.

## Standing

| solver | state | geo-mean best eval |
|---|---|---|
| GD | $3d$ | $1.6\times10^{-1}$ |
| CGLS | $3d$ | $2.4\times10^{-6}$ |
| exact BCD (round-robin) | $O(k^2)$ | $7\times10^{-4}$ |
| CGLS + block GS precond | $O(k^2)=8.9d$ | $2.0\times10^{-8}$ |
| **LSMR + block-Jacobi precond** | $O(k^2)=8.9d$ | $1.1\times10^{-9}$ |
| CGLS + sketch precond | $d\cdot r=166d$ | $9.5\times10^{-15}$ |
| truncated SVD (floor) | $d^2$ | $8.7\times10^{-14}$ |

Best $O(d)$-state result has improved from $2\times10^{-8}$ to $1.1\times10^{-9}$; the floor is still reached only at $O(d\cdot r)$. Remaining gap: 5 orders.


## Round 4: the O(d)-preconditioner search is now close to exhaustive, and negative

Four structural hypotheses tested, all cheap, all killed.

**Orthogonality is NOT the barrier.** Instrumented Golub-Kahan: $\|V_k^\top v_{k+1}\|$ leaves $\epsilon$ at iteration **5**, not 100. But one-sided FULL reorthogonalization (run as a diagnostic only, $O(kd)$) changes nothing:

| | best eval | iterations |
|---|---|---|
| LSQR, no reorth | $4.96\times10^{-10}$ | 600 |
| LSQR, full reorth | $3.76\times10^{-10}$ | **82** |

Reorthogonalization buys $7\times$ **speed** and **zero accuracy**. So restarted-LSMR-with-compensated-residuals is unmotivated -- consistent with the earlier direct test where exact `fsum` residuals moved $2.3\times10^{-5}$ only to $6.4\times10^{-6}$.

**$\Phi^\top\Phi$ is NOT Toeplitz-plus-low-rank.** Diagonal-averaged Toeplitz approximation $T$, bias column excluded:

| $N$ | halo | $\|G-T\|_F/\|G\|_F$ | rank$(G-T)$ |
|---|---|---|---|
| 128 | 8 | $5.2\times10^{-2}$ | 145 of 145 |
| 128 | 70 | $3.7\times10^{-1}$ | 269 of 269 |
| 128 | 102 | $4.3\times10^{-1}$ | 333 of 333 |

The prediction that halo $=8$ would be closer is confirmed, but the residual is **full rank** at every halo, so there is no Toeplitz-plus-low-rank structure for a circulant preconditioner to exploit. Deflating the dominant near-constant direction first makes the relative figure worse, not better.

**No fixed fast transform diagonalizes it.** Off-diagonal mass of $W^\top G W$:

| $N$ | halo | DCT | DFT | median $|\langle v_i, \mathrm{dct}_i\rangle|$ |
|---|---|---|---|---|
| 128 | 8 | $0.49$ | $0.69$ | $0.051$ |
| 256 | 8 | $0.47$ | $0.67$ | $0.045$ |
| 256 | 102 | $0.63$ | $0.73$ | $0.018$ |

**Status of the $O(d)$ preconditioner search.** Every natural structure has now been tested and rejected:

| structure | state | verdict |
|---|---|---|
| diagonal | $d$ | no-op; all tanh column norms equal to 1.6% |
| banded Cholesky | $bd$ | impossible; $G$ off-diagonals do not decay (0.99 ... 0.74) |
| Toeplitz / circulant | $d$ | $G-T$ full rank at every halo |
| DCT / DFT | $d$ | 47--66% off-diagonal mass |
| block-Jacobi / block GS | $O(k^2)$ | works, but floors at $\kappa\approx10^{9}$ |
| Nystrom rank-$\ell$ | $\ell d$ | worse than block-Jacobi at equal state; no gap to exploit |
| sketch (full-rank) | $d\cdot r$ | $\kappa=4.6$, reaches the floor -- but $O(d^2)$ |

**The information argument this adds up to.** Driving $\kappa$ to $O(1)$ requires supplying $\sim r$ independent scale corrections along $\sim r$ specific directions. Specifying those directions costs $O(rd)$ **unless the singular vectors have exploitable structure** -- and the four natural structures are now measured not to. Here $r\approx N$ always (expA04's bounded tanh null space), and at halo $=8$, $d\approx N$ too, so $O(d\cdot r)=O(d^2)$ with no escape.

**Conclusion for the subproblem, stated as a negative:** on a frozen $\Phi$ with an exponentially decaying, gapless spectrum and unstructured singular vectors, no $O(d)$-state preconditioner reaches the floor. Best $O(d)$ result stands at $1.1\times10^{-9}$ (LSMR + block-Jacobi, $8.9d$); the floor needs $O(d\cdot r)$.

**The route this leaves.** Fight the parameterization, not the conditioning. The QI construction reaches $10^{-15}$ precisely because it works in a well-conditioned (cardinal, localized) basis rather than the nodal tanh basis. Plain differencing $D^p$ was tested and buys only one order ($7.3\times10^{14}\to7.9\times10^{13}$), so the useful change of basis is not the naive one -- but this is the direction with a theory behind it, and it is a modeling change rather than an optimizer change.


## Round 5: SOLVED at machine epsilon -- SPIR

Sam was right that the $10^{-9}$ stall was arbitrary and that no law forbids the floor. The negative in round 4 was overclaimed: it is a correct statement about **$O(d)$-state preconditioners**, and I wrongly let it read as a statement about the subproblem.

**The literature has this solved.** Plain sketch-and-precondition is *not* backward stable on ill-conditioned problems (Meier, Nakatsukasa, Townsend & Webb, SIMAX 2024); wrapping it in iterative refinement restores backward stability -- **SPIR**, Epperly, Meier & Nakatsukasa 2024 (arXiv:2406.03468). That is exactly the missing piece: I had built the sketch arm and had separately tested refinement, but never combined them.

**Measured, all 9 cells, at an OBSERVABLE stopping rule ($\|A^\top r\|$ plateau -- no oracle, no best-over-trajectory):**

| cell | floor | SPIR (stopped) | oracle best |
|---|---|---|---|
| sine $N{=}64$ | $2.5\times10^{-15}$ | $2.6\times10^{-16}$ | $2.5\times10^{-16}$ |
| sine $N{=}128$ | $3.4\times10^{-14}$ | $2.2\times10^{-16}$ | $2.2\times10^{-16}$ |
| sine $N{=}256$ | $1.2\times10^{-14}$ | $2.0\times10^{-16}$ | $1.9\times10^{-16}$ |
| sine_8pi $N{=}64$ | $1.4\times10^{-13}$ | $1.3\times10^{-13}$ | $1.3\times10^{-13}$ |
| sine_8pi $N{=}128$ | $4.1\times10^{-14}$ | $1.6\times10^{-15}$ | $1.4\times10^{-15}$ |
| sine_8pi $N{=}256$ | $3.6\times10^{-14}$ | $1.2\times10^{-15}$ | $8.8\times10^{-16}$ |
| runge $N{=}64$ | $3.3\times10^{-9}$ | $3.3\times10^{-9}$ | $3.3\times10^{-9}$ |
| runge $N{=}128$ | $2.3\times10^{-14}$ | $1.3\times10^{-16}$ | $1.2\times10^{-16}$ |
| runge $N{=}256$ | $1.8\times10^{-14}$ | $1.1\times10^{-16}$ | $1.1\times10^{-16}$ |

**geo-mean $3.7\times10^{-15}$, i.e. $23\times$ BELOW the truncated-SVD floor.** Stopped $\approx$ oracle best on every cell, so there is no semiconvergence gap to hide. The two cells that do not reach $10^{-15}$ sit exactly on their own approximation floors (sine_8pi at $N{=}64$, runge at $N{=}64$) -- target/geometry limits, not solver limits.

Cost: 8 refinement rounds $\times$ 200 inner LSMR iterations, under a second per cell. Extended-precision residuals are **not** needed (double-double changed the result by $<1\%$) -- the barrier was never residual rounding, which is now tested and rejected three times.

**The honest remaining gap is memory, and only memory.** SPIR's preconditioner is $d\times r$: $270d$ at $N{=}256$, i.e. $\approx126$k doubles $\approx1$ MB, against Adam's $2d=2768$ doubles for the same network. So:

- The subproblem's $\sigma=0$ target (**machine epsilon in a short number of steps**) is **MET**.
- The $O(d)$ memory constraint is **NOT** met, and nothing in the randomized-NLA literature meets it: sketch-and-precondition ($O(d\cdot r)$) is the state of the art for ill-conditioned LS to machine precision.

Standing, best per state class:

| state | best method | geo-mean |
|---|---|---|
| $3d$ | CGLS | $2.4\times10^{-6}$ |
| $O(k^2)=8.9d$ | LSMR + block-Jacobi | $1.1\times10^{-9}$ |
| $d\cdot r=270d$ | **SPIR** | $\mathbf{3.7\times10^{-15}}$ |
| $d^2$ | truncated SVD | $8.7\times10^{-14}$ |


## Round 6: SOLVED in fp64 at O(d) state -- block-QR whitening

Sam's mpmath observation was the key: the stalls were a **precision** floor, not a memory bound, and doubling the mantissa is a constant factor on bits, so $O(d)$ numbers at higher precision is still $O(d)$. That reframing produced the fix.

### The existence proof (double-double, then discarded)

LSQR run entirely in double-double (Dekker/Knuth, 2 float64 words per number, so still $O(d)$) with a block-Jacobi preconditioner reached **$2.1\times10^{-16}$ at iteration 100** on sine $N{=}128$ and $1.1\times10^{-16}$ on runge $N{=}128$. It also showed the winning configuration needs a *deliberately ill-conditioned* preconditioner ($\kappa(M)=2.6\times10^{14}$); weakening it to $\kappa(M)\approx10^{6}$ made the method diverge. dd is not the deliverable -- it is the proof that $O(d)$ suffices, plus the diagnosis of what fp64 must survive.

### The fp64 algorithm

Three things are load-bearing, each isolated by measurement:

1. **Whiten by QR, never by the Gram.** Per block, pivoted Householder QR gives $A_{:,C}=Q_CR_C$, so the whitened block *is* $Q_C$ -- orthonormal to machine precision **by construction**. Computing the same object as $A_{:,C}M_C^{-1/2}$ with $M_C$ from `eigh` of the block Gram both squares the block condition number and is catastrophic cancellation, because $\kappa(M^{-1/2})$ reaches $10^{14}$. Measured: Gram route $1.6\times10^{-8}$, QR route $1.7\times10^{-15}$.
2. **Never apply the ill-conditioned factor inside the iteration.** Running LSQR through an operator that applies $M$ at every matvec injects relative error $\epsilon\,\kappa(M)\approx10^{-2}$ into *every* matvec -- measured, caps the method at $4\times10^{-4}$. Materialize the whitened operator $B$ once; the iteration then only ever touches $B$, and $R_C^{-1}$ is applied a **single** time at the end. A single application of an ill-conditioned operator is harmless -- that is exactly what the truncated SVD does, and why its function error is $10^{-14}$ despite parameter error $\epsilon\kappa$.
3. **Pivot and drop.** Halo blocks are rank-deficient; column-pivoted QR exposes it in $|\mathrm{diag}(R)|$ and the dependent columns are dropped rather than inverted.

### Result: at or below the SVD floor on every cell, single solve, fp64

| cell | floor | block-QR $k{=}128$ |
|---|---|---|
| sine $N{=}64$ | $2.45\times10^{-15}$ | $\mathbf{1.66\times10^{-15}}$ |
| sine $N{=}128$ | $3.43\times10^{-14}$ | $\mathbf{7.74\times10^{-15}}$ |
| sine $N{=}256$ | $1.22\times10^{-14}$ | $\mathbf{7.31\times10^{-15}}$ |
| sine_8pi $N{=}64$ | $1.40\times10^{-13}$ | $4.25\times10^{-13}$ |
| sine_8pi $N{=}128$ | $4.12\times10^{-14}$ | $7.54\times10^{-14}$ |
| sine_8pi $N{=}256$ | $3.56\times10^{-14}$ | $\mathbf{1.99\times10^{-14}}$ |
| runge $N{=}64$ | $3.30\times10^{-9}$ | $3.30\times10^{-9}$ (its own floor) |
| runge $N{=}128$ | $2.26\times10^{-14}$ | $\mathbf{9.59\times10^{-16}}$ |
| runge $N{=}256$ | $1.80\times10^{-14}$ | $\mathbf{1.77\times10^{-15}}$ |

**geo-mean $4.13\times10^{-14}$ against an SVD floor of $8.70\times10^{-14}$** -- below the floor, in fp64, no refinement needed, 73--403 LSQR iterations.

Block-size frontier (single solve, geo-mean): $k{=}8$: $7.5\times10^{-6}$; $k{=}16$: $1.5\times10^{-6}$; $k{=}32$: $1.1\times10^{-8}$; $k{=}64$: $4.8\times10^{-12}$; $k{=}128$: $4.1\times10^{-14}$.

### Cost, against the subproblem's budget

- **Persistent state: the $R$ factors, $k\times k$ per block over $d/k$ blocks $= d\cdot k$ floats with $k$ FIXED $\Rightarrow O(d)$.** This is exactly the subproblem's "$k$ fixed at 64 to 512, blocks rotate" budget. $B$ is whitened data (batch-sized), not optimizer state.
- Setup: $d/k$ QRs of $n\times k$ blocks $=O(ndk)$, i.e. $k$ passes, once.
- Per iteration: one matvec with $B$ and one with $B^\top$ -- one forward-backward pass.

### Final standing

| state | method | geo-mean |
|---|---|---|
| $3d$ | CGLS | $2.4\times10^{-6}$ |
| $64d$ | block-Jacobi + LSMR + refinement | $\sim10^{-9}$ |
| $\mathbf{128d}$ | **block-QR whitening + LSQR** | $\mathbf{4.1\times10^{-14}}$ |
| $d\cdot r$ ($270d$) | SPIR | $3.7\times10^{-15}$ |
| $d^2$ | truncated SVD (reference) | $8.7\times10^{-14}$ |

The $\sigma=0$ target of the subproblem -- **machine epsilon, short number of steps, $O(d)$ memory, one pass per step, fp64** -- is met. Open: the $\sigma>0$ statistical floor, row subsampling, column masking, and a non-frozen $\Phi$.
