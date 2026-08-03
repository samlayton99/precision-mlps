# Sam's spec: machine-precision rank-deficient least squares under O(m+n) memory

**Status: SUPERSEDED, kept for the record. Sam wrote this spec in July 2026. Its central question -- can an O(m+n)-memory iterative method reach the unit-roundoff floor on these systems -- was then tested and answered NEGATIVELY in expD11 (`results/checkpoint_D_optimizers/expD11_batching/expD11_results.md`): on gapless spectra, reaching the fp64 floor requires stored orthogonality of size c ~ r, i.e. Theta(d^2) state. The consequence is the tiered design in expD10. Read this for the framing, not the prescription.**

---


## Problem

Given $A \in \mathbb{R}^{m\times n}$ (dense, numerically rank-deficient, $m$ and $n$ in any relation) and $b \in \mathbb{R}^m$, compute

$$x_\lambda = \arg\min_{x\in\mathbb{R}^n} \; \|b - Ax\|_2^2 + \lambda^2\|x\|_2^2$$

to relative forward error $\approx u$ (unit roundoff of the working precision).

**Constraints:** $A$ is the only matrix that may be stored. All auxiliary storage is $O(m+n)$. No $A^\top A$, no factorization, no sketch.

## Method

**Iterative refinement on the damped augmented system, with residuals evaluated in extended precision, and damped LSMR as the inner solver.**

Three components, each load-bearing:

| Component | Purpose |
|---|---|
| Tikhonov damping $\lambda$ | Makes the solution unique despite rank deficiency; caps the condition number at $\kappa_\lambda = \sigma_1/\lambda$, which sets the inner solver's convergence rate |
| Extended-precision residual | The only source of forward accuracy beyond $\kappa_\lambda u$; everything else runs in working precision |
| Damped LSMR | Matrix-free inner solve in $O(m+n)$; never forms $A^\top A$ or $[A;\lambda I]$ |

## Notation

- $u$ — unit roundoff, working precision ($\approx 1.1\times10^{-16}$ for float64)
- $u_r$ — unit roundoff of the residual evaluation. Set $u_r = u^2$ (double-double)
- $\kappa_\lambda := \sigma_1(A)/\lambda$
- $\bar A := \begin{bmatrix} A \\ \lambda I\end{bmatrix} \in \mathbb{R}^{(m+n)\times n}$, never materialized

The augmented system being refined:

$$\begin{bmatrix} I & A \\ A^\top & -\lambda^2 I\end{bmatrix}\begin{bmatrix} r \\ x\end{bmatrix} = \begin{bmatrix} b \\ 0\end{bmatrix}$$

Its solution is exactly $(b - Ax_\lambda,\; x_\lambda)$. It is a linear system, not a least-squares problem, which is why standard iterative-refinement theory applies without the Golub–Wilkinson small-residual restriction.

## Algorithm

```
INPUT:  A, b, λ > 0, outer_steps ≈ 3, inner_tol ρ
OUTPUT: x with ‖x − x_λ‖ / ‖x_λ‖ ≈ u

x ← 0                                    # R^n
r ← b                                    # R^m

for k = 1 .. outer_steps:

    # ---- STEP 1: augmented residuals, EXTENDED PRECISION (u_r = u²) ----
    f ← b − r − A·x                      # R^m
    g ← A^T·r − λ²·x                     # R^n
    round f, g back to working precision

    # ---- STEP 2: correction, WORKING PRECISION ----
    # solve  min_δ ‖ Ā·δ − [f ; −g/λ] ‖₂   by damped LSMR
    δx ← LSMR(A, rhs_top=f, rhs_bot=−g/λ, damp=λ, tol=ρ)
    δr ← f − A·δx

    # ---- STEP 3: update ----
    x ← x + δx
    r ← r + δr

    # ---- RECEIPT ----
    η ← ‖A^T·r − λ²·x‖ / (‖A‖_F · ‖r‖)
    if η ≲ u and ‖δx‖/‖x‖ ≲ u: break
```

**Inner solver matvec rules** (Golub–Kahan bidiagonalization on $\bar A$, no stacked matrix):

$$\bar A v = \begin{bmatrix} Av \\ \lambda v\end{bmatrix}, \qquad \bar A^\top w = A^\top w_{1:m} + \lambda\, w_{m+1:m+n}$$

## Guarantees

**Limiting forward error:**

$$\frac{\|x_\lambda - \hat x\|}{\|x_\lambda\|} \;\approx\; \kappa_\lambda u_r + u$$

With $u_r = u^2$, the first term is negligible for all $\kappa_\lambda < 1/u \approx 10^{16}$, so the error converges to $u$.

**Contraction:** each outer step multiplies the forward error by approximately the inner solver's relative accuracy $\rho$. From $x_0 = 0$, reaching $u$ needs $\lceil \log(1/u)/\log(1/\rho)\rceil$ outer steps. With $\rho = 10^{-6}$, that is 3.

**Reference:** Björck (BIT 1967) for the augmented-system formulation; Carson & Higham (SISC 2018) for the $\text{cond}\cdot u_r + u$ limiting-accuracy result; Carson & Daužickaitė (SIMAX 46(2), 2025) for a comparison of the two-precision variants.

## Cost

**Memory beyond $A$:** ~6 vectors of length $m$, ~6 of length $n$. Double-double doubles the footprint of $f$, $g$, and the two accumulators. Total $O(m+n)$.

**Compute per outer step:** 2 matvecs in extended precision (≈4–8× the flops of a working-precision matvec) plus $\sim\!\sqrt{\kappa_\lambda}\log(1/\rho)$ LSMR iterations at 2 matvecs each.

**Choosing $\lambda$:**

| $\lambda/\sigma_1$ | $\kappa_\lambda$ | LSMR iters / correction | Bias $\|x_\lambda - x^\dagger\|$ |
|---|---|---|---|
| $10^{-2}$ | $10^2$ | ~50 | large |
| $10^{-4}$ | $10^4$ | ~500 | moderate |
| $10^{-6}$ | $10^6$ | ~5,000 | small |
| $10^{-8}$ | $10^8$ | ~50,000 | very small |

Forward error reaches $u$ in every row. $\lambda$ trades **compute against regularization bias**, not against numerical accuracy. Obtain $\sigma_1$ by a few matrix-free power iterations.

## Implementation pitfalls

1. **Compiler FMA contraction silently destroys double-double arithmetic.** Compile the residual path with `-ffp-contract=off`, or use explicit 2Sum/2Product. Symptom: $\|\delta x\|/\|x\|$ stalls several orders above $u$.
2. **Stock LSMR (e.g. `scipy.sparse.linalg.lsmr`) accepts `damp` but assumes the right-hand side is $[c;0]$.** This algorithm requires a nonzero second block. It is roughly a 20-line change inside the bidiagonalization loop.
3. **Initialize $x_0 = 0$.** LSMR's Krylov space then lies inside $\mathcal{R}(A^\top)$, so it cannot acquire null-space components; this is what makes the undamped limit the minimum-norm solution.
4. **Only Step 1 uses extended precision.** Running the inner solve in high precision is wasted work and destroys the cost model.
5. **The scalar $\eta$ is the convergence certificate,** not the residual norm. Its floor is $\approx u$; a floor above that means Step 1 is not actually extended-precision.

## What the target is

The output is $x_\lambda$ — the exact minimizer of the *regularized* objective — to full working precision. The gap $\|x_\lambda - x^\dagger\|$ to the unregularized pseudoinverse solution is regularization bias, is fixed entirely by $\lambda$, and is typically $10^{-3}$–$10^{-6}$. Refinement drives rounding error to $u$ and does not touch bias.

---

## Transfer notes for a stochastic optimizer

**The transferable mechanism:** the error in an iterate is recoverable from a residual/gradient quantity, but subtraction of nearly equal terms destroys it. Evaluating that one quantity at higher precision than everything else recovers it, at negligible memory cost. This is the same principle behind fp32 master weights under bf16 compute and behind Kahan-compensated summation in low-precision Adam states. The general pattern: **precision asymmetry — cheap everywhere, expensive only at the cancellation site.**

**$\lambda$ maps onto damping / weight decay** and plays the identical structural role: it caps the effective condition number, which is what actually controls the iteration count.

**What does not transfer.** The $\kappa_\lambda u_r + u$ result assumes deterministic, exact operator applications. Under minibatch sampling, the accuracy floor is set by gradient variance, which is many orders of magnitude above $u$; refinement cannot go below sampling noise, and extended precision buys nothing there. The technique is relevant to a stochastic optimizer only where a deterministic subproblem is being solved inside the loop — a preconditioner fit, a linear solve against a fixed curvature estimate, a projection step — not to the outer stochastic iteration itself.