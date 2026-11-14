# Resolving the theoretical-versus-practical solver mismatch

Status: internal research result and proposed proof amendment, 2026-11-09. No manuscript or external response has been changed or submitted.

## What the evidence supports

The ordinary unscaled SVD recipe is sufficient for the high-accuracy networks in these tests. Reference scaling provides no consistent accuracy advantage when the SVD driver and relative cutoff are matched. A solver-independent posterior argument supplies continuous error bounds for the actual returned networks. This closes the practical theory/experiment connection without requiring the implementation to use the conservative cutoff from the original recovery proof.

The original wording still needs correction: the implementations are not identical. The existing scaled recovery theorem remains an a priori sufficient construction; the simpler solver is supported by the comparison and by bounds on its actual output. These are distinct mathematical statements.

## Why the first comparison looked bad

The first experiment changed both the column scaling and the singular-value cutoff. Its theoretical arm used the explicit worst-case approximation bound (A.51), plus a conservative realization of (A.63), to enforce the sufficient condition (A.65). At $N=128$, that cutoff was about $4.075$ and removed the entire matrix. At $N=1024$, its matrix-error allowance alone was about $1.10\times10^{-9}$, overwhelming the $1.27\times10^{-12}$ analytic cutoff component. It deliberately discarded directions that the practical solver could use accurately.

That result is real for the implemented sufficient recipe, but it does not establish that reference scaling is necessary, that ordinary SVD is unstable, or that a large cutoff is necessary. The original diagnostic and its data are retained. The follow-up isolates scaling and then evaluates the practical solver independently of the recovery construction.

## Controlled comparison

The primary comparison retains the original six functions, $N=128,256,512,1024,2048$, $\lambda=0.25$, $R=\lceil\sqrt N\rceil$, and $M=8N+1$. Both arms use single-target LAPACK GELSD, the same samples, and relative cutoff $2^{-52}$. One fits $A$; the other fits $AD/\sqrt M$ with labels $y/\sqrt M$ and exports through $D$. The comparison changes no hyperparameters based on evaluation error. Evaluation uses 16,384 midpoint inputs, with a second independent 32,768-point midpoint grid.

The declared robustness grid adds $\lambda\in\{0.2,0.3,0.4\}$ at $N\in\{256,512,1024\}$, doubled sample density, and relative cutoffs $2^{-50}$ and $2^{-48}$. It comprises 23 parameter settings and 138 paired function cases. Non-dyadic-bandwidth robustness runs use NumPy tanh identically in both arms; the primary comparison uses correctly rounded FP64 features. These are comparisons, not per-function parameter searches.

- Unscaled SVD has no larger measured error in 100 of 138 comparisons.
- Its worst error ratio relative to scaled SVD is 1.63; scaled SVD's worst ratio relative to unscaled SVD is 12.45. Ratios near the floating-point floor should not be overinterpreted.
- For the primary comparison at $N\ge256$, the largest unscaled error is $5.92\times10^{-13}$; the largest scaled error is $1.19\times10^{-12}$.
- Both arms retain the same approximation/aliasing behavior when bandwidth is varied. For example, the larger errors at $\lambda=0.4$ occur in both arms.

The new figure's dashed lines mean **reference-scaled SVD with the same practical cutoff**, not the original (A.65) cutoff. It must not be relabeled as an unchanged implementation of that sufficient recipe.

## Minimal proof amendment

Let $q_c(x)=b+\sum_jw_j\tanh(\gamma(x-x_j))$ be any returned network, regardless of how its readout $c=(w,b)$ was calculated. Let $T\ge\|c\|_1$ and $R_X\ge\|f-q_c\|_{2,X}$ be valid upper bounds.

**Solver-independent transfer corollary.** Under the analytic, sampling, input-rounding, and evaluation assumptions in Appendix A.3.5, its continuous error-transfer statements apply to $q_c$ with these values of $T$ and $R_X$.

**Proof.** On each prescribed complex ellipse, $|q_c(z)|\le C_{\rm ell}T$, hence $|f(z)-q_c(z)|\le B+C_{\rm ell}T$. Starting with this bound, equations (A.69) onward use only the coefficient and residual bounds and the stated sampling/evaluation assumptions. No step uses the construction of $D$, the SVD coordinates, or (A.65). The reference-scaled recovery result is one sufficient way of producing $T,R_X$ beforehand; bounding them for a computed network supplies the same inputs afterward.

This is the smallest conceptual change to the paper. The ordinary SVD solve need not change. The original $s=8,p=1$ transfer estimate is valid but too loose for a useful numerical certificate, so the verification below uses a sharper, separate evaluation rule. It does not change the training data or fitted coefficients.

## A useful continuous error bound for the returned network

Partition $[-1,1]$ into cells $I_k$ of lengths $h_k$. Let $g=f-q_c$. Suppose $g$ is analytic on a neighborhood of each cell's Bernstein ellipse of parameter $\rho>1$, and $|g(z)|\le M_k$ there. A degree-$p$ Chebyshev truncation $P_k$ satisfies

$$\|g-P_k\|_{\infty,I_k}\le e_k:=\frac{2M_k}{\rho-1}\rho^{-p}.$$

Let $Q_k$ denote normalized $(p+1)$-point Gauss–Legendre quadrature, with positive weights summing to one. It integrates $P_k^2$ exactly. Therefore

$$\begin{aligned}
\|g\|_{2,I_k}&\le\|P_k\|_{2,I_k}+e_k\\
&=\sqrt{Q_k(P_k^2)}+e_k\\
&\le\sqrt{Q_k(g^2)}+2e_k.
\end{aligned}$$

Here the cell norm is normalized by cell length. Applying Minkowski across cells gives the normalized whole-domain RMS bound

$$\boxed{\|g\|_{2,[-1,1]}\le\left[\sum_k\frac{h_k}{2}Q_k(g^2)\right]^{1/2}+2\left[\sum_k\frac{h_k}{2}e_k^2\right]^{1/2}.}$$

For this dictionary, choose $\rho=8$ on the core cells. The largest imaginary preactivation is

$$\frac{\gamma h}{4}(\rho-\rho^{-1})=\frac{0.25}{4}(8-1/8)=0.4921875<\frac\pi4.$$

Thus every tanh feature has magnitude at most one on these ellipses. If $B_k$ bounds the target locally, take $M_k=B_k+T$. These local target bounds are small compared with the global analytic bounds used in the a priori witness estimate. Set $p=24$ and use 25 Gauss nodes per cell; $\rho^{-p}=2^{-72}$. The same argument applied to $f$ yields a positive lower bound on its RMS norm, giving a relative error upper bound.

The derivation uses no reference scaling, no special SVD cutoff, no particular optimizer, and no training-grid sampling-density condition. It validates the returned function, not the algorithm that produced it.

## Numerical verification and FP64 evaluation

The verifier uses 256-bit MPFR directed arithmetic to enclose the 25 Legendre roots and quadrature weights. Disjoint sign-changing intervals establish all roots of the degree-25 polynomial. The weights are enclosed through $2/[(1-z^2)P'_{25}(z)^2]$ and divided by two to normalize them. Interval moment tests verify exactness through degree 49. Coefficient norms, positive weighted sums, square roots, and final divisions use directed arithmetic.

Tanh values on the verification lattice are prepared at 256 bits and rounded to FP64. Prediction uses explicit pairwise summation, with a conservative allowance for feature storage, multiplication, summation, target storage, and residual subtraction. Root and MPFR-oracle uncertainty are bounded separately and checked below an absolute $10^{-40}$ allowance. The implementation records the derivative/operation bounds used for this allowance; it is not treating a high-precision calculation as automatically exact. Saved centers and slopes are checked against the verification lattice, and each result records its coefficient-file hash. Analytic remainders are added to the residual upper bound and subtracted from the target-norm lower bound.

The first bound concerns the exact real network defined by the stored FP64 coefficients. A second bound covers the controlled forward evaluator used in this experiment. For representable input $t$, centered subtraction followed by the power-of-two slope gives an argument perturbation bounded by $u|a|$ plus subnormal allowances. Since $\sup_{y\ge0}y\operatorname{sech}^2y<1$, the activation error including correctly rounded tanh is at most $3u$ plus negligible subnormal terms. With $n=W+1$ readout terms, pairwise evaluation contributes

$$E_{\rm ev}=\left[3u+\Gamma_n^{\rm pw}(1+3u)\right]T+\beta.$$

For an arbitrary real input rounded to $Qx$, $|q_c(Qx)-q_c(x)|\le\gamma Tu$, because $\|q'_c\|_\infty\le\gamma T$. Adding these allowances to the exact-network RMS bound gives a continuous bound for this controlled FP64 pipeline. The measured coefficient sizes and argument ranges exclude overflow. This contract does not certify arbitrary tanh libraries, BLAS summation orders, or alternate affine evaluation orders.

For the unscaled solver at $N=1024$:

| Target | Exact stored-network relative RMS upper bound | Controlled FP64 relative RMS upper bound |
|---|---:|---:|
| $e^x$ | $2.31\times10^{-14}$ | $1.34\times10^{-13}$ |
| $\sin(4\pi x)$ | $8.28\times10^{-14}$ | $3.12\times10^{-13}$ |
| $\sin(24\pi x)$ | $5.23\times10^{-13}$ | $1.90\times10^{-12}$ |
| Chirp | $2.70\times10^{-13}$ | $1.19\times10^{-12}$ |
| Weak high-frequency mixture | $7.82\times10^{-14}$ | $2.63\times10^{-13}$ |
| Gaussian-windowed sine | $1.34\times10^{-13}$ | $5.15\times10^{-13}$ |

These are upper bounds, not measured grid errors. Verification was run for both methods at all five primary widths, giving 60 network bounds. The fitting procedure was never changed by the verifier.

## Claim to make, and claim to avoid

The supported claim is: **For the tested analytic targets, standard unscaled truncated SVD achieves comparable accuracy to reference-scaled SVD. A solver-independent posterior argument bounds the actual returned networks, so reference scaling is unnecessary for the demonstrated accuracy.**

The unsupported claim is that deleting $D$ automatically preserves the original unconditional width-independent coefficient theorem. Generic least-squares theory does not give that. In exact arithmetic, a witness with $\|c^\star\|_1\le C$ and sampled error at most $E$ gives unscaled TSVD bounds $\|c_\tau\|_2\le C+E/\tau$ and residual at most $E+\tau C$, but converting to $\ell_1$ generically introduces $\sqrt n$. The current tanh dictionary might admit a sharper specialized argument; this work does not assume one.

All 30 primary unscaled solutions also satisfy the original reference-coordinate norm budget $\|D^{-1}c\|_2\le2BL$ numerically. Two wider robustness cases do not satisfy that particular budget. The new posterior argument does not require it, so this is neither concealed nor used as a universal claim.

The concrete paper changes are to correct the implementation description, retain the existing a priori result with its proper scope, add the solver-independent corollary/proposition, and report the matched-solver comparison plus bounds on the actual experimental networks. No claim of identical algorithms is needed.

## Reproduction

- `experiments/expC14_theory_solver_comparison/ablation.py --robustness`: 138 paired comparisons and the new figure.
- `experiments/expC14_theory_solver_comparison/posterior.py`: independent continuous bounds from stored coefficients.
- `experiments/expC14_theory_solver_comparison/test_comparison.py` and `test_posterior.py`: seven tests, including interval moments and nonzero recovery/export.
- Data and figures are in `scaling_ablation/`; continuous bounds are in `posterior_verification/`. `resolution_summary.json` records the headline statistics. The original conservative-cutoff experiment remains at this directory's root.
