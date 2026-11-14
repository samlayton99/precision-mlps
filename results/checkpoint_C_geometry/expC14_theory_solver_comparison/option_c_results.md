# Option C: comparison under the original construction theorem — draft-pending-Sam

## TL;DR

- The original reference-scaled recovery recipe achieves comparable accuracy to simple SVD on the six-function suite after choosing an admissible larger halo and improving validated numerical bounds. Original (A.51) and cutoff condition (A.65) are unchanged.
- Across all 30 function/width pairs, the theoretical error is at most 2.754 times the simple-SVD error. For $N\ge256$, maximum errors are $5.62\times10^{-13}$ and $5.88\times10^{-13}$, respectively. Simple SVD has lower error in 7 of 30 cases; neither identical outputs nor a universal accuracy ordering is claimed.
- All original cutoff, geometry, coefficient-bound, and sampled-error checks pass. Twenty tests pass; doubling the independent midpoint grid changes reported errors by at most 1.41%.
- This instantiates the **general theorem in Appendix A.1**, with $a_H=11/4$, rather than the main text's displayed special case $a_H=1$. Fitting and evaluation use FP64; scalar bound verification and parameter enclosures use 256-bit MPFR. This is not evidence that every step of bound generation can be performed using a single FP64 number.

## Question

Can a procedure satisfying the original construction theorem and its original cutoff achieve accuracy comparable to the practical unscaled SVD, without replacing the cutoff, adding a norm-guard algorithm, or invoking a posterior-only proof?

## Experiment design

Both arms approximate the same six functions on $[-1,1]$: $e^x$, $\sin(4\pi x)$, $\sin(24\pi x)$, $\sin(8\pi(x+1)^2)$, $\sin(2\pi x)+10^{-3}\sin(40\pi x)$, and $e^{-8(x-0.2)^2}\sin(8\pi x)$. The grid of interior interval counts is $N\in\{128,256,512,1024,2048\}$.

Fix $\lambda=1/4$, analytic-neighborhood radius $\delta=1$, enlarged-real-interval margin $\Delta=9/16$, and halo constant $a_H=11/4$. Define spacing $h=2/N$, halo count $R=\lceil a_H\sqrt N\rceil$, slope $\gamma=\lambda/h$, centers $x_j=-1+jh$ for $j=-R,\ldots,N+R$, and hidden width $W=N+2R+1$. Both solvers receive exactly these neurons. Actual hidden widths are 193, 345, 639, 1201, and 2299; comparing at the same $N$ also compares at the same $W$ within every pair.

At the smallest resolution, $16a_H^2=121\le128$, the halo extent $(R+1/2)h=0.5078125<\Delta$, $\gamma\delta=16>4\pi$, and $\lambda R=8>2\log2$. All larger configurations satisfy the same admissibility conditions. The constant was chosen from the analytic bound before evaluating function errors: retaining (A.51) exactly gives $E_{\rm rep}/B=8.46\times10^{-17}$ already at $N=128$, rather than the large value produced by the earlier $a_H=1$ choice. The same constant is used for every function and width.

Training uses $M=8N+1$ uniform real samples. The paper's sampling choices $s=8$, polynomial degree $p=1$, and ellipse parameter $\varrho=2$ satisfy (A.55). Primary errors are measured on 16,384 independent uniform midpoints, with 32,768 separate midpoints for validation. These are measured relative grid $L^2$ errors, $\|q-f\|_2/\|f\|_2$, not tight continuous-domain certificates.

**Simple SVD.** The existing main-figure procedure is reproduced on the matched geometry: single-target FP64 `scipy.linalg.lstsq(A,y,cond=2**-52,lapack_driver="gelsd")`, with tanh features and an output-bias column. There is no reference scaling or theorem-bound calculation in this arm.

**Original theorem recipe.** Coordinate envelopes and $D=\operatorname{diag}\sqrt\alpha$ follow (A.56)–(A.57), including outermost halo corrections and the bias envelope. The matrix is $F=AD/\sqrt M$, and labels are $y_X=y/\sqrt M$. The thin SVD, pairwise projection, retained-coordinate division, and native coefficient export are all FP64. A native Jacobi SVD supplies the factors. The analytic approximation bound is the original $E_{\rm rep}$ in (A.51), with no substitution of observed fit or test error.

Let $S=B\sqrt{\sum_j\alpha_j}$ be the reference scaled-coefficient bound. The verified numerical bounds $a_W$ and $b_W$ concern the same nearby matrix and labels as (A.61)–(A.62). The code uses an enclosed lower bound $S_{\rm lo}\le S$ and an enclosed upper bound $\overline E_W\ge E_{\rm rep}$ to choose

$$\tau=\operatorname{round\ upward}\left[a_W+\frac{\overline E_W+b_W}{S_{\rm lo}}\right].$$

It verifies $(\tau-a_W)S_{\rm lo}-\overline E_W-b_W\ge0$ with directed arithmetic. This implies the original (A.65) condition $\tau S\ge\overline E_W+a_WS+b_W$. The excess allowance $\rho_W$ is bounded using an upper enclosure of $S$. Singular values are retained exactly when $\sigma_i>\tau$. There is no practical-cutoff substitution, norm-based acceptance branch, or coefficient refinement.

**Code & data**

- `experiments/expC14_theory_solver_comparison/option_c.py`, `option_c_config.yaml`: complete comparison.
- `theory_parameters.py`: outward enclosures of the unchanged analytic bound and envelopes.
- `tight_products.py`: enclosed matrix products and norm bounds; `jacobi_probe.py`: native FP64 SVD configuration and matrix-only probe.
- Final data, weights, summaries and PNG/PDF/SVG: `results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_jacobi/`.
- Earlier permitted-driver trials are retained in `option_c/` and `option_c_gesdd/`; the initial $a_H=1$ experiment and practical-cutoff variants remain separate.
- Reproduce into a fresh output directory: `MPLCONFIGDIR=/private/tmp/precisionmlps-c14-mpl VECLIB_MAXIMUM_THREADS=4 .venv/bin/python experiments/expC14_theory_solver_comparison/option_c.py --output /absolute/path/to/new/results`.
- Plot: `.venv/bin/python experiments/expC14_theory_solver_comparison/plot_option_c.py /absolute/path/to/results`.

## Results

At $N=1024$:

| Function | Simple SVD | Original theorem recipe |
|---|---:|---:|
| $e^x$ | $6.54\times10^{-14}$ | $3.31\times10^{-14}$ |
| $\sin(4\pi x)$ | $9.35\times10^{-14}$ | $3.26\times10^{-14}$ |
| $\sin(24\pi x)$ | $3.26\times10^{-13}$ | $2.13\times10^{-14}$ |
| Chirp | $2.20\times10^{-13}$ | $8.78\times10^{-14}$ |
| Weak high-frequency mixture | $3.28\times10^{-14}$ | $3.09\times10^{-14}$ |
| Gaussian-windowed sine | $9.76\times10^{-14}$ | $2.43\times10^{-14}$ |

At $N=128$, the more oscillatory functions remain less accurate in both methods, reaching errors near $10^{-10}$; their curves closely track. From $N=256$ onward, both methods remain below $6\times10^{-13}$ on every tested function. At $N=512$, simple SVD is better on several targets; at $N=1024$ and 2048, the theoretical recipe is better on all six. These finite-precision differences do not establish a universal ordering.

All 30 instances satisfy the original cutoff inequality. All geometry and sampling checks pass, as do the derived coefficient and sampled-error bounds. Stored coefficients are FP64. The width-independent reference-envelope constant from (A.48)–(A.50) is also checked against the computed envelope sums. The paper's full error allowance may be very loose, especially with $s=8,p=1$ and global analytic class bounds; the figure reports measured accuracy rather than presenting that allowance as a sharp prediction.

The remaining computed-network assumptions in A.3.5 also have a direct justification for this evaluator. Core endpoints and centers are exactly representable dyadic numbers; $\gamma=N/8$ is a power of two. For a representable input $t$, let $a=\gamma(t-x_j)$ be the exact preactivation. Rounded centered subtraction followed by the power-of-two multiplication perturbs it by at most $u|a|$ plus subnormal allowances. The inequality $\sup_{v\ge0}v\operatorname{sech}^2(v)<1$ bounds the resulting tanh perturbation by $u/(1-u)$ plus those allowances. Correct rounding of tanh adds at most $u$, so the paper's uniform activation contract holds with $\epsilon_\phi=3u$ plus an explicit negligible subnormal term. Products and output sums use the prescribed pairwise arithmetic. For arbitrary real inputs, nearest rounding clamped to $[-1,1]$ has displacement at most $u$ and preserves each core cell because its endpoints are exactly representable. Argument ranges and stored coefficient magnitudes exclude overflow. Thus the original input/evaluation allowance can be applied; no alternate posterior error theorem is needed. Its numerical upper bound is not asserted to equal the much smaller measured errors in the figure.

### Figures

- **comparison:** One function color; solid simple-SVD curves and dashed original-theorem curves; horizontal $N$, vertical relative $L^2$ error. Both methods use the larger halo. This is the option-C figure to use, rather than the earlier matched-practical-cutoff figure.
- **cutoff_diagnostics:** The analytic, matrix, and label/projection contributions to the original cutoff for $e^x$. The analytic term is below the numerical terms throughout this design.

## Why these changes are permitted

The halo choice is explicitly permitted by the general geometry in Appendix A.1. It changes an admissible constant, not the $\sqrt N$ scaling or the proof. The main text's special-case halo description would need to identify the experimental constant rather than implying $a_H=1$.

The original text describes (A.63) as a **sufficient realization** of the required numerical interface. It permits validated bounds on the same nearby-problem quantities. The new implementation tightens those bounds instead of using the first experiment's loose dimension-based estimates.

For clarity, let $\widehat U,\widehat V$ be the returned FP64 singular-vector matrices and $\Sigma$ the diagonal matrix of returned singular values. Define their Gram defects $H=\widehat U^T\widehat U-I$ and $G=\widehat V^T\widehat V-I$. Let $o_U,o_V<1$ be validated upper bounds on their spectral norms. Let $\sigma_{\max}$ be the largest singular value and $V_+=\sqrt{1+o_V}$.

Write $\widehat U=U_0(I+H)^{1/2}$ and $\widehat V=V_0(I+G)^{1/2}$, where $U_0,V_0$ have orthonormal columns. With $t_s(e)=e^2/[8(1-e)^{3/2}]$, Taylor's theorem yields the valid bound

$$a_W=e_F+r_F+\left[\tfrac12\|H\Sigma\|_2+t_s(o_U)\sigma_{\max}\right]V_++\tfrac12\|\Sigma G\|_2+t_s(o_V)\sigma_{\max}.$$

Here $e_F$ bounds ideal-to-stored matrix formation error and $r_F$ bounds the residual of the stored SVD factors. This avoids multiplying every orthogonality defect—including defects in nearly null directions—by the largest singular value.

For label recovery, let $\widetilde y$ be the stored normalized labels, $p$ the actual working-precision pairwise projection, and $q=\widehat U^T\widetilde y$ the exact projection of stored arrays. With $t_i(e)=3e^2/[8(1-e)^{5/2}]$, a valid projection discrepancy is

$$d_p=\|p-q\|_2+\tfrac12\|Hq\|_2+t_i(o_U)\|q\|_2.$$

Adding the label-formation error and division-rounding allowance gives $b_W$. All products in these bounds are enclosed; the actual solver projection $p$ is not replaced by a higher-precision answer. These formulas bound the same objects as the original proof and leave (A.65) unchanged.

The matrix-product enclosures split FP64 inputs into fixed-point chunks whose products and dot sums are exact at the supported dimensions, then account for reconstruction and decomposition tails. Spectral bounds are derived from enclosed Gram products. When exact normalization is unavailable because of tiny entries, a validated Frobenius bound is used; entries and error terms are not dropped.

## Driver choice and arithmetic disclosure

The first GESVD/GESDD trials satisfied the original conditions but still had some accuracy gaps. Their actual factorization errors, not merely loose diagnostic estimates, limited the permitted cutoff. A matrix-only probe compared a standard native FP64 Jacobi driver. It used no target labels or test errors for selection. The selected configuration is `DGEJSV` with `JOBA=C, JOBU=U, JOBV=J, JOBR=N, JOBT=N, JOBP=N`; the complete thin factors and LAPACK scale are checked. [LAPACK's DGEJSV documentation](https://netlib.org/lapack/explore-html/d8/d78/group__gejsv_gaca7ba7f1e8002c7a1d5bffa4ccbb541f.html) specifies this SVD driver; the integer interface was checked against [SciPy's wrapper source](https://raw.githubusercontent.com/scipy/scipy/v1.17.0/scipy/linalg/flapack_other.pyf.src).

Model matrix formation, SVD, pairwise projection, retained divisions, coefficient export and evaluation remain FP64. Analytic parameter enclosures and scalar numerical-bound bookkeeping use 256-bit MPFR, while diagnostic matrix products use FP64 expansions. Real target values are prepared offline and rounded once. The original theorem allows validated bounds with outward rounding; this use is recorded explicitly. The experiment should not be described as proving an entirely single-FP64-scalar bound-generation pipeline or as reproducing the paper's all-operations-at-$p$-bits experiment.

Tests compare matrix-product enclosures with high-precision references, validate weighted polar bounds on noncommuting examples, check the unchanged analytic bound and cutoff, and verify Jacobi factors/scaling on rank-deficient and extreme-scale inputs. Independent read-only review checked the correspondence with (A.61)–(A.65). These checks are evidence of a faithful numerical implementation, not a formal proof-assistant verification.

## Conclusions

Option C succeeds on this suite: a procedure satisfying the original general construction theorem achieves comparable measured accuracy to simple unscaled SVD. The simple implementation avoids the reference envelopes and numerical-bound machinery. The supported wording is “comparable accuracy, with simple SVD sometimes more accurate,” rather than claiming identical algorithms or that simple SVD is generally more accurate.

## Open questions

Before a paper-ready response, decide how to describe the general halo constant in the main text and explicitly disclose bound-verification arithmetic. This experiment does not establish the same accuracy for every analytic target, every numerical backend, or the main text's fixed $a_H=1$ special case at the same small widths.
