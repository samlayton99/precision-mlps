# Why the eigenvalue ratios change with gamma

Status: diagnostic calculations complete; mathematical identities checked independently; general conclusions beyond the tested geometry remain open. September 22, 2026.

## What this investigation establishes

- For the actual finite matrix, the change in a normalized eigenvalue is exactly the difference between that eigenvalue's fractional growth and the largest eigenvalue's fractional growth. The smaller resolved eigenvalues often grow much faster in percentage terms.
- Uniform integration over centers gives an exact positive Fourier integral for each centered quadratic form. Its logarithmic derivative contains the explicit factor $H_\gamma(\omega)=2(z\coth z-1)$, where $z=\pi|\omega|/(2\gamma)$. This factor increases with frequency. It does not require Fourier waves to be eigenvectors of the sampled kernel.
- The actual eigendirections were used to evaluate that integral, with finite-halo, mean-direction coupling, and discrete-center corrections accounted for separately. These corrections are small for the examined moderate-gamma modes. Their smallness was measured, not assumed from the heatmap.
- Not every ratio improves: resolved ranks 131 and 132 decrease at sufficiently large gamma in the default geometry. The earlier sharp-step-minus-identity approximation is not accurate enough to explain the observed spectrum in this gamma range.

## Question

Why do many eigenvalues divided by the largest eigenvalue increase by orders of magnitude as gamma increases, even when the relevant eigendirections change comparatively little? The calculation must distinguish an exact relation, a numerical check on this geometry, and a general monotonicity claim.

## Model and exact calculation

There are $m$ training inputs $x_i$, $W$ centers $c_j$, and common slope $\gamma$. The normalized design matrix and kernel are

$$J_\gamma=\frac1{\sqrt m}[\mathbf1,\Phi_\gamma],\qquad (\Phi_\gamma)_{ij}=\tanh(\gamma(x_i-c_j)),\qquad K_\gamma=J_\gamma J_\gamma^T.$$

Geometry, readout coordinates, sample normalization, and the bias feature are fixed. The default is $N=128$ interior intervals, $h=2/N$, $129$ interior centers and $12$ halo centers on each side, hence $W=153$; $m=263$ uniform training points. No optimizer training is performed. The gamma grid has 78 values in $[0.25,128]$. Eigenvalues are found by rectangular SVD of $J$, not by forming a rounded Gram matrix and diagonalizing its numerical tail. The display excludes ratios below $10^{-14}$; this display threshold is not a certified error bound.

### 1. The exact quantity that determines whether a ratio improves

Write the ordered eigenpairs as $K_\gamma u_i=\lambda_i u_i$, $\|u_i\|=1$, with $\lambda_1$ largest. For a simple positive eigenvalue,

$$\lambda_i'=u_i^TK_\gamma'u_i.$$

Differentiate $K_\gamma u_i=\lambda_i u_i$ and multiply by $u_i^T$. The two terms containing $u_i'$ cancel because $u_i^TK_\gamma=\lambda_i u_i^T$. This is the usual symmetric-matrix eigenvalue derivative; see [MIT's matrix-calculus notes](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes-and-readings/). At repeated eigenvalues, individual eigenvectors are not unique; use differentiable eigenvalue branches or diagonalize the derivative within the repeated eigenspace.

Define fractional eigenvalue growth per fractional gamma increase,

$$a_i(\gamma):=\frac{\gamma\lambda_i'}{\lambda_i}=\frac{d\log\lambda_i}{d\log\gamma}.$$

Then the exact ratio identity is

$$\boxed{\frac{d\log(\lambda_i/\lambda_1)}{d\log\gamma}=a_i-a_1.}$$

This is not a statement about absolute eigenvalue increments. A small eigenvalue can have a tiny absolute increment but a very large fractional increment. With $\eta_\gamma=0.5/\lambda_1$, the one-step residual factor is $1-\frac12\lambda_i/\lambda_1$, so these are precisely the ratios entering normalized readout GD.

For numerical stability, if $J=U\Sigma V^T$ and $\lambda_i=\sigma_i^2$, compute

$$a_i=\frac{2\gamma}{\sigma_i}u_i^TJ_\gamma'v_i,$$

where the bias column of $J_\gamma'$ is zero and its hidden columns are

$$\frac1{\sqrt m}(x_i-c_j)\operatorname{sech}^2(\gamma(x_i-c_j)).$$

### 2. Why the Fourier multiplier can now enter that derivative explicitly

Let $Q=I-\mathbf1\mathbf1^T/m$, which subtracts the sample mean. Define the centered whole-center-line matrix

$$A_\gamma=-\frac{2}{hm}Q\big[(x_i-x_j)\coth(\gamma(x_i-x_j))\big]Q.$$

The diagonal value inside the brackets is $1/\gamma$. This is a well-defined positive matrix on mean-zero sample vectors. It is not the full finite kernel. For any real vector $q$ with $\mathbf1^Tq=0$,

$$q^TA_\gamma q=\frac1{hm}\int_{\mathbb R}\left|\sum_jq_j\tanh(\gamma(x_j-c))\right|^2dc.$$

The sum tends to zero at both center-coordinate tails because $\sum_jq_j=0$. Its step-function version is supported between the extreme training inputs, so the smoothing and Parseval arguments use square-integrable functions. The transform of a smoothed step difference is the transform of the step difference multiplied by $M_\gamma(\omega)=z/\sinh z$. Consequently,

$$\boxed{q^TA_\gamma q=\frac1{2\pi hm}\int_{\mathbb R}\frac{4M_\gamma(\omega)^2}{\omega^2}\left|\sum_jq_j e^{-i\omega x_j}\right|^2d\omega.}$$

This is the useful extra structure provided by uniform whole-line center integration: the frequency contributions to the quadratic form are nonnegative. The samples remain finite and need not be periodic. No claim about Fourier waves being sample-matrix eigenvectors is used.

The apparent singularity at zero frequency is removable: the numerator's exponential sum is $O(\omega)$ for zero-sum $q$.

### 3. The explicit frequency-dependent fractional gain

Since $z=\pi|\omega|/(2\gamma)$,

$$\gamma\frac{\partial}{\partial\gamma}\log M_\gamma(\omega)^2=2(z\coth z-1)=:H_\gamma(\omega).$$

The function $H$ is zero at frequency zero and strictly increasing in $|\omega|$. Indeed, its derivative with respect to $z>0$ is $2(\coth z-z\operatorname{csch}^2z)>0$. Its useful limiting forms are

$$H_\gamma(\omega)\sim\frac{\pi^2\omega^2}{6\gamma^2}\quad(|\omega|/\gamma\to0),\qquad H_\gamma(\omega)\sim\frac{\pi|\omega|}{\gamma}-2\quad(|\omega|/\gamma\to\infty).$$

Thus increasing gamma produces a larger *fractional* increase in the Fourier weight at higher frequencies. This is stronger and more relevant to eigenvalue ratios than saying only that high frequencies are suppressed.

For an actual finite-matrix eigenvector $u_i$, put $q_i=Qu_i$ and $D_i=u_i^TA_\gamma u_i$. When $D_i>0$, normalize the integrand above to a probability density $w_i(\omega)$:

$$w_i(\omega)=\frac{4M_\gamma(\omega)^2|\sum_j(q_i)_j e^{-i\omega x_j}|^2}{2\pi hm\,\omega^2D_i},\qquad\int_{\mathbb R}w_i(\omega)d\omega=1.$$

Holding the vector fixed when differentiating the matrix gives the exact identity

$$\boxed{\gamma u_i^TA_\gamma'u_i=D_i\int H_\gamma(\omega)w_i(\omega)d\omega.}$$

The eigenvector need not diagonalize $A_\gamma$. This does **not** identify the right side with the total derivative of $D_i(\gamma)=u_i(\gamma)^TA_\gamma u_i(\gamma)$; that total derivative would contain an additional $u_i'$ term. We instead apply the eigenvalue derivative to the actual $K$ first, then split $K'$. That distinction is essential.

### 4. Keep the corrections instead of assuming them away

Let $K_c$ be the finite uniform-center integral over the same halo bounds, including the bias. Decompose the actual matrix exactly as

$$K_\gamma=A_\gamma+T_\gamma+B_\gamma+S_\gamma,$$

where

$$T_\gamma=QK_cQ-A_\gamma,\qquad B_\gamma=K_c-QK_cQ,\qquad S_\gamma=K_\gamma-K_c.$$

- $T$ accounts for the omitted center tails. It is negative semidefinite.
- $B$ accounts for the constant sample direction and its coupling to the centered directions. The constant bias itself has zero gamma derivative.
- $S$ accounts for replacing the center integral with the finite center sum.

Therefore, with

$$c_i=\frac{\gamma}{\lambda_i}u_i^T(T_\gamma'+B_\gamma'+S_\gamma')u_i,$$

the exact finite-kernel formula is

$$\boxed{\frac{d\log(\lambda_i/\lambda_1)}{d\log\gamma}=\frac{D_i}{\lambda_i}\int H_\gamma w_i+c_i-a_1.}$$

Every object refers to the current gamma and actual eigenvector. The equation has not frozen eigendirections. Odd symmetry makes the $B'$ contribution vanish for odd eigenvectors. Our symmetric geometry preserves the odd subspace, and the mixed-sine target is odd. We also evaluated actual even modes and retained their $B'$ contribution.

The exact equation supplies a sufficient condition requiring only a frequency tail. For any threshold $\Omega>0$, define $\tau_i=\int_{|\omega|\ge\Omega}w_i(\omega)d\omega$. Monotonicity and nonnegativity of $H$ give

$$\boxed{\frac{d\log(\lambda_i/\lambda_1)}{d\log\gamma}\ge\frac{D_i}{\lambda_i}\tau_i H_\gamma(\Omega)-|c_i|-a_1.}$$

A positive right side proves local ratio improvement, provided the quantities or their bounds are established. There is no assumption that eigenvalue rank alone orders frequency content. Numerical evaluation of these quantities is evidence for a particular geometry; it is not a geometry-independent theorem or an interval-arithmetic certificate.

**Code and data.** Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/run.py` and `integrate_growth.py`. Reproduce with the project Python environment. Saved arrays and numerical checks are under this folder's `data/`: `ratios.npz`, `fourier_modes.npz`, `summary.json`, and `integrated_growth.json`. All figures are PNGs beside this writeup.

## Results for the actual geometry

At gamma 16, the largest eigenvalue has fractional growth $a_1=0.010960864$. The table compares the pure Fourier weighted average $\int H w_i$ to the actual fractional eigenvalue growth. Here the selected odd ranks correspond to full ranks 2, 6, 12, 20, and 32. The pure average deliberately omits $D_i/\lambda_i$ and all finite-model corrections; these are retained in the exact accounting and in the saved data.

| Full eigenvalue rank | Fourier weighted average | Actual $a_i$ | Actual ratio growth $a_i-a_1$ |
|---:|---:|---:|---:|
| 2 | 0.01463423 | 0.01463424 | 0.00367337 |
| 6 | 0.35105663 | 0.35105679 | 0.34009592 |
| 12 | 1.48709604 | 1.48709713 | 1.47613626 |
| 20 | 3.54684319 | 3.54684847 | 3.53588761 |
| 32 | 6.90227140 | 6.90229545 | 6.89133459 |

The actual improvement is therefore a difference in **fractional** eigenvalue growth. For example, infinitesimally near gamma 16, a 1% gamma increase produces about a 6.9% increase in the rank-32 eigenvalue, but only a 0.011% increase in the leading eigenvalue.

For rank 32 at gamma 16, about 96.43% of the Fourier-integral weight lies above $|\omega|=8\pi$. The threshold inequality gives a numerical lower bound of about 2.888 on its ratio-growth derivative, compared with the measured 6.891. The signs do not depend on comparing nearly equal terms: the positive margin is large relative to the corrections.

Across full ranks 2–40, including even modes, adding the mean-coupling term to the Fourier contribution leaves a maximum relative discrepancy in the eigenvalue-growth rate of about 0.77% at gamma 8, 0.0012% at gamma 16, and 0.0059% at gamma 64. These discrepancies are the remaining finite-halo and center-grid terms, plus numerical error. The mean-coupling term itself is not negligible for every low mode, especially at gamma 8.

### Finite changes, not only derivatives at one point

The exact integration formula is

$$\log\frac{(\lambda_i/\lambda_1)(\gamma_b)}{(\lambda_i/\lambda_1)(\gamma_a)}=\int_{\log\gamma_a}^{\log\gamma_b}(a_i-a_1)\,d\log\gamma.$$

We integrated both the exact derivative and the Fourier estimate $\int H w_i-a_1$ along the **actual changing eigendirections**. No growth rate was fitted. This is a mechanism audit; it does not predict eigendirections without solving the matrices.

| Gamma change | Full rank | Actual ratio improvement | Integrated Fourier estimate, omitting corrections |
|---|---:|---:|---:|
| 8 → 16 | 12 | 5.9814× | 5.9790× |
| 8 → 16 | 20 | 49.2169× | 49.1050× |
| 8 → 16 | 32 | 1277.7239× | 1263.3445× |
| 16 → 64 | 12 | 2.15237× | 2.15237× |
| 16 → 64 | 20 | 7.41750× | 7.41751× |
| 16 → 64 | 32 | 74.07037× | 74.07064× |

Including all terms recovers the endpoint changes to about $2.2\times10^{-14}$ relative in this calculation. Agreement of 16- and 32-point integration is a numerical convergence check, not a rigorous enclosure of a continuum of gamma values.

### Important exceptions

The default finite kernel does not have universal ratio monotonicity. Rank 131 has ratio approximately $2.46\times10^{-9}$ at gamma 64 and $1.09\times10^{-10}$ at gamma 128, a decrease by about 23×. Rank 132 also decreases. These values are well above the display cutoff, and the negative derivatives are confirmed by independent finite differences. The step limit can lose directions as halo neurons saturate; we have not separately attributed every declining tail mode to a particular neuron combination.

The approximation $K_\gamma\approx K_\infty-2I/(hm\gamma)$ is also insufficient in the measured range. At gamma 16 it predicts 221 negative eigenvalues in the continuous $263\times263$ matrix. The true matrix is positive semidefinite. Its operator error relative to the largest eigenvalue is only about 0.0051, illustrating why a visually small or leading-eigenvalue-relative matrix error can destroy the small-eigenvalue explanation. The finite-center integral itself agrees closely with much of the actual spectrum, while the sharp-step shift does not.

### Figures

- [Eigenvalue ratios and exact growth rates](eigenvalue_ratio_growth.png): top left shows actual normalized eigenvalues versus gamma, including a decreasing tail ratio; top right shows their logarithmic derivatives. Bottom left compares fractional eigenvalue growth with the leading eigenvalue's growth at gamma 16, including the Fourier-plus-mean calculation for odd and even modes. Bottom right accounts for five odd modes using the Fourier, halo, center-grid, and normalization terms.
- [Fourier distributions and eigenvalue growth](fourier_eigenvalue_growth.png): columns use gamma 8, 16, and 64. Top panels show the normalized positive-frequency weight in the quadratic form for actual odd eigenvectors. These are not target residual spectra. Bottom panels compare the pure Fourier weighted averages with actual eigenvalue growth, using common logarithmic vertical limits.
- [Integrated ratio improvements](integrated_ratio_growth.png): compares endpoint improvements with the integrated Fourier estimate for gamma 8 → 16 and 16 → 64. Actual eigendirections are followed along each path; this does not assume fixed target projections or fixed eigenvectors.
- [Check of the earlier sharp-step approximation](approximation_check.png): compares the discrete kernel, finite-center integral, and sharp-step diagonal-shift approximation at three gamma values. Negative predicted eigenvalues are counted above each panel and omitted from the logarithmic axes.

## Verification and mathematical limits

The identities and finite-model correction formula were independently reviewed in this task. An independent centered finite difference in log gamma checks the SVD-derived logarithmic eigenvalue derivatives at eight gamma values; the maximum absolute discrepancy for ratios above $10^{-12}$ is below $4\times10^{-7}$. Whole-line frequency integration uses 16,385 points and is compared with 8,193. The checked energy and derivative integrals agree to about $10^{-13}$ or better. The frequency cutoff is $30\gamma$; for a unit sample vector, the omitted energy is bounded analytically by $4/[h\gamma(\exp(30\pi)-1)]$. Center integration uses composite Gauss–Legendre quadrature over the actual center cells, comparing 8 and 16 nodes per cell, and is cross-checked against the closed integral matrix. All calculations are FP64, without directed-rounding interval certification.

For symmetric outer integration bounds $[-B,B]$, the centered finite-halo error has the rigorous norm bound

$$0\preceq A_\gamma-QK_cQ,\qquad\|A_\gamma-QK_cQ\|_2\le\frac{2}{h\gamma}e^{-4\gamma(B-1)}.$$

This follows by bounding the squared tail of each centered tanh combination. It can be much too loose relative to a small eigenvalue, so the investigation evaluates individual directional corrections rather than using this global bound as if it certified all modes.

Increasing $M_\gamma$ implies increasing positive eigenvalues of the centered whole-line matrix. It does not, by itself, order their ratios. A sufficient comparison is stochastic dominance of the frequency weights because $H$ is increasing. A larger mean frequency or more zero crossings alone is not that theorem. This investigation evaluates the relevant weighted averages and an explicit tail-based sufficient condition instead of assuming a universal rank-to-frequency correspondence.

There cannot be a generic normalized-ratio monotonicity theorem for arbitrary uniform-center finite networks. An exact counterexample uses samples $(-1,1)$ and centers $(-3/2,0,3/2)$, including the bias and the normalization $1/m$. Let $a=\tanh(5\gamma/2)$ and $b=\tanh(\gamma/2)$. Then

$$\lambda_+=1+\frac{(a+b)^2}{2},\qquad\lambda_-=\tanh^2\gamma+\frac{(a-b)^2}{2}.$$

Their ratio has the asymptotic expansion $\lambda_-/\lambda_+=1/3+(4/9)e^{-\gamma}+O(e^{-2\gamma})$, so it eventually decreases. This analytic example establishes the limitation separately from the observed tail reversals in the default network.

## Conclusions supported by these calculations

For the examined resolved modes of the default geometry, larger gamma improves the readout GD ratios because the frequency weights associated with those modes acquire much larger fractional gains than the leading eigenvalue. The exact derivative decomposition and measured corrections account for this effect quantitatively. The claim concerns specified modes and gamma ranges; it is neither a theorem that every ratio increases nor a closed-form formula for the entire finite eigendecomposition.

## What remains open

Derive geometry-dependent bounds on the eigenvectors' Fourier weights without first diagonalizing the finite matrix, and control the corrections over a continuous gamma interval. That would turn this exact conditional mechanism and checked numerical account into an a priori theorem for a specified family of uniform geometries. Width, jitter, and alternative readout-coordinate maps have not been varied in this investigation.

## Same-geometry check: slow GD despite an adequate readout (September 23)

Sam clarified that the primary goal is an impracticality argument for stable raw-coordinate GD at small gamma, not a perfect spectral predictor. The polynomial-bound route is excluded because its previous bounds were too loose. The following check applies the positive-slow-band idea from the new `gamma_access_collaborator_note.pdf` to **this investigation's** W=153, m=263 geometry; it does not import numerical values from that note's W=559, m=8193 geometry.

Use the fixed mixed-sine target, zero readout, and eta=0.5/lambda_max. A mode's per-update rate is eta*lambda_i. If a fraction p of squared target norm lies in positive rates (a,b], then the exact spectral GD formula gives

$$E(n)\ge\sqrt p(1-b)^n,\qquad n\ge\left\lceil\frac{\log(\sqrt p/\epsilon)}{-\log(1-b)}\right\rceil\quad(p>\epsilon^2).$$

Only positive eigenvalues enter p, so this portion is representable. Numerical values below come from rectangular SVD, not executed GD; no interval certificate is claimed.

| Gamma | Positive rate band | Target energy fraction | Necessary steps to 1% from this band | Full spectral forecast to 1% |
|---:|---:|---:|---:|---:|
| 4 | (1e-14, 1e-11] | 0.02735465123 | 280,573,583,534 | 1,007,616,497,993 |
| 8 | (1e-9, 1e-6] | 0.04352213951 | 3,037,926 | 19,697,563 |
| 16 | (1e-9, 1e-6] | 5.5257e-7 | No obstruction to 1% from this band | 66,209 |
| 64 | (1e-9, 1e-6] | 1.6695e-6 | No obstruction to 1% from this band | 18,272 |

The gamma-4 band was selected during this diagnostic, not preregistered. The gamma-8 band matches the collaborator note's thresholds. A zero lower count only says that the selected band alone does not obstruct 1%; it does not predict instantaneous convergence.

Capacity was also checked constructively. At gamma 4, solving only resolved singular directions with eta*lambda>=1e-14 produces an explicit readout of Euclidean norm 4625.807 and relative residual 0.0005850007883. Thus an error substantially below 1% is attainable with this same geometry. Direct 70-digit evaluation of the saved-double coefficients and nominal-real tanh gives 0.0005850007883319744. Independent `gesvd` and `gesdd` SVD routines give band masses 0.027354651231198325 and 0.0273546512311828. These checks assess numerical robustness, not directed-rounding certification.

Increasing a constant scalar step to the linear stability limit eta<2/lambda_max increases every rate by less than four relative to the saved step. The same gamma-4 band then still forces approximately 70 billion updates. This statement concerns ordinary frozen-feature Euclidean GD, not preconditioned, momentum, or jointly changing-feature algorithms.

The new check supplies the required-target and representability part of the argument on the same geometry as the Fourier eigenvalue diagnostics. The latter explain measured ratio changes through the nonnegative Fourier integral and its explicit multiplier, retaining finite-center corrections. The gamma-4 necessary count is supported by the spectral calculation; the existing detailed Fourier attribution is strongest at gamma 8 and above. No polynomial approximation enters either calculation.

Reproduction: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/slow_band_check.py`. Numerical results: `data/slow_band_check.json`. No new figures or optimizer trajectories were produced.

## Step prediction versus the actual tanh spectrum (September 23)

The preceding full spectral counts already use the actual finite tanh matrix; they are not independent Fourier predictions. To measure approximation accuracy, we now separately diagonalize the uniform-center integral over the actual halo cells. Both models use the same learning rate, $\eta=0.5/\lambda_{\max}(K_{\mathrm{finite}})$, the same samples, and the same target. SVD of a quadrature feature matrix avoids forming a Gram matrix before resolving small eigenvalues. No training updates are executed.

For each matrix, compute $E(n)^2=\sum_i p_i(1-\eta\lambda_i)^{2n}$, where $p_i=|u_i^\top y|^2/\|y\|^2$, including the nullspace contribution. Integer bisection finds the first $n$ with $E(n)\le0.01$. This gives the exact-arithmetic GD trajectory of that matrix, subject to numerical decomposition error; it is not a prediction fitted to a training trace.

| Gamma | Uniform-center integral prediction | Actual finite-tanh spectral count | Relative difference |
|---:|---:|---:|---:|
| 4 | $1.00901448\times10^{12}$ | $1.00761650\times10^{12}$ | +0.13874% |
| 8 | 19,698,115 | 19,697,563 | +0.00280% |
| 16 | 66,209 | 66,209 | Same integer crossing |
| 64 | 18,219 | 18,272 | -0.29006% |

The finite-center integral predicts the 1% count within 0.30% for these four cases. This is a check of our spatial integral approximation, not a reproduction of the collaborator note's periodic Fourier construction. It uses the integral matrix's own eigenvectors and target projections. Composite quadrature with 8 versus 16 nodes per center cell gives identical crossing counts at gamma 8, 16, and 64; at gamma 4 the difference is 57 steps out of about a trillion. Consequently the trillion-step figures should not be interpreted as single-step-certified counts. No directed-rounding certificate or floating-point trillion-step training trajectory is claimed.

The earlier slow-band lower bounds serve a different purpose. At gamma 4 the bound is about 28% of the full spectral count; at gamma 8 it is about 15%. They establish necessary delay without claiming to estimate the crossing time tightly.

- [Step prediction comparison](step_prediction_check.png): left, relative training error versus step, with solid curves from the finite tanh spectrum and dashed curves from the independently constructed center-integral matrix. Colors identify gamma. Right, steps to 1% and the integral prediction's relative error. Close agreement makes the curves overlap. No executed GD is shown.

Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/step_prediction_check.py`. Saved counts and complete plotted arrays: `data/step_prediction_check.json` and `data/step_prediction_check.npz`.

## Real-space illustration of the collaborator's periodic reference

[Periodic feature illustration](periodic_feature_illustration.png) shows the exact periodic reference described in Appendix A of the updated collaborator note, not the final boundary-corrected construction. This is an analytic illustration, not a training experiment. It uses interval length $L=2$, reference period $2L=4$, and gamma 4. Top: ordinary tanh and its smooth square-wave replacement for centers 0 and 0.875, with samples inside $[-1,1]$ marked. Bottom: their 33-by-16 sampled core feature matrices, with entries normalized by $1/\sqrt{33}$ and no bias column. Both heatmaps use the same color scale. Inputs increase downward and centers increase to the right.

The periodic reference is evaluated using the paired tanh image identity from Appendix A, retaining 12 image pairs; its period and antiperiod are checked numerically. No periodic boundary condition is imposed on the original training problem. The final construction adds corrections to the reference and treats boundary columns, halo features, bias, and the endpoint row explicitly. Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/periodic_feature_illustration.py`.
