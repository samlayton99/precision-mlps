# Direct two-sided eigenvalue intervals — draft-pending-Sam

## TL;DR

- Both eigenvalue and ratio bounds are now evaluated at a single gamma. No old eigenvalues, reference eigenvectors, or measured difference from the finite kernel is used in their construction.
- The whole-line Fourier integral is corrected for omitted halo centers and discrete center sampling. Both remaining errors have analytic bounds. The resulting intervals cover the tested finite-kernel ratios from gamma 4 to 64.
- The positive Fourier multiplier controls the integral operator's eigenvalues. The finite corrections and normalization remain explicit; the theorem does not assert universal monotonicity of every finite normalized eigenvalue.

## Question

Can the direct Fourier upper bound be completed with a useful lower bound at the same gamma, and can the same construction remain informative when the previous uniform grid-error estimate becomes loose?

## Experiment design

Use the existing geometry: 128 interior intervals, center spacing $h=1/64$, 153 tanh neurons including halo, and $m=263$ samples. Features and kernel are

$$
B_{aj}=m^{-1/2}\tanh(\gamma(x_a-c_j)),\qquad B_{a0}=m^{-1/2},\qquad K_\gamma=B_\gamma B_\gamma^\top.
$$

Sweep 33 logarithmically spaced gamma values from 4 to 64. Plot ranks 13, 20, 26, and 33, including both symmetry classes. This is a target-independent spectrum experiment; there is no training. The same theorem applies to arbitrary sample locations and finite lattice-center subsets; the implementation uses a contiguous center block covering the sample interval and includes a jittered-sample validation.

Let $Q\in\mathbb R^{m\times(m-1)}$ have orthonormal columns spanning $\mathbf1^\perp$. Write $t_\gamma(c)=(\tanh(\gamma(x_a-c)))_{a=1}^m$. The whole-line integral, restricted to this subspace, is

$$
H_\gamma=\frac1{hm}\int_{\mathbb R}Q^\top t_\gamma(c)t_\gamma(c)^\top Q\,dc.
$$

There are two finite-geometry corrections:

1. **Omitted centers.** The whole lattice contains centers absent from the actual finite network. Retain their outer products in the positive-semidefinite matrix $T_F=(1/m)\sum_{c\in F}Q^\top t_\gamma(c)t_\gamma(c)^\top Q$, where $F$ is a finite set of omitted centers. The still more distant omitted centers have matrix $T_{\rm tail}$ satisfying $0\preceq T_{\rm tail}\preceq\tau_\gamma I$.
2. **Center sampling.** A lattice sum differs from the center integral. Retain its first $p$ paired Poisson terms $A_{\gamma,k}$ explicitly. The remaining matrix has norm at most $\delta_{\gamma,p}$.

Thus define the corrected integral matrix

$$
S_\gamma=H_\gamma+\sum_{k=1}^{p}A_{\gamma,k}-T_F.
$$

Every term uses the specified geometry and this one gamma. The theorem below is exact for any finite truncation choices; truncation changes the known error allowances.

**Two-sided finite-network theorem.** Let $\beta_1(\gamma)\ge\cdots\ge\beta_{m-1}(\gamma)$ be the ordered eigenvalues of $S_\gamma$. Then

$$
S_\gamma-(\delta_{\gamma,p}+\tau_\gamma)I
\preceq Q^\top K_\gamma Q
\preceq S_\gamma+\delta_{\gamma,p}I.
$$

Consequently, for $2\le i\le m-1$,

$$
\boxed{
[\beta_i-\delta_{\gamma,p}-\tau_\gamma]_+
\le\lambda_i(K_\gamma)
\le\beta_{i-1}+\delta_{\gamma,p}.
}
$$

For scalar bounds $0<\ell_\gamma\le\lambda_1(K_\gamma)\le L_\gamma$, this gives

$$
\boxed{
\frac{[\beta_i-\delta_{\gamma,p}-\tau_\gamma]_+}{L_\gamma}
\le\frac{\lambda_i(K_\gamma)}{\lambda_1(K_\gamma)}
\le\min\left\{1,\frac{\beta_{i-1}+\delta_{\gamma,p}}{\ell_\gamma}\right\}.
}
$$

The endpoint eigenvalues have the corresponding one-sided interlacing bounds; a universal lower bound for the last eigenvalue is zero. No actual eigenvector is assumed to have zero mean. The different indices on the two sides are the cost of removing one sample-space dimension.

The simplest normalization bounds are $\ell_\gamma=1$ and $L_\gamma=W+1$, where $W$ is the number of tanh neurons. The displayed numerical intervals use sharper scalar bounds described below. They require only feature matrix-vector products, not the actual finite spectrum.

**Code & data**

- Driver: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_interval.py`.
- Reused integral/error routines: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_upper_bound.py`.
- Output folder: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_interval/`.
- `data.json`: all bounds, actual spectra at plotted ranks, correction remainders, and validation records.
- `direct_ratio_interval.png`: ratios and bound tightness.
- `direct_eigenvalue_interval.png`: unnormalized kernel eigenvalues and bound tightness.
- Previous strip-error proof: sibling `direct_ratio_upper_bound/REPORT.md`.

## Results

For rank 26:

| Gamma | Ratio lower bound | Actual ratio | Ratio upper bound |
|---:|---:|---:|---:|
| 4 | $6.562\times10^{-13}$ | $2.088\times10^{-12}$ | $2.262\times10^{-12}$ |
| 8 | $7.719\times10^{-8}$ | $1.487\times10^{-7}$ | $1.617\times10^{-7}$ |
| 16 | $2.532\times10^{-5}$ | $3.715\times10^{-5}$ | $4.039\times10^{-5}$ |
| 32 | $2.995\times10^{-4}$ | $3.826\times10^{-4}$ | $4.158\times10^{-4}$ |
| 64 | $6.919\times10^{-4}$ | $8.357\times10^{-4}$ | $9.081\times10^{-4}$ |

Across the displayed ranks, the lower side is about 31–47% of the actual ratio at gamma 4 and 79–84% at gamma 64. The upper side is about 8–61% above actual at gamma 4 and 9–12% above actual at gamma 64. Most visible interval width comes from interlacing and scalar normalization, not the analytic truncation remainders.

The even ranks in this symmetric geometry have zero-mean eigenvectors and unusually sharp upper interlacing bounds. Ranks 13 and 33 were included specifically to also test eigenvectors with nonzero constant components. Their intervals remain valid and useful; the construction does not assume the actual eigenvectors have zero mean.

The earlier gamma-64 grid-error allowance was 80.5 in kernel eigenvalue units and made the bounds useless. Keeping eight explicit paired Poisson terms reduces the remaining analytic allowance to $9.8\times10^{-21}$. At gamma 4 and 8 no Poisson terms need to be retained. These corrections are computed from the Fourier formula, not measured from the difference against the actual finite matrix.

### Figures

- **Eigenvalue-ratio interval:** left panel has gamma on a log horizontal axis and normalized eigenvalues on a log vertical axis. Colors identify ranks. Actual ratios are solid, lower bounds dashed, upper bounds dotted; shading fills the interval. The right panel divides each bound by its actual ratio.
- **Eigenvalue interval:** the same layout for the unnormalized eigenvalues of the already sample-normalized kernel. This separates the eigenvalue enclosure from the additional scalar uncertainty in normalizing by the largest eigenvalue.

## Additional details

### Where the lower bound comes from

Let $C_\infty$ denote the centered kernel obtained by summing over every lattice center. The finite kernel satisfies

$$
Q^\top K_\gamma Q=C_\infty-T_F-T_{\rm tail}.
$$

The Fourier/Poisson calculation gives $C_\infty=H_\gamma+\sum_{k=1}^p A_{\gamma,k}+R_p$, with $\|R_p\|\le\delta_{\gamma,p}$. Therefore $Q^\top K_\gamma Q=S_\gamma+R_p-T_{\rm tail}$. This proves both matrix inequalities. Weyl's inequalities followed by compression interlacing prove the eigenvalue and ratio bounds. All bounds are at a single gamma.

### Explicit grid corrections and their remainders

Write $d_{ab}=x_a-x_b$ and $\nu_k=2\pi k/h$. The function of center location

$$
G_{ab}(c)=\tanh(\gamma(x_a-c))\tanh(\gamma(x_b-c))-1
$$

is integrable. For $\nu\ne0$, its ordinary Fourier transform is

$$
\widehat G_{ab}(\nu)=
-\frac{4M_\gamma(\nu)}{\nu}
\sin(\nu d_{ab}/2)\coth(\gamma d_{ab})
e^{-i\nu(x_a+x_b)/2},
\qquad M_\gamma(\nu)=\frac{\pi|\nu|/(2\gamma)}{\sinh(\pi|\nu|/(2\gamma))}.
$$

Its diagonal limit is $-2M_\gamma(\nu)e^{-i\nu x_a}/\gamma$. The formula follows by writing the product-minus-one as a coth factor times a difference of tanhs and applying the tanh transform to that integrable difference.

For lattice origin $c_0$, the paired real correction is

$$
A_{\gamma,k}=\frac2{hm}Q^\top\operatorname{Re}
\left[e^{i\nu_k c_0}\widehat G(\nu_k)\right]Q.
$$

Individual correction matrices need not be positive semidefinite. For any strip angle $0<\vartheta<\pi/2$, the existing analytic contour bound gives

$$
\delta_\gamma=
\frac{8\gamma\|x-\bar x\mathbf1\|^2\sec^4\vartheta}
{3hm[\exp(2\pi\vartheta/(\gamma h))-1]},
\qquad
\delta_{\gamma,p}=\delta_\gamma\exp\left(-\frac{2\pi\vartheta p}{\gamma h}\right).
$$

The latter simply sums only the unretained Fourier indices $k>p$. The implementation uses $\vartheta=\arctan(\pi/(2\gamma h))$ and enough terms for this analytic remainder to be at most $10^{-18}$.

For the exterior-center remainder, suppose the first unretained right and left centers are $c_R>x_{\max}$ and $c_L<x_{\min}$. From $1-\tanh t\le2e^{-2t}$, cancellation of the constant tails, Cauchy–Schwarz, and a geometric series,

$$
\tau_\gamma=
\frac{4\left[e^{-4\gamma(c_R-x_{\max})}+e^{-4\gamma(x_{\min}-c_L)}\right]}
{1-e^{-4\gamma h}}
$$

is a valid operator upper bound. Any omitted interior centers must be included explicitly in $T_F$. This implementation retains the first $\lceil18/(\gamma h)\rceil$ omitted centers on each side.

### Bounds on the leading eigenvalue

Let $e=\mathbf1/\sqrt m$. In the orthonormal basis $[e,Q]$, the actual kernel has blocks

$$
\begin{pmatrix}a&b_{\rm vec}^\top\\b_{\rm vec}&C\end{pmatrix},
\qquad a=\|B_\gamma^\top e\|^2,\quad
b_{\rm vec}=Q^\top B_\gamma(B_\gamma^\top e),\quad C=Q^\top K_\gamma Q.
$$

The mean-direction Rayleigh quotient gives $\ell_\gamma=a$. With $c=\beta_1+\delta_{\gamma,p}$ and $b=\|b_{\rm vec}\|$, the upper matrix bound for $C$ gives

$$
L_\gamma=\frac{a+c+\sqrt{(a-c)^2+4b^2}}2\ge\lambda_1(K_\gamma).
$$

Computing $a,b$ uses the finite feature matrix but not its eigenvalues or any measured correction between matrices. This use of the actual finite mean block is part of the bound, not independent validation. The actual SVD is separate and used only to check and display the intervals.

### Relation to gamma and high frequencies

For a zero-mean sample vector $u$,

$$
\frac1{hm}\int\left|\sum_a u_a\tanh(\gamma(c-x_a))\right|^2dc
=\frac2{\pi hm}\int
\frac{M_\gamma(\omega)^2}{\omega^2}
\left|\sum_a u_a e^{-i\omega x_a}\right|^2d\omega.
$$

This is the quadratic form of $H_\gamma$. All gamma dependence of this integral is in $M_\gamma^2$. At frequencies much smaller than gamma the multiplier is near one; at frequencies much larger than gamma it is exponentially small. Raising gamma increases the integral for every fixed vector. Min–max therefore proves that every ordered eigenvalue of $H_\gamma$ increases, without fixing or identifying its eigenvectors.

The finite-network statement retains the halo removal and lattice corrections, then bounds their uncomputed remainder. Thus the Fourier calculation is connected to actual finite eigenvalues by a proved interval. It does not merely restate that the finite feature matrix changes. Nonetheless the corrected matrix's eigenvalues still require calculation; no closed scalar expression in rank and gamma has been proved here. Nor does monotonicity of $H_\gamma$ alone prove monotonicity of every ratio of the full finite kernel.

The leading kernel eigenvalue is always between 1 and $W+1$ because the bias supplies a unit Rayleigh contribution and the bounded features bound the trace. Its normalization cannot grow without bound with gamma. The sharper bounds retain its actual finite mean/coupling dependence. The displayed intervals themselves establish the large ratio differences between the tested regimes.

### Numerical verification

Compute the whole-line integral through a square-root quadrature factor and use its singular-vector basis. Form the corrected matrix there as a diagonal integral spectrum plus the explicit alias matrix minus exterior-feature outer products. This reduces cancellation compared with first forming the full Gram matrix.

Increasing quadrature order from 10 to 16 and the integration padding from $20/\gamma$ to $24/\gamma$ changes the plotted corrected eigenvalues by at most $5.8\times10^{-8}$ relative at the checked slopes; that worst value occurs for the smallest gamma-4 modes. Independent actual-feature SVD drivers agree within $7.2\times10^{-12}$ relative at these ranks. The separately assembled centered finite matrix agrees in operator norm with the corrected construction to roughly $10^{-13}$, consistent with FP64 arithmetic and quadrature. An additional nonuniform-sample check passes.

All displayed ratio inequalities pass numerically. The mathematical theorem is an exact-arithmetic statement. These are checked FP64 evaluations, not interval-certified values: analytic truncation allowances do not include floating-point or quadrature-rounding uncertainty. A plotted raw upper eigenvalue may differ from an equal actual value by last-digit rounding.

## Conclusions

At the tested geometry, the single-gamma construction gives nontrivial two-sided intervals for the selected finite eigenvalues and ratios across gamma 4–64. Its lower bound needs the omitted-center correction; explicitly retaining Poisson terms fixes the large-gamma weakness of the previous grid-error estimate.

## Open questions

- A general closed rank/gamma formula or universal monotonicity theorem for the finite normalized ratios is not established by these intervals.
- Rigorous numerical enclosures would be required to promote the evaluated tiny eigenvalue bounds to interval-certified numerical statements.
