# Finite-gamma eigenvalue ratio lower bound — data-obvious for the tested geometry

## TL;DR

- A fixed reference at gamma 8 produces useful lower bounds on the actual ratios of ranks 12, 20, and 32 throughout a sweep to gamma 128.
- At gamma 64, the lower bounds are 77%, 64%, and 41% of the respective actual ratios. All tested bounds remain below the actual ratios.
- These are FP64 evaluations of a proved inequality, with corrections measured from the actual finite kernel. They are not interval-certified or predictions made from gamma alone.

## Question

Does the finite-change theorem capture the increase of $\lambda_i(\gamma)/\lambda_1(\gamma)$ when the reference eigenspace stays fixed, including the finite-halo and discrete-center corrections?

## Experiment design

The normalized feature matrix has entries $(B_\gamma)_{aj}=m^{-1/2}\tanh(\gamma(x_a-c_j))$ and a bias column $m^{-1/2}\mathbf 1$. The actual kernel is $K_\gamma=B_\gamma B_\gamma^\top$. Geometry matches the earlier matrix investigation: 128 interior intervals with spacing $h=1/64$, 129 interior centers, 12 halo centers per side, and 263 uniform samples in $[-1,1]$. There are 153 tanh neurons plus the bias.

Gamma is sampled at 83 distinct values from 8 to 128, including 16, 32, and 64. Eigenvalues are sorted in descending order. The displayed ranks are 12, 20, and 32. Every point uses the same reference gamma $\gamma_0=8$ and its same leading eigenspace. No eigenvector matching, refitting to the sweep, target function, or optimization trajectory is involved. Actual eigenvalues are computed by squaring singular values of the rectangular feature matrix, avoiding Gram-matrix eigenvalue cancellation.

The spatial decomposition is exact when its correction is retained:

$$K_\gamma=K_\gamma^{(0)}+E_\gamma,\qquad (K_\gamma^{(0)})_{ab}=\frac{154}{m}-\frac{2}{hm}(x_a-x_b)\coth(\gamma(x_a-x_b)).$$

The diagonal limit of the final factor is $1/\gamma$. The correction $E_\gamma$ contains the finite-halo and discrete-center differences established in the earlier spatial decomposition.

For each rank $i$, let $U_i$ contain the first $i$ orthonormal eigenvectors of $K_8$. Define

$$C_i(\gamma)=\operatorname{diag}(\lambda_1(8),\ldots,\lambda_i(8))+U_i^\top(K_\gamma^{(0)}-K_8^{(0)})U_i,$$

$$R_i(\gamma)=U_i^\top(E_\gamma-E_8)U_i,\qquad \varepsilon_i(\gamma)=\left\|C_i^{-1/2}R_iC_i^{-1/2}\right\|_2.$$

We use the absolute row-sum upper bound $L_\gamma=\max_a\sum_b|(K_\gamma)_{ab}|\ge\lambda_1(\gamma)$, with no new eigenvalues in its calculation. The plotted lower bound is

$$\ell_i(\gamma)=\frac{[1-\varepsilon_i(\gamma)]_+\lambda_{\min}(C_i(\gamma))}{L_\gamma}\le\frac{\lambda_i(\gamma)}{\lambda_1(\gamma)}.$$

The finite kernel at the new gamma is used to evaluate $R_i$ and $L_\gamma$. Its eigenvalues are used only for comparison and validation. This is a measured-correction test of the theorem, not an independent forecast that avoids constructing the new kernel.

Code & data:

- Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_bound.py`.
- Data and configuration: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/data/finite_ratio_bound.json`.
- Figure: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_bound.png`.

## Results

At gamma 64, with the reference still fixed at gamma 8:

| Rank | Initial ratio at gamma 8 | Lower bound at gamma 64 | Actual ratio at gamma 64 | Bound / actual |
|---|---:|---:|---:|---:|
| 12 | $4.247\times10^{-4}$ | $4.211\times10^{-3}$ | $5.468\times10^{-3}$ | 0.770 |
| 20 | $4.479\times10^{-6}$ | $1.048\times10^{-3}$ | $1.635\times10^{-3}$ | 0.641 |
| 32 | $4.934\times10^{-9}$ | $1.930\times10^{-4}$ | $4.670\times10^{-4}$ | 0.413 |

Across the entire sweep, the lower bound is at least 77.0%, 64.0%, and 40.8% of the actual ratio for the respective ranks. The curves flatten at larger gamma along with the actual ratios. At the reference itself, the bound is about 82.5% of the actual ratio because the row-sum bound on the largest eigenvalue is conservative; this initial gap is not a halo correction.

### Figures

- **Finite ratio lower bound:** three columns for eigenvalue ranks 12, 20, and 32. The upper row plots the actual ratio in blue, the lower bound in dashed green, and the starting ratio in dotted gray, against gamma on logarithmic axes. Upper-row vertical limits differ by rank. The lower row plots bound divided by actual on a common linear scale from zero to one. A lower bound above the horizontal starting ratio establishes an improvement over the starting ratio, subject to the stated floating-point limitation.

## Additional details: why the inequality is valid

The gamma dependence in the spatial difference has the all-vector Fourier identity

$$u^\top(K_\gamma^{(0)}-K_8^{(0)})u=\frac{2}{\pi hm}\int_{\mathbb R}\frac{M_\gamma(\omega)^2-M_8(\omega)^2}{\omega^2}\left|\sum_a u_a e^{-i\omega x_a}\right|^2\,d\omega,$$

where $M_\gamma(\omega)=z/\sinh z$ and $z=\pi|\omega|/(2\gamma)$. This finite difference is valid for every real sample vector $u$, including vectors with nonzero mean. At zero frequency the multiplier quotient has the finite limit $\pi^2(8^{-2}-\gamma^{-2})/12$. For gamma at least 8 the integrand is nonnegative, so the matrix difference is positive semidefinite. Since $\lambda_i(8)>0$ for these ranks, $C_i$ is positive definite.

By definition of $\varepsilon_i$, $R_i\succeq-\varepsilon_i C_i$. Also $U_i^\top K_\gamma U_i=C_i+R_i$. The min–max principle and positive semidefiniteness of $K_\gamma$ give

$$\lambda_i(K_\gamma)\ge\lambda_{\min}(U_i^\top K_\gamma U_i)\ge[1-\varepsilon_i]_+\lambda_{\min}(C_i).$$

Dividing by the upper bound $L_\gamma$ proves the displayed ratio bound. New eigenvectors may rotate or cross; they need not remain in the old leading eigenspace. The eigenvalues of $C_i$ still require a small matrix calculation. The result is not a universal assertion that every normalized eigenvalue increases with gamma.

Implementation checks compared the whitened correction norm with a separate generalized eigenvalue calculation, checked the exact compressed-matrix reconstruction, and checked both min–max inequalities at every point. All passed; correction-norm calculations differed by at most $2.1\times10^{-13}$. These checks validate the FP64 implementation, not interval enclosures of the reported bounds.

## Conclusions

On this fixed geometry and these three ranks, the evaluated bound captures the large increase of normalized eigenvalues while remaining within a factor of 2.46 of the actual ratios. No training was executed.

## Open questions

How useful is the bound across other widths, halo sizes, sample grids, and deeper eigenvalue ranks? Can a sufficiently tight analytic correction bound replace evaluation of the finite kernel? Neither question is settled by this plot.

## Follow-up: four widths and four reference gammas

Sam requested a 4-by-4 grid with starting gamma 16, 8, 4, and 2 down the rows; $N=64,128,256,512$ across columns; and four colored rank fractions $i=N/32,N/16,N/8,N/4$ in each panel. The rank sets are $(2,4,8,16)$, $(4,8,16,32)$, $(8,16,32,64)$, and $(16,32,64,128)$ respectively. Solid curves show actual ratios, and matching dashed curves show lower bounds. A separate figure shows bound divided by actual in the same layout. The earlier single-width figure is preserved.

The halo remains $\lceil\sqrt{N}\rceil$ centers per side, and sample counts are the first prime larger than $2N$: 131, 263, 521, and 1031. Thus the actual tanh-neuron counts are 81, 153, 289, and 559. Each row sweeps from its reference gamma to 128 on the applicable part of a 61-point logarithmic gamma grid. These are fixed ranks per width, not target-dependent quantities.

At small starting gamma and larger ranks, some reference singular values are below reliable FP64 resolution. The computed orthonormal reference vectors are still legitimate trial vectors for min–max; the compressed baseline is evaluated as $(U_i^\top B_{\gamma_0})(U_i^\top B_{\gamma_0})^\top$ rather than replacing it with a numerically tiny diagonal. This equals the diagonal baseline for exact reference eigenvectors, and the min–max argument remains valid for any orthonormal trial basis. No claim is made that numerically unresolved starting eigenvectors are individually identified.

Actual eigenvalues are hidden when their singular value is at most $10\epsilon_{\rm mach}\max(m,\text{feature count})\sigma_1$. Bounds are hidden when the smallest eigenvalue of the compressed main matrix is below a conservative cancellation threshold, when two correction-norm calculations disagree, or when the numerical min–max check fails. The thresholds are numerical screens, not interval error bounds. There are 232 unresolved compressed-matrix evaluations among 2944 requested rank/reference/width/gamma combinations; the other 2712 have positive evaluated bounds. The actual spectrum has 92 below-resolution evaluations, counted across reference rows. Missing segments are not zero bounds.

The broader grid shows that tightness deteriorates at larger fractional ranks, larger widths, and smaller starting gamma. For example, at gamma 128 the $i=N/4$ bound/actual ratio is approximately 0.815 for $(N,\gamma_0)=(64,16)$, 0.0653 for $(512,16)$, and 0.0008 for $(512,8)$. Therefore the tightness of the original three selected ranks does not extend uniformly to this larger grid.

Additional code & data:

- Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_width_grid.py`.
- Data: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/data/finite_ratio_width_grid.json`.
- Ratio figure: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_width_grid.png`.
- Tightness figure: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_width_grid_tightness.png`.

## Follow-up: doubling and explicit starting/final-gamma comparisons

The next requested presentation retains the same four widths, sample counts, halo convention, and fractional ranks. It makes the starting and final gammas explicit rather than treating the starting eigenspace as an incidental choice.

The doubling figure has widths as columns. Its horizontal axis is the starting gamma $\gamma_a\in[2,64]$, and every point compares the actual final ratio $\lambda_i(2\gamma_a)/\lambda_1(2\gamma_a)$ with the lower bound built at $\gamma_a$. The upper row overlays solid actual and dashed lower-bound curves; the lower row plots their quotient. Each of the 51 logarithmically spaced starting gammas is marked by a dot. No smoothing or curve fitting is applied. The reference eigenspace is recomputed at each starting gamma.

Two 4-by-4 heatmaps have columns for width and rows for $i=N/32,N/16,N/8,N/4$. Within each panel, horizontal position is starting gamma from $\{2,4,8,16,32,64\}$ and vertical position is final gamma from $\{4,8,16,32,64,128\}$. Only comparisons satisfying $\gamma_b\ge2\gamma_a$ are evaluated. The first heatmap shows lower bound divided by actual final ratio on a linear zero-to-one color scale. The second shows lower bound divided by actual starting ratio on a logarithmic color scale; values above one establish improvement under the theorem, with the same FP64 qualification as earlier figures. Values above one are bold. White cells are comparisons outside the selected domain; gray cells are numerically unresolved.

The starting-subspace screen is stricter than in the earlier width grid: both the starting singular value $\sigma_i$ and the cutoff gap $\sigma_i-\sigma_{i+1}$ must exceed $10\epsilon_{\rm mach}\max(m,\text{feature count})\sigma_1$. This excludes the arbitrary numerical starting directions identified in the subsequent audit. The compressed-matrix, independent correction-norm, and min–max checks are retained. Of 1056 distinct rank/width/gamma-pair evaluations, 947 give positive evaluated bounds and 109 have unresolved starting subspaces. The heatmaps contain 281 positive and 55 unresolved cells; the doubling curves contain 752 positive and 64 unresolved evaluations, with some comparisons shared between the two presentations. All admitted bounds passed the numerical inequality checks. These screens are not interval certification.

The positive evaluated bounds range from about 1.2% to 83.6% of the actual final ratios. Some cells with loose relative accuracy still establish large increases over their starting ratios. The figures separate those two questions without maximizing over starting gammas or selecting a favorable reference after the calculation.

Additional code & data:

- Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_transitions.py`.
- Raw values and statuses: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/data/finite_ratio_transitions.json`.
- Doubling figure: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_doubling.png`.
- Tightness heatmap: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_transition_tightness.png`.
- Improvement heatmap: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/finite_ratio_transition_gain.png`.

## Presentation extracts: eigenvalues versus target coefficients, and actual kernels

The saved original mixed-sine component comparison was replotted as a two-panel step-count figure. Both panels retain the actual spectral GD count. The left freezes $p_i=|u_i^\top y|^2/\|y\|^2$ at gamma 8 and changes normalized eigenvalues; the right freezes normalized eigenvalues at gamma 8 and changes $p_i$. These are the original saved arrays, paired by descending eigenvalue rank, with their unresolved-direction ambiguity shading. The threshold remains 1% relative L2, the initial readout is zero, and the learning rate is $0.5/\lambda_1(\gamma)$. No training or spectral sweep was repeated for this extract.

The eigenvalue-only change accounts for the main orders-of-magnitude change in this threshold-crossing count. This does not establish that target alignment is irrelevant at finer tolerances: the original fixed-budget squared-error comparison separates substantially at large gamma. Generality across targets, reference gammas, and matching conventions remains a separate question.

A companion image shows the actual finite $B_\gamma B_\gamma^\top$ matrices at gamma 4, 8, 16, and 64, on the same sample and center geometry and a shared color scale. These are direct finite sums, including bias and training normalization. The images illustrate matrix changes; their appearance alone does not establish an eigenvalue-ratio or training-time bound.

Additional code & data:

- Source: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/presentation_checks.py`.
- Existing counterfactual data: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/data/default.json`.
- Step-count comparison: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/eigenvalue_vs_alignment_steps.png`.
- Actual-kernel image: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/kernel_gamma_comparison.png`.
- Actual-kernel arrays: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/data/kernel_gamma_comparison.npz`.
