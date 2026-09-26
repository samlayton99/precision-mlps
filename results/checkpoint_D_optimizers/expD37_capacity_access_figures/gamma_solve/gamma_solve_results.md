# Gamma solve — implemented; interpretation pending Sam

## TL;DR

- The interactive viewer compares the finite tanh kernel against an integrated uniform-center kernel, with shared scales and an explicit training-domain box.
- The second view implements all five requested spectral comparisons, with configurable target, reference gamma, fixed step count, and component matching.
- Default calculations and PNGs are saved. Fourteen numerical tests pass; browser interactions and PNG export were exercised.

## Question / hypothesis

How closely does a uniform-center integral approximate the finite feature kernel? When gamma changes, how do the measured GD curves compare with curves that hold either target projections or normalized eigenvalues at a reference value?

## Experiment design

The default has $N=128$ interior intervals, $h=2/N$, $H=\lceil\sqrt N\rceil$ halo centers on each side, and $m=263$ training samples. Centers are $c_j=-1+jh$, $j=-H,\ldots,N+H$, so $W=N+1+2H$. Jitter scales fixed random offsets, bounded by 0.49 original spacings, with endpoints pinned to preserve ordering and support. The physical model includes the bias feature.

The discrete kernel is $K_{ii'}=[1+\sum_j\tanh(\gamma(x_i-c_j))\tanh(\gamma(x_{i'}-c_j))]/m$. The continuous comparison replaces the sum with $h^{-1}\int_a^b$, with $a=c_{\min}-h/2$, $b=c_{\max}+h/2$. These limits and density satisfy $(b-a)/h=W$ and make the sum the composite midpoint rule for the integral. The same bias and empirical normalization apply to both. Heatmaps extend into the halo; spectra and target weights use only the training samples in $[-1,1]$.

The default target is $\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)$. The gamma sweep covers 0.25–128 with 41 logarithmic values and the exact reference $\gamma_0=8$ inserted. All spectral calculations use FP64. The original feature matrix is decomposed by rectangular SVD. With zero initial readout, $p_i=|u_i^Ty|^2/\|y\|^2$ and $a_i=\eta_\gamma\lambda_i=\lambda_i/(2\lambda_{\max})$ give $E_\gamma(n)^2=\sum_i p_i(1-a_i)^{2n}$, including nullspace energy. The fixed-budget plots use $n=10^6$ by default; the app exposes this control. Step-count plots solve for the first integer crossing of $E\le0.01$ up to $10^{15}$ updates without executing training.

The default comparison pairs modes by descending eigenvalue rank. A second option matches adjacent eigendirections by maximum total squared overlap. A local finite-difference check at the reference gamma tests the eigenvector derivative formula on isolated modes. Unresolved-direction counterfactual ambiguity is shaded rather than interpreted as an established change of eigenvector alignment.

**Code & data**

- Source and usage: [gamma_solve README](../../../../experiments/expD37_capacity_access_figures/gamma_solve/README.md).
- Saved arrays: [default.npz](data/default.npz); configuration, comparisons and derivative checks: [default.json](data/default.json).
- Desktop launcher: [Gamma Solve.app](Gamma%20Solve.app), also installed on the laptop Desktop. Its calculations stay on the mini.

## Results

The requested numerical calculations and visualizations are available. No optimizer or gamma-barrier conclusion is asserted here; the component-swapping plots are ready for inspection.

### Figures

- [Kernel comparison](kernel_comparison.png): left finite sum, middle matched integral, right normalized eigenvalue spectra. Viridis heatmaps use the same scale; red outlines identify the training square.
- [Spectrum and alignment](spectrum_alignment.png): a shows target weights versus per-step rates colored by gamma; b/c compare fixed-budget squared errors with one component held at the reference; d/e show the associated steps to 1% relative L2. Black is the actual spectral prediction, blue fixes target weights, orange fixes normalized eigenvalues.

## Additional details

The integral is evaluated analytically, with numerically stable near-diagonal and small-gamma branches, and checked against independent adaptive quadrature. The midpoint convergence test refines a fixed physical integration interval. The GD identity is checked against an executed readout-coefficient trajectory. Reference equality, integer crossings, matching invariance of the actual curve, and counterfactual ambiguity bounds are tested separately. Browser checks exercised all five geometry controls, target and matching changes, the continuity panel, and PNG export.

These numerical predictions are not the technical note's interval-certified Fourier construction. Missing 1% crossings mean budget censoring. The counterfactual curves depend on mode matching, especially in unresolved or nearly degenerate directions, and do not form a unique causal decomposition.

## Conclusions

The viewer implements the requested comparisons with matched kernel normalization and explicit conventions for spectral pairing. Scientific interpretation remains pending Sam's review.

## Open questions

- Does the observed comparison persist across the two component-matching conventions and changes in reference gamma?
- Where does center discretization materially separate the finite kernel from its uniform-center integral?
