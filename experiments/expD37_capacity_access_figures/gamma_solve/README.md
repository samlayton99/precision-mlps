# Gamma solve

Two interactive experiments: finite versus integrated tanh kernels, and the separate effects of normalized eigenvalues and target projections on frozen-readout GD. Open **Gamma Solve.app** on the laptop Desktop. It starts the viewer on the mini if needed, establishes a local SSH tunnel, and opens the browser. Code and saved arrays stay in this repository. It requires the existing `home` SSH connection/Tailscale; it is not an offline app.

On the mini, run `bash experiments/expD37_capacity_access_figures/gamma_solve/start.sh`, then open `http://127.0.0.1:8067`. The viewer uses the existing project environment and locally installed Plotly, with no external browser assets. Default geometry: N=128 interior intervals, m=263 training samples, gamma=16, gamma reference=8, zero readout. Gamma is swept over 0.25–128, including the reference exactly. The default fixed budget is one million updates. The default target is the note's mixed sine, not the older D36 mixture.

## Kernel normalization and integration bounds

Let $h=2/N$, $H=\lceil\sqrt N\rceil$, and $c_j=-1+jh$ for $j=-H,\ldots,N+H$. There are $W=N+1+2H$ neurons. Integration covers the union of their midpoint cells: $a=c_{\min}-h/2$, $b=c_{\max}+h/2$. Thus $(b-a)/h=W$ exactly. The two kernels are

$$k_{\mathrm{sum}}(x,x')=1+\sum_j\tanh(\gamma(x-c_j))\tanh(\gamma(x'-c_j)),$$
$$k_{\mathrm{integral}}(x,x')=1+\frac1h\int_a^b\tanh(\gamma(x-c))\tanh(\gamma(x'-c))\,dc.$$

The finite sum is the composite midpoint approximation to the integral divided by $h$. For $f(c)=\tanh(\gamma(x-c))\tanh(\gamma(x'-c))$, its absolute error is bounded by $(b-a)h\sup|f''|/24$. In particular, refinement at fixed physical gamma and fixed integration bounds gives second-order relative quadrature convergence. Small gamma*h supports close agreement; the agreement is not guaranteed at arbitrarily sharp transitions or large jitter. The integral replaces the discrete center distribution by a uniform one; it does not integrate over the training inputs and is not the note's two-input smoothing integral. Both the discrete sum and the continuum approximation use the exact same bias of one.

The app defaults to displaying the training normalization $K=k/m$. The checkbox removes this normalization from both heatmaps together. The spectrum always shows $\lambda_i/\lambda_{\max}$, unchanged by that scalar. Heatmaps include exterior points to show the halo; eigenvalues and target weights use only the $m$ training samples in $[-1,1]$. Heatmaps display at most 227 sample positions, while the spectral calculation uses all training points. Both heatmaps use the same color scale and actual coordinate axes, including nonuniform data positions. Rows increase from top to bottom and columns from left to right, so the main diagonal runs from top-left to bottom-right.

Center jitter and data jitter multiply fixed seeded random offsets by a continuously adjustable factor in $[0,1]$. Each offset is at most 0.49 of the original spacing, so points cannot exchange order. Training endpoints, centers at ±1, and extreme halo centers are pinned. The integral's density and limits remain fixed as centers jitter. Changing N changes both the spacing and the prescribed square-root halo. The data slider can select any integer m; the default avoids having all center locations coincide with training samples.

The interactive spectrum includes dotted ordinary least-squares fits of $\log_{10}(\lambda_i/\lambda_{\max})=a+bi$ against linear eigenvalue rank. Labels give the slope in decades per rank, the fitted adjacent-eigenvalue ratio $10^b$, and $R^2$ in log space. Both fits use the same ranks, requiring resolved integral eigenvalues and normalized values at least $10^{-16}$ in both spectra. At least three shared values are required. The fitted range is displayed and updates with geometry; these are descriptive exponential fits, not power-law exponents. Fit lines and labels are included in the viewer's PNG download.

## Spectral comparisons

Compute a rectangular SVD of $J=[\mathbf 1,\Phi]/\sqrt m$; do not diagonalize an explicitly rounded Gram matrix to obtain the discrete spectrum. With singular values $s_i$ and left singular vectors $u_i$, define $\lambda_i=s_i^2$, $p_i=|u_i^Ty|^2/\|y\|^2$, and $a_i=\eta_\gamma\lambda_i=\tfrac12(s_i/s_1)^2$. The squared relative error is $E_\gamma(n)^2=p_{\mathrm{null}}+\sum_i p_i\exp(2n\log(1-a_i))$. The omitted left-nullspace energy is measured directly as a projection residual, not by subtracting two nearly equal scalar energies.

- **a:** scatter of $p_i(\gamma)$ versus $a_i(\gamma)$, colored by gamma with viridis. Log axes display the range. Only points below the stated display limits are hidden; saved arrays retain them.
- **b:** actual error versus $\sum_i p_i(\gamma_0)(1-a_i(\gamma))^{2n}$.
- **c:** actual error versus $\sum_i p_i(\gamma)(1-a_i(\gamma_0))^{2n}$.
- **d/e:** the corresponding first integer step to 1% relative L2, found by doubling and integer bisection up to $10^{15}$. A missing crossing is budget-censored, not a proof of impossibility. The threshold for squared error is $10^{-4}$.

The bias and nullspace are included. Fixed eigenvalues means fixed **normalized eigenvalues**, since the step is $0.5/\lambda_{\max}$. These swapped curves are synthetic comparisons, not independently executed optimizers and not a unique causal decomposition. The actual curve uses the same discrete spectral GD identity as the note but is a numerical prediction from the original feature matrix, not a certified Fourier approximation.

The default pairs components by descending eigenvalue rank. The alternate mode follows adjacent eigenvectors by maximum total squared overlap (one-to-one assignment), moving outward from the reference gamma. Results can depend on matching. At equal eigenvalues individual eigenvectors are not unique; subspaces are the appropriate object. Singular directions below $s_i/s_1=10^{-12}$ are flagged as unresolved for interpreting eigenvector changes, but their small singular values are retained in the actual decay calculation. The omitted nullspace is an aggregate zero-rate component. Shaded counterfactual ranges place the unresolved block's total target weight on its fastest or slowest assigned rate; they do not claim rigorous floating-point error certification. Reported individual swapped lines in a broad band should not be treated as uniquely established alignment effects.

The local continuity check compares eigenvector perturbation derivatives with sign-aligned centered finite differences at gamma reference for the first 12 sufficiently isolated modes. It includes the omitted nullspace in the derivative formula. This validates the local calculation, not differentiability through all crossings. The adjacent-overlap display shows the first 24 reference-labeled components.

## Verification and saved outputs

Run `.venv/bin/python -m pytest -q experiments/expD37_capacity_access_figures/gamma_solve/test_gamma_solve.py`. Tests cover ordering, midpoint-cell mass, independent adaptive quadrature, midpoint refinement, executed coefficient GD, reference equality of counterfactuals, matching invariance of the actual curve, integer crossing, ambiguity bounds, and local eigenvector derivatives.

Run `.venv/bin/python experiments/expD37_capacity_access_figures/gamma_solve/server.py --snapshot` to save the default arrays and two PNGs in `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/`. The app's save button adds a configuration-keyed NPZ and JSON under `data/`; identical configurations overwrite their own files. PNG downloads go to the laptop browser's normal download location. No sweep runs are launched by merely opening a PNG.

Files: `core.py` is the numerical calculation; `server.py` serves the local API and assets; `index.html` is the viewer; `figures.py` saves the static previews; `start.sh` starts the local server; `make_laptop_app.py` builds the Desktop launcher. No existing experiments are modified.
