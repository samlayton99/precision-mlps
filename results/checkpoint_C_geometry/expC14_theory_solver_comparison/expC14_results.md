# expC14: Theoretical versus existing readout solver — draft-pending-Sam

**Follow-up:** [solver_resolution.md](solver_resolution.md) separates scaling from cutoff selection, reports 138 matched-cutoff comparisons, and derives solver-independent posterior error bounds for the returned networks. The initial experiment below changes both scaling and cutoff. Its negative result must not be interpreted as showing that reference scaling is necessary or that ordinary SVD is inadequate.

## TL;DR

- The curves do not coincide for this implementation of the theorem's recovery recipe. At $N=1024$, existing-solver relative $L^2$ errors are $1.24\times10^{-14}$ to $4.28\times10^{-13}$; theoretical-procedure errors are $2.67\times10^{-12}$ to $4.57\times10^{-10}$.
- The theoretical cutoff discards every singular direction at $N=128$. Its analytic component controls the small-$N$ regime; the conservative numerical allowances dominate at $N\ge1024$.
- Five implementation tests pass. Doubling the independent evaluation grid changes any reported error by at most 0.159%. This is an algorithmic comparison, not an interval-certified continuous-accuracy result.

## Question

Does the Appendix A.3 reference-scaled, bound-thresholded SVD procedure deliver the same measured approximation accuracy as the existing FP64 readout solver on matched networks and samples?

## Experiment design

The authoritative reference is the supplied `QI_MLPs___ICLR_2027_Submission (18).pdf`, Appendix A.1/A.3, especially equations (A.1), (A.3), (A.51), and (A.55)–(A.68). The locally matching construction appendix source was also checked. The experiment uses $f(x)=e^x$, $\sin(4\pi x)$, $\sin(24\pi x)$, $\sin(8\pi(x+1)^2)$, $\sin(2\pi x)+10^{-3}\sin(40\pi x)$, and $e^{-8(x-0.2)^2}\sin(8\pi x)$ on $[-1,1]$.

Both arms use $N\in\{128,256,512,1024,2048\}$, $h=2/N$, $R=\lceil\sqrt N\rceil$, $W=N+2R+1$, $\lambda=0.25$, $\gamma=\lambda/h$, and centers $x_j=-1+jh$ for $j=-R,\ldots,N+R$. Set $\delta=1$, $\Delta=0.25$, and $a_H=1$. All configurations satisfy (A.3). Training uses $M=8N+1$ uniform core samples; $s=8,p=1,\varrho=2$ satisfies (A.55). The main grid has 32,768 uniform midpoints, and the validation grid has 65,536 distinct uniform midpoints. Error is the discrete ratio $\|\hat f-f\|_2/\|f\|_2$ on each grid, not a certified continuous norm or worst-case error.

**Actual arm.** The existing main width-figure solver: single-target `scipy.linalg.lstsq(A,y,cond=2**-52,lapack_driver="gelsd")`, with $A_{ij}=\tanh(\gamma(x_i-x_j))$ and a final bias column. The solver is reproduced on the shared theoretical geometry/sample grid, not the published fixed-halo configuration.

**Theory arm.** Build the coordinate envelopes from (A.56), adding boundary corrections at the outermost halo slots, and include the bias envelope. Set $D=\operatorname{diag}\sqrt\alpha$, $L=\sqrt{\sum_j\alpha_j}$, $S=BL$, $F=AD/\sqrt M$, and $y_X=y/\sqrt M$. Set $\overline E_W$ to the explicit bound $E_{\rm rep}$ in (A.51), rather than a measured residual. Use a thin SVD, explicit pairwise projections, and retain exactly $\sigma_i>\tau$, with $\tau$ rounded upward from $(\overline E_W+a_WS+b_W)/S$. Native weights are exported using explicit pairwise products/sums and $D$.

The analytic target bounds are $e^{1+\Delta+\delta}$, $\cosh(4\pi\delta)$, $\cosh(24\pi\delta)$, $\cosh(16\pi(2+\Delta+\delta)\delta)$, $\cosh(2\pi\delta)+10^{-3}\cosh(40\pi\delta)$, and $e^{8\delta^2}\cosh(8\pi\delta)$, respectively. All are valid on a rectangle containing the analytic tube. They are fixed across $N$ and not fitted to observed errors.

The (A.63) numerical allowances combine observed Frobenius SVD residuals and orthogonality defects with standard $\gamma_k$ dot-product rounding allowances, formation errors, and pairwise projection errors. Exact formulas and all components are saved in code/JSON. These are deliberately conservative choices within the numerical-bound construction, not uniquely mandated constants or optimized bounds. Both arms use correctly rounded FP64 MPFR tanh values and pairwise output summation. Dyadic grids make preactivation arithmetic exact. Target samples and analytic constants are prepared offline at 160 bits, then rounded; the readout solve and predictions remain FP64. Constants are checked independently at higher precision in tests.

**Code & data**

- Code/config/tests: `experiments/expC14_theory_solver_comparison/`
- Existing solver/style references: `experiments/expC09_bandwidth_figures/convergence.py` and `combined.py`
- Data: `results/checkpoint_C_geometry/expC14_theory_solver_comparison/N*.json`, `coefficients_N*.npz`, `summary.csv`, `manifest.json`, `validation.json`
- Figure: `results/checkpoint_C_geometry/expC14_theory_solver_comparison/comparison.{png,pdf,svg}`
- Reproduce: `MPLCONFIGDIR=/private/tmp/precisionmlps-c14-mpl VECLIB_MAXIMUM_THREADS=4 .venv/bin/python experiments/expC14_theory_solver_comparison/run.py --output /absolute/path/to/new/output`
- Tests: `.venv/bin/python -m pytest -q experiments/expC14_theory_solver_comparison/test_comparison.py`

## Results

Measured relative $L^2$ error at $N=1024$:

| Function | Actual | Theory |
|---|---:|---:|
| chirp | 1.954e-13 | 4.044e-10 |
| exp | 1.236e-14 | 2.791e-11 |
| packet | 1.042e-13 | 2.672e-12 |
| sine24 | 4.277e-13 | 4.569e-10 |
| sine4 | 6.665e-14 | 2.771e-10 |
| weak_high_frequency | 6.808e-14 | 1.385e-10 |

At $N=128$, the analytic cutoff is approximately $4.075$ and the theoretical solver retains no directions, producing relative error one for every target. At $N=512$, the analytic cutoff is approximately $1.75\times10^{-5}$ and dominates the numerical term. At $N=1024$, the analytic component is approximately $1.27\times10^{-12}$ but the matrix-error allowance is approximately $1.10\times10^{-9}$. At $N=2048$, this allowance grows to approximately $3.89\times10^{-9}$, explaining why the theoretical curves turn upward. The cutoff also includes a target-dependent label/projection term. The saved cutoffs, ranks, and coefficient norms allow direct inspection.

### Figures

- **Comparison:** One axis with $N$ horizontally and relative $L^2$ error vertically; six function colors, solid actual curves, dashed theoretical curves. Font family, palette, line weights, and axis treatment follow the paper's Figure 2 code. Dashed lines plot measured network errors, not theoretical upper bounds.

## Additional details

Independent read-only review identified an export-rounding bound that initially used the norm of $V$ where entrywise absolute products were needed. It was corrected before the recorded run. The final implementation uses the explicit absolute product mass. Tests cover independent high-precision envelope formulas and boundary indexing, admissibility, exact versus truncated reparameterization, cutoff conditions, nonzero coefficient recovery/export, correctly rounded features, and target-oracle consistency. All recorded geometry, cutoff, transformed-norm, sample-error-bound, and coefficient-bound checks pass.

This experiment is not a complete interval certificate: constants/oracle values and stored-array norms are not all enclosed by directed-rounding interval arithmetic. In addition, the valid $s=8,p=1$ sampling choice yields a very loose continuous-error-transfer term ($e_p^{\rm an}=B+T$). Measured accuracy must not be described as theorem-certified near-machine continuous precision. Numerical allowances can potentially be tightened substantially; the observed gap is for the explicit choices above, not proof that every admissible realization of the theorem performs poorly. Changing $\delta$, the analytic witness bound, or the numerical bounds is a new declared comparison, not a post-hoc edit to these curves.

## Conclusions

This comparison does not provide the hoped-for empirical equivalence. With the explicit witness bound and conservative numerical allowances used here, the theoretically specified solver is substantially less accurate than the existing fitting method. The data identify the cutoff as a concrete issue to investigate before claiming agreement.

## Open questions

Can sharper valid approximation/numerical bounds lower the theorem-compliant cutoff enough to recover the practical solver's accuracy? Alternatively, can the proof directly cover the practical cutoff without losing the claimed coefficient control?
