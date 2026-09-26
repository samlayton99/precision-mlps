# Note figure manifest

All PNGs use saved calculations except the kernel heatmap, which directly
recomputes B_gamma B_gamma^T at gamma=1,4,8,16 on the saved geometry.
No smoothing, fitting, new training, or change of experimental geometry was used. Thin joining lines
connect actual sample values. Source code is `note_figures.py`.

| Figure | What it establishes | What it does not establish |
|---|---|---|
| `ratios_versus_target_weights.png` | At 1% error in the default mixed-sine experiment, changing eigenvalue ratios while freezing rank-indexed target weights reproduces the main orders of magnitude in step-count improvement. | This is a rank-paired counterfactual, not eigenvector tracking or a theorem that target alignment is irrelevant. Shading represents unresolved-direction sensitivity, not a confidence interval. |
| `finite_kernels.png` | Actual finite kernels at gamma=1,4,8,16, on identical geometry and a shared viridis color scale. | Kernel images do not themselves establish improved normalized eigenvalues. |
| `doubling_ratios.png` | Direct ratios and lower bounds for gamma_a -> 2 gamma_a, at all four preselected width/rank configurations. Bottom row exposes looseness. | Not all lower bounds exceed starting ratios; the old-space restriction and denominator upper bound can be conservative. Corrections are measured using the new finite kernel. |
| `step_counts_and_capacity.png` | Finite-center integral spectra forecast actual finite-tanh spectral first-hit counts accurately, while explicit readouts already attain below 1% error. | These are not counts from executed training and are not an independent consequence of the ratio lower-bound theorem. No rigorous error enclosure is inferred from the observed prediction errors. |
| `predicted_learning_curves.png` | The entire finite-tanh and center-integral spectral curves agree closely until 1% acquisition. | Agreement does not certify unresolvable machine-precision tail dynamics. |

## Geometry and normalization

Default mixed-sine experiment: 128 interior intervals (h=1/64), 153 tanh
neurons plus a bias, 263 uniformly spaced samples on [-1,1], 12 halo centers
per side, center-cell interval [-1.1953125,1.1953125]. Target is
sin(2 pi x) + 0.5 sin(6 pi x) + 0.25 sin(10 pi x). B is divided by sqrt(m);
loss is half the mean squared error. Initial readout is zero. Both the finite
and center-integral predictions use eta=0.5/lambda_max(K_finite).

The width sweep's N denotes **interior intervals**, not total tanh neurons.
For N=64,128,256,512 the total neuron counts are 81,153,289,559 and m is
131,263,521,1031. Halo per side is ceil(sqrt(N)). Rank colors use
i=N/32,N/16,N/8,N/4. Caption this convention explicitly to avoid conflict
with theoretical notation in which N may denote total neurons.

## Step-count evidence

The center integral is evaluated with 16 Gauss-Legendre points per center
cell. Results agree closely with the independently saved 8-point calculation.
These approximate the finite center sum; they are not the collaborator's
periodic Fourier construction. Eigenvalues are obtained from feature SVDs,
which resolve small rates more accurately than diagonalizing an explicitly
formed Gram matrix. Error is evaluated by the exact fixed-matrix spectral
formula, with log1p used for tiny learning rates. Integer first-hit counts
are found by bracketing and binary search; no fitted trajectory rates occur.

| Gamma | Actual finite spectral steps | Integral prediction | Prediction error | Explicit readout error |
|---|---:|---:|---:|---:|
| 4 | 1,007,616,497,993 | 1,009,014,477,487 | +0.13874% | 5.85001e-04 |
| 8 | 19,697,563 | 19,698,115 | +0.00280% | 2.92449e-06 |
| 16 | 66,209 | 66,209 | +0.00000% | 1.55713e-07 |
| 64 | 18,272 | 18,219 | -0.29006% | 5.97120e-05 |

At gamma 4, a saved positive-rate band with eta*lambda in (1e-14,1e-11]
contains 0.02735465123 of target squared norm. It yields a necessary-step
bound of 280,573,583,534 for 1% error. This is separate from eigenvalue-ratio
LOWER bounds: a lower bound on a new eigenvalue does not yield a necessary
training delay. The saved explicit readout has norm 4625.807 and uses 30
resolved modes; its relative error is 5.85000788e-4. Main count is 1.0076e12.

## Numerical audit and limitations

An independent direct feature construction with the alternate `gesdd` SVD
recomputes all four finite spectral counts and all four explicit readout
errors. Results and relative differences are in `figure_data.json`. Every
resolved transition record was also checked for lower<=actual and arithmetic
consistency of its tightness/gain metrics. This audits the numbers; it is
not interval certification of rounding errors.

Unresolved starting eigenspaces are omitted from lower-bound curves. The
saved filter requires both the singular value and cutoff gap to exceed
10*eps*max(shape)*s1; additional compressed-matrix/correction consistency
checks are applied. Blank lower-bound regions are not zero bounds.

## Sources

- `../data/step_prediction_check.json` and `.npz`: step counts and learning curves.
- `../data/slow_band_check.json`: explicit readout errors and necessary-step bands.
- `../data/finite_ratio_transitions.json`: all doubling ratios and bounds.
- `../../data/default.json`: rank-paired target-weight controls, gamma0=8.
- `kernel_gamma_comparison.npz`: actual kernel matrices at gamma=1,4,8,16, recomputed from the sample/center arrays in `../data/kernel_gamma_comparison.npz`. The earlier file is unchanged.
- `kernel_audit.json`: direct symmetry and 1/m normalization checks for these four matrices.
