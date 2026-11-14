# D59 — QUILL accuracy scaling: 20 smooth functions

Status: completed November 12, 2026. Fifteen fixed-geometry least-squares factorizations produce 300 measured function/width pairs. All scalar models, configurations, source hashes, and evaluation data are saved.

## Main result

All 20 functions, including the nonanalytic smooth compact bump, reach relative L2 below 1e-12 at width 1,617, including halos. The worst dense-grid relative L2 at that width is 4.46e-13. The compact bump improves more slowly than the analytic targets and reaches 8.75e-14 there. These are independent-grid FP64 measurements, not continuous-domain error certificates.

## Question and design

Use the standard sample-fitted QUILL recipe on the smooth portion of the 24-function scale-acquisition suite. Include 19 analytic targets and the C-infinity compact bump. Exclude the two cusps and two discontinuous targets. Fit uniform tanh geometry with lambda=.25, R=max(20,ceil(sqrt(N))), an explicit bias column, and rank-revealing `gelsd` least squares with cutoff 1e-14. N ranges from 8 to 2,048 intervals, giving widths 49–2,141. All 20 columns share the same factorization at each resolution.

See [the full protocol](../../../experiments/expD59_quill_smooth_scaling/README.md) and [configuration](../../../experiments/expD59_quill_smooth_scaling/config.yaml).

## Selected measurements

| N | Total width | Functions below 1e-12 | Largest relative L2 |
|---:|---:|---:|---:|
| 128 | 169 | 13/20 | 5.93e-5 |
| 256 | 297 | 17/20 | 1.26e-6 |
| 512 | 559 | 18/20 | 9.76e-9 |
| 1024 | 1089 | 19/20 | 5.35e-12 |
| 1536 | 1617 | 20/20 | 4.46e-13 |
| 2048 | 2141 | 20/20 | 7.90e-13 |

The compact bump is the largest-error target in the first four rows. Degree-12 Legendre is largest in the final two. More width does not monotonically improve the floating-point floor.

## Figures and records

- [All-function scaling figure](figures/quill_scaling.png), with a [PDF export](figures/quill_scaling.pdf).
- `scaling_scalar.csv`: all 300 primary scalar-deployment measurements.
- `scaling.csv`: original multi-output evaluation of the same fitted coefficients.
- `best_scalar_observed.json`: per-function minima over the declared width grid; descriptive, not a separate model-selection validation study.
- `models/`: actual exported model at each width and fit diagnostics.
- `evaluation_data.npz`: fixed normalization, dense-grid, random-grid, and feature/boundary probe data.
- `manifest.json`, `scalar_audit_provenance.json`, `figures/figure_provenance.json`: configurations and hashes.

## Verification and limitations

The scalar evaluator was checked against independent scalar MLP objects using identical coefficients; saved-model reload preserves evaluated values. Separate versus shared solves agree on the well-resolved controls. The coarse bump's large readout mass exposes ordinary floating-point cancellation and is retained rather than hidden. Evaluation uses 65,536 held-out midpoint samples, random samples, and boundary/feature probes. Coarse half-grid comparisons and sampled maximum errors are stored separately.

This establishes the observed fixed-geometry least-squares accuracy. It does not validate the direct analytic contour encoder on Gamma's exterior halo, apply analytic convergence rates to the bump, or claim that a trained model finds these weights.

## Interpretation

The measured curves support high-accuracy construction across the requested smooth suite, with a slower resolution requirement for the smooth nonanalytic bump and finite-precision plateaus after resolution. Broader scientific conclusions remain for Sam to assess.
