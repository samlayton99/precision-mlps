# Correctly rounded forward inference at width 1024 — data-obvious

## TL;DR

- Recomputed all 46 precision points with inputs, every model parameter and every forward operation rounded to the prescribed significand precision.
- The high-precision plateau remains near $10^{-13}$. Unrounded hidden-layer evaluation was not its sole cause.
- Coefficient recovery remains an offline FP64 SVD; the error metric is measured in FP64. This is a full forward-precision experiment, not a low-precision solver experiment.

## Question

Does the precision curve persist when the complete network forward evaluation uses the stated precision, including hidden-layer parameters and activation arguments?

## Experiment design

The target, total width, halo, sampling and bandwidth rule match the preceding experiment: $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$, $W=1024$, 24 halo centers per side, 4,801 fitting points, 8,001 evaluation points, and every integer $p=8,\ldots,53$. The same refined bandwidths use $\bar\omega=16\pi$ and $e_{\mathrm{tol}}=2^{1-p}$.

Let $Q_p$ denote round-to-nearest, ties-to-even at $p$ binary significand bits. Stored parameters are $\tilde c_j=Q_p(c_j)$, $\tilde\gamma=Q_p(\lambda/h)$, $\tilde w_j=Q_p(w_j)$ and $\tilde b=Q_p(b)$. Inputs are $\tilde x=Q_p(x)$. The centered-tanh forward pass computes

$$d_j=Q_p(\tilde x-\tilde c_j),\qquad z_j=Q_p(\tilde\gamma d_j),\qquad \phi_j=Q_p(\tanh(z_j)),$$
$$s_0=\tilde b,\qquad s_{j+1}=Q_p\left(s_j+Q_p(\tilde w_j\phi_j)\right).$$

Every operation uses MPFR at precision $p$, including correctly rounded tanh, separate multiplication and addition, and the same sequential accumulation at $p=53$. Saturated tanh values are returned as exactly $\pm1$ only beyond a conservative threshold where this equals the correctly rounded result. No fused multiply-add or FP64 intermediate product is used in inference. MPFR's exponent range is left unrestricted; this is a significand-precision model rather than a particular hardware exponent format. FP64 arrays serialize $p$-bit-representable values without adding numerical information.

The fitting matrix is regenerated using those same rounded parameters, inputs and feature operations. Labels are rounded to $p$ bits, the offline FP64 SVD uses cutoff $2^{1-p}$, and recovered coefficients are rounded before inference. The relative error compares the resulting output with the FP64 target at the original evaluation coordinates, so input-quantization error is included. The dashed reference keeps slope $-1$ in log-base-2 error and refits only its intercept over $p=16,\ldots,40$, giving $C=19.52428$ in $E=C2^{-p}$.

**Code & data:** evaluator `experiments/expC09_bandwidth_figures/strict_precision.py`, MPFR kernel `strict_arithmetic.c`, runner `strict_precision_law.py`, renderer `combined.py`, pinned dependency `requirements-strict.txt` (gmpy2 2.3.1, bundling MPFR 4.2.2), and checks `tests/test_strict_precision.py`. Install the pinned dependency in the Python environment or expose it through `PYTHONPATH`; run the strict runner and then the renderer. Source fingerprints, predictions, measurements, summaries, validation and stored model/output arrays are in `results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W1024_strict/data/`. The new figure is `results/checkpoint_C_geometry/expC09_bandwidth_figures/combined/figures/all_targets_chirp_precision_law_rule_W1024_strict.png`. The old figure and provenance are preserved in `combined/archive/precision_law_W1024_mixed_precision/` under the same experiment results directory.

## Results

| Working precision | Corrected relative $L^2$ error |
|---:|---:|
| 8 | $8.45\times10^{-2}$ |
| 24 | $1.30\times10^{-6}$ |
| 40 | $1.36\times10^{-11}$ |
| 53 | $9.29\times10^{-14}$ |

The descending region and eventual plateau remain. This experiment does not isolate the remaining contributions from coefficient recovery, finite geometry and evaluation rounding.

### Figures

- **Corrected three-panel PNG:** panels (a) and (b) retain their native FP64 measurements; panel (c) uses the fully rounded forward evaluation at all 46 precisions. The reference intercept is recomputed from the corrected curve. Existing layout and legend wording are retained.

## Additional details

Thirteen inference tests passed: feature and readout comparisons against independent 200-bit operation-by-operation oracles; exact agreement of separate-product/sequential-sum readout with native FP32; and a constructed $p=52$ case that detects incorrect double rounding through FP64. The 21 existing bandwidth and target checks also passed, totaling 34. All 46 measurements are newly calculated, finite and complete; each saved model's coefficients, centers, features and outputs are checked for $p$-bit representability. The resulting PNG was visually inspected.

## Conclusions

With every forward operation rounded to the requested precision, the measured precision curve still has a high-precision floor near $10^{-13}$.

## Open questions

The relative contributions of offline FP64 coefficient recovery and finite geometry to the remaining floor have not been isolated. Hardware-specific tanh approximations, exponent limits and reduction orders are separate questions.
