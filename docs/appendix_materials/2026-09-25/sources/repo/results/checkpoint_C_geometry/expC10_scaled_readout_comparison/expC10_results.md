# Standard versus envelope-scaled least squares

## 1. Question and figure

The requested [4 × 6 figure](comparison.png) compares standard raw-readout least squares with the new implementation note's envelope-scaled solve. Columns are functions; rows are interior resolutions. Both axes are logarithmic, with identical limits throughout.

Blue circles mark the standard selection \(\lambda=0.25\). Orange diamonds mark the note's predicted bandwidths, selected before fitting. Curves show the two methods at the same bandwidth, separating scaling effects from the selection rule.

## 2. Matched geometry and samples

| Interior intervals \(N\) | Tanh neurons \(W=N+49\) |
|---:|---:|
| 64 | 113 |
| 128 | 177 |
| 256 | 305 |
| 512 | 561 |

Both methods use \(h=2/N\), identical uniform centers, and 24 halo neurons per side. This preserves Sam's standing halo choice instead of testing the note's separate square-root halo rule.

All widths use the same 2,049 endpoint-inclusive training points and 8,191 independent test midpoints on \([-1,1]\). There are no exterior target observations or derivatives. Training has the note's \(sN+1\) form, with \(s=32,16,8,4\).

The six targets are \(\sqrt2\sin(2\pi x)\), \([\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}\), \(\sqrt5x^2\), \(1/(1+25x^2)\), the same mixed sine multiplied by \(e^{-x^2/(2\cdot0.4^2)}\), and \(e^x\).

## 3. Implementation

Sweep 81 logarithmically spaced bandwidths from 0.03 to 1.5, adding the exact standard and predicted choices. This produces 87 bandwidths in the first three rows and 88 in the last.

Both arms use FP64 SciPy gelsd truncated least squares, the same training-loss normalization, and relative SVD cutoff \(10^{-14}\), matching the recent D36 baseline:

- Standard: solve \(Ac/\sqrt{M_X}\approx y/\sqrt{M_X}\).
- Adjusted: solve \(ADa/\sqrt{M_X}\approx y/\sqrt{M_X}\), then recover \(c=Da\).

All test errors evaluate the physical coefficients \(c\). There is no iterative training.

The adjusted scaling \(D_{jj}=\sqrt{\alpha_j(\lambda)}\) is recomputed at **each current bandwidth**, as in the new note. This differs from D37's optimizer study, which held the reference scaling fixed at 0.25.

## 4. Declared parameters and invalid regions

The short note delegates numerical constants to its full theorem. Here the effective aliasing budget is FP64 epsilon, the SVD cutoff is \(10^{-14}\) for both methods, and \(\delta=0.25\) for the entire targets. Runge uses \(\delta=0.19\), below its poles at distance 0.2. Square roots are rounded upward by one FP64 step.

The envelope formula requires \(\delta>\pi/(N\lambda)\). **Gray regions violate this condition, so there is no orange curve there.** Negative allowances are not clipped or replaced by a different scaling.

This is a controlled practical implementation, not a claim that every cell satisfies every hypothesis of the full floating-point theorem or uses its uniquely certified cutoff.

## 5. Bandwidth selection

Use the note's amplitude-weighted mean absolute frequency, retaining the DC mode and dropping the repeated right endpoint. Sine and mixed sine use their known analytic means; the other four functions use the FFT of the 2,048 distinct samples. Solve the exact equation from the new note on \([0.03,1.5]\), using logarithms to avoid overflow.

For \(e^x\), the endpoint jump in its period-two extension gives an estimated frequency of about 404.80 radians per unit. At \(N=64,128,256\), the equation has no admissible high-precision branch at that frequency. Those panels have no orange diamond, although the scaled solver can still be tested at manually supplied bandwidths. At \(N=512\), its predicted bandwidth is approximately 0.136.

The other predictions range from approximately 0.242 to 0.280. All 21 available predictions have positive, valid envelopes.

## 6. Results

The adjusted method does **not** show consistent improvement. The two curves generally meet near 0.25, while the scaled solve often has larger error at smaller admissible bandwidths.

Of the 21 available predicted selections, eight give lower error than the standard selection; none improves by more than a factor of two, while three worsen by more than two. Many differences are near the numerical recovery floor, so these counts describe this run rather than establish a robust ranking.

At \(N=64\), mixed sine and Gaussian envelope improve by approximately 1.56 and 1.61 times. At \(N=128\), their improvements are approximately 1.61 and 1.95 times. Larger-width selections mostly land near \(10^{-14}\)–\(10^{-13}\).

The selected-point CSV records both the overall standard/note error ratio and the raw/scaled error ratio at the **same** predicted bandwidth.

## 7. Validation

Four focused tests passed: the Fourier mean including DC and endpoint removal; the original scalar budget equation; the envelope products against a 70-digit calculation; and preservation of the untruncated fit under coordinate changes on a well-conditioned dictionary.

Native and physical predictions were compared at selected bandwidths. The largest discrepancy relative to the target is \(1.5\times10^{-12}\) at \(N=64\), where coefficients can be large, and below \(2.7\times10^{-15}\) in the other rows. Plotted errors always use the recovered physical coefficients.

Fitting all widths took approximately 20 seconds on this machine. Existing results were preserved.

## 8. Saved data and reproduction

The data directory contains metadata and predictions, one compressed file per width with all curves, coefficients, scales, ranks and singular values, plus [selected-point comparisons](data/selected_points.csv) and a JSON summary. The supplied PDF is preserved in papers/optimization_notes/high_precision_tanh_implementation_note.pdf.

From the repository root:

    .venv/bin/python experiments/expC10_scaled_readout_comparison/run.py
    .venv/bin/python experiments/expC10_scaled_readout_comparison/run.py --plot-only
    .venv/bin/python -m pytest -q experiments/expC10_scaled_readout_comparison/test_comparison.py

Completed widths are reused if the configuration matches.

## 9. Uniform hidden-column scaling control

[Three-way comparison](comparison_sqrt_h.png) adds a green dashed curve using \(D=\operatorname{diag}(1,\sqrt h,\ldots,\sqrt h)\), where \(h=2/N\). Every hidden-neuron column, including the halo, receives the same factor; the bias stays unscaled. Samples, centers, bandwidths, targets, precision and relative SVD cutoff match the original comparison. The blue and orange curves reuse the original saved fits.

The new curve mostly follows the raw baseline, with no consistent accuracy gain. At \(\lambda=0.25\), the median raw-error / scaled-error ratio across the 24 panels is 1.001. Unlike the full envelope scaling, this control does not produce the same substantial degradation at smaller bandwidths. Many differences near the best fits are at numerical recovery precision.

The additional fits took about nine seconds. Saved-array checks confirm unchanged source data and matching bandwidth grids, the intended column factors, and finite errors. Additional coefficients, ranks, singular values and errors are stored under data/sqrt_h; selected comparisons are in data/selected_points_sqrt_h.csv. The original figure is preserved.

To reproduce the control or redraw it:

    .venv/bin/python experiments/expC10_scaled_readout_comparison/sqrt_h.py
    .venv/bin/python experiments/expC10_scaled_readout_comparison/sqrt_h.py --plot-only

## 10. Note geometry and scaling with authorized baseline numerical defaults

The initial comparison was an adaptation of the short implementation note. It did not implement the square-root halo rule, derive the SVD cutoff from a recovery budget, or certify all arithmetic. Its empirical ranking must not be presented as a verdict on the complete theorem-based implementation.

[Halo-rule comparison](note_geometry/comparison.png) preserves the original raw solve in blue and the earlier envelope-scaled solve in dashed orange, both with R=24. Purple uses the note's halo rule with a_H=1, matching the merged PR's geometry constructor: R=8,12,16,23 and W=81,153,289,559 for N=64,128,256,512. It computes every envelope, including the boundary corrections and bias, with 70-decimal interval arithmetic, then rounds the upper square-root endpoint upward to binary64. The older method only nudged the square root of a float64 envelope upward by one increment.

This control retains the same training/test points, bandwidths, frequency predictions, delta values, and relative SVD cutoff 1e-14. Keeping the cutoff fixed isolates the halo and rounding corrections. At the marked predictions, the change gives mixed improvements and regressions; it does not produce a consistent improvement across the grid. Curves include numerical fits outside sufficient theorem conditions. Gray shading indicates only an undefined envelope, not the complete theorem admissibility test.

The older full source, theorem_for_sam.pdf (SHA-256 c198b97918435c7868e41bc8d87cd2779476f5bec58e807b8e13d2c7a967054a), was recovered from the earlier attachment. Its Appendix E supplies a constrained-fitting procedure and a posteriori certificates, rather than the newer note's scaled-SVD cutoff rule. It additionally requires gamma*delta >= 4*pi, lambda*R >= 2*log(2), lambda <= 1, and other geometric, sampling and arithmetic conditions. Only five of the 21 marked predictions pass the checked geometric inequalities with the declared delta values, all in the N=512 row. Passing those inequalities alone is not certification.

The exact newer recovery-budget cutoff and effective aliasing-budget prescription are not supplied. Sam explicitly resolved this by authorizing baseline defaults wherever uncertain. Accordingly, this completed numerical comparison retains relative cutoff 1e-14, effective aliasing budget 2.220446049250313e-16, delta 0.25 (Runge 0.19), bracket [0.03,1.5], the existing samples, and the baseline FP64 evaluator. It implements the short note's geometry, envelope formulas and scaled-SVD workflow with these declared defaults. The ordinary NumPy/BLAS evaluator is not the older theorem's fused-affine, correctly rounded tanh and exact-accumulation evaluator, and the cutoff has not been certified from its missing recovery budget; no full numerical certificate is claimed. The source question is resolved by this explicit fallback authorization.

Six focused checks passed: independent 120-digit envelope comparisons at four geometries, rejection of an invalid analytic neighborhood, and agreement with the actual PR's boundary-slot assignment. Source data hashes stayed unchanged. Additional fits took about 15 seconds. Data, coefficients, scales, singular values, ranks and geometric eligibility masks are saved under note_geometry/data.

Reproduce:

    .venv/bin/python experiments/expC10_scaled_readout_comparison/note_geometry.py
    .venv/bin/python -m pytest -q experiments/expC10_scaled_readout_comparison/test_note_geometry.py
