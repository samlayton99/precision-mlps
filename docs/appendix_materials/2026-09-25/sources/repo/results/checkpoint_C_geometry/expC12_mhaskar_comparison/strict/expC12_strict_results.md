# Mhaskar versus QUILL with p-bit construction and inference — data-obvious

## TL;DR

- This supersedes the parameter/output-only quantization comparison. Both methods now use correctly rounded $p$-bit construction and inference, with an FP64 linear-solve exception.
- QUILL fits its readout using the features produced by its already-rounded affine hidden parameters. Mhaskar's projection, polynomial conversion, derivative recurrence, and stencil assembly also run at $p$ bits.
- The width budget, degree/step candidates, validation points, and held-out reporting grid are retained. Every saved model is reloaded and its predictions checked bit for bit.

## Question / hypothesis

How do the two constructions compare when precision applies to arithmetic throughout the model pipeline, rather than only to the completed parameters and final predictions?

## Experiment design

The target is $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$, with every integer precision from 8 to 53 bits and a maximum width of 1024 neurons. QUILL uses 24 halo centers per side. Mhaskar chooses among the same 34 polynomial degrees, from 0 through 511, and 65 difference steps from $10^{-4}$ through 4. Selection uses 1024 validation midpoints; reported relative $L^2$ error uses 8001 separate uniform points.

**Arithmetic contract.** Every model calculation uses round-to-nearest, ties-to-even at $p$ significant bits. Both models use the affine form $c+\sum_j a_j\tanh(w_jx+b_j)$. Inputs and parameters are rounded on entry; each multiplication, addition, and tanh is rounded separately. Readout accumulation is sequential, beginning with the output bias. There is no fused multiply-add. MPFR implements correctly rounded operations, not native low-precision hardware execution. Its internal implementation of an elementary function does not expose higher-precision intermediates to the network.

External sampling coordinates, target observations, and hyperparameter-grid values are generated in FP64 and rounded when entering construction or inference. Reference target evaluation, error metrics, and model-selection metrics remain FP64 and do not feed unrounded values back into model arithmetic. Completed model parameters must be exactly representable in their FP64 storage containers; candidates outside that range are invalid. The experiment varies significand precision, not IEEE exponent formats.

**QUILL.** Compute spacing, centers, the refined bandwidth rule, hidden slope, and hidden biases at $p$ bits. This includes the rule's logarithms, exponentials, and bisection arithmetic. Generate the fitting matrix using the same rounded affine multiplication/addition/tanh operations used at inference. Round the observed target values to $p$ bits. The SVD solve alone uses FP64, with the existing relative cutoff $2^{1-p}$. Round its returned readout coefficients immediately. No readout is fit to a higher-precision feature matrix.

**Mhaskar.** Use the centered tanh specialization of Lemma 3.2 of [Mhaskar (1996)](https://doi.org/10.1162/neco.1996.8.1.164). Compute $b=\log(2)/2$ at $p$ bits. Replace the earlier FP64 DCT with a direct discrete Chebyshev projection: round the supplied Chebyshev nodes and observations, evaluate the Chebyshev recurrence, and accumulate its coefficient sums at $p$ bits. Convert each polynomial truncation into monomials at $p$ bits. Compute $c_r=\tanh^{(r)}(b)/r!$ from $y'=1-y^2$ using rounded products, sums, and divisions. Assemble the normalized centered-difference coefficients and merge identical hidden slopes using rounded operations. No FP64 coefficient construction, logarithmic weight fallback, or fitted readout is used for this method.

The polynomial front end remains a practical discrete Chebyshev specialization, rather than the paper's filtered operator with its unspecified degree-dependent constants. The difference step and degree are selected empirically within the stated grid.

**Pruning without changing the winner.** Initially evaluate every candidate on every eighth validation point. Dividing the norm of those residuals by the full validation target norm gives a lower bound on the full relative error. Skip a full evaluation only if this lower bound exceeds the best complete error, with a $10^{-12}$ relative safety margin. The runner checks that subset predictions agree bit for bit with full predictions, and that every rejected candidate's lower bound exceeds the final winner's error. Thus this is not approximate shortlist selection.

**Code & data**

- Current runner: `experiments/expC12_mhaskar_comparison/strict.py`.
- Arithmetic and construction: `experiments/expC12_mhaskar_comparison/pbit.py`, `pbit_kernels.c`; the existing correctly rounded readout kernel is reused.
- Usage and dependency instructions: `experiments/expC12_mhaskar_comparison/README.md`.
- Protocol, source hashes, measurements, and checks: `data/config.json`, `data/summary.json`, `data/validation.json`.
- Full candidate screening errors, complete validation errors, pruning bounds, and statuses: `data/mhaskar_search.npz`. Pruned full errors are absent, not fabricated measurements.
- Intermediate rounded construction arrays: `data/construction_p*.npz`; saved rounded models: `models/*.npz`.
- PNG figures: `figures/three_panel_mhaskar_comparison.png`, `figures/precision_law_comparison.png`, `figures/mhaskar_diagnostics.png`.

## Results

The precision curve now measures the stated $p$-bit arithmetic contract for both methods. Its numerical values and selected Mhaskar degrees/steps are recorded for every precision in the summary file. This experiment does not force monotonic error or reuse the previous QUILL curve.

| Bits | QUILL relative error | Mhaskar relative error | Selected Mhaskar degree |
|---:|---:|---:|---:|
| 8 | $1.51\times10^{-1}$ | $0.976$ | 1 |
| 24 | $1.57\times10^{-6}$ | $0.960$ | 6 |
| 40 | $2.11\times10^{-11}$ | $0.958$ | 7 |
| 53 | $8.76\times10^{-14}$ | $0.959$ | 7 |

All 46 precisions completed. All 92 saved models replay bit for bit, with parameters and predictions exactly representable at their prescribed significand lengths. QUILL uses all 1024 neurons; at 53 bits the selected Mhaskar degree-7 model uses 15 neurons. At 13 bits, the selected difference step is the smallest sampled value, so that point should not be read as an interior or globally optimized step choice.

### Figures

- **Three-panel comparison:** retains the saved width-scaling and bandwidth-selection panels and replaces panel (c) with the new strict-arithmetic observations. Teal is QUILL, purple is the selected Mhaskar network. The dashed line has fixed slope $-1$ in $\log_2$ error versus precision; only its intercept is fit, using bits 16–40.
- **Standalone precision law:** enlarges the corrected panel (c). The label $O(\log(1/\varepsilon))$ denotes the reference relationship between bits and inverse error, not a lower-bound theorem proved by this sweep.
- **Mhaskar diagnostics:** compares the 53-bit Chebyshev coefficients evaluated directly in FP64 as a diagnostic with the network's best sampled screening-grid error at each degree. The right panel shows 53-bit network error versus step on the 128-point screening subset. These diagnostic screening errors do not replace the full validation errors used for selection or the separate held-out errors in the precision plot. Errors above the displayed vertical limits are clipped by the plotting range.

## Additional details

The verification suite passes 76 tests. The new checks cover affine features, polynomial projection, scaled derivatives, stencil assembly, complete forward evaluation, and a constructed case where FP64-then-round gives the wrong result through double rounding. An independent higher-precision oracle rounds after each specified operation for testing only. The production code does not use that oracle. End-to-end low-degree polynomial tests also verify the expected second-order dependence on the difference step.

The previous and corrected curves need not coincide. In addition to the inference change, QUILL now constructs its geometry and bandwidth rule at $p$ bits, while Mhaskar's entire polynomial front end is also restricted. Low-precision inputs and hidden parameters can coincide after rounding. These effects are part of the stated experiment, not plotting noise.

## Conclusions

This comparison enforces a common $p$-bit construction/inference contract, with the FP64 SVD exception explicitly isolated. Its conclusions concern the implemented constructions and finite parameter search; it does not establish a necessary precision bound for all implementations of either method.

## Open questions

- How much of each curve's floor comes from construction, parameter representation, and inference separately?
- How do alternative valid Mhaskar biases or equivalent summation conventions change the measured errors under the same precision contract?
