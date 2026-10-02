# Spectral encoder implementation audit

Reviewed `spectral/run.py`, `spectral/validate.py`, `spectral/metrics.json`, and `spectral/validation.json` against the derivation in `theory.md`. This was a read-only review of implementation and recorded results; the review did not rerun or expand the experiment.

No material implementation error was found. The predictor uses the analytic reference Fourier inverse and the p-variable correction. The dense new-geometry least-squares routine appears only in independent validation.

## Checked mathematical invariants

- With centers \(c_j=-1+2j/N\), feature rows equal the first-column amplitude times \(e^{-2\pi ikj/N}\). This matches `synthesize` and the FFT signs in `encode`.
- The circulant Gram eigenvalue is \(N\sum_{k\equiv l\pmod N}|A_{k0}|^2\), including every Fourier alias retained by the cutoff. The inverse factors and N factors in the reference encoder are consistent with NumPy's FFT normalization.
- The \(1/L\) in each feature is correct for periodic normalized derivative kernels and their primitives. The objective uses the sum of squared Fourier amplitudes rather than L times that sum; this common factor does not alter the fitted coefficients or relative errors.
- The constant coefficient mode is explicitly excluded. The Helmert matrix used by the independent validator is an orthonormal basis for the same zero-sum coefficient space. The predictor's a, B, and recovered readouts stay in that space. This is a declared constrained inverse, not an unqualified truncated-pseudoinverse substitution.
- The implemented M, C, leakage W, and whitened small system match the exact derivation. The first-order and transport-only comparisons use the expected signs. The omitted-residual comparison retains the leakage Gram and removes only its target residual forcing.
- The uniform sample grid starts at −1. Multiplication of its FFT coefficients by \(e^{i\pi k}\) correctly converts them to the global Fourier convention used by the features.
- The saved encoder arrays depend only on geometry and the reference. The target spectra and target samples enter only through subsequent encoding.

The recorded validation reports spectral-synthesis discrepancies at most \(8.04\times10^{-16}\), reference-inverse discrepancies at most \(6.98\times10^{-15}\), and loss-decomposition discrepancies at most \(5.24\times10^{-16}\). Four-edit coefficient predictions agree with the independent constrained dense solver to approximately \(10^{-13}\). Across the 30 main target/geometry/activation cases, the largest relative coefficient difference is \(1.72\times10^{-11}\), and every validation retains all 47 allowed coefficient directions.

## Qualifications required in reporting

1. **Relative function errors concern the mean-zero target component.** `target` omits frequency zero and `objective` divides by the norm of the remaining target coefficients. For periodic Runge the separately fitted output mean is approximately 0.196116. Its reported relative error is therefore not the relative error normalized by the full target including that mean. This does not affect coefficients or their prediction errors.
2. **The sample encoder requires spectral resolution.** More than twice the cutoff in uniform samples ensures that the retained frequencies lie below Nyquist; it does not rule out aliasing from an arbitrary unresolved target. The tested smooth periodic targets agree with their analytic Fourier encodings to roughly \(10^{-14}\). General sample input should be described as encoding its resolved/truncated periodic Fourier approximation.
3. **The collision experiment demonstrates approaching ill-conditioning, not a realized failure in the tested range.** All tested distances retain rank 47 and keep the rank-warning flag false. The geometry-only conditional energy falls to approximately \(4.35\times10^{-6}\), while the dense design condition rises to approximately \(3.11\times10^5\). Coefficient prediction remains accurate to about \(10^{-12}\). At exact collision an independent direction is lost, but exact collision was not included in these numerical cases.
4. **This spectral run covers tanh and GELU.** The theoretical note also discusses ReLU; any numerical ReLU result must be attributed to its separate experiment, not these metrics.

The refinement from frequency cutoff 512 to 1024 produces no coefficient change at recorded precision for the tested four-edit mixture cases. This supports adequacy of that Fourier truncation for those cases; it is not a universal truncation-error bound.
