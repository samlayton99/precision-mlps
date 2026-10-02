# Tents with eightfold width

Requested follow-up: repeat the constant-target tent comparison with much stronger overlap. Main tent is T(x/8), with integer-spaced centers, so its support spans 16 center spacings instead of 2. No output bias. We fit continuous squared error on a periodic domain of length 128, and use the minimum-Euclidean-norm readout with relative SVD cutoff 1e-12. Repeated all cases with domain length 256. Every tent corner is a quadrature breakpoint; Gaussian quadrature exactly integrates the piecewise-quadratic feature products and squared residuals, up to floating-point roundoff.

## Main result and exact explanation

For each residue class r modulo 8,

\[
\sum_{m\in\mathbb Z} T((x-r-8m)/8)=1.
\]

There are eight interleaved subgrids, each capable of reproducing the constant by itself. On a periodic domain whose length is a multiple of 8 this is also an exact finite identity. The full feature matrix has seven exact null directions (rank 121 for 128 columns). These are genuine redundancies, not numerical loss of rank.

- **Original:** every readout is 1/8. Relative L2 error 1.10e-15.
- **Delete center zero:** the minimum-norm exact solution assigns zero to every center divisible by 8, including distant ones still present in the model; every other readout is 1/7. Relative L2 error 8.57e-16. Numerically obtained coefficients agree with this exact formula within 7.5e-14.
- **Halve the selected width:** its optimal coefficient is zero; the other coefficients reproduce the deletion solution. The narrower tent introduces a component outside the original span, while the unchanged neurons already reproduce the constant exactly. Relative L2 error 1.18e-15.
- **Double the selected width:** the constant is again fitted to 9.84e-16. The selected coefficient is 0.1361702, weights at centers +/-8 are 0.0680851, other multiples of 8 are 0.1361702, and all other centers have weight 0.1234043. These are the minimum-norm values for the length-128 circle, not a uniquely determined infinite-line response. For M=period/8, the selected weight is M/(8M-10.5); the other values follow from the identity T(x/16)=T(x/8)+[T((x-8)/8)+T((x+8)/8)]/2.

The periodic minimum-norm convention is material. On the infinite line these global periodic readout changes do not belong to the square-summable perturbation class used for the earlier width-1 deletion formula. One must not identify the two variational problems. The figure explicitly reports the periodic domain. Domain doubling preserves the original, deletion, and half-width readouts to about 1e-13, but changes the double-width readout by about 0.00582 because the global minimum-norm objective depends on the number of repeated cells.

## Nearby width control

The integer width-to-spacing ratio creates a special exact redundancy. We repeated the same four fits with half-width 8.25 (total support 16.5 spacings) to distinguish this from overlap alone. There are now no exact null directions in the baseline, and even the original constant fit has a small spacing-periodic ripple.

| Case | Relative L2, period 128 | Maximum evaluated absolute error |
|---|---:|---:|
| Original | 0.0011855 | 0.0027534 |
| Delete center | 0.0012333 | 0.0041501 |
| Halve selected width | 0.0012188 | 0.0034599 |
| Double selected width | 0.0011832 | 0.0032542 |

The global repeated zero-weight family becomes a modulated coefficient pattern. The whole readout cannot be represented as a single smooth line, even for a constant target. Period doubling changes local coefficients by up to 0.00088 for deletion and 0.00702 for the half-width case, so these nearby-width figures are finite periodic examples, not certified infinite-line limits. Original readouts are invariant to the tested period change.

## Artifacts

- Main figure: `tent_overlap_w8.png` and `.pdf`.
- Nearby-width control: `tent_overlap_w8.25.png` and `.pdf`.
- Reproducible script: `tent_overlap_check.py`.
- Fits, ranks, readouts, residual errors and domain comparison: `tent_overlap_verification.json`; per-case NPZ arrays beside it.

The result is stronger compensation with larger overlap, but its exact periodic pattern is governed by the tent's grid alignment and the solution convention. It is not a general theorem that wider kernels always improve approximation or that every deleted neuron can be compensated exactly.
