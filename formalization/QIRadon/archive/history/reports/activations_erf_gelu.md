# Standard erf, normal CDF, and GELU: verified activation examples

## Status

`QIRadon/Activations/Erf.lean` and `QIRadon/Activations/GELU.lean` compile cleanly with Lean 4.33.0 and the pinned mathlib revision. They use the actual Gaussian Schwartz map supplied by `Activations/Gaussian.lean` and prove concrete activation identities and envelopes. No tanh-specific inversion identity is used.

All names below have namespace `QIRadon.Activations`.

## Standard normal density and CDF

The actual real Schwartz kernel is

`normalPDF(x) = exp(-x²/2)/sqrt(2π)`.

`normalPDF_mass` proves its integral is one. `normalPDF_fourier` proves its cycles-frequency Fourier transform is `exp(-2π²ξ²)`, deriving this from the actual mass-preserving rescaling of the self-Fourier Gaussian.

The activation is exactly

`normalCDF(x) = 1/2 + integral_0^x normalPDF(t) dt`.

`normalCDF_first_derivative` and `normalCDF_contDiff` prove the derivative-kernel identity and smoothness through every finite order. `normalCDF_neg` proves `normalCDF(-x)=1-normalCDF(x)` from the even density.

Its actual envelopes are:

- `normalPDFSpatial`: exponent μ=2, C=1/sqrt(2π), b=1/2.
- `normalPDFSpectral`: exponent ν=2, c=C=1, A=a=2π².

Thus the generic first-order construction applies with derivative order r=1, mass multiplier m=1, and an ordinary output-bias anchor.

## Standard erf

`erf` is defined as `2 normalCDF(sqrt(2)x)-1`.

`erf_integral_formula` proves this is exactly the conventional function

`erf(x) = (2/sqrt(π)) integral_0^x exp(-t²) dt`.

The actual Schwartz kernel `erfKernel` is `exp(-x²)/sqrt(π)`, as proved by `erfKernel_standard_formula`. Its mass is one, and `erf_first_derivative` proves `erf'=2 erfKernel`.

`erfKernel_fourier` proves the exact transform `exp(-π²ξ²)`. The concrete envelopes `erfSpatial` and `erfSpectral` have μ=ν=2. Their spectral constants are c=C=1 and A=a=π²; the spatial exponent constant is b=1, with the normalization written equivalently as sqrt(2)/sqrt(2π).

Thus the generic construction applies with r=1, m=2, again with an ordinary output bias.

## Exact standard GELU

The activation `gelu(x)=x normalCDF(x)` is exactly standard GELU. `gelu_standard_erf_formula` proves

`gelu(x) = (x/2)(1+erf(x/sqrt(2)))`.

The Schwartz kernel is constructed as `normalPDF - normalPDF''`, not postulated to be a Schwartz function. Direct derivative calculations yield

`geluKernel(x) = (2-x²) normalPDF(x)`

and `gelu_second_derivative` proves `gelu''=geluKernel`.

This kernel is signed: positivity in physical space is not needed by the theory. Its Fourier transform has no zero:

`geluKernel_fourier`: `Khat(ξ)=(1+(2πξ)²) exp(-2π²ξ²)`.

The derivation uses actual Fourier differentiation twice on integrable Schwartz derivatives. `geluKernel_mass` proves integral one.

The proved explicit envelopes are:

- `geluSpatial`: μ=2, C=6/sqrt(2π), b=1/4.
- `geluSpectral`: ν=2, c=1, C=5, A=2π², a=π².

The polynomial factors are absorbed with the explicit inequality `(a+by)exp(-2y)≤(a+b)exp(-y)` for nonnegative a,b,y. No informal polynomial-versus-exponential asymptotic is assumed.

## The GELU anchor is a genuine shallow MLP

Second-order reconstruction needs an affine anchor. It does not need an external polynomial feature for GELU:

`gelu_sub_reflection`: `gelu(x)-gelu(-x)=x`.

`gelu_anchorPolynomial_two` rewrites the complete order-two anchor as

`a0 + a1 gelu(x) - a1 gelu(-x)`.

`gelu_anchored_network` applies this to the actual generic construction. If S is the original finite GELU sum, its anchored version is exactly

`S(x) + [g(0)-S(0)] + [g'(0)-S'(0)] gelu(x) - [g'(0)-S'(0)] gelu(-x)`.

It therefore uses precisely two additional activation slots and one output bias. On a ridge spoke, the same identity is applied to the scalar projected coordinate. The multidimensional assembly may count two additional slots per spoke; combining affine anchors more efficiently is a separate possible simplification, not part of these files.

## Connection to the established rate

All three examples have μ=ν=2. Therefore the already-proved generic exponent `κ=max(1,1/μ+1/ν)` equals one. The scalar local-analytic construction has the certified logarithmic width `O(log(1/ε))`, including its halo. For GELU the two anchor neurons add a fixed scalar overhead and preserve that exponent.

This verifies concrete nonempty examples for the fast generic class. It does not certify the tanh approximation commonly used as a numerical surrogate for exact GELU, all arbitrary activations, or floating-point implementations of coefficient integrals.

## Verification

Command: `lake build QIRadon.Activations.Erf QIRadon.Activations.GELU`.

Sixteen primary identities/envelope definitions were inspected with `#print axioms`; only `propext`, `Classical.choice`, and `Quot.sound` occur. Source scans contain no `sorry`, `admit`, or custom axiom declarations.
