# General-kernel Lean formalization — completed analytic certificate

Owned implementation: `QIRadon/Kernel.lean` and `QIRadon/Kernel/Audit.lean`.
Dependency supplied by coordinator: `QIRadon/Anchoring.lean`.

## Verified result and scope

`QIRadon.Kernel.finite_activation_function_certificate` proves the full scalar cutoff–alias–finite-halo error certificate, followed by the sharp Taylor anchoring factor `E*T^r/r!`, for the explicitly defined finite activation sum. It does **not** assume the desired network approximation bound, a Poisson identity, summability, or a reconstruction oracle. The necessary summability and Poisson hypotheses are derived from the original Schwartz-kernel / compact-frequency assumptions.

The input assumptions are:

- A real activation `φ` is globally `C^r` and satisfies `iteratedDeriv r φ u = m*K u` for an actual real Schwartz kernel `K` and `m != 0`.
- The positive-bandwidth use case is allowed; the proved identity actually requires only `γ != 0`.
- `pHat : ℝ → ℂ` is integrable. The real part of its explicitly defined inverse Fourier transform equals the target's r-th derivative on `[-T,T]`.
- `T > 0`, frequency cutoff `B >= 0`, grid spacing `h > 0`, and a finite retained index set `J : Finset ℤ`.
- The Fourier transform of the explicitly constructed scaled kernel is nonzero throughout `[-B,B]`.
- The target is `ContDiffOn ℝ r` on the closed target interval. No global smoothness or endpoint-crossing differentiability of the target was added.
- An explicit cutoff–alias–halo expression is at most `E` on the target interval. This is a bound on integrals and series of the supplied spectrum and kernel, **not a network error assumption**. Deriving geometric/algebraic rates from particular kernel envelopes remains T3 work.

The constructed density is the real part of the finite-band inverse spectrum. The raw network is

`S(s) = h*γ/(m*γ^r) * Σ_{j∈J} ρ(h*j) φ(γ*(s-h*j))`.

The `h*γ/(m*γ^r)` formulation works for `r=0`; it avoids the incorrect natural-number exponent `γ^(r-1)` at `r=0`.

The explicit anchor is

`P(s) = Σ_{k<r} (g^(k)(0)-S^(k)(0))*s^k/k!`.

The proved function is `Q=S+P`, with `|Q(x)-g(x)| <= E*T^r/r!`.

For `r=0`, `P=0`. For `r=1`, `P` is exactly an ordinary output bias; both identities are proved. For `r>=2`, **pure-MLP realization of the anchor polynomial remains an explicit activation-dependent architecture obligation**. The file does not present a polynomial side channel as a pure MLP.

## Fourier normalization

The Lean implementation uses mathlib's Fourier transform

`F f ξ = ∫ exp(-2π i ξ x) f(x) dx`.

Paper angular frequency is `ω=2πξ`. Thus:

- The inverse transform is an integral in `ξ` with no `1/(2π)` prefactor.
- Alias shifts are `k/h`, corresponding to `2πk/h` in angular-frequency notation.
- The cutoff `B` in Lean is a cutoff in cycles-frequency.
- `complexScaledKernel_fourier` proves the exact bandwidth identity `F[K_γ](ξ)=F[K](ξ/γ)`.

The sign of the alias shift is `ξ-k/h`; summation is over every nonzero integer, so this is the same full set as the paper's `ξ+k/h` convention.

## Actual three error terms

`kernelErrorCertificate` is the sum of:

1. `∫_{ξ∉[-B,B]} |pHat ξ|`.
2. The sum over **every nonzero integer k** of the finite-band absolute spectral overlap integral `∫ |b(η)| |Khat(η-k/h)| dη`, where `b=1_{[-B,B]} pHat/Khat`.
3. `h*(∫|b|)*Σ_{j∉J}|K_γ(s-hj)|`.

Every series used in the proof is proved absolutely convergent. The alias term is written as a sum of nonnegative one-dimensional integrals, rather than the Tonelli-equivalent integral of their sum. No nearest-alias truncation is used. The retained set is arbitrary finite `J`; a specific floor-based halo choice and its width/asymptotic bounds remain downstream choices.

## Substantive proved components

- `norm_synthesize_sub_band_le`: sharp-cutoff inverse Fourier error bounded by omitted spectral mass.
- `norm_cutoffDensity_le`: explicit density bound used in halo control.
- `integrable_band_quotient`: compact-band quotient integrability derived from integrable spectrum, continuous kernel transform, and no zeros.
- `synthesize_convolve`, `cutoffDensity_convolve`: genuine Fubini deconvolution identities. The sharp-cutoff density itself is not required to be integrable.
- `fourier_synthesize_mul_reflect`: actual Fourier transform of the density times translated kernel; this is the spectral-alias formula.
- `synthesize_mul_reflect_spatial_decay`, `synthesize_mul_reflect_fourier_decay`: derive quadratic decay on both sides from compact spectrum and Schwartz kernel. This avoids falsely treating a sharp-cutoff inverse density as Schwartz.
- `poisson_scaled_synthesize_mul_reflect`: actual arbitrary-positive-spacing Poisson identity for that integrand.
- `aliasMass_decay`, `sum_fourier_alias_le_mass`: all-alias absolute overlap is summable and dominates the oscillatory error.
- `sum_spatial_tail_le_density_bound`: the omitted-center contribution is bounded by the actual kernel tail and density bound.
- `finite_spectral_kernel_certificate`: complete finite kernel-sum error bound from those components.
- `complexScaledKernel`: explicit real-Schwartz-to-complex-scaled-Schwartz construction, not a supplied existence witness.
- `finite_spectral_real_kernel_certificate`: real-part projection preserves the complex certificate.
- `rawActivationNetwork_iteratedDeriv`: actual activation derivative equals the finite kernel sum with the correct bandwidth/mass factor.
- `rawActivationNetwork_derivative_certificate`: complete derivative error bound for the actual finite real activation network.
- `anchorPolynomial_jet`, `anchorPolynomial_top_derivative`: all low-order jets exactly restored, top derivative unchanged.
- `finite_activation_function_certificate`: complete sharp function-value certificate using coordinator's local Taylor theorem.

## Verification

`lake build QIRadon.Kernel` completed successfully, warning-free.

`lake env lean QIRadon/Kernel/Audit.lean` completed successfully. For all audited main results, `#print axioms` returned only:

`[propext, Classical.choice, Quot.sound]`.

There is no `sorryAx`, no custom axiom, no `admit`, and no unsafe proof shortcut. Definitions use Lean's ordinary `noncomputable` real integrals; this does not certify a floating-point implementation or quadrature evaluator.

## Not claimed complete by this file

- Kernel-envelope-specific geometric/algebraic scaling rates.
- Local analytic profile preparation and Radon boundary extension.
- Angular quadrature or optimal allocation.
- Pure activation realization and neuron counts for generic high-order anchor polynomials.
- Numerical coefficient quadrature, finite precision, or automatic evaluation from arbitrary point samples.

These are not hidden as assumptions of a headline theorem. The file proves the scalar master certificate that those later theorems must instantiate.

## Additional coordinator-assigned support: actual Jacobi beta moments

Owned additional file: `QIRadon/Quadrature/BetaMoments.lean`.

Proved (namespace `QIRadon.Quadrature`):

- `real_beta_intervalIntegrable`: real beta density integrability for both shape parameters positive, including singular endpoint exponents.
- `real_beta_integral`: real beta integral equals the ratio of real Gamma functions.
- `jacobi_even_half_intervalIntegrable`, `jacobi_even_intervalIntegrable`: actual weighted even monomials are integrable for every exponent `a > -1`.
- `jacobi_weight_intervalIntegrable`: actual weight `(1-x^2)^a` integrability on `[-1,1]`, including `a=-1/2`.
- `jacobi_even_half_moment`, `jacobi_even_moment`: the actual integral of `x^(2k)*(1-x^2)^a` equals `Gamma(k+1/2)*Gamma(a+1)/Gamma(k+a+3/2)` on `[-1,1]`.
- `jacobi_odd_moment`: odd weighted moments vanish by reflection.

The proof uses the already proved complex beta integral, explicit real conversion, the actual substitution `u=x^2`, and reflection. No desired moment identity is a hypothesis.

`lake build QIRadon.Quadrature.BetaMoments` succeeds warning-free. The important results were added to `Kernel/Audit.lean`; axiom output remains exactly `[propext, Classical.choice, Quot.sound]`.
