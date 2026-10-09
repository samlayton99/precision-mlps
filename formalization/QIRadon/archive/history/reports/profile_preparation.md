# Local analytic profile preparation and actual finite-network rate

## Verified scope

The local profile preparation and its connection to the actual scalar activation construction are now proved. The original target needs a bounded holomorphic extension only to its local Bernstein ellipse. No global Fourier-tail assumption on the original target, supplied polynomial-approximation oracle, or assumed reconstruction identity is used.

Files:

- `QIRadon/ProfilePreparation.lean`
- `QIRadon/ProfilePreparation/PeriodicStrip.lean`
- `QIRadon/ProfilePreparation/ChebyshevApproximation.lean`
- `QIRadon/ProfilePreparation/GaussianWindow.lean`
- `QIRadon/ProfilePreparation/HorizontalFourier.lean`
- `QIRadon/ProfilePreparation/DerivativeSynthesis.lean`
- `QIRadon/LocalProfileRate.lean`

This is the local analytic scalar theorem and its supporting foundation. It does not by itself prove the complete multidimensional Radon theorem, all activation examples, an algebraic local-profile theorem, numerical quadrature error for coefficient integrals, or floating-point stability.

## The actual construction

The symbol `N` in these files is the **preparation polynomial order**, not the final number of neurons. This distinction is essential when reporting the scaling law.

1. Take exactly `N+1` real target samples at Chebyshev roots `cos((2j+1)π/(2(N+1)))`.
2. Form the explicit discrete-cosine coefficients and the Chebyshev polynomial `P_N`. The proof establishes equality with actual Lagrange interpolation and exact reproduction of all polynomials of degree at most `N`.
3. Form the entire prepared profile

   `G_N(z) = P_N(z) exp(-N(z/T₀)^2) sum_{k=0}^N (N(z/T₀)^2)^k/k!`,

   where `T₀>1` is a fixed boundary margin in normalized coordinates. The finite Gaussian window is close to one on `[-1,1]` and decays on the entire real line.
4. Compute the actual Fourier transform of `G_N`, using cycles frequency. The derivative spectrum is `(2πiξ)^r Ghat_N(ξ)`.
5. Apply the actual cutoff/deconvolution readout-density construction from `ScalarRate.lean`, with spectral cutoff `B=L/π`, suppression `L=(growth(T₀)+3)N`, kernel scaling `γ=B/L^(1/ν)`, center spacing `h=1/(HB)`, and physical halo length `D=Z L^(1/μ)/γ`.
6. Retain exactly the integer centers `j h` for `|j|≤floor((1+D)/h)`. The final count theorem includes both halos.
7. Apply the activation construction and its explicitly defined anchoring polynomial. For derivative order one the anchor is an output constant. Higher-order polynomial anchors remain a separate pure-MLP realization/count obligation.

All coefficient quantities are determined by the original samples, fixed construction constants, and finite one-dimensional coefficient integrals. The proofs do not solve a global least-squares problem. Numerical approximation of those integrals is not certified in these files.

## Assumptions of the completed network theorem

`QIRadon.LocalProfileRate.local_analytic_activation_error` assumes:

- `δ>0` and `g` holomorphic on a neighborhood of `bernsteinEllipse δ = cos({z : |Im z|≤δ})`.
- `|g(z)|≤M` on that ellipse. The actual target is `Re g(x)` on `[-1,1]`.
- An activation `φ` with continuous derivatives through order `r` and `φ^(r)=m K`, where `m≠0` and `K` is an actual real Schwartz kernel.
- A two-sided spectral envelope for that actual kernel transform:

  `c exp(-A|ξ|^ν) ≤ |Khat(ξ)| ≤ C exp(-a|ξ|^ν)`,

  with `c,C,a,ν>0` and `A≥0`.
- A spatial envelope `|K(u)|≤Cs exp(-b|u|^μ)` with `Cs,b,μ>0`.
- A fixed margin `T₀>1`, integer order `N≥1`, alias-separation choice `H≥2` with `a(H/2)^ν≥A+4`, and halo choice `Z≥0` with `(b/2)Z^μ≥A+3`.

These are primary activation and local-target hypotheses. The proof derives the intermediate spectrum, moment, reconstruction, alias, and halo estimates; it does not assume them as the conclusion.

## Error and width actually proved

Write

- `v(T₀)=T₀^(-2)-1-log(T₀^(-2)) > 0`,
- `growth(T₀)=log(2(3+4T₀))+3/(2T₀²)`,
- `b_local=min(δ/2, v(T₀)/2, 2)>0`,
- `κ=max(1,1/μ+1/ν)`.

The theorem proves, for the actual finite anchored activation network `F_N`,

`sup_{|x|≤1} |F_N(x)-Re g(x)| ≤ C_local exp(-b_local N)`.

Every component of `C_local` is explicit:

`C_local = M(6+8/δ)/(1-exp(-δ)) + 2M(1+2/v(T₀)) + C_moment * envelopeConstant/r!`,

where

`C_moment = (2M sqrt(4πT₀²)) (2π)^r (2r!/π^(r+1))`.

`envelopeConstant` is the explicit positive spectral/spatial envelope constant in `ScalarRate.lean`.

The actual retained neuron count obeys

`width ≤ (2H/π+2HZ+1) (growth(T₀)+3)^κ N^κ`.

Therefore the certified scalar width is `O(log(1/ε)^κ)`. It is not universally `O(log(1/ε))`. For example, the exponents `μ=ν=2` give `κ=1`, whereas `μ=ν=1` give `κ=2` for this generic proof. No activation-specific improvement is smuggled into the general statement.

`local_analytic_activation_accuracy` chooses the integer order explicitly:

`N = ceil(max(1, max(1,log(C_local/ε))/b_local))`.

It proves error at most `ε` for every `ε>0`, including the zero-target-constant case. `local_analytic_accuracy_width` substitutes this exact integer choice into an explicit logarithmic width bound, including the ceiling overhead and all halo neurons.

`local_analytic_activation_error_scaled` gives the physical interval `[-T,T]` for any `T>0`: sample `g(T node)`, evaluate the normalized constructed network at `x/T`, and assume analyticity on the correspondingly scaled local ellipse. Physical centers and halo lengths multiply by `T`; slopes divide by `T`. No global extension assumption appears.

## Proof chain and key names

In namespace `QIRadon.ProfilePreparation`:

- `samplePolynomial_eq_interpolate`, `samplePolynomial_interpolates`, `samplePolynomial_reproduces`: this is an actual sample-based interpolation operator.
- `stripCoefficient_bound`: finite Cauchy–Goursat rectangles shift each periodic Fourier coefficient; the vertical sides cancel by periodicity, giving the exponential coefficient bound.
- `stripCoefficient_hasSum`: actual AddCircle Fourier inversion recovers the analytic periodic function.
- `analytic_cosinePolynomial_error`: pairing positive and negative frequencies gives Chebyshev approximation on the local ellipse.
- `samplePolynomial_geometric_error`: combine that analytic approximant with the actual sample interpolation operator, then absorb its polynomial prefactor. Final error is `M(6+8/δ)/(1-exp(-δ))*exp(-δN/2)`.
- `preparedProfile_real_error`: combines the interpolation estimate with the actual finite Gaussian-window error.
- `preparedProfile_horizontal_bound`: direct bounds on the actual polynomial and window give `|G_N(x+iy)|≤Envelope_N exp(-N x²/(4T₀²))`.
- `preparationEnvelope_exp_form`: the derived envelope is an explicit linear prefactor times `exp(constant*N)`.
- `angularFourier_gaussian_strip_bound`: a finite contour shift followed by the real-line limit proves actual Fourier exponential decay. Gaussian bounds make the vertical integrals vanish and justify the limit.
- `preparedProfile_fourier_moment`: integrates the derived decay, with the exact exponential moment `2r!/c^(r+1)`.
- `preparedProfile_derivative_synthesis`: actual inverse Fourier inversion and all polynomial moments prove the `r`th derivative is the inverse derivative spectrum.

In namespace `QIRadon.LocalProfileRate`:

- `prepared_moment_exp_bound`: derives the usable weighted spectrum bound `C_moment exp((growth+1)N)`.
- `prepared_activation_geometric_error`: selecting scalar suppression `(growth+3)N` leaves a certified network-to-prepared-profile error `C exp(-2N)`.
- `local_analytic_activation_error`: combines interpolation, window, and actual finite-network error.
- `local_analytic_grid_width`: counts the exact symmetric finite grid including the halo.
- `local_analytic_activation_accuracy`, `local_analytic_accuracy_width`: explicit epsilon selection and count.
- `local_analytic_activation_error_scaled`: physical-interval rescaling.

## Verification

`lake build QIRadon.ProfilePreparation` and `lake build QIRadon.LocalProfileRate` succeed with Lean 4.33.0 and the pinned mathlib revision. The headline declarations were audited using `#print axioms`; their only dependencies are the standard foundational axioms `propext`, `Classical.choice`, and `Quot.sound`. No `sorryAx`, custom axiom, or assumed approximation conclusion occurs.

The formal analysis uses exact real/complex arithmetic and exact integrals. Turning it into a reliable floating-point implementation requires separate quadrature and roundoff bounds. The full multidimensional theorem also still needs its angular construction and anchors composed and counted by the project coordinator.
