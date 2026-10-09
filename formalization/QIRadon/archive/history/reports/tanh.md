# Verified tanh certificate and separate corrected-halo obligation

Completed files owned by kernel agent:

- `QIRadon/Activations/TanhFourier.lean`
- `QIRadon/Activations/TanhEnvelope.lean`
- `QIRadon/Activations/Tanh.lean`

The local-scaling worker owns `TanhSchwartz.lean`; its independent report is `reports/tanh_schwartz.md`.

## Actual derivative kernel and transform

The kernel is the actual function K(x)=1/[2 cosh²(x)], with tanh'(x)=2K(x). The activation parameters for the general construction are r=1 and m=2.

The Fourier transform is proved from the integral, not assumed:

- The actual differentiable bijection t=(1+tanh x)/2 maps the real line to (0,1), has Jacobian K(x), and satisfies log[t/(1−t)]=2x.
- Applying the one-dimensional change-of-variables theorem turns the Fourier integral into Beta(1−iπξ,1+iπξ).
- The complex beta/gamma integral identity and gamma reflection formula evaluate this as π²ξ/sinh(π²ξ).
- Frequency zero is evaluated separately and gives 1; the quotient is never interpreted as an unproved 0/0 limit.

`tanhKernel_fourier` is the resulting everywhere-defined actual Fourier identity in mathlib's cycles convention.

## Quantitative envelopes

The spectrum is positive. The proved bounds are

`exp(−π²|ξ|) ≤ Khat(ξ) ≤ 4 exp(−(π²/2)|ξ|)`.

The lower estimate follows from `exp(−t)sinh(t)=(1−exp(−2t))/2` and exp(−2t)≥1−2t. For the upper estimate, exp(2t)≥1+2t yields t/sinh(t)≤(1+2t)exp(−t); the linear factor is absorbed using 1+2t≤4exp(t/2).

`tanhSchwartzSpectral` therefore supplies c=1, C=4, A=π², a=π²/2, ν=1 for the actual Fourier transform of the bundled kernel. `tanhSpatial` supplies C=2, b=2, μ=1 from the worker's actual exponential bound. The worker proves every Schwartz seminorm using derivative polynomials, bounded tanh range, and the exponential tail. Global activation smoothness is also proved.

These are the actual hypotheses needed by the general kernel theorem, hence its sufficient scalar exponent is κ=max(1,1/μ+1/ν)=2. The theorem can now be specialized to ordinary tanh with one output-bias anchor. This does not claim tanh intrinsically needs κ=2.

## Independent stronger tanh boundary correction

The approved `tanh_certificate.md` gives a different finite construction: complex-shift density, finite rectangle contours, midpoint vertical contributions, tanh-pole alias residues, finite endpoint moments, and prescribed rational partial fractions placed into existing halo slots. Its claimed κ=1 estimate is not a consequence of the general global Poisson/tail theorem.

The formula has been inspected. The elementary finite-contour identity has the correct orientation/sign when the coth residues and two horizontal tanh shifts are tracked. The rational correction includes the physical boundary term **minus the vertical midpoint contribution**; omitting the latter would change the construction. The finite moment/rational approximation and its halo exponent are mathematically separate from the tanh Fourier calculation.

This stronger contour/corrected-halo theorem is **not yet formalized here**. Mathlib has rectangle Cauchy–Goursat but not the complete needed finite-pole midpoint-residue and explicit rational-correction package ready to invoke. A faithful completion must prove those identities and bounds; assuming its final halo error as a premise would not complete the requested corollary. No such premise has been added.

## Verification

`lake build QIRadon.Activations.Tanh` succeeds. The Fourier identity, actual spectral bounds, both envelopes, derivative normalization, and Schwartz bundle were audited with `#print axioms`; only `[propext, Classical.choice, Quot.sound]` occur. No sorry/admit/custom axioms are present.
