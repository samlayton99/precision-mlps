# Numerical acquisition audit for the actual analytic source

This is an audit and proposed proof route, not a claim that source cubature costs have already been verified. The exact source construction and its geometric scalar/angular certificates are proved. T8 still needs a complete acquisition procedure with certified finite work and dimension dependence.

## The integrals actually required

For each real spoke direction, the unfiltered projection is a finite radius integral and a (d−1)-dimensional transverse-ball integral. Scalar derivatives can be obtained by complex Cauchy circles inside the proved scalar tube. Even dimensions add the finite singularity-subtracted inner Hilbert integral, endpoint logarithm, and the actual finite spatial tail on `B_R ∩ {|θ·x|>S}`. Kernel preparation subsequently uses the acquired scalar profile; its perturbation analysis does not by itself prove acquisition accuracy.

The source uses the explicit normalized radius bump. It is smooth and flat at the radius endpoints but not analytic there. An analytic quadrature estimate on the entire closed radius interval would therefore be invalid.

## A quantitative route for the radius variable

Let W=R−R₀. The normalizer Z obeys the elementary positive lower bound

`Z ≥ (W/2) exp(−16/(3W²))`,

by integrating over the middle half of the interval. Two endpoint pieces of width a≤W/4 have normalized total mass at most

`(4a/W) exp(16/(3W²)−1/(aW))`.

For L>0, choose

`a = min(W/4, 1/[W(L+16/(3W²))])`.

Then the omitted normalized mass is ≤exp(−L). On the retained radius interval, a complex neighborhood of radius a/4 avoids both endpoint singularities. There the real part of `(z−R₀)(R−z)` is positive, so the unnormalized exponential bump has modulus at most one. With the additional target/square-root margins, ordinary analytic quadrature needs O(L²) radius nodes. This is a proposed rigorous route; the quantitative endpoint and complex-radius lemmas must still be proved and combined with the coordinator's quadrature certificate.

Normalization error must be propagated using the explicit lower bound on Z. It cannot be hidden by saying the density integrates to one in exact arithmetic. Rescaling the raw bump by a positive constant before evaluating numerator and denominator can avoid underflow without changing the normalized density.

## Transverse-ball integration

The integrand is holomorphic in the transverse point on a uniform complex neighborhood supplied by the original target margin. A natural positive cubature is radial Gauss–Jacobi times the already verified spherical rule. For a (d−1)-ball, total polynomial degree O(L) suggests O(L^(d−1)) cubature nodes, with dimension and tube-width constants made explicit.

The required proof is not merely that the rule integrates low-degree polynomials. It must derive the relevant approximation/analytic certificate for the actual shifted, complex-valued target integrand. The complex-valued case needs either a direct complex norm proof or separately controlled real and imaginary parts. The existing real-angular orbit certificate cannot simply be fed a complex-valued target while pretending its real-angle values are real.

Together with the proposed radius rule, one unfiltered projection value can be acquired in O(L^(d+1)) target evaluations by straightforward independent cubature. This is a conservative acquisition estimate, not an inference-cost estimate.

## The even-dimensional tail requires its own domain treatment

The proved identity rewrites the tail as a finite spatial integral over a ball with a central slab removed. Multiplying the integrand by that region's indicator and applying full-sphere analytic quadrature destroys angular analyticity at the cap boundary. That is not a valid geometric-convergence argument.

Two plausible exact reparameterizations are:

1. Ball-mixture radius a, cap angle t=a cos v, and transverse unit-ball coordinates. The square-root endpoint disappears and the Jacobian becomes an integer power of sin v. The angle bound arccos(S/a) is analytic in a because a≥R₀>S.
2. Spherical coordinates for the spatial integral, separating the radial cutoff from the cap. The cap opening near its threshold requires an additional square-root change of variable or an appropriate Jacobi weight.

Both need actual Jacobians, domain identities, analytic margins, and finite quadrature error bounds. Their integrands and domains are explicit and calculable; the current Lean development does not yet certify that cubature.

## Scalar derivatives and dimension dependence

Cauchy-circle quadrature is a natural way to evaluate profile derivatives without numerically differentiating noisy samples. The already proved scalar tube provides the contour. The required absolute tolerance must include the explicit factor r! radius^(−r), with r=d−1 (and any further derivative order required by the activation). An analytic periodic trapezoid argument would use O(log(C/tolerance)) contour samples once its annular margin is proved.

All cubature loops can stream nodes, so their memory need not contain a dense neuron-by-sample matrix. But the time cost does not follow from the neuron count. Independently evaluating every profile at every spoke repeats substantial transverse integration. Multiplying the per-profile work by the number of profile queries can nearly double the exponent involving d. Reuse, polynomial transforms, or a fast Radon transform could change that cost, but none should be claimed without a separate algorithm and proof.

The exact known-function construction therefore establishes a strong representation and an explicit integral-based coefficient recipe. It does not yet establish coefficient acquisition in the same asymptotic time as evaluating the finished network. The hard remaining T8 step is certified high-order cubature for these actual ball/cap/radius integrals, not just floating-point perturbation propagation.
