# Verified activation contrasts

`QIRadon/Activations/ReLUSpline.lean` gives an actual shallow ReLU construction
for a C² scalar function on [a,b]. For N positive, h=(b-a)/N, and x_j=a+jh,
its exact formula is

  f(a) + Σ_{j=0}^{N-1} [(f(x_{j+1})-f(x_j))/h]
           [ReLU(x-x_j)-ReLU(x-x_{j+1})].

The file supplies explicit flat arrays indexed by Fin(N*2), proves equality
with `QIRadon.reluNetwork`, and derives its interpolation behavior from the
ReLU formula. With |f''|≤M, the proved uniform error is

  M(b-a)²/(8N²).

The curvature constant 1/8 comes from actual convex and concave quadratic
barriers. `network_uniform_error_of_contDiff` is the direct C² endpoint.
There are 2N neuron terms before combining duplicate centers. Together with
`QIRadon.relu_quadratic_lower_bound`, this gives a real algebraic W^-2 rate
and rules out geometric convergence for arbitrary shallow ReLU readouts,
even on an analytic quadratic target.

`QIRadon/Activations/PolynomialObstruction.lean` treats every fixed polynomial
activation P with degree≤m. Its actual network polynomial is proved to have
degree≤m, independently of width and all trainable affine parameters. On target
x^(m+1) over [0,1], define m+2 fixed nodes t_i=i/(m+1) and

  S_m = Σ_i |1 / ∏_{j≠i}(t_i-t_j)|.

The file proves S_m>0 and uniform error≥1/S_m for every width and parameter
choice. The lower bound follows from the exact leading-coefficient Lagrange
identity applied to x^(m+1)-networkPolynomial. No approximation obstruction or
finite-difference cancellation is supplied as a hypothesis.

Both modules use only the standard Lean foundations; no additional axioms,
`sorry`, or `admit` are used.
