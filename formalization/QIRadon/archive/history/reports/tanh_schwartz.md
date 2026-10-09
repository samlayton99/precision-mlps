# Actual tanh derivative kernel is Schwartz

Implemented `QIRadon/Activations/TanhSchwartz.lean`, importing the separately proved actual `TanhFourier` formula. Namespace: `QIRadon.Activations`.

The kernel is exactly

`K(x)=tanhKernelValue x=1/(2*cosh(x)^2)`.

The proof does not infer Schwartz regularity from smoothness and value decay alone. It constructs derivative polynomials

`P_0=1`, `P_(n+1)=(1-X^2) P_n' - 2X P_n`

and proves the actual formula

`iteratedDeriv n K x = K(x)*P_n(tanh(x))`.

For |y|≤1, a polynomial is bounded by the absolute sum of its coefficients. Since |tanh(x)|<1 and the actual kernel satisfies

`|K(x)| <= 2 exp(-2|x|)`,

every derivative has an exponential envelope. The elementary factorial bound gives

`|x|^k |K(x)| <= 2 k!`.

Consequently every Schwartz seminorm is explicitly bounded by

`2 k! * sum_j |coefficient(P_n,j)|`.

These derivative-level estimates and the proved global smoothness construct the actual `tanhKernel : Schwartz(ℝ,ℝ)` witness.

Principal interfaces:

- `tanhKernelValue_contDiff (n : WithTop ℕ∞)`
- `tanh_contDiff (n : WithTop ℕ∞)`
- `tanhKernelValue_hasDerivAt`
- `tanhDerivativePolynomial`
- `tanhKernelValue_iteratedDeriv`
- `polynomial_unit_abs_bound`
- `tanhKernelValue_spatial_bound`
- `tanhKernelValue_weighted_bound`
- `tanhKernel`, `tanhKernel_apply`
- `tanh_first_derivative`: the actual activation matches r=1, m=2.

The kernel worker owns the spectral/spatial envelope bundle and the resulting generic kappa=2 corollary; this file supplies the required actual Schwartz object. It does not claim the separate stronger tanh boundary-correction theorem.

Verification: `lake build QIRadon.Activations.TanhSchwartz`, then principal axiom audit in `scratch/AuditTanhSchwartz.lean`.

Build succeeded without warnings in the new module. All six audited declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No proof holes or project axioms were introduced.
