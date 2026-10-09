# Actual complex rotations and word bounds

`QIRadon/Quadrature/ComplexPlaneRotation.lean` complexifies the actual real coordinate-plane rotation from `PlaneRotation.lean`. The namespace is `QIRadon.ComplexRotation`.

## Concrete objects

`realEmbedding` is the coordinatewise `ofReal` map, bundled and proved to be a real-linear isometry from real Euclidean space to complex Euclidean space.

`complexPlaneRotate i j z x` uses the coordinates

- coordinate i: `cos(z) x_i - sin(z) x_j`;
- coordinate j: `sin(z) x_i + cos(z) x_j`;
- other coordinates unchanged.

When i=j it is the identity. `complexPlaneRotate_real` proves exact agreement with the existing real rotation after applying `realEmbedding`. The signs match the angular Casimir convention.

`complexPlaneRotation` bundles the same operation as an actual complex continuous linear operator.

## Derived bounds

The module proves

`norm(R(z)x) ≤ exp(2 norm(z)) norm(x)`,

`norm(R(z)x-x) ≤ (exp(2 norm(z))-1) norm(x)`.

The corresponding operator-norm inequalities are proved too. Constants do not depend on ambient dimension.

The argument decomposes the actual matrix as `I+(cos(z)-1)P+sin(z)J`, where the actual plane projection P and skew generator J have operator norm at most one. The complex exponential series gives

`norm(cos(z)-1) ≤ exp(norm(z))-1`,

`norm(sin(z)) ≤ exp(norm(z))-1`.

Finally `2(exp(r)-1)≤exp(2r)-1` yields the displayed conservative factor two. This proof deliberately preserves vanishing displacement at z=0; it does not insert a fixed factor per rotation.

## Whole rotation words

`rotationWord pairs z x` accepts `pairs : Fin l → (ι × ι)` and one complex angle per position `z : Fin l → ℂ`. It applies the ordered product `R_0(z_0)(R_1(z_1)(...x))`.

`rotationWord_norm` and `rotationWord_displacement` prove

`norm(R_word(z)x) ≤ exp(2 sum_i norm(z_i)) norm(x)`,

`norm(R_word(z)x-x) ≤ (exp(2 sum_i norm(z_i))-1) norm(x)`.

The more general `indexedWord` permits an arbitrary list of parameter indices, including repetitions, and proves the corresponding bounds with the sum over that actual list.

`complexPlaneRotate_analytic_comp` proves holomorphic dependence even when both the angle and input vector depend holomorphically on common parameters. `rotationWord_entire` therefore establishes actual joint analyticity in all complex angles. At the zero parameter vector the word is exactly the identity.

These facts provide the geometric bridge for the mixed-Cauchy estimate: a sufficiently small sum of complex-angle magnitudes keeps the actual rotated vector inside the available complex tube. The tube radius and the bound on the function being evaluated still come from the actual profile analyticity proof; they are not replaced by a derivative-growth assumption here.

## Verification

The module is compiled against the existing real rotation implementation and the pinned Lean/mathlib toolchain. Its principal embedding, norm, displacement, analyticity, and real-compatibility statements are inspected with `#print axioms`. Only the standard foundational axioms are permitted; no proof placeholders or custom axioms are introduced.
