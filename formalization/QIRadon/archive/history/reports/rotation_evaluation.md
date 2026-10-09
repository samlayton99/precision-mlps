# Actual scalar evaluation under complex rotations

Completed in `QIRadon/Radon/RotationEvaluation.lean`, namespace `QIRadon.Radon`.

The scalar argument is defined using the actual complex bilinear coordinate product

`complexBilinear u v = sum_i u_i * v_i`,

and

`rotationEvaluation pairs theta x angles = complexBilinear (rotationWord pairs angles (realEmbedding theta)) (realEmbedding x)`.

The varying rotated direction is not conjugated. With x fixed and real this bilinear expression agrees with the Hermitian inner product whose conjugate slot is the fixed embedded x; that identity is used only to obtain the dimension-independent norm estimate.

Proved results:

- `rotationEvaluation_entire`: joint entire dependence on all complex angles.
- `rotationEvaluation_zero`: exact original coordinate `inner theta x` at zero angles.
- `rotationEvaluation_displacement`: scalar displacement is bounded by `(exp(2 sum_i norm angles_i)-1) * norm theta * norm x`, derived from the actual vector rotation-word displacement.
- `rotationEvaluation_unit_displacement`: unit-direction specialization removes `norm theta`.
- `rotationEvaluation_mem_scalarTube`: for unit theta, norm x≤1, L≥1, strict angle budget sum norm angles<eta, and exp(2eta)-1≤delta, the actual scalar argument lies in `scalarTube L delta`. The strict angle budget preserves membership in the **open** tube even when the margin inequality is an equality.
- `scalarRotationBudget delta=log(1+delta)/2`, with positivity for delta>0 and exact exp-margin equality.
- `rotationEvaluation_ofReal`: at real angles the argument equals the real inner product with `totalRealRotationWord pairs angles theta`, including diagonal identity pairs and the original outermost-first word ordering.

The module imports `RotationLift` for its existing exact real-word identities. It adds no ambient-extension assumption, no maps-to oracle, and no dimension- or word-length factor to the existing angle budget. It is intended for the actual `RotationOrbit` source assembly.

Verification: `lake build QIRadon.Radon.RotationEvaluation`; principal declarations audited in `scratch/AuditRotationEvaluation.lean`.

Build succeeded. All six audited declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No `sorry`, `admit`, or project axiom was introduced.
