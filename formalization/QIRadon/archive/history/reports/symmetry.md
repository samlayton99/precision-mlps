# Actual symmetric Gauss–Jacobi nodes and weights

Owned file: `QIRadon/Quadrature/JacobiSymmetry.lean`.

The actual Jacobi moment functional is proved reflection-invariant using the actual interval integral and even density. The reflected monic orthogonal polynomial `(-1)^n p(-X)` is proved monic, degree n, and orthogonal. Previously proved uniqueness gives its equality to p; negation therefore preserves its roots.

For an injective nodal grid, the proof constructs a reflection permutation from this root closure and proves it involutive. Interpolation uniqueness proves the reflection identity for the Lagrange basis; applying the symmetric moment functional proves equality of paired weights.

The principal consumer interface is:

`QIRadon.Gaussian.gaussJacobi_reflection_of_exact a ha hn node hinj hexact`

It accepts the existing exact Gauss–Jacobi rule API and returns an actual permutation `σ : Fin n ≃ Fin n` satisfying:

- `Function.Involutive σ`;
- `node (σ i) = -node i`;
- `weight (jacobiMomentGeneral a ha) node (σ i) = weight (jacobiMomentGeneral a ha) node i`.

Orthogonality is recovered from quadrature exactness applied to the nodal polynomial times a lower-degree polynomial; it is not added as an unproved requirement to this interface. The full constructed-rule wrapper is `exists_symmetric_gaussJacobi_rule_general`.

`reflection_fixed_iff_node_zero` explicitly records that scalar zero nodes are reflection-fixed. Odd-order Jacobi rules must not be folded by pretending every scalar node has a distinct negative partner. A full sphere-node involution has no fixed points because its unit vectors are nonzero; the recursive base-index antipodal pairing must remain included.

Verification: `lake build QIRadon.Quadrature.JacobiSymmetry` succeeds warning-free. Axiom audits of polynomial parity and both main reflection existence theorems return only `[propext, Classical.choice, Quot.sound]`. There are no sorry/admit/custom axioms.
