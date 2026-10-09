# Optimizing all-orders Casimir projection estimates

`QIRadon/Quadrature/CasimirDecay.lean` proves the scalar optimization and weighted harmonic-energy consequences needed by the angular argument. It does not itself derive the projection estimates from holomorphy or a Casimir operator. That derivation is a separate, necessary input from the angular modules.

All names have namespace `QIRadon.CasimirDecay`.

## Input and optimized coefficient bound

Suppose `B≥0`, `c>0`, `b_0≤B`, and for every integer `k≥1` and every integer `j≥0`,

`b_k ≤ B (c*j/k)^(2j)`.

`optimized_positive_index` explicitly chooses `j=floor(k/(2c))`. Then `c*j/k≤1/2`, while `j>k/(2c)-1`. Consequently,

`b_k ≤ B 4^(-j) ≤ 4B exp(-(log 2/c)k)`.

The proof includes the case where the floor is zero. It uses the `j=0` projection estimate, whose right side is B. No positivity assumption on that floor is inserted.

`all_orders_decay` adds the constant layer `k=0`, using the given `b_0≤B`, and proves the exponential bound for every k.

## Actual weighted energy

For nonnegative `b_k` (the intended application is an actual coefficient norm), define

`a=log 2/(2c)`,

`A=4B/sqrt(1-exp(-log 2/c))`.

`all_orders_energy_certificate` proves both

- `Summable (fun k => exp(2*a*k)*b_k²)`, and
- `sum_k exp(2*a*k)*b_k² ≤ A²`.

This is derived from the all-orders estimates. It is not a repackaged hypothesis asserting weighted energy or exponential decay.

The termwise majorant is `16B² exp(-(log 2/c)k)`, and its sum is the exact geometric-series expression `16B²/(1-exp(-log 2/c))`. The denominator is positive because c is positive. `energyRate_pos`, `energyConstant_nonneg`, and `energyConstant_sq` expose the facts needed by the canonical harmonic-expansion interface.

If the angular calculation supplies `c=4d/δ`, these constants give `a=δ log 2/(8d)`. This substitution is only an interpretation of the generic proved theorem; the actual angular projection bound still needs to be supplied.

## Verification

`lake build QIRadon.Quadrature.CasimirDecay` succeeds. The primary floor optimization, all-index decay, weighted-series, and energy-certificate theorems were inspected with `#print axioms`; only `propext`, `Classical.choice`, and `Quot.sound` occur. There are no proof placeholders or custom axioms.
