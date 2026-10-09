# General-kernel QI–Radon formalization

The main mathematical construction, geometric/algebraic width laws, and resource-allocation results are verified in Lean. See `STATUS.md` for exact theorem coverage and assumptions. The optional stronger tanh boundary correction is not verified. Numerical-acquisition cost analysis was stopped at Sam's request on 2026-10-09.

The specification is the historical [approved theorem map](../../docs/radon_scaling_theory_2026_10_08/theorem_map_for_approval.md); later user instructions narrow the remaining work as recorded in `STATUS.md`. Lambda optimization is excluded.

Main raw-input endpoint: `AutomaticConstruction.known_target_gelu`. It starts with a real target and bounded local analytic extension, constructs the boundary setup and finite geometry, and proves uniform accuracy for one ordinary shallow MLP with all halo and anchor units counted. General-kernel and algebraic endpoints are listed in `STATUS.md`.

Build with `lake build`, then audit with `lake env lean Audit.lean`. Pinned environment: Lean 4.33.0 and Mathlib `db584cd6d46c92f209a44c0f1c829460d327499d`.

Only completed modules are imported by `QIRadon.lean`. No project axioms, `sorry`, `admit`, or assumed approximation conclusions are permitted. Standard Lean foundations are allowed. Compilation and axiom checking are complemented by statement-level reviews in `reports/assumption_audit.md`, `reports/automatic_construction_audit.md`, and `reports/source_assumption_independent_review.md`.
