# Formalization status

Coordinator: Codex /root. Started 2026-10-08; updated 2026-10-09.

Sam authorized the general-kernel theorem map, excluding lambda optimization. On 2026-10-09 he prioritized the main mathematical goals and explicitly set aside coefficient-computation cost and secondary extensions. Numerical-acquisition research and the optional improved tanh correction were stopped. They must not be counted as verified.

## Mathematical coverage

| Approved component | Current result |
|---|---|
| T1: Radon and finite boundary setup | Verified from the original bounded local analytic target. Explicit normalized radius density and all margin parameters are constructed. Scalar profiles and centered angular regularity are conclusions, not hypotheses. |
| T2: finite general-kernel certificate | Verified actual cutoff, every alias, finite halo, and r-fold anchoring. General higher-order ordinary-MLP claims require a separately certified anchor realization and count its error/units. |
| L1 / T3: local analytic preparation and scalar rate | Verified actual sampled Chebyshev preparation, finite window, Fourier bounds, actual network error and halo-inclusive O(log^kappa(1/epsilon)) count. |
| T4: algebraic scalar rates | Verified both global polynomial-Fourier O(epsilon^(-1/s)) and local real-sample polynomial-approximation O(epsilon^(-kappa/s)) classes. No claim that mere nonanalyticity implies a rate. |
| T5: angular quadrature | Verified actual positive Gauss–Jacobi rule, exactness, folding, M=q^(d-1), canonical harmonic expansion, geometric bound and sharp Sobolev uniform-norm loss (d-1)/2. |
| T6: multidimensional laws | Verified uniform fixed-network constructions, exact origin anchoring, actual finite width including halos, and the stated fixed-d geometric/algebraic laws. Ordinary GELU raw-input endpoint is verified with two additional anchor neurons total. General r<=1 needs no extra anchor neurons. |
| T7: allocation | Verified continuous direction/row optimum, constraints, high-accuracy split, general-kappa per-spoke multiplier existence/uniqueness, finite search brackets, feasible rounding, and connection to actual finite row errors and grid cardinalities. Optimality concerns the proved continuous certificate, not all integer geometries or all neural networks. |
| Activation examples/obstructions | Actual Gaussian, erf/CDF, GELU and generic tanh envelopes verified. Actual ReLU upper/lower algebraic bounds and fixed-degree polynomial obstruction verified. |
| C_data: split selection | Verified actual finite-max residual coverage/noise certificate and near-best tested width; genuine Bernstein held-out squared-L2 certificate and near-best tested width; finite-sample impossibility without quantitative assumptions. |
| T8: numerical perturbation pieces | Actual parameter/evaluation/summation error, sufficient binary accuracy, readout mass, and sample-preparation stability are verified. Full coefficient-acquisition algorithms and computation costs are deferred at Sam's request. |

## Main endpoints

- `KnownFunctionConstruction.general_kernel_accuracy_and_width`: raw analytic source to actual finite general-kernel approximation; no supplied profile or harmonic-energy oracle.
- `KnownFunctionConstruction.firstOrder_mlp_accuracy_and_width`: ordinary MLP for the general r=0/r=1 classes.
- `GenericAnchoredConstruction.known_target_with_anchor`: arbitrary derivative order with an explicitly supplied realization/approximation of the actual integration polynomial; full target approximation is derived, and all anchor error and units are charged.
- `KnownFunctionConstruction.gelu_accuracy_and_width` / `gelu_total_width_bigO`: actual ordinary GELU MLP and complete count.
- `AutomaticConstruction.known_target_gelu`: only raw target, bounded local analytic extension, R>1 and sigma>0 are supplied; boundary density, margins and geometry constants are explicit.
- `LocalAlgebraicConstruction.local_centered_uniform_accuracy_and_width` / `local_centered_total_epsilon_bigO`: local polynomial-approximation profile class plus centered angular Sobolev class.
- `AllocatedConstruction.allocated_rows_certificate`: optimized continuous certificate attached to the actual rounded finite rows.
- `TanhCorollary.accuracy` / `width_bigO`: explicit prescribed ordinary tanh network, exact origin, and generic log^(d+1) halo-inclusive width. This verified corollary is included at Sam's request; the substantially longer improved-boundary proof remains deferred.

## Exact scope of rates

For spectral/spatial exponents nu,mu, kappa=max(1,1/nu+1/mu). The general analytic rate is W=O(log^(d-1+kappa)(1/epsilon)). Gaussian/erf/GELU have kappa=1 and hence log^d. Generic tanh has kappa=2 and hence log^(d+1). The separate corrected-halo tanh improvement to kappa=1 is **not proved** and has been deferred.

The algebraic full-width exponent is (d-1)/t+1/s for the global Fourier-moment class, and (d-1)/t+kappa/s for the local polynomial-approximation class. Constants depend explicitly on fixed dimension, regularity, kernel envelopes, and boundary margins.

The source rotation estimate uses exp(2 sum|angle|), so the certified angular radius is conservatively smaller than the informal note's sharper radius. This constant difference was disclosed; target assumptions and headline exponents were not changed. The original sharp Hilbert endpoint logarithm is proved.

## Verification and preservation

Lean 4.33.0; Mathlib db584cd6d46c92f209a44c0f1c829460d327499d. Root aggregator imports completed modules only. Final `lake build` and `lake env lean Audit.lean` both passed, including the generic higher-order anchor assembly and explicit tanh corollary. Principal audits use only propext, Classical.choice, and Quot.sound. No proof placeholders or custom axioms were found. Results are preserved in `reports/final_verification.md` and `reports/final_axiom_audit.txt`.

Unfinished secondary exploration is preserved under ignored `scratch/deferred_numerics/` and `scratch/deferred_tanh_correction/`, outside the verified library. External lean-SCV was inspected but never imported; no toolchain upgrade was required. Existing user work outside this formalization was preserved. No commit or push was made.

## Final polish — 2026-10-09

A fresh independent semantic audit of the principal source-to-network chain found no correctness or hidden-assumption defect. Canonical harmonic mean preservation is now proved once and used by both analytic and Sobolev angular estimates, removing their former factor two. The smaller constants propagate to the actual angular-order choices. Five final construction proofs share one positive-weight error-assembly step. The explicit integer-order factor is sharpened from 2+1/b to 1+1/b; halo counts, scalar anchoring, hypotheses and convergence exponents remain intact.

The final full build and 69-declaration axiom audit passed. `reports/final_polish_2026_10_09.md` records the exact changes, and `reports/polish_fresh_audit.md` records independent findings. The local algebraic profile class remains explicitly polynomial approximability, without an additional raw-C^s-to-profile theorem.
