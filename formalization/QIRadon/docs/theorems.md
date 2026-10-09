# Verified theorem map

All names below are prefixed by `QIRadon.`. Import `QIRadon` to access the
complete verified library. [The argument](argument.md) defines the mathematical
objects and explains how these statements compose.

The PDF is organized around four main theorems. Its corrected tanh versions of
the analytic and local algebraic width laws use a scalar contour argument proved
in the note but not yet formalized. The table below maps the **existing Lean
statements**, including the weaker generic tanh exponent, rather than claiming
the complete corrected variant is verified. Allocation and numerical error
results are already generic in the scalar cost exponent, so their statements
do not change when the corrected construction supplies exponent one.

| Result | Main assumptions and conclusion | Declaration / source |
|---|---|---|
| Raw known-target endpoint | Bounded local analytic extension on a larger ball, R>1 and positive complex margin. Constructs boundary parameters and one ordinary GELU MLP; uniform target error ≤epsilon. | `AutomaticConstruction.known_target_gelu`; [source](../QIRadon/Construction/AutomaticConstruction.lean) |
| General-kernel construction | The same analytic source plus an r-th-derivative kernel, spectral/spatial envelopes, and stated margins. Derives finite error and all-halo grid count. | `KnownFunctionConstruction.general_kernel_accuracy_and_width`; [source](../QIRadon/Construction/KnownFunctionConstruction.lean) |
| Ordinary MLP, r≤1 | The general-kernel assumptions and r≤1. No additional anchor neurons. | `KnownFunctionConstruction.firstOrder_mlp_accuracy_and_width`; same source |
| Ordinary MLP, higher r | Explicit approximation of the actual degree-below-r integration polynomial. Its error and neurons are charged in the final conclusion. | `GenericAnchoredConstruction.known_target_with_anchor`; [source](../QIRadon/Construction/GenericAnchoredConstruction.lean) |
| Geometric full-width law | Fixed dimension/regularity/kernel parameters; kappa=max(1,1/nu+1/mu). Actual rounded orders and halos give log^(d−1+kappa) width. GELU includes two anchor neurons and gives log^d. | `LocalScaling.local_total_grid_epsilon_bigO`, `KnownFunctionConstruction.gelu_total_width_bigO`; [scaling](../QIRadon/Construction/LocalScaling.lean) |
| Global algebraic law | Polynomially weighted derivative Fourier moment and angular Sobolev norm. Width exponent (d−1)/t+1/s. | `AlgebraicConstruction.algebraic_accuracy_and_width`, `MultidimensionalScaling.algebraic_total_grid_epsilon_bigO`; [construction](../QIRadon/Construction/AlgebraicConstruction.lean) |
| Local algebraic law | Uniform scalar polynomial approximability with rate n^(−s), bounded profile values, and centered angular Sobolev norm. Actual sample-only construction; exponent (d−1)/t+kappa/s. | `LocalAlgebraicConstruction.local_centered_uniform_accuracy_and_width`, `LocalAlgebraicConstruction.local_centered_total_epsilon_bigO`; [source](../QIRadon/Construction/LocalAlgebraicConstruction.lean) |
| Finite kernel certificate | Actual kernel and derivative spectrum, finite spectral cutoff, positive spacing, and finite grid. Bounds cutoff, all aliases, and the full omitted-center complement. | `Kernel.finite_spectral_kernel_certificate`, `Kernel.finite_activation_function_certificate`; [source](../QIRadon/Kernel/Certificate.lean) |
| Canonical angular estimate | Continuous even angular integrand, actual canonical analytic/Sobolev norm. Derives the error of the actual positive folded rule. | `SharpAngular.canonical_quadrature_error`, `Gaussian.canonical_foldedSphere_sobolev_error`; [analytic](../QIRadon/Quadrature/SharpAngular.lean), [Sobolev](../QIRadon/Quadrature/CanonicalConvergence.lean) |
| Allocation | Positive cost/error constants and the sufficient continuous error model. Unique optimum and high-accuracy split; no arbitrary-network optimality claim. | `Allocation.algebraicSplit_strict_min`, `Allocation.existsUnique_analytic_optimizer`, `Allocation.analytic_optimizer_fraction_tendsto`; [source](../QIRadon/Allocation/Uniform.lean) |
| Unequal-spoke allocation | General power cost, positive per-spoke rates and prescribed minimum orders. Proves the multiplier solver and attaches it to actual rounded rows. | `AllocationPower.existsUnique_spoke_root`, `AllocatedConstruction.allocated_rows_certificate`; [solver](../QIRadon/Allocation/AllocationPowerSolver.lean), [assembly](../QIRadon/Construction/AllocatedConstruction.lean) |
| Activation examples and limits | Verified Gaussian, erf/CDF, GELU, and generic tanh envelopes; separate actual ReLU upper/lower algebraic bounds and fixed-degree polynomial obstruction. | `TanhCorollary.accuracy`, `TanhCorollary.width_bigO`, `ReLUSpline.network_uniform_error_of_contDiff`, `PolynomialActivation.network_error_lower_bound`; [activations](../QIRadon/Activations/) |
| Selection from data | Explicit coverage/noise conditions or held-out sampling assumptions. Near-best tested width with the corresponding error certificate. | `DeterministicSelection.exists_sample_selected_near_best`, `StatisticalSelection.validation_selection_certificate`; [source](../QIRadon/Validation/) |
| Numerical perturbations | Explicit parameter, evaluation, summation, or preparation perturbation bounds. No complete coefficient-acquisition/runtime theorem. | `NumericalPrecision.rounded_network_error`, `PreparationStability.sampled_network_error`; [source](../QIRadon/Numerics/) |

## Scope boundaries

- The original target assumptions are discharged into the scalar and centered
  angular hypotheses in the analytic endpoint. They are not left as profile or
  approximation oracles.
- The algebraic profile class is not a separately proved raw-C^s-to-profile
  theorem. Its polynomial-approximation and angular Sobolev assumptions remain
  explicit.
- Generic higher-order activations need an anchor realization. The theorem
  never silently treats a polynomial-plus-network expression as an ordinary MLP.
- Generic tanh has the Lean-verified log^(d+1) count. The corrected-halo log^d
  argument is included in the main human proof and used by the experiments;
  its finite-contour and rational-correction lemmas remain unformalized.
- These are exact-real approximation results. Coefficient acquisition, complete
  runtime complexity, and an unknown-PDE solver are outside their conclusions.
