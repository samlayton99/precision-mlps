# Assumption and scope audit against the approved theorem map

Date: 2026-10-08. Reviewer: `lean_local_scaling`, delegated by the coordinator.

This is a statement-level, read-only audit of the implemented mathematics against `docs/radon_scaling_theory_2026_10_08/theorem_map_for_approval.md`. I inspected theorem premises, conclusions, construction definitions, and the relevant proof steps. A successful build by itself was not treated as proof that a theorem matches the requested scope. Work was still advancing while this review ran; the latest inspected additions include `AlgebraicConstruction.lean` and the sharp Hilbert endpoint bound.

I wrote `LocalScaling.lean` and `LocalGELU.lean`, so review of those two modules is self-review, not independent review. Review of the other modules is independent of their implementation. This audit does not replace a final check of the subsequently completed source-to-angular bridge.

## Finding in one sentence

I found no incorrect cutoff/alias/halo normalization, hidden approximation-error premise, or lost halo count in the inspected main proof chain. The existing conditional construction theorems are substantial and accurately bounded. There are, however, several definite gaps between those theorems and the complete approved package; two assembly details also need to be stated correctly before reporting a uniform, centered, optimally allocated final theorem.

## Ranked findings

### High significance: T8 is still a separate, substantial numerical-realizability obligation

`Kernel.realReadoutDensity` is an exact finite-band integral of a quotient of Fourier transforms. `ProfilePreparation.samplePolynomial` is genuinely determined by the finite Chebyshev samples. The actual Gaussian-polynomial preparation has a proved transform and moment bound. These are meaningful constructive mathematical definitions, not a least-squares oracle.

But the main error theorems currently use exact samples, exact quadrature nodes/weights, exact integrals, exact readouts, and exact activation evaluation. `Certification.finite_network_perturbation` and `activation_network_perturbation` propagate already supplied coefficient/feature error bounds; they do not yet establish the number of working bits or work required to obtain those inputs. In particular, no current theorem composes target-sample perturbations through the Chebyshev/window preparation, kernel inversion, density quadrature, anchor evaluation, and final summation into a complete epsilon budget and finite compute cost.

This is acknowledged incompleteness, not a defect in the exact-arithmetic theorems. It is nevertheless material to Sam's requirement that the construction remain calculable in practice. The current headline is a neuron-count theorem in exact arithmetic; it is not yet the approved certified numerical construction or an evaluation-cost-limited algorithm. `noncomputable` Lean definitions are normal for real analysis and are not themselves evidence of impossibility; the missing point is the quantitative implementation bridge.

### Medium significance at audit time: multidimensional accuracy statements were pointwise

**Resolved in subsequent implementation by this reviewer:** `UniformConstruction.local_analytic_uniform_accuracy_and_width` and `algebraic_uniform_accuracy_and_width` fix the constants and network before quantifying x. The centered analogue and an actual fixed ordinary-GELU corollary are also now proved. Those follow-up edits are self-reviewed, not an independent second review.

`LocalMultidimensional.local_analytic_accuracy_and_width`, `CanonicalConstruction.canonical_analytic_accuracy_and_width`, and `AlgebraicConstruction.algebraic_accuracy_and_width` each accept a particular input x and a continuous angular function F representing that input's angular integrand. They prove accuracy at that x.

The same construction can be certified uniformly once the angular norm assumptions hold for every x in the ball with the **same** constants. But this quantifier order must be recorded: choose q, N, and all network parameters once, then prove the bound for every x. Allowing a different a or A at each x would not establish the requested sup norm or one fixed finite network. The coordinator agreed that a wrapper with uniform canonical regularity certificates is required; this reviewer was subsequently assigned to add it. The original-target bridge must itself deliver uniform constants.

### Medium significance at audit time: T5's centered and full-profile integrands differed

**Analytic centered version subsequently implemented by this reviewer:** `CenteredConstruction` uses only centered canonical angular norms, has exact origin anchoring, derives the factor-two scalar transport, and chooses the scalar budget epsilon/4. `CenteredGELU.canonical_uniform_accuracy_and_width` gives one actual ordinary GELU MLP uniformly approximating the canonical Schwartz target with all units counted. The centered/full-profile distinction remains explicit; the algebraic uniform wrapper currently uses full-profile angular norms.

The approved map states angular regularity for

`H_x(theta) = g_theta(theta·x) - g_theta(0)`

and an exact reference-value output constant. The implemented main constructions and canonical norm hypotheses instead use

`F_x(theta) = g_theta(theta·x)`.

Both produce valid approximation methods. They are not literally the same certificate. Controlling the full F_x also controls the direction-dependent origin profile component; regularity of the centered H_x alone does not imply that condition. The current finite construction does not anchor its overall output exactly at f(0): it quadratures the full profiles. Under the intended analytic-source hypotheses both should be controllable, but the extra origin component can affect constants.

To claim the exact centered T5 statement, one needs its centered construction/corollary and corresponding scalar error transport. It is enough to add the reference constant in the output bias; no tanh-difference activation is required. The existing full-profile result is valid as a separate version and should not be silently presented as the centered theorem.

### Medium significance at audit time: T7's per-spoke optimizer covered linear resolution cost

**Resolved subsequently (2026-10-09):** `AllocationPower` and `AllocationPowerSolver` now prove and construct the general-kappa continuous optimizer including baselines. This reviewer added `AllocatedConstruction`, deriving the solver's D/a/c from actual local row certificates, proving the actual sampled-row weighted error budget after ceiling, and bounding the actual sum of all halo grids by 2^kappa times the continuous optimum. See `reports/allocated_construction.md`. This is certificate optimality with a rounding guarantee, not exact integer-global/network-global optimality. The integration module is self-reviewed.

`Allocation.existsUnique_analytic_optimizer`, its constrained version, and its asymptotic fraction/ratio correctly handle the two-factor continuous cost with general kappa. The algebraic ratio also has the desired dimension and smoothness dependence. Their conclusion is optimality of the stated continuous certificate, not a lower bound for arbitrary neural networks; source comments correctly make that distinction.

The per-spoke formulas `spokeSplit_global_min` and `waterfill_global_min`, however, minimize **linear** resolution cost subject to errors `D_i exp(-a_i n_i)`. This matches the kappa=1 family after the appropriate width factor. For the generic kappa>1 local construction, actual sufficient row cost grows as `C_i n_i^kappa`; the displayed water-filling formula is not its optimizer. No generic nonlinear-cost per-spoke optimizer is currently present.

There is also not yet one theorem tying the continuous optimum and its rounding to the exact `localGrid` cardinalities and the actual construction orders. `rounded_product_width` proves the stated numerical rounding inequality for q and H, but it does not identify H with a particular chosen local preparation order/grid. This is an integration obligation, not an error in the optimization calculus. Reporting a general-kernel, actual-neuron-count optimal per-spoke construction would currently overstate the result.

### Medium significance at audit time: local finite-smooth preparation was missing

**Resolved in subsequent implementation by this reviewer (2026-10-09):** `ProfilePreparation.FilteredSamples` proves an actual norm-3 real-sample filter with exact low-degree reproduction and error≤4 times best polynomial error. `LocalAlgebraic`, `LocalAlgebraicScaling`, and `LocalAlgebraicConstruction` compose the actual Gaussian preparation, finite network, centered uniform angular Sobolev theorem, and exact all-halo width law epsilon^(-((d-1)/t+kappa/s)). No original-target complex continuation or global Fourier hypothesis is added. See `reports/local_algebraic.md`. These follow-up modules are self-reviewed.

The scalar global polynomial-Fourier-moment theorem and the actual canonical Sobolev angular theorem are proved, and `AlgebraicConstruction` now combines them into an actual finite epsilon construction. This is a real algebraic theorem.

The separate L1/T4 variant promised for targets specified only through local polynomial approximation is not proved by those results. That variant needs a real-sample filtered polynomial approximation with its error and stability cost propagated through preparation. Its expected generic exponent is epsilon^(-kappa/s), not epsilon^(-1/s). The current notes correctly distinguish these classes; the final report must preserve that distinction rather than label the global-moment theorem a proof for every finite-smooth local target.

### Remaining approved theorem families, currently not complete

- ReLU: the lower bound is proved for the actual W-neuron formula with arbitrary parameters. The promised constructive spline upper bound is absent.
- Fixed-degree polynomial activations: no approximation obstruction theorem is present yet.
- Tanh: neither the generic mu=nu=1 envelope corollary nor the separate improved boundary-correction corollary is implemented in the inspected activation files. The Gaussian/erf/exact-GELU family is implemented.
- C_data: the deterministic coverage certificate and minimum-width-among-passing-candidates theorem are proved. The statistical Bernstein alternative and the finite-sample narrow-bump impossibility result remain absent. The approved near-best-among-candidates statement with the explicit coverage/noise margin also needs to be linked to the deterministic selector rather than inferred from its name.
- General r anchors: the generic finite activation theorem deliberately retains its integration polynomial. For an actual ordinary MLP claim at arbitrary r, supply a realization/approximation theorem and count/error. Standard GELU at r=2 is closed by exactly two global neurons, and r=0/r=1 have the expected zero-unit/output-bias cases. There is no claim that all activations realize every anchor exactly.

These items are not newly discovered contradictions. They are the remaining scope needed before calling the entire approved map complete.

## Checks that passed

### Frequency conventions and actual coefficients

`Kernel.lean` uses cycles frequencies, with the inverse transform phase exp(2 pi i xi x). Thus its aliases occur at k/h, and the finite-band density has no angular-frequency 1/(2 pi) prefactor. The approved note uses angular frequencies and therefore has aliases at 2 pi k/h and the 1/(2 pi) prefactor. These are consistent after the change of variables omega=2 pi xi.

I checked the potentially dangerous mixed-convention location: `ScalarRate.analytic_grid_width_cycles` explicitly substitutes `H_angular = 2*pi*H_cycles` into the older `KernelRates` width helper and proves the resulting spacing equality. The actual finite construction uses h=1/(H B). I found no dropped 2 pi factor in that bridge.

The actual coefficient factor is `h*gamma/(m*gamma^r)`, so it is valid even at r=0; it is not the incorrect natural-number exponent version of gamma^(r-1). `complexScaledKernel_fourier` proves the positive-bandwidth multiplier `Khat(xi/gamma)`. `finite_spectral_kernel_certificate` derives all aliases and the omitted-center tail before the rate lemmas estimate them.

### No circular approximation assumptions in the main completed chain

The general finite certificate assumes an integrable target spectrum and nonvanishing kernel spectrum on the used band. Its network error is a conclusion. The uniform premise in `finite_activation_function_certificate` bounds the explicitly defined cutoff/alias/halo quantity, not the network error; subsequent envelope theorems derive it.

The local preparation constructs the actual Chebyshev sample polynomial, proves its interpolation error, multiplies by the actual finite window, derives its Fourier moments, and invokes the finite construction. It does not assume that an unspecified polynomial approximation or network exists.

The canonical angular theorem uses the actual orthogonal harmonic coordinates of F and their weighted norm. It no longer assumes a supplied convergent expansion or an approximation rate. Such a norm hypothesis is legitimate for the conditional regularity-class theorem. It must still be derived by T1 before invoking the original local-target theorem. That bridge is explicitly active work and is not counted here as an audit defect.

### Halos, ordinary MLPs, and counting

The counted local grid is literally the symmetric integer interval `[-floor((T+D)/h), floor((T+D)/h)]`; `symmetric_grid_card` proves its exact cardinality. The tensor index is the retained sphere-rule index times that grid. All halo neurons are counted. The fold has exactly q^(d-1) retained directions.

`Architecture.network` is an ordinary single-hidden-layer MLP. Its dense parameter count is `(d+2)*W+1`. The general anchored evaluator is not silently equated with such an MLP when r>=2. `LocalGELU.local_gelu_is_network` does prove that equivalence globally in x for exact GELU and charges two extra neurons total. That part of this review is self-review, as noted above.

The constants μ,ν and the kernel envelope determine kappa=max(1,1/μ+1/ν). Thus the generic result is log^(d-1+kappa), not log^d for every activation. Exact GELU's signed second-derivative kernel has the proved nonvanishing Gaussian-polynomial spectrum and μ=ν=2; the signs do not invalidate the spectral argument.

### Domain, preparation scale, and asymptotics

`LocalProfileRate` is normalized to the target interval [-1,1], with a separately supplied Bernstein ellipse parameter delta. Its `T₀>1` is the Gaussian flat-window scale. Unlike the informal L1 wording, its samples are in [-1,1], not [-T₀,T₀]. The proof controls the resulting polynomial's off-interval growth explicitly, so this is not an unjustified extension of the target. The scaled theorem samples g(T times node) and correctly rescales input, centers, slopes, and halo length for [-T,T].

This notation should be explained in the final note: code `T₀` is not automatically the target's analytic real-domain radius; code `N` is the preparation degree/order, not the exact number of interior centers; code `H` is an oversampling factor, not the total row width used by the theorem map. The actual neuron cardinality theorems avoid relying on these labels.

The local width laws require fixed rates and constants as epsilon tends to zero. Dimension enters the exponent and constants, including the harmonic trace constant. The code does not assert a dimension-independent Big-O constant or remove the curse of dimensionality. The sharp Sobolev angular theorem uses the correct uniform-norm loss `(d-1)/2`.

### Quadrature is mathematically constructed, but not yet a certified floating-point algorithm

The Gaussian-rule proof obtains the monic orthogonal polynomial, its real simple interior roots, and integrated Lagrange weights; positivity, normalization, and degree exactness are proved. The recursive sphere rule derives actual normalized sphere integration from moments. The resulting node definitions use classical choice to enumerate roots. This is ordinary exact mathematical construction and does not assume quadrature correctness, but root isolation precision/work belongs to T8.

The present circle base is the half-offset recursive Chebyshev/Jacobi version. Its exactness and folding are proved. Equivalence to arbitrary phase-shifted circle arrays remains a separate optional identification. The approved map only requires the recursive positive Gauss–Jacobi scheme, so this is not a headline rate failure; do not claim exact identity with an older concrete direction array without the identification.

## Overall conclusion

The completed exact-arithmetic conditional theory has not secretly replaced the neural network with an unrelated solver or assumed its own approximation conclusion. The principal remaining risk is reporting scope too broadly: pointwise versus uniform accuracy, full versus centered angular profiles, general kappa versus linear-cost per-spoke allocation, and exact-real neuron count versus certified finite arithmetic. The ongoing source bridge plus these named assembly/coverage obligations should remain visible in the completion criteria.

## Follow-up verification

The coordinator assigned the two assembly repairs identified above back to this reviewer. New files are `QIRadon/CenteredConstruction.lean`, `QIRadon/CenteredGELU.lean`, and `QIRadon/UniformConstruction.lean`. They do not alter existing theorem hypotheses or replace the outstanding original-target regularity bridge. The centered construction's target equality is separately proved from actual canonical Radon reconstruction, including derived angular integrability. Its local-agreement version only requires agreement of the scalar continuation with the canonical profile for |s|≤1, avoiding a hidden global continuation assumption.
