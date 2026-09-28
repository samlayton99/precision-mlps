# Section 3: recovered appendix material

Prepared 25 September 2026. This is a source collection, not an assembled appendix. It follows **Section 3 of the supplied current draft**, gives newer work priority, and leaves arithmetic circuits and the later sections out.

**The main additions are the finite-kernel spectral proof, the neighboring-feature conditioning proof, the geometry/readout identities and drift bounds, and the recent numerical protocols and audits.** Most of the representation and tanh bandwidth-selection proofs are already in the supplied collection. They should not be added again under an older title.

- [All source files, individually named and linked](SOURCES.md)
- [Search coverage, other tasks, and provenance](SEARCH_SCOPE.md)
- [Superseded material and remaining gaps](EXCLUSIONS_AND_GAPS.md)
- [Checksummed source manifest](MANIFEST.json)

The packet contains unchanged snapshots. Supporting data and code remain in the repository; their original locations are linked in the source manifest. Existing proof reviews are included, but this inventory is not a new line-by-line certification of every proof.

## What the supplied collection already covers

| Source | Material already available | Place in the current draft |
|---|---|---|
| [Current paper](supplied/current_paper.pdf) | The construction, current theorem statements, figures, and an existing Appendix A containing representation and recovery material. | Selection criterion throughout. |
| [Fixed-lambda three-theorem note](supplied/fixed_lambda_tanh_three_theorems.pdf) | Analytic reference density; signed midpoint/pole expansion; finite-interval halo correction; coefficient bounds; constrained recovery; sampled-to-continuous certification. | 3.1–3.2. |
| [FP64 scaled-SVD recovery proof](supplied/fp64_scaled_svd_recovery_proof.pdf) | Coefficient-envelope scaling, arithmetic allowances, SVD recovery, export and evaluation error, and continuous RMS transfer. | 3.2. |
| [Comparison derivations](supplied/comparison_bounds.tex) | Staircase, Mhaskar, Costarelli–Spigler, ChebNet, and the conditions needed for the QUILL row. | Table 1 in 3.2. |
| [Bounded-slope accessibility](supplied/bounded_slope_readout_accessibility.tex) | Polynomial-tail access, coefficient cost, gradient-flow and discrete-GD delay, readout-coordinate controls, and damping. | An alternative 3.4 route already supplied; do not duplicate the older polynomial notes. |
| [High-precision bandwidth prediction](supplied/predicting_high_precision_bandwidths.tex) | Finite-contour tanh aliasing, first-pair formula, scalar selection rule, frequency estimation, approximation conditions, inversion and sensitivity, and recovery transfer. | 3.3. |

The comparison derivations match `docs/quill_table_bounds_appendix.tex` line for line; their file hashes differ only because of newline formatting. The newer FP64 recovery note and the fixed-lambda note also use different recovery/evaluation contracts. Preserve that distinction when combining them; do not silently substitute the earlier constrained solve or exact accumulator for the newer all-binary64 algorithm.

## 3.1 — Construction: one useful practical source

### A. The concise QuILL algorithm from the other task

**Recovered:** [Practical QuILL pseudocode](sources/recovered/practical_quill_algorithm.tex), from **Add QuILL setup algorithm**. This is the final short LaTeX version from that task, preserved without alterations.

It gives domain normalization, grid placement, target sampling, bandwidth selection, hidden weights/biases, the readout solve, and mapping back to the original interval. This is useful implementation-facing material, not another approximation theorem.

**Before reuse:** its practical convention is 24 halo nodes per side and samples proportional to total width, whereas the current theorem uses a square-root halo and the draft describes samples proportional to the number of interior cells. Its generic SVD instruction is not a specification of the certified scaled-SVD procedure. Keep it as a recipe source and reconcile those conventions during appendix assembly.

## 3.2 — Realizability: the useful additions are implementation evidence

### B. The true-p-bit construction and solve

**Primary sources:** [operation-by-operation specification](sources/repo/experiments/expC11_true_precision_law/SPEC.md), [completed chirp report](sources/repo/results/checkpoint_C_geometry/expC11_true_precision_law/expC11_results.md), and [Runge extension](sources/repo/results/checkpoint_C_geometry/expC11_true_precision_law/runge25_comparison/expC11_runge25_results.md).

This work is newer than the original precision plots and directly relevant to Figure 2(c). It specifies rounding in geometry formation, feature evaluation, least squares, coefficient storage, and evaluation. The implementation includes a precision-controlled LAPACK DGELSS port, independent replay, arithmetic checks, and native-format comparisons. The saved results cover the full 8–53-significand-bit sweeps for chirp and Runge.

**What it adds beyond the supplied proof:** an explicit experimental arithmetic contract and reproducibility evidence for an actual low-precision construction pipeline. The theorem alone does not document this implementation.

**Scope:** it uses its documented DGELSS/cutoff and evaluation procedure, not the complete certified reference-scaled algorithm of the new FP64 note. A successful precision sweep is evidence of realizability; it is not an implementation certificate for a different solver. The earlier C09 plots round fewer stages and retain an FP64 solve. They must not be substituted for this result.

The [current combined figure](sources/repo/results/checkpoint_C_geometry/expC11_true_precision_law/figures/all_targets_chirp_precision_law_true.png) is included. The specification and reports point to the implementation and tests; no tests or training were rerun for this inventory.

### C. The comparison implementations and numerical-rescue controls

**Newest source:** [C13 five-method specification](sources/repo/experiments/expC13_five_method_comparison/SPEC.md). It describes a shared precision model and nonzero-parameter budget for QUILL, Mhaskar, staircase, Costarelli–Spigler, and ChebNet, with faithful variants separated from implementation improvements.

This work is more relevant to the current Table 1 than a generic literature dump. The repository already contains code, figures, models, and partial measurements. **There was no completed C13 report at the time of this sweep.** The saved summary was still incomplete, so the source is collected as a protocol and work-in-progress result, not as a finished comparison.

Two completed earlier reports are also useful:

- [C12 strict comparison](sources/repo/results/checkpoint_C_geometry/expC12_mhaskar_comparison/strict/expC12_strict_results.md): documents the approximation-to-MLP conversion and precision experiment. Its QUILL solve has an explicit FP64 exception; C13 removes that exception.
- [C12 numerical rescue](sources/repo/results/checkpoint_C_geometry/expC12_mhaskar_comparison/rescue/expC12_rescue_results.md): tests compensated summation/dot products, symmetric coefficient formation, and higher-order finite-difference extrapolation. It separates accumulation error from errors already present in the constructed network. This is useful evidence that the chosen baseline was investigated rather than dismissed after one poor implementation.

The [comparison bibliography](sources/repo/docs/quill_comparison_references.bib) is an accessory missing from the pasted proof. The older broader comparison appendix is superseded by the supplied version. None of these empirical comparisons proves a necessary precision lower bound for every competing construction.

### D. The envelope-scaling implementation audit

**Source:** [C10 scaled-readout comparison and corrections](sources/repo/results/checkpoint_C_geometry/expC10_scaled_readout_comparison/expC10_results.md), recovered with the context in **Junmi Optimization**.

This records raw versus envelope-scaled versus square-root-spacing-scaled solves, followed by the corrected square-root halo and outward-rounded envelopes. It also records which plotted geometries meet the theorem's sufficient conditions.

**Why retain it:** it identifies exactly where our practical experiment differed from the theorem: halo rule, relative SVD cutoff, chosen alias budget, analytic-neighborhood parameter, and evaluation arithmetic. It prevents using a negative scaling experiment as evidence against a certified recovery theorem that was not actually implemented. This is an audit/protocol appendix source, not an additional recovery theorem.

### E. Optional: the readout magnitude law

**Sources:** [readout norm derivation](sources/repo/docs/theory_magnitude_rule.md) and [readout structure measurements](sources/repo/results/checkpoint_A_numerics/expA06_readout_structure/expA06_results.md).

The distinct result is the link between target derivatives and solved coefficient norms in an ideal uniform-grid model, including the leading tanh relationship between a coefficient and the target derivative. It also explains why near-Nyquist content and numerical truncation change the simple norm scaling.

This is older and conditional on its lattice/sampling assumptions. The current representation note already supplies the coefficient bound needed for Theorem 3.1. Use this source only if an empirical explanation of readout magnitudes is wanted; it is not a replacement proof for the finite-network bound.

## 3.3 — Bandwidth selection: preserve extensions, not another tanh derivation

### F. The revised activation appendix and selector

**Sources:** [Revision 7 appendix PDF](sources/repo/docs/lambda_theorem_compatibility/choosing_optimal_lambda/choosing_optimal_lambda_appendix.pdf), [LaTeX](sources/repo/docs/lambda_theorem_compatibility/choosing_optimal_lambda/choosing_optimal_lambda_appendix.tex), [scope/reproduction](sources/repo/docs/lambda_theorem_compatibility/choosing_optimal_lambda/README.md), and [revision history](sources/repo/docs/lambda_theorem_compatibility/choosing_optimal_lambda/revision_notes.md).

Most tanh content is superseded by the bandwidth note just supplied. The additional material worth preserving is the **GELU kernel transform and practical selector**, activation-specific examples, and the existing two-activation validation figure/recipe. The finite-contour theorem is tanh-specific; the general activation recipe does not inherit that theorem automatically.

The standalone selector and figure sources remain in the original directory linked by the manifest. The supplied newer note should control the tanh formula and interpretation: replacing the full spectrum by a representative frequency is a practical approximation, and minimizing a non-aliasing envelope under an aliasing budget is not a proof of globally minimal measured error.

### G. The exact lattice aliasing and projection results

**Primary source:** [lambda-rule theory](sources/repo/docs/lambda_rule_theory.md), especially Sections 1–5 and 8. Read it with the [consolidated corrections](sources/repo/results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/hardened_rule.md) and [independent audit](sources/repo/results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/SKEPTIC_REVIEW_2026-09-08.md).

Useful material not identical to the new finite-interval tanh proof:

- The Fourier-fiber projection formula for the best approximation in a translate span.
- The exact aliasing floor for bandlimited targets in the stated whole-line/periodic setting.
- The difference between the least-squares floor and the larger cardinal-interpolation error.
- Riesz/conditioning formulas and the transform table for several activations.

This is a supporting explanation or activation extension. It is **not** an additional proof that the exact infinite-lattice floor holds unchanged on our finite interval. The later signed-contour/halo note handles the finite construction. Several old two-error-model claims were withdrawn; the audit travels with this source for that reason.

### H. The recent width/bandwidth measurements and a useful failure case

**Sources:** [C09 protocol](sources/repo/results/checkpoint_C_geometry/expC09_bandwidth_figures/expC09_results.md), [local semilog refinement fits](sources/repo/results/checkpoint_C_geometry/expC09_bandwidth_figures/refinement_rates/manual_intervals_results.md), [underresolved sine controls](sources/repo/results/checkpoint_C_geometry/expC09_bandwidth_figures/sine24_panels/sine24_results.md), and [C08 anchor-rule evidence](sources/repo/results/checkpoint_C_geometry/expC08_anchor_rule/expC08_results.md).

These are the sources for the empirical side of Sections 3.2–3.3: geometric refinement over specified width ranges, bandwidth curves, predicted markers, and checks when the prediction is poor. The underresolved sine example is particularly useful because it distinguishes a successful alias budget from a successful total-error prediction when recovery is ill-conditioned. The semilog slopes are fits over disclosed intervals, not new asymptotic proofs.

The newer C11 report controls the precision panel. C09 remains the source for the width and bandwidth panels it measured.

## 3.4 — Frozen-geometry training: the largest missing proof bundle

### I. Our two-sided finite-network ratio theorem and its full appendix

**Primary sources:** [expanded Revision 4 PDF](sources/repo/docs/frozen_geometry_capacity_access/gamma_ratio_note/gamma_ratio_note_v4.pdf), [expanded Markdown](sources/repo/docs/frozen_geometry_capacity_access/gamma_ratio_note/gamma_ratio_note_v4.md), and [latest compact version](sources/repo/docs/frozen_geometry_capacity_access/gamma_ratio_note/gamma_ratio_note_compact.pdf).

This is the closest match to the current Theorem 3.2 and its proof sketch. The supplied collection does not contain it. Preserve these exact parts:

| Part | Material to salvage |
|---|---|
| Main Sections 1–2; Appendix A | Gradient descent residual recurrence, normalized eigenvalues, center-integral matrix, and the explicit distance–coth expression. |
| Appendix B | Positive Fourier quadratic form; precisely where the tanh multiplier enters. |
| Appendix C | Explicit lattice corrections, the bound on omitted aliases, finite-halo subtraction, and the remaining matrix error. |
| Appendix D | Interlacing, two-sided finite eigenvalue/ratio intervals, and bounds for the largest eigenvalue. |
| Appendix E.1–E.2 | Converting ratio endpoints to error and necessary/sufficient step counts; the extension to a stated scalar learning-rate class. |
| Appendix E.3 | A counterexample to universal finite-ratio monotonicity; keeps the final statement within its actual scope. |
| Appendix F | Numerical construction, target weights, unresolved modes, and reproducibility. |

The [mathematical review](sources/repo/docs/frozen_geometry_capacity_access/gamma_ratio_note/revision4_math_review.md) and [argument review](sources/repo/docs/frozen_geometry_capacity_access/gamma_ratio_note/revision4_argument_review.md) explain the corrected inequality directions and numerical limits. Upper ratio endpoints give lower residual bounds and necessary times; lower endpoints give upper residual bounds and sufficient times. Earlier anchor-dependent lower-ratio versions are not the source to import.

**Status:** a written finite-network theorem with explicit hypotheses and reviewed derivations. Its displayed numerical evaluations are checked floating-point calculations, not directed-rounding certificates. It still uses a corrected-matrix spectrum and actual finite-kernel target projections. It does not claim a closed scalar gamma-to-rank law or universal improvement of every normalized ratio.

**Figure provenance matters:** the [included interval figures](sources/repo/results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/note_interval_figures/MANIFEST.md) use 153 hidden neurons, 263 samples, and a mixed sine with a 10π final frequency; the curves are spectral calculations. The current paper's Figure 3 specifies width 512, a 14π final frequency, and executed updates. These are different experiments. I did not locate a matching current-Figure-3 run manifest in this repository sweep.

### J. Neighboring features and coordinate conditioning

**Primary source:** [neighbor-difference conditioning proof](sources/repo/docs/neighbor_difference_conditioning.md). Supporting sources are the [PR implementation review](sources/repo/docs/frozen_geometry_capacity_access/pr_conditioning_review.md), [scale-learning consolidation](sources/repo/results/checkpoint_D_optimizers/expD06_fixed_center_scales/scale_learning_consolidation.md), and [matched frozen-coordinate ablations](sources/repo/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/expD36_ablation_results.md).

This contains distinct, useful mathematics:

- Width-independent whole-line Riesz bounds for neighboring tanh differences at fixed relative bandwidth.
- The coefficient-coordinate transformation and the width-dependent conditioning penalty it removes.
- The remaining exponential small-bandwidth factor: localization improves coordinates without removing all smoothing-related difficulty.
- A separate near-null direction created by saturated halo features and the bias on a finite interval.
- The exact combinations of reference scaling and neighboring, including their effect on physical gradient-descent updates.

This is more detailed than the supplied accessibility note's general reparameterization discussion. It is useful for explaining the conditioning confound and documenting the control experiments. The whole-line interior result is not a guarantee for the entire finite sampled dictionary including halo and bias; the source explicitly separates those objects.

### K. Keep the collaborator's latest full note as an alternative, not a duplicate of ours

**Source:** [From gamma to slow target directions: the complete argument](sources/attachments/gamma_optimization_full_note.pdf), recovered through **theorems (pt 2)**.

It adds an explicit sampled Fourier factorization, distant-transition and harmonic-truncation bounds, a target-overlap theorem, and a full-curve perturbation bound. The main useful distinction is between its **simplified analytic delay theorem** and its **accurate full spectral forecasts**. The updated note evaluates the former directly: the disclosed cutoff scan yields only 17/13/12/11 necessary updates, versus observed counts of roughly 15.8 million/186 thousand/62 thousand/16 thousand. Its accurate forecasts are a different calculation.

Also worth salvaging if needed: Section 9 gives a coefficient-space block-power identity and a tanh Gram identity used for an independent ball-arithmetic endpoint check. The note reports certification of particular primary timing endpoints, not all figures or all claims. The checker code and its certificate outputs were not located in this repo sweep, so this packet preserves the mathematical description and the author's stated scope rather than presenting a newly verified certificate.

The sampled-Fourier method permits center configurations beyond our lattice construction. That is an additional method, but the present draft's distance–coth proof sketch follows our route. Do not merge the two proofs or their numerical examples under one theorem without making the change explicit.

## 3.5 — Geometry acquisition: useful pieces, incomplete overall claim

### L. Exact loss/projection identities and the limited-drift proof

**Most useful sources:** [energy-bound audit](sources/repo/docs/gamma_barrier_synthesis/energy_bound_audit.md), [projection audit](sources/repo/docs/gamma_barrier_synthesis/projection_audit.md), and [theory framework](sources/repo/docs/gamma_barrier_synthesis/theory_framework.md).

These contain existing mathematics worth recovering:

- The exact least-squares split into approximation error and unfinished readout fitting, and the derivative of the approximation term along joint training.
- The distinction between the current-readout projected gradient and the VarPro gradient evaluated at solved coefficients. They are not interchangeable.
- A loss-energy bound on raw-parameter travel, its discrete sufficient-descent version, and its learning-rate/parameterization dependence.
- Conditional width scaling for the time needed to reach a specified collection of large slopes, and the stronger late-time bound using the remaining actual loss.

For fuller earlier derivations, retain the [gamma-barrier handoff](sources/attachments/gamma_barrier_handoff_v2.pdf): Sections 3–4 for projection and local sensitivity, Section 5 for drift, and Section 6 for the whitening distinction. The [shorter readout–scale note](sources/attachments/readout_scale_competition.pdf) adds the exact bilinear allocation toy and a controlled small-preactivation tanh remainder. These are supporting pieces; newer audits determine which interpretations survived.

**What they do not provide:** a complete theorem that ordinary joint training cannot reach any sufficiently accurate geometry. A distance bound to the QI regime does not show that every accurate representation requires reaching that regime. Nor does the loss decomposition by itself prove persistent removal of useful geometry gradients. The supplied draft's Section 3.5 is still only a heading; these are ingredients for it, not a completed replacement section.

### M. The measurements that support or limit those claims

Use the [evidence ledger](sources/repo/docs/scale_mismatch_evidence_ledger.md) to navigate the older experiments and the [session catalogue](sources/repo/docs/expD24_session_catalogue.md) for the full figure list. The more recent, directly relevant additions are:

- [D35 projection diagnostic](sources/repo/results/checkpoint_D_optimizers/expD35_projected_scale_gradient/expD35_results.md): actual raw-slope gradient split over 10,000 steps; records numerical resolution of the small term.
- [D31 split Adam](sources/repo/results/checkpoint_D_optimizers/expD31_split_adam/expD31_results.md): includes the later 10,000-step and dynamic-ratio runs. It shows useful geometry changes under an intervention, with explicit cutoff and optimizer caveats.
- [D33 current-readout split](sources/repo/results/checkpoint_D_optimizers/expD33_current_readout_split/expD33_results.md): the J versus J* comparison, avoiding a false identification of those directions.
- [D34 Newton–Schulz penalty](sources/repo/results/checkpoint_D_optimizers/expD34_ns_residual_penalty/expD34_results.md): the approximate-projector alternative and comparison protocol.

Treat these as controlled observations and interventions. The older claims that geometry never improves, that the cubic scale law governs every raw-slope update, or that useful movement necessarily requires instability were not established and should not be resurrected in Section 3.5.

## Suggested order for appendix assembly later

1. Keep the supplied finite-lambda representation and newer FP64 recovery proofs as the basis for 3.2; reconcile their contracts with the actual implementation.
2. Use the supplied bandwidth note for tanh in 3.3. Add the GELU/fiber material only if the appendix needs that extension.
3. Import the latest finite-kernel ratio appendix for 3.4, with its normalization, target weights, and numerical-scope statements intact.
4. Place neighboring/scaling and the precise experimental protocols beside the claims they control.
5. For 3.5, select the existing identities and conditional drift result only after deciding the exact statement. The source collection does not yet justify the full joint-training claim.

No new theorem, optimizer experiment, or training run was created for this collection.
