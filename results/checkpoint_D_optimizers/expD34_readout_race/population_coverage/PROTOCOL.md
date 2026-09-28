# Does the population persistence theorem cover the archived dynamics?

This retrospective audit tests the sufficient conditions in
[the population theorem](../../../../docs/d34_population_persistence_theorem.md)
and identifies which violations matter to observed slope motion. It uses
existing checkpoints and ordinary-GD continuations. There is no new training,
target selection, or claim of fresh held-out confirmation.

## Questions and fixed panels

The first question is whether the theorem's alignment, parity, and loading
conditions hold. The second is whether violating particles carry outward
motion themselves or change the shared compensation acting on other
particles. The third is whether failure of a structural condition coincides
with failure of the underlying cubic or quintic effective-force model.

The checkpoint panels contain the existing 23-target development and
confirmation archives, the 100k/400k/600k cross-function checkpoints, and
the six-target width panel at physical widths 177, 705, and 1409. Exact
duplicate states are counted once; repeated checkpoints remain longitudinal
observations rather than additional target diversity. The temporal panel
uses the 40 natural-GD branches in the persistence campaign, with all eight
saved horizons from zero through 20,000 additional updates.

## Normalization and eligibility

Use the stored construction spacing $h$, actual width $W$, and empirical
mean inner product. Parameters are ordered $(a,b,c,d)$, and rescaled neuron
coordinates are $\sqrt W(a,b,c)$.

First subtract the empirical target mean from both the target and output
bias. This exact symmetry leaves residuals and gradients unchanged and
prevents a harmless constant target term from being counted as a parity
failure. Normalize the global target/readout sign so the affine contribution
$p$ is nonnegative. Test nonconstant even loading separately.

For sector-distance diagnostics, allow both neuron sign orientations.
Project each slope/readout pair onto the union of the cones
$0\le\alpha\le\zeta$ and $0\ge\alpha\ge\zeta$, and include hidden-bias
and centered output-bias displacement. This is the closest distance without
the coarse-moment constraint. A reference obtained by subsequently rescaling
both projected coordinates to preserve $p$ is a chosen compatible reference,
not a claimed constrained nearest point.

Report signed products, alignment defects, readout-dominance defects,
coarse conditioning, target parity, cubic sign, generated cubic moment,
fifth-loading margin, and their magnitudes. Floating-point near-zero
loadings are unresolved exact identities unless supported by construction
or a separate error allowance. No population-level condition is declared
true because only a small fraction of particles violate it.

## Motion and shared compensation

Freeze three disjoint particle groups at each fork: negative aligned
product; nonnegative product but insufficient readout magnitude; and
sector-compliant. Retain these labels over the continuation.

Read cumulative per-neuron positive and negative scale travel, signed
effective-force and tracking contributions, crossing corrections, and first
hits from the saved natural branches. Verify that positive minus negative
travel equals endpoint displacement and equals the sum of the signed force
channels and crossing correction. Positive travel is not additive across
force channels; only the signed decomposition is.

At every sampled state, partition the raw fine gradient by source group.
Compute each group's compensation with the full population's coarse Gram
matrix. Sum the components back to the original effective field. Report
their signed influence and absolute magnitude on all receiver neurons and
the upper slope tail. Removing neurons or recomputing a smaller Gram matrix
would be a different intervention and is not part of this audit.

Compare the existing cubic and quintic effective fields with the exact full
fine-complement field at the same actual states. Report absolute and relative
slope-vector error, direction, and outward-motion error for the full
population and the largest 10% of slopes. Keep tracking separate. Do not
anchor the fields to conceal their initial discrepancies.

## Interpretation and verification

Exact alignment coverage supports applying the sufficient mechanism only
when its loading and transfer conditions also hold. Small violations with
small direct travel and small outward compensation motivate a defect-budget
extension. Concentrated exceptional motion motivates a population-quantile
theorem. Accurate polynomial forces despite sector failure indicate a
structural-condition gap. Poor polynomial forces indicate a surrogate gap.
These interpretations are fixed before reading the new audit results.

Analytic fixtures check neuron/global sign symmetries, constant-shift
invariance, both-cone projection, loading signs, and additive compensation.
Run numerical tests and analysis in FP64 CPU Slurm allocations on the
existing Runpod account. Save source/input hashes, per-case rows, missing
inputs, numerical failures, and reconstruction discrepancies. Sampled
temporal diagnostics are evidence about persistence, not certified regional
bounds. No GPU time or new optimization traces are required.
