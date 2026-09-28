# D36 handoff: metric restoration and geometry learning

## Stopped state

The user requested that Runpod GPUs be released on September 20, 2026. All
campaign jobs had already completed. A subsequent `squeue -u junmiaoh` check
showed no running or pending jobs. Do not submit further Runpod work until the
user makes the GPUs available again. Local analysis and documentation do not
require a GPU allocation.

Jobs 640–675 used 10,240 allocated GPU-seconds, or 2.844 H200 GPU-hours. This
includes compilation, invalid initial intervention pilots, replay diagnostics,
and failures. The authorized ceiling was ten GPU-hours, with at most two GPUs
simultaneously. Unused budget does not override the pause.

## What the evidence currently says

The [findings report](README.md) contains the equations, comparisons, figures,
and numerical limitations. The central result is that restoring the original
parameter metric can make an apparently favorable geometry direction too stiff
to follow. Across 17 late switching states, accepted bandwidth movement becomes
a median $8.68\times10^{-9}$ of the movement with the retained SSBroyden metric.
True directional curvature predicts the mixture's accepted length to within
0.17%. The normalized gradient still points mostly into bandwidth coordinates.

This narrows the hypothesis: measure useful exposure of the residual **per
feasible step**, not just per unit direction. It does not establish that an
accurate Newton metric is suppressing geometry acquisition. None of these
switching states passes the operational 25% curvature-prediction check.

Early metric mixtures, wider confirmations, and 100k continuations provide no
consistent improvement across seeds and targets. Low-gradient neuron replacement
also damages the fitted function in the mixed-target cases. These are distinct
negative results, not proof that all metric restoration or neuron replacement
must fail. Small endpoint differences are additionally limited by observed
sensitivity to roundoff-sized perturbations.

## The next restricted experiment

Start from saved states, before scheduling another broad sweep. Keep the model,
physical initialization, objective, and parameter scales fixed. Do not supply
the construction bandwidth or solved coefficients to the live network.

1. Reuse the 17 saved pre-intervention states and their true-Hessian/Gauss–Newton
   probes. For each candidate direction, measure loss slope
   $a=-g^Td$, curvature $c=d^T\nabla^2L\,d$, exposure
   $E=\nabla_\lambda G_\tau^Td_\lambda$, predicted length $a/c$ when $c>0$,
   and the actual line-search displacement. Keep the reference residual fixed
   when measuring $G_\tau$ changes. Compare against identical history restarts
   retaining the learned metric.
2. Test restoration restricted to weak directions of the full normalized
   Jacobian, rather than adding the identity in every direction. If $P$ projects
   onto right-singular vectors with singular values at most $\sigma_c$, then
   $\|JPv\|^2\leq\sigma_c^2\|Pv\|^2$. This bounds Gauss–Newton sensitivity,
   but not the residual-weighted part of the true Hessian; check that term too.
   Use a few disclosed relative cutoffs as a diagnostic ablation, not as
   theoretically established constants.
3. Require a numerically resolved finite accessibility improvement and an
   acceptable actual loss step before promoting a rule to training. Positive
   $E$ or larger parameter motion alone does not qualify. Retain both 20k and
   100k readout-flow diagnostic budgets; do not normalize them anew at a fork.
4. Only if these saved-state tests show a benefit, run matched 20k continuations
   on the two existing targets and individual/neighboring coordinates. Freeze
   the rule before fresh-seed confirmation. Run all branches of a parent in one
   worker with one compiled kernel. Record accepted-step counts, evaluations,
   actual loss, geometry paths, finite accessibility changes, and allocated time.

Keep neuron replacement separate. Its next useful test would compensate the
measured function jump before comparing recovery. Selecting a neuron solely by
small gradient does not establish that it contributes little to the function;
setting its readout to zero also initially zeros its slope gradient. A broader
bandwidth redraw must remain labeled as injected geometry.

## Files and restart details

The feature branch is `experiment/ssb-geometry-switching`, based on `321540d`.
The isolated checkout is `/tmp/precision-mlps-ssb-geometry`. The implementation
is in `experiments/expD36_ssb_geometry_switching/`; its README describes the
analysis commands and the checked numerical definitions.

Remote code remains at
`/workspace/junmiaoh/experiments/precision-mlps/ssb-geometry-code`.
Raw checkpoints remain at
`/workspace/junmiaoh/experiments/precision-mlps/runs/ssb_geometry_switching`.
An audited local cache of all final checkpoints, accepted-step traces, parameter
snapshots, and report inputs is preserved under the original checkout at
`results/checkpoint_D_optimizers/expD36_ssb_geometry_switching/evidence`.
The full remote archive also contains historical matrices not all copied
locally. The optional bulk transfer was stopped after the required evidence
passed its audit. `evidence_inventory.json.gz` identifies the cached files at
audit time; it does not claim to enumerate the entire remote directory.
The remote training revision is `e544161`; later local commits curate analysis
and documentation. Source hashes and manifests are recorded per job.

The worked update-12,500 example in the report is case
`ssbroyden_N128_s1_individual_mixed_09a8bddbea07`. Start from its
`solver_000012500.npz`, which was saved before applying the metric mixture.
`probe_curvature_000012500.json` records the matched finite-step measurements.
The other 16 states are indexed by `curvature_probes` in
`analysis/findings.json`. This is the smallest concrete starting set for the
next diagnostic; it does not require retraining cold initializations.

Within each case, `solver.npz` is the latest complete optimizer checkpoint;
`solver_STEP.npz` preserves branch states. `snapshot_STEP.npz` stores parameters,
`diagnostic_STEP` stores the residual/spectrum/gradient evidence, and
`ssb_trace_*` stores every accepted update. The runner resumes the saved solver
and controller. Parent forks also require the recorded checkpoint SHA-256.
Never initialize a fresh optimizer when claiming exact continuation.

The manifests distinguish original invalid intervention pilots from corrected
`primed_guard_v2` runs. Use `memory_pinned_*` for common-secant replay; the
original replay used the wrong first-self-scaling convention. Do not pool
either obsolete diagnostic variant into the corrected comparisons. Numerical
failures are explicit endpoints, not convergence claims or permission to restart
silently. Slurm job completion alone is not a scientific completion check;
use the case frontiers and artifact audit.

Validation recorded 21 focused tests passing. The full suite recorded 698
passes, 11 skips, and 17 failures matching the publication baseline's failure
set. See [verification.json](verification.json), [jobs.json](jobs.json), and
[artifact_inventory.json](artifact_inventory.json).
