# Population energy redistribution: completed campaign

The campaign tests whether concentration explains available fine sensitivity,
how relative growth advantage evolves, and whether sensitivity gains produce
population-scale acquisition. The self-contained exposition and proofs are in
[`docs/d34_population_concentration.md`](../../../../../../docs/d34_population_concentration.md).

The result is a conditional population explanation with a causal qualification:
redistributing energy can accelerate effective fine dynamics, but concentration
alone does not determine expansion. The feature arrangement and its target
alignment matter. All sampled intervention errors remain above 11.5%, despite
some accelerated trajectories.

## Completed design

- Replay 72 archived trajectories, including 58 native-GD cases spanning 23
  targets, six effective-flow pairs, and eight earlier step-refinement runs.
- Continue 324 baseline/intervention trajectories: six targets, three
  width/seed cohorts, nine starting states, and two dynamics.
- The nine states are baseline plus two paths each of energy equalization and
  2×, 4×, 10× increases in $\sqrt{\chi_6}$, preserving the full hidden Gram
  matrix. No readout fitting is performed.
- Repeat 16 representative interventions at half the integration step and the
  same physical duration.

Every intervention starts at total update 25k and runs for 200 flow-time units,
equivalent to 100k updates at the native GD step 0.002. Effective flow uses
RK4 with step 0.02. The new half-step runs use 200k actual GD updates or 20k
RK4 steps. Replays extend another 5k equivalent updates to compare archived
endpoints; reported mechanisms and bounds use the first 100k.

No construction failed and no trajectory stopped early. The five GPU jobs
used 7403.5 seconds of recorded function time, within the approved three
GPU-hour budget. Numerical execution and plotting were remote on Modal;
local work did not load scientific arrays. GPU host memory had an 8 GiB hard
cap, and CPU analysis had a 4 GiB hard cap.

## Evidence and checks

The sibling `energy_audit_20260925`, `energy_development_20260925`,
`energy_validation_20260925`, `energy_wide_20260925`, and
`energy_refinement_20260925` directories contain raw scalar histories,
construction records, execution logs, and source/input hashes. Only initial
and final intervention parameter arrays were retained as sparse local
artifacts; they are unnecessary for the scalar analysis here. No dense
optimization trace was stored.

This directory contains the signed audit, intervention effects, separate and
joint coupled comparisons, accumulated-moment comparisons, refinement checks,
sampling checks, curated figures, and `facts.json`. `execution.json` records
the successful 22-test verification and the exact analysis source hashes.
`inputs_manifest.json` identifies every input file and its hash.

The numerical comparisons use sampled structural histories. The proofs are
conditional statements; the GD comparison plots do not certify the unsampled
positive discrete-defect allowance. `bound_offset` gives the last reported
bound time when a comparison stops between output checkpoints, whereas
`horizon` is the interpolated stopping time. Raw error and population scale
use the same checkpoint convention in every intervention arm.

The runner is `experiments/expD34_readout_race/population_energy_run.py`,
launched through `population_window_modal.py` with studies `energy_audit`,
`energy_development`, `energy_validation`, `energy_wide`, and
`energy_refinement`. The `energy_analysis` study consumes the five resulting
directories and the six original `window_*_20260925` input directories listed
in the manifest. Use a fresh output directory for each invocation.
