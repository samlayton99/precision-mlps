# Population evolution and persistent output error

This campaign implements the five-hour execution plan approved on 24 September
2026. Its scientific question is whether collective properties of the evolving
ODE explain persistent output error. Agreement with a frozen or Taylor model
is not the definition of persistence. The limit is five wall-clock hours and
five aggregate GPU-hours, including controls and retries.

## Evidence and outcomes

The archive audit covers the available 23 target functions, keeping the six-target
wide panel separate from the broader late panel. The primary error is the
attached network's unsmoothed relative $L^2$ error. Training and independent-grid
errors are separate. Report several tolerances, not one privileged acquisition
threshold. An archived sparse crossing is not an exact first-hit time.

Analyze collective second and sixth moments, concentration, readout-weighted
nonlinear output capacity, signed mixed moments, exact moment derivatives,
coarse conditioning, and tracking's effect on the output. Evaluate exact-tanh
output bounds before choosing a sharper population persistence condition.
Archived-state inequalities are retrospective evidence, not certificates that
hold between checkpoints or prospective theorems.

Adam analysis uses archived first and second moments and the actual next-update
denominator. Compare residual energy with both ordinary and adaptively weighted
Jacobians, and reconstruct actual output progress including momentum and
nonlinear step effects. A preconditioned spectrum alone is not an Adam theorem.

## Large-scale interventions

Use center-preserving geometry multipliers $3.2,10,32,100$ with fixed final
readout references $c_0$ and $c_0/s$. Retain the existing exact coarse-balance
repair and stationarity gates. Continue ordinary FP64 GD at rate 0.002 for
20,000 updates. Repair failures, unresolved diagnostic decompositions, and
nonfinite training are different outcomes. No smaller multiplier or altered
learning rate silently replaces a failed arm.

The initial panel has 70 archived checkpoints. Broaden width 705 to the remaining
17 target functions at seeds 30 and 31, trained for 20,000 updates using the
existing initialization and target normalization. Keep checkpoint ages explicit.
The six prespecified development functions are degree five, mixed sine, left
Gaussian, right bump, right step, and absolute-value kink. At width 705, their
first seed has $1,10,100$ branches continued to 100,000 additional updates with
both applicable readout references. Half-step controls use the same six-target
development subset and matched flow time at multipliers $1,10,100$.

Distinguish injected scale and output improvement from subsequent learning.
Report absolute physical and normalized scales, readout repair displacement,
post-repair loss, population moments, and renewed tracking. Force growth with
little error improvement, sustained useful learning, contraction, and failed
preparation are all reportable results.

## Theory and verification

Derive moving collective envelopes, signed concentration drift, and energy and
tracking budgets from the evolving exact vector field. Seek a first-exit proof
or closed differential inequality. An unbounded higher moment is a closure gap;
do not replace it silently with individual-neuron confinement. Conditional
assumptions and initial-data consequences are labeled separately.

Checks cover independent gradients and projections, directional moment
derivatives, nonlinear-output bounds, Adam update replay, spectral energy
accounting, repair constraints, ordinary-GD accounting, and step-size sensitivity.
All numerical computation uses Runpod Slurm. Preserve unique sparse evidence,
source hashes, manifests, and scheduler accounting; remove only reproducible
duplicates. Reports are written directly after inspecting the evidence.
