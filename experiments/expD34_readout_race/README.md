# Ordinary-GD readout competition

## Proposed effective-force perturbations

**Current protocol: matched feedback, with the original 13-target panel complete
through 200k additional updates.** The
[200k report](../../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/existing_200k/README.md)
reports all targets and both seed cohorts. A
[prospectively locked ten-function panel](../../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/heldout_protocol.md)
tests transfer across five new families; fresh seeds on the original functions
do not constitute that test.

This specification supersedes the broad mode-deletion proposal retained below.
The question is now whether evolving errors or evolving sensitivities supplies
the feedback that limits scale acquisition. The identified object remains
$F_a=T_ae_H$; coarse disequilibrium and the orthogonal residual remain measured
corrections, not newly proposed dominant drivers. The
[main note](../../docs/d34_coarse_balance_stagnation.md) supplies the mechanism
and the [detailed note](../../docs/d34_coarse_balance_stagnation_details.md)
supplies the existing degree-9 persistence and acquisition results.

### Example: one fixed coupling succeeds and another does not

The degree-9 two-error model predicts slope displacement over a long ordinary-GD
continuation with nearly fixed effective sensitivities. Sine's sensitivities
change substantially, and the analogous frozen forecast fails. This contrast
motivates a sharper test than removing arbitrary modal penalties: give three
branches the same initial slope signal, then change which factor of that signal
is allowed to respond. The campaign does not assume that the degree-9 closure
works for every target, nor treat a failed sine prediction as support for it.

### Theory: three branches share the complete first update

At a fork $\theta_s$, abbreviate $T_s=T_a(\theta_s)$ and
$e_s=e_H(\theta_s)$. The selected finite empirical basis is fixed throughout.
Every branch uses the full original loss gradient in the non-slope blocks.
Its applied slope direction is the corresponding force below plus the same
state-dependent remainder field
$R_a(\theta)=J_{a,C}^Tz_C+g_{a,\perp}$:

| Arm | Applied effective slope force at its own current state $\theta$ |
|---|---|
| `joint` | $T_a(\theta)e_H(\theta)$ |
| `freeze_map` | $T_se_H(\theta)$ |
| `clamp_residual` | $T_a(\theta)e_s$ |

The remainder is recomputed from the branch's own state. It is not held fixed,
borrowed from the baseline, or omitted. The `clamp_residual` arm changes only
the residual supplied to this slope-force term; it does not hold the actual
network residual constant. Both modified arms remain coupled to ordinary
readout, hidden-bias, and output-bias updates. They need not descend a scalar
objective. The readout-size observation is a secondary diagnostic, with no
constraint imposed on the readouts.

At the common first next state $\theta_1=\theta_s-\eta g(\theta_s)$, the
second slope updates have the exact differences

$$
a_2^{\rm freeze\_map}-a_2^{\rm joint}
=\eta[T_a(\theta_1)-T_s]e_H(\theta_1),
$$

$$
a_2^{\rm clamp\_residual}-a_2^{\rm joint}
=\eta T_a(\theta_1)[e_H(\theta_1)-e_s].
$$

These identities check the intervention and distinguish the first effects of
map drift and error drift. They do not by themselves explain long persistence.
Later differences include feedback through all blocks. Baseline-small tracking
can grow after an intervention; measure that growth rather than assuming its
continued irrelevance or reinterpreting it as a baseline driver.

The fork-only affine predictor differentiates each actual applied direction,
denoted $g^{(b)}$, including the residual-weighted curvature terms:

$$
x_0^{(b)}=0,\qquad
x_{n+1}^{(b)}=[I-\eta Dg^{(b)}(\theta_s)]x_n^{(b)}-\eta g(\theta_s),
\qquad \widehat\theta_n^{(b)}=\theta_s+x_n^{(b)}.
$$

This is a discrete forecast fixed at the fork. It does not replay future
Hessians, import future residuals, or substitute the Gauss–Newton matrix for
the full derivative. A separate fixed-$T$, fixed-Schur-coupling residual
forecast retains all chosen fine modes. Its pure effective dynamics and its
constant-remainder correction are reported separately. Neither model is
relabelled as ordinary nonlinear GD.

### Predictions: distinguish the hypotheses before reading outcomes

For every fork, issue the three predicted signed per-neuron displacement curves
and effective-force curves before any continuation, together with derivative
rates, numerical checks, supported horizons, and source hashes. The fork
prediction files and their issuance digest must be retained. Long matrix-power
forecasts that overflow are unsupported; unstable modes are not silently
discarded to obtain a plausible curve.

| Proposed explanation | Required discriminating prediction |
|---|---|
| Error evolution supplies the relevant limiting feedback. | The precomputed `freeze_map` trajectory approximates `joint`; `clamp_residual` has a resolved, predicted signed departure. |
| Sensitivity evolution supplies that feedback. | The precomputed `clamp_residual` trajectory approximates `joint`; `freeze_map` has a resolved, predicted signed departure. |
| Both factors must evolve. | Both modified branches have distinct predicted departures, with the sign, timing, and per-neuron displacement specified before continuation. |

A predicted contrast smaller than the numerical uncertainty does not
discriminate. A verified branch with the opposite sign, or a displacement
outside a justified forecast envelope, rejects that prediction. If all local
forecasts fail, the local closure fails; this is not positive evidence that
unspecified coupling explains the result. Failure of a forecast does not
invalidate the exact force decomposition. Agreement at the first two updates
is implementation verification, not independent support for persistence.

Numerical uncertainty comes from independent derivative/reconstruction checks,
basis refinement, and matched-time step refinement. These checks do not bound
model error. A prospective model envelope requires the explicit neighborhood
remainder bound or an independently calibrated error model, frozen before
held-out validation. Report absolute errors and unresolved contrasts when no
such envelope is available; do not choose a percentage tolerance after seeing
the branch. Keep the predeclared horizons even if a later curve is inconvenient.

### Locked cases, horizon tiers, and common budget

The primary scientific horizon is **1k–200k additional updates**, with total
updates since initialization reported separately. At the 100k fork, those
endpoints are 101k–300k total; at the 600k fork, they are 601k–800k total.
The aim is a useful finite-time explanation, not indefinite forecast accuracy.
Million-update continuations are optional stress tests. After the user's
horizon clarification, the long panel was stopped after its completed 500k
and 2m tiers; the uncompleted 5.4m tier is not a required deliverable or a
failed prediction. Its consumed time remains in the shared accounting.

Use the 13 targets in `adam_forces.TARGETS`: sine, Runge, moments 3, 5, and 9,
mixed sine, localized sine, chirp, moment 4, and the four existing blends.
Preserve their original training-grid normalization, width 177, FP64,
2,048 midpoint training points, and $\eta=0.002$. Independent evaluation uses
8,192 midpoint points and does not select endpoints. Use the fixed empirical
orthogonal-polynomial basis through degree 65. Its orthogonal residual remains
in $R_a$; a degree-129 control changes the intervention as well as its diagnosis.

| Cohort | Starts and branches | Additional-update tiers |
|---|---|---|
| Existing trajectories | 13 targets × seeds 0–4 × forks 100k, 400k, 600k: **195 starts, 585 branches**. | 1k, 10k, 50k, 200k. |
| Fresh validation | Seeds 20 and 21 supply **26 ordinary-GD backbones** from the unchanged initialization. Fork all targets at 100k, 400k, 600k: **78 starts, 234 branches**. | The same 1k, 10k, 50k, 200k tiers. |
| Locked long panel | Sine, moment 9, moment 5, mixed sine, chirp × seeds 0 and 20 × forks 400k and 600k: **20 starts, 60 branches**. | 500k, 2m, 5.4m if throughput and the shared ceiling permit. |

The table preserves the original locked design. The ten new functions and
their 180 primary branches are specified in the linked held-out protocol;
the long-panel status above records the subsequent horizon decision. Neither
change selects targets by whether their outcomes favor stagnation.

Complete broad common tiers before expanding their horizon. The long panel is
fixed by target, seed, and fork rather than selected for an interesting result.
Report eligible, completed, failed, and budget-censored cases separately.
An uncompleted tier is not a failed scientific prediction, and a planned
5.4-million-update endpoint is not a claim that it was reached.

The existing forks come from
`results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/raw/primary_{seed}/`.
Resolve each case from manifest labels and snapshot steps, never array position.
Record source hashes, normalization, initial parameters, backend, source
revision, and the prediction issuance record. Fresh backbones preserve the
original initialization and training setup; issue each fork's forecast before
continuing its three branches. Reuse the same fork and branch state when
advancing a horizon tier.

The entire campaign shares **10 allocated GPU-hours**, including compilation,
verification, unsuccessful allocations, fresh backbones, and controls:

| Budget reservation | GPU-hours |
|---|---:|
| Verification and throughput measurement | 0.75 |
| Existing-trajectory panel | 2.50 |
| Fresh backbones and validation | 2.00 |
| Locked long panel | 2.00 |
| Degree-129 and half-step controls | 1.50 |
| Reserve | 1.25 |

These are subdivisions of one ceiling, not separate authorizations. Reserve
allocations in the common ledger before launch; account for elapsed allocated
GPU time across providers. Runpod permits at most two concurrent GPUs and
Modal at most four; both draw from this same 10-GPU-hour ceiling. Forecast
preparation can use CPU. Throughput determines which locked horizon tiers fit the ceiling;
it does not change the target list, select favorable cases, or authorize
additional GPU time. Adam is outside this GD campaign.

### Measurements, checks, and the acquisition theorem

Accumulate signed effective, tracking, and orthogonal slope contributions,
the exact absolute-value crossing correction, and positive and negative gamma
travel **at every update for every neuron**. Record first hits and initial
occupancy at gamma 1, 3.2, and 16, alongside the complete distribution of
displacement and original-objective training/evaluation errors. Those scales
are reference diagnostics, not universal precision thresholds. A sparse
forecast endpoint difference is not cumulative positive travel; the current
affine forecast explicitly leaves that quantity unavailable.

Save offsets 0, 1, 2, 10, 100, 1k and each horizon tier, with intermediate
1k checkpoints. Verify identical starts and first updates, the exact second
update contrasts, independent full-gradient derivatives, decomposition
reconstruction, resolved coarse inversion, and signed-travel accounting.
Compare degree 65 with 129 and $\eta=0.002$ with $0.001$ at equal physical
time on the five long-panel targets, seed 0, forks 400k and 600k: 10 starts
and 30 branches for each control. Begin at the primary 10k-update physical
horizon, using 20k updates for the half-step control; extend to the 50k and
200k primary horizons only within the control reservation. Compare with the
existing same-case, same-arm baseline, including ordinary GD. Retain the
discrepancy as evidence rather than fitting the prediction to it.
A singular/unresolved coarse solve invalidates that
intervention case and is reported, not silently regularized.

The existing degree-9 local-loss-floor theorem applies to ordinary GD; it
does not automatically apply to either modified field. The new fork-only
enclosure likewise states its scope explicitly. Uniform tanh derivative
bounds give a GD-map Lipschitz constant $\beta_R$ in a radius-$R$ ball. If

$$
Q_K=\eta\|g_s\|\sum_{j=0}^{K-1}\beta_R^j<R,
$$

a first-exit argument bounds every ordinary-GD update through $K$. Comparing
$Q_K$ with the distance to an acquisition event excludes that event. Taylor
remainders also bound the discrepancy from the ordinary affine forecast
while this neighborhood is enclosed. These real-arithmetic inequalities are
evaluated in FP64, not directed-rounding arithmetic. The bound may close only
over a short interval even when a much longer forecast is numerically finite.
Do not replace that shorter theorem horizon by the plotted forecast horizon.

Exact measured outward travel gives a retrospective acquisition audit. A
prospective travel claim needs an independently controlled future envelope.
The enclosure's maximum simultaneous occupancy is not the number of distinct
neurons ever crossing a threshold. Finally, confinement of a modified branch
does not establish confinement of ordinary GD without an additional bound on
the discrepancy between their update fields.

Completion means verified matched evidence and a decision about the stated
forecasts, including rejection or unresolved cases. It does not require the
desired mechanism, large slopes, or precision recovery to appear.

### Executable entry points

The following preparation and prediction commands run from the repository
root. The prediction directory is immutable once issued; resume trajectory
tiers rather than regenerating predictions after seeing their outcomes.

```sh
D34_DATA=results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu .venv/bin/python -m experiments.expD34_readout_race.effective_feedback prepare \
  --source results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/raw \
  --output "$D34_DATA/existing_inputs.npz"
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.expD34_readout_race.effective_feedback predict \
  --inputs "$D34_DATA/existing_inputs.npz" --output "$D34_DATA/existing_predictions"
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu .venv/bin/python -m experiments.expD34_readout_race.effective_feedback prepare \
  --fresh --seeds 20,21 --starts 0 --output "$D34_DATA/fresh_inputs.npz"
```

Run the continuation command only inside its reserved Slurm GPU step; the
provided `effective_feedback.sbatch` supplies the execution and reservation
environment. This example shows `joint`; use distinct output directories for
the other two arms and the same issued prediction manifest.

```sh
python -m experiments.expD34_readout_race.effective_feedback run \
  --inputs "$D34_DATA/existing_inputs.npz" --output "$D34_DATA/runs/joint" \
  --arm joint --backend slurm --degree 65 --eta .002 --horizon 1000 \
  --predictions "$D34_DATA/existing_predictions/manifest.json"
```

The runner's `--horizon` is measured in **reference updates at step 0.002**.
For the half-step control, keep the same horizon and pass `--eta .001`; the
runner doubles the actual update count internally. For example,
`--horizon 200000` means 200k actual updates at 0.002 or 400k at 0.001.
Doubling both the horizon and the update multiplier would run beyond the
intended matched physical time.

Fresh backbones use ordinary GD. Export each saved fork before issuing its
three-branch forecasts; the source below is a backbone run directory. Do not
run modified branches directly from initialization as a substitute for the
specified late forks.

```sh
python -m experiments.expD34_readout_race.effective_feedback export-fork \
  --source "$D34_DATA/runs/fresh_joint" --offset 100000 \
  --output "$D34_DATA/fresh_fork_100000.npz"
python -m experiments.expD34_readout_race.effective_feedback analyze \
  --source "$D34_DATA/runs" --output "$D34_DATA/analysis"
```

For Modal, `effective_feedback_modal.py` provides the corresponding bounded
single-GPU invocation. It requires an explicit budget reservation and records
input, prediction, and source hashes. The same scientific protocol applies
on either provider; command availability does not authorize extra allocations.

The new-function helper preserves its own target arrays during fork export.
Its analysis uses the already-issued forecasts and writes evidence arrays,
tables, and summaries; reports are authored separately in Markdown.

```sh
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu .venv/bin/python -m experiments.expD34_readout_race.effective_feedback_holdout prepare \
  --output "$D34_DATA/inputs/heldout_initial.npz"
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu .venv/bin/python -m experiments.expD34_readout_race.effective_feedback_holdout export \
  --inputs "$D34_DATA/inputs/heldout_initial.npz" \
  --source "$D34_DATA/raw/heldout_backbone" --offset 100000 \
  --output "$D34_DATA/inputs/heldout100k.npz"
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.expD34_readout_race.effective_feedback_holdout_analysis \
  --raw "$D34_DATA/raw/heldout" --inputs "$D34_DATA/inputs/heldout_all.npz" \
  --predictions "$D34_DATA/predictions/heldout_all" --out "$D34_DATA/analysis/heldout"
```

These preparation commands are for a fresh execution directory. Do not
overwrite the completed campaign's immutable input packs or forecasts.

<details>
<summary>Superseded modal-attenuation proposal (historical specification, not the current run matrix)</summary>

**Status: specified, not executed.** This study tests selective attenuation of
the effective fine slope force identified in the existing trajectories. Its
theoretical basis is the [main note](../../docs/d34_coarse_balance_stagnation.md)
and the [response propositions](../../docs/d34_coarse_balance_stagnation_details.md#11-perturbing-the-identified-effective-force).
The completed `stagnation_run` experiments instead changed lower-mode loss
penalties in every parameter gradient. Neither those results nor the earlier
`plateau_probes` specification supply outcomes for these new interventions.

### Example: weaken the modes that contract slopes

In the degree-9 trajectories, modes 2 and 3 explain most inward effective
slope motion. Removing lower-mode penalties reveals an outward force, but
one too small to produce appreciable travel over the tested interval. The
next question is whether selective attenuation has the predicted immediate
effect and whether coupling subsequently sustains, cancels, or amplifies it.

At each current state, decompose the original full-loss GD gradient as

$$
g_a=F_a+J_{a,C}^Tz_C+g_{a,\perp},\qquad F_{a,I}=T_{a,I}e_I.
$$

For a removed fraction $\varepsilon$, the slope-only intervention is

$$
a_{n+1}=a_n-\eta\bigl(g_a-\varepsilon F_{a,I}\bigr).
$$

The other blocks use their ordinary gradients at the same pre-update state.
Recompute every component on the branch's own trajectory. This is generally
not GD on a modified scalar loss. The matched full-parameter intervention is

$$
\theta_{n+1}=\theta_n-\eta g+\eta\varepsilon T_Ie_I.
$$

Both have the same first slope update, but only the full-parameter field
satisfies $J_CT_I=0$. Slope-only attenuation introduces the leading coarse
output perturbation $\eta\varepsilon J_{a,C}F_{a,I}$. At a finite step, both
statements have nonlinear output remainders; preserving coarse output to first
order does not preserve $z_C$ because its equilibrium also moves. Later
differences between the two arms include both coarse rebalancing and the
other parameter feedback changed by the full-parameter intervention.

### Theory: distinguish immediate response from coupled response

At a fixed state, the predicted signed slope velocity is affine in
$\varepsilon$. Evaluate that prediction for
$\varepsilon\in\{0,0.01,0.1,0.5,1\}$ and mode groups 2, 3, 2–3, 4–8, 9,
10–65, and all retained effective modes 2–65. Record direction-reversal
thresholds and the denominator determining each threshold before any
continuation. Compute the exact next-step change in $|a_j|$, including zero
crossings. A reversed direction alone does not predict appreciable acquisition.

For the coupled response, let $E_a$ insert a slope vector into the full
parameter vector. Along the ordinary-GD baseline, propagate

$$
\chi_{n+1}=
\bigl(I-\eta\nabla^2L(\theta_n)\bigr)\chi_n
+\eta E_aF_{a,\{2,3\}}(\theta_n),\qquad \chi_0=0.
$$

Then $\varepsilon\chi_n$ predicts the first-order parameter displacement
caused by attenuation. Use the full Hessian action, not its Gauss–Newton
replacement. This prediction uses the baseline trajectory; an autonomous
surrogate must separately evolve its own residuals and sensitivities. It
cannot obtain them from the branch it is supposed to predict.

From every starting state below, attenuate modes 2–3 by
$\varepsilon=0.01,0.005,0.0025$ for 1,000 updates. These are 36 perturbed
continuations, sharing ordinary-GD baselines. Compare the parameter remainder
$\|\theta_n^\varepsilon-\theta_n^0-\varepsilon\chi_n\|$ at offsets 1, 10,
100, and 1,000. Test for quadratic reduction as $\varepsilon$ is halved,
before floating-point error dominates. Separately record changes in readout,
hidden bias, modal residuals, and the force map to locate the feedback.

### Starting states and nonlinear continuations

Use `sine` and `moment9`, seeds 0–2, and forks at updates 100,000 and 600,000:
12 starting states. Preserve width 177, FP64 arithmetic, the original
2,048-point midpoint training grid, target normalization, and physical GD
step $\eta=0.002$. Evaluate on the independent 8,192-point grid; this checks
deterministic approximation and does not select endpoints.

The common source bundle is
`results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/raw/primary_{seed}/`.
Select each case through `manifest.json` by target, `optimizer == "gd"`, and
`eta == 0.002`; locate the fork through `snapshots.npz["steps"]`. Do not
hardcode case positions. The parameter array `p[case, checkpoint]` has 532
entries in $(a,b,c,d)$ order, with 177 entries per hidden block. Record hashes
of both source files, selected case metadata, source checkpoint, implementation
revision, backend, grid, and normalization. Require identical initial
parameters across matched arms.

Use the existing empirical orthogonal-polynomial basis through degree 65,
constructed once from the original training measure and held fixed across
branches and time.
The residual outside that span remains the separate $g_{a,\perp}$ contribution
and is not attenuated. Do not silently substitute the complete-complement
projection for a finite-basis intervention. Degree 129 supplies a basis
sensitivity check below.

Run 20,000 additional updates for each of these nine arms from each starting
state: 108 primary continuations. All cases are fixed in advance.

| Arm | Removed contribution |
|---|---|
| Ordinary GD | None. |
| Slope all-effective off | $\varepsilon=1$, modes 2–65, slope block only. |
| Slope mode 2 off | $\varepsilon=1$, mode 2, slope block only. |
| Slope mode 3 off | $\varepsilon=1$, mode 3, slope block only. |
| Slope modes 2–3 off | $\varepsilon=1$, modes 2–3, slope block only. |
| Slope modes 2–3 half | $\varepsilon=0.5$, modes 2–3, slope block only. |
| Slope complementary force off | $\varepsilon=1$, modes 4–65, slope block only. |
| Full-parameter modes 2–3 off | $\varepsilon=1$, modes 2–3, all parameter blocks. |
| Full-parameter complementary force off | $\varepsilon=1$, modes 4–65, all parameter blocks. |

Accumulate exact per-neuron positive and negative gamma travel, signed channel
contributions, zero-crossing corrections, and first threshold crossings at
every update. Save states and diagnostics at offsets 0, 1, 10, 100, then every
200 updates through 20,000, including offset 1,000. Diagnostics include modal
residuals, sensitivities, tracking and orthogonal corrections, projection
conditioning, original full-loss error, evaluation error, and physical readout
magnitudes, with output bias reported separately.

For gamma thresholds 1, 3.2, and 16, report initial occupancy and new crossings
separately. These are reference geometric scales, not universal necessary
conditions for approximation. Compare each new crossing's initial threshold
deficit with its accumulated outward travel. A change in mean gamma does not
resolve whether a small population acquires large scales. For sine, record
network and target modal coefficients separately: its cubic residual includes
a genuine target component.

### Predictions and what each outcome would change

| Observation | Interpretation or next refinement |
|---|---|
| Removing modes 2–3 reverses motion but leaves tiny outward travel. | These modes explain contraction; the remaining force still does not deliver appreciable acquisition. |
| The response grows beyond the accumulated immediate effect. | Coupled feedback matters; test the variational prediction before assigning a mechanism. |
| Slope-only and full-parameter arms produce different coarse transients. | Test the predicted initial coarse leakage; later differences also include other parameter feedback. |
| An autonomous surrogate misses a verified intervention response. | Refine its residual modes or evolving sensitivities; controlled baseline corrections become candidates only if new measurements show that they matter. |
| Readouts change substantially while effective force changes little. | Magnitude alone does not explain motion; its relevance must appear through $T_ae_H$. |
| Original loss increases after attenuation. | Selective force removal need not preserve descent; the ordinary-GD loss-floor theorem does not transfer automatically. |

The exact outward-travel audit bounds observed threshold crossings. A
prospective finite-time claim additionally requires a bound on future force,
a controlled response remainder, or a closed trajectory enclosure. Report
which of these is supplied; do not call measured past travel a forecast.

### Numerical checks and the later Adam test

Before interpreting continuations, verify reconstruction of the original
gradient, identical starts, unchanged non-slope first updates in slope-only
arms, equal first slope perturbations in matched full-parameter arms, and
$J_CT_I$ tangency. Check derivatives independently, including near-cancellation
and slope sign crossings. Report unresolved coarse inversions; do not silently
regularize them into evidence for the proposed regime.

For seed 0, both targets, and both forks, repeat ordinary GD, slope modes 2–3
off, and full-parameter modes 2–3 off at $\eta=0.001$ for 40,000 updates.
These 12 controls compare equal physical time. Compare degree-65 and degree-129
decompositions and applied intervention directions at every starting state
and saved endpoint. If the differences materially affect attribution or the
predicted displacement, run representative continuations with degree 129
before interpreting the primary contrast. A changed basis can change the
intervention itself, not just its diagnostic label.

A later Adam litmus test should attenuate the raw-gradient contribution
**before both moment updates**, preserving the fork's optimizer history.
Counterfactual branches must evolve their own second moments and denominators.
Assess sustained signed acquisition and original-objective error alongside
moment evolution. Large coarse-channel oscillations can cancel in signed
travel while still changing the second moment. The GD response propositions
do not establish an Adam theorem.

Completion means verified interventions, complete matched evidence, and an
interpretation of both positive and negative outcomes. Recovery of large
slopes or precision is not an acceptance requirement. This section proposes
future runs; the implementation of the present plan changes documentation only.

</details>

## Effective-force plateau investigation

The section below records the earlier campaign specification. Its resource
ceiling is historical; the new protocol above does not launch or allocate runs.

The [interim evidence report](../../results/checkpoint_D_optimizers/expD34_readout_race/force_plateaus/README.md)
walks through the four completed audit figures and records the queued stages.

The new exploration has a 10 GPU-hour ceiling, including verification and
interventions, with at most two concurrent Runpod GPUs. All numerical work,
tests, and plotting run through Slurm; local work is editing and inspection.

The primary question is why the effective fine force can remain nearly
constant, and whether its persistence predicts stalled or useful scale
acquisition. Four competing hypotheses guide the first measurements: weak
coupling with slow state evolution; replenishment balancing residual fitting;
rotation or concentration hidden by a flat norm; and correction of generated
lower-order output dominating access to a hard target. A constant nonzero
force alone does not imply a permanent barrier.

`plateau` measures the exact complete-residual force and its directional
derivative, separating geometry-map change, readout-map change, residual
evolution from effective steps, and residual evolution from tracking steps.
It also compares force directions across saved states, separates target and
generated-output forces, and makes parameter-free frozen-tangent GD forecasts.
Adam derivatives follow its actual next-step direction and are not assumed
to define a valid continuous-time approximation; the finite-step remainder
is reported explicitly. The initial audit uses all 13 targets and five seeds
at 20k, 100k, 200k, 400k, and 600k updates.

`plateau_dense` resumes every primary case for 512 adjacent updates at both
100k and 600k, preserving the complete optimizer history. It measures lag-one
and lag-two vector cosines, norm variation, and net-vector/path coherence
separately for raw effective/tracking forces and their actual optimizer steps.
These windows distinguish a constant force vector from reversals hidden by
a constant norm. The observer is checked against the unchanged training kernel.

`plateau_run` continues the existing 600k states to 6 million updates for all
13 targets, GD and Adam, and seeds 0–2. Optimizer history is preserved. One
batched job per optimizer permits two-GPU execution. Every-update force/path
accumulators and interval extrema are retained, with full states every 100k.
This tests plateau persistence without selecting favorable targets or seeds.
Long-run allocations are capped initially at two GPU-hours; diagnostic CPU
work runs alongside them. Remaining budget is reserved for paired mechanism
interventions selected from the diagnostic contrasts, with the selection
recorded before running those interventions.

The paired GD probes are fixed before execution: degrees 3, 4, 5, and 9,
mixed sine, localized sine, and chirp; seeds 0–2; forks at 100k and 600k;
500k further updates. `plateau_probes` compares joint GD, frozen readout,
clamping the fine-residual input to the effective slope force at its fork
value, and removing only the slope tracking correction. The last two are
artificial diagnostic dynamics. Clamping keeps the current force map and
ordinary tracking force; it does not freeze the whole gradient. Frozen-readout
results also measure the balance using the remaining movable blocks.

For pure degrees 4, 5, and 9, additional losses multiply the squared residual
in degrees 2 through $k-1$ by 0, 0.1, or 10; weight 1 is the joint-GD arm.
The target, initialization, and residual basis stay fixed. These are 37 paired
cases per fork and seed. Exact positive/negative slope travel, signed force
channels, zero-crossing correction, force coherence, and unresolved projections
are retained. Changing the lower-mode penalty tests generated-mode opposition;
it is not evidence about ordinary GD unless the unmodified arm agrees.

Independent confirmation uses seeds 20–24, the seven GD anchors, and Adam on
degree 9, mixed sine, and chirp through 1.1 million updates. Parameter-free
constant-force and frozen-Jacobian GD predictions are written at 100k and
600k, before the next 500k updates. No coefficients are fitted to future
trajectories. The same intervention matrix starts at 600k for these new seeds.
The prespecified comparisons are force-vector error, norm ratio, signed
outward movement, and error reduction. Prediction failure on recovering targets
is a scientific outcome, not a reason to change the validation set. No
conditional inequality is promoted to a predictive theorem without independent
control of its future assumptions.

`plateau_stage.sbatch` launches the discovery probes, independent continuations,
independent probes, or numerical controls. The controls repeat both seed-0
forks with half the GD step at matched physical time and with 4096 training
midpoints while preserving the original target function and normalization.
`plateau_gate` rejects incomplete or unresolved source bundles before dependent
GPU stages. The initial allocation caps are 2 GPU-hours for long runs, 3 for
discovery probes, 1 for independent continuations, 1 for independent probes,
and 4/3 for numerical controls: 8 1/3 GPU-hours, leaving 1 2/3 hours unallocated.
There are no automatic retries; partial checkpoints are retained.

`plateau_results --root <campaign> --output <analysis>` evaluates available
endpoints on the original and 8192-point independent grids, audits signed
motion, compares matched intervention arms, and scores the issued forecasts.
It retains incomplete and failed cases explicitly. The numerical controls
compare the same arm at the same physical horizon; they do not silently compare
different amounts of training. Its output is data, not an automatically written
scientific conclusion.

Training uses the original 2048 midpoints, fixed target definitions, width
177, FP64, and physical rate 0.002. Independent-grid errors check deterministic
approximation, not statistical generalization. No held-out model-selection
claim is made. Verification checks force reconstruction, derivative finite
differences, frozen linear GD against explicit updates, and archived-state
continuation. Failed hypotheses remain reportable outcomes. A completed
deliverable consists of interpretable force trajectories, tested mechanism
contrasts, and a note stating which prediction or conditional bound the
evidence supports, including failures of frozen-force predictions.

## Adam force attribution and additional targets

`frozen_readout_scale --output <directory>` checks physical readout magnitudes
on frozen construction-center sine dictionaries at $N=64,128,256$ and
$\gamma h=0.25,0.5,1$. Zero-start GD uses the finite-time SVD recurrence;
ordinary Adam at rates 0.002 and 0.0002 uses a QR representation of the same
empirical loss. All cases run through 600k updates. Individual weights are
reported relative to $h=2/N$, with interior, core, halo, and bias distinguished.
Consecutive terminal Adam windows and direct-feature controls check numerical
oscillations. Run locally with `JAX_ENABLE_X64=true JAX_PLATFORMS=cpu`.

`frozen_scale_gap --root <Adam-evidence> --output <comparison-directory>`
compares learned Adam geometry with common frozen gammas through 256 and with
multiples of its own learned slopes. The latter multiplies both $a$ and $b$,
preserving feature centers and relative slope magnitudes. Every geometry gets
the same zero-initialized readout-GD assay at 0.002 through 600k steps, evaluated
by the verified linear recurrence. Scale selection uses training-grid error;
independent-grid error and doubled-grid checks are reported separately. This
measures the best tested scale under the stated assay, not a universal optimal
gamma. Run the numerical work and figure generation in CPU-only Runpod Slurm.

The [implemented plan](../../docs/d34_adam_force_plan.md) specifies the target
definitions, comparisons, measurements, and interpretation gates. The
[completed report](../../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/README.md)
collects all 184 cases and the distinction between force activity and net growth.

`adam_forces.py` defines thirteen targets and an exact full-residual split into
effective force and coarse-tracking correction. The higher-mode residual is
the entire complement of the constant/linear projector, with no activation or
polynomial truncation. All parameter blocks enter the coarse kernel, including
the output bias. The five original targets retain their exact definitions;
new mixed-sine, localized-sine, and chirp targets use unit training-grid RMS.

Auxiliary first-moment buffers propagate the force components through ordinary
Adam using its actual shared second-moment denominator. Their steps add to the
actual optimizer step. They are measurements, not independent Adam optimizers
or interventions in training. A separate diagonal-mobility balance can be
evaluated with the actual Adam preconditioner; it is not a claim that Adam
obeys the GD tracking theorem. Unresolved coarse solves retain the full gradient
in an explicitly unresolved channel rather than changing the training update.

Focused checks compare gradients and coarse Jacobians with PyTorch autodiff,
reconcile the full-complement split with the retained-basis split, and check
ordinary Adam and component histories against PyTorch under cancellation,
tiny gradients, and zero-gradient intervals:

```bash
JAX_ENABLE_X64=true .venv/bin/python -m pytest -q tests/test_expD34_adam_forces.py
```

`adam_run.py` runs five paired primary seed bundles, two Adam rate-sensitivity
bundles, three control bundles, and one epsilon-sensitivity bundle: 184 cases
through 600k updates. Primary GD and Adam use 0.002; Adam rates 0.0002 and 0.001
are separate sensitivity cases. Adam uses betas (0.9, 0.999), epsilon 1e-8,
and zero initial moments. Controls use bias-corrected EMA momentum without
adaptive scaling, or Adam with beta1 zero. Epsilon 1e-12 is a separate control.
The existing raw physical coordinates and half-MSE are unchanged.

Every-update accumulators retain raw-force allowances, actual component path
lengths, signed outward contributions, actual positive/negative travel, and
reconstruction errors. Trace rows describe the gradient state before the last
update of their interval; interval minima and maxima retain intervening
excursions. Snapshots and the resumable state retain all moment buffers. A
nonfinite training case is frozen and explicitly marked failed; unresolved
decomposition alone does not alter training. Loss increases are recorded, not
used for selecting a checkpoint or stopping a finite run.

Use `adam.sbatch <campaign-root> <stage> [end-step]` with arrays `0-4%2` for
`primary`, `0-1%2` for `rates`, `0-2%2` for `controls`, and index 0 for
`epsilon`. Queue dependent stages to keep at most two campaign GPUs active;
each array task requests one GPU and preserves Slurm's allocation mask. The
new extension has a six allocated GPU-hour ceiling within the earlier total
eight-hour allowance. Include compilation and unsuccessful allocations.

`adam_analyze --source <completed-bundle> --output <analysis-directory>` exports
state diagnostics, exact motion-window accounting, modal refinement, and
frozen-geometry curves. It requires the full 600k horizon. `--construction`
instead exports the common-gamma references. The Adam balance check uses the
virtual next update's bias-corrected moments and denominator, retaining both
the momentum-lag forcing and actual finite-step coarse-residual defect.
The virtual update at the terminal state is not part of the trained trajectory;
actual-step attribution comes from online traces and cumulative measurements.
These measurements do not assume that the GD balance is Adam's equilibrium.
Analysis tables retain all cases and failure flags; scientific interpretation
and figure captions are written after inspecting the outputs.

`adam_analysis.sbatch <campaign-root>` runs the numerical analysis, verification,
curation, and figures in a CPU-only Slurm allocation on Runpod. It explicitly
disables GPUs. `adam_summarize --root <evidence> --figures-only` redraws the
figures from merged tables and curated traces. Curated traces retain selected
last-state values and exact minimum/maximum envelopes over each combined
interval; cumulative movement comes from every-update accumulators, not
quadrature of these plotting samples. `adam_verify` checks the original GD
replays and fixed-state grid refinement; `--curate` exports selected complete
optimizer states and the coarsened traces with source hashes.

## Signal-recovery audit

`recovery.py` analyzes whether renewed slope signal produces signed movement
and population-scale acquisition. It reads the existing evidence packages,
including exact cumulative signed-force summaries; it never integrates sparse
plotting traces. The primary targets are sine, degree three, and degree nine.
The 20k state is the fixed comparison baseline, not a selected signal minimum.

```bash
MPLCONFIGDIR=/tmp/race-mpl python -m experiments.expD34_readout_race.recovery \
  --evidence results/checkpoint_D_optimizers/expD34_readout_race \
  --output results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery
```

The original raw archive was retired after curation. To recover neuronwise
diagnostics, `replay_recovery.py` replays only width 177, seeds 0–4, those three
targets, and equal physical rates 0.002. It uses the unchanged D34 vector field,
checks original initial/data hashes, and checks archived scalar observations
at 20k, 100k, and 600k. Run it in a GPU Slurm step with `JAX_ENABLE_X64=true`,
`--evidence <curated-evidence> --output <compact-replay-directory>`.
The executable verifies the allocation and GPU mask. It stores selected states
and per-neuron upward/downward travel accumulated at every update, rather than
recreating the retired dense archive. Pass its `compact_states.npz` to the
analysis command with `--replay`.

New measurements separate signed alignment, concentration of positive travel,
endpoint net movement, residual filtering, coefficient amplification, and
output-bias effects. Hessian-block attribution is an instantaneous flow
diagnostic at a GD state. Reference population bounds are conditional on a
justified error radius; observed prediction errors are not certificates.
Raw slope thresholds and construction bandwidths remain diagnostics, not
universal necessary conditions for approximation.

Focused checks: `JAX_ENABLE_X64=true python -m pytest -q tests/test_expD34_recovery.py`.

The completed [focused report](../../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md) provides the compact replay, per-seed tables, population budgets, and verification records. Its [theory companion](../../docs/d34_scale_acquisition_theory.md) states the actual-initialization scope. Early integrated attribution uses 20-update trapezoids through update 2,000, checks a 40-update coarsening, and reports its discrepancy from the observed GD log decline. It is an approximate integration of flow diagnostics, not an exact finite-step attribution. The generic energy-budget comparison checks the final update in addition to the archived 599,999 descent ratios.

This experiment tests whether readout adaptation depletes residual moments before a substantial population of hidden slopes can grow. The evidence sought is an initialization-only prediction of signed scale trajectories and their changes across readout rates. Successful execution does not require the hypothesis to hold.

Measured findings and limitations are in the [results report](../../results/checkpoint_D_optimizers/expD34_readout_race/REPORT.md), with [curated evidence](../../results/checkpoint_D_optimizers/expD34_readout_race/README.md) that regenerates the figures without training.

**Notation.** These symbols refer to physical raw coordinates throughout.

| Symbol | Meaning |
|---|---|
| $f=d+\sum_j c_j\tanh(a_jx+b_j)$ | Network with independently trained slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=\lvert a_j\rvert$ | Reported slope magnitude; neuron indices and live signs are preserved. |
| $\lambda_j=(2/N)\gamma_j$ | Reporting scale tied to the reference budget, not to sample or evolving-center spacing. |
| $\eta$, $\kappa$ | Geometry rate and readout/geometry rate ratio. Output bias uses the readout rate. |
| Polynomial reference | Independently evolved degree-1/3/5/7 activation, using only its own state and fixed target moments. |

## Locked protocol

All four parameter blocks update simultaneously from the old state, using the loss $L=\tfrac12\operatorname{mean}(f-y)^2$. There is no momentum, normalization of training residuals, refit, clipping, freeze, parameter remapping, or schedule. Reported MSE is $2L$.

The main matrix uses $(N,H,W)=(64,12,89),(128,24,177),(256,48,353)$, seeds 0–4, and $\kappa\in\{10^{-4},10^{-3},10^{-2},0.1,1,10,100\}$. The geometry rate is $0.002$. All finite scientific trajectories receive at least 20,000 updates. Training has 2,048 midpoints on $[-1,1]$ at every width; evaluation uses 8,192 different midpoints. Two additional seed-0, width-177 baseline comparisons retain D28's 1,024 training samples for sine and Runge. Independent evaluation diagnoses interpolation error and is not used to choose a winning optimizer, rate, or checkpoint.

Initialization calls the unchanged D28 helper. Slopes and hidden biases are independent uniform Xavier draws with bound $\sqrt{6/(W+1)}$ from NumPy's stream seeded by `[seed, N]`. Readouts use a separate `[seed, N, 24]` stream and the same bound. Output bias is zero. There is no tanh gain, readout norm rescaling, reference-envelope multiplier, or sign canonicalization. Copies of these physical arrays initialize every paired rate and reference.

Experiment A uses the upstream sine and Runge target amplitudes. Experiment B uses empirical orthonormal Legendre polynomials fitted once on the training grid:

$$
y_k=0.3\phi_0+0.4\phi_1+\sqrt{0.75}\phi_k,\qquad k=3,5,9.
$$

Every target has training RMS one and the same first two target coefficients. The saved polynomial coefficient map evaluates the same function on independent grids. The affine references must agree across all three targets; the cubic references must agree for $k=5,9$. Actual tanh trajectories can differ immediately. The main matrix contains 525 tanh and 2,100 reference trajectories.

## Mathematical verification

`references.py` computes polynomial coefficients and the exact moment-gradient pullback. Its separate affine Gram recurrence retains the quadratic finite-step term and reconstructs individual states with a common $3\times3$ map. Higher-order references retain all parameters. `core.py` implements simultaneous GD and analytic tanh gradients; its stable expression for $\operatorname{sech}^2$ avoids saturation cancellation. FP64 is enabled by the experiment entry point or test environment, not as an import side effect of the kernels.

Tests compare the analytic gradients with independent sample-space differentiation, the affine recurrence with explicit polynomial GD at extreme rates, the initialization and early trajectory with the unchanged PyTorch D28 trainer, and the state-conditioned moment remainder bound. A reference loss computed from moments can suffer cancellation; it must be checked against direct sample evaluation at saved states and never silently clipped as a training objective.

## Execution and evidence

GPU computation must run inside Slurm on the allocated devices, with at most two H200s concurrently. The initial allocation budget is two GPU-hours. Core comparisons precede half-step, equal-geometry-time refinements. Remaining time funds balanced width-177 continuations toward 100k, seeds 5–9, then further common-horizon continuations. A finite trajectory interrupted by the allocation remains incomplete, not converged. Polynomial breakdown does not terminate its independently trained tanh counterpart.

The diagnostic record includes signed mean and median scale changes, quantiles, population thresholds, residual moments, coarse and remainder signed slope forces, parameter gradients and actual changes, readout norms, full states, reference prediction errors, and independent-grid errors. Hypothesis assessment must use paired rate contrasts and reference-validity checks, not just declining loss or a few large slopes. Numerical findings belong in the results report after execution.

`JAX_ENABLE_X64=true python -m experiments.expD34_readout_race.run --root <run-root> --stage core --frontier 20000` runs the core matrix. Stages `baseline`, `refine`, `continue`, and `replicate` select the two historical-grid checks, width-177 seed-0 half-step comparisons, width-177 common continuations, and seeds 5–9 respectively. Use `--frontier 40000` for the equal-time refinement. `--worker 0 --workers 2` and `--worker 1 --workers 2` split complete seed/width bundles between the two workers. Reinvocation automatically resumes the saved GD state without changing its rate or initialization.

Each bundle stores its immutable protocol and physical arrays once. Subdirectories `p0,p1,p3,p5,p7` contain independent resumable states, full every-update scalar traces, gradients and states at updates 0–20 and every 20 updates thereafter, and the first coarse-threshold crossing states. `p0` is tanh. Snapshot step $n$ and trace step $n$ refer to the state before update $n\to n+1$. The final state is also retained in `state.npz`. Normalized residual moments and sensitivities can be reconstructed from the stored unnormalized moments, gradient norms and half-MSE; undefined zero-residual ratios must remain undefined.

Submit `run.sbatch` only after deploying a committed source snapshot into the isolated remote `race-code` directory and recording its commit in `.source_commit`. The launcher preserves Slurm's device mask and the runner checks an active GPU job/step and exactly one visible GPU. The deadline writes a resumable incomplete trajectory; it never promotes a partial run into the scientific comparison. Raw NPZ evidence stays outside Git; curated numerical summaries and figures accompany the final report.

`python -m experiments.expD34_readout_race.analyze --root <run-root> --output <evidence-directory> --end 20000` exports tables and compact plotting arrays. Pass `--bundles` to analyze named bundles independently. Use `analyze.sbatch` for remote CPU analysis. Common-horizon checkpoints at each 20k boundary support comparisons after training continues. Incomplete and failed trajectories are identified explicitly and excluded from endpoint rate contrasts.

Sample-space checks use updates 0–20 and the saved states nearest 61 logarithmically spaced times. They record direct versus moment-computed loss, exact readout and geometry contributions to coarse-residual velocity, and state-conditioned slope-gradient remainder intervals. Reference trajectory errors and matched-target differences use every saved state. The interval is a pointwise statement at the observed state, not a guarantee that an independently evolving reference remains nearby.

For long continuations, `--sparse-snapshots` restricts the analyzed full states to updates 0–20 and 61 logarithmic times; the analysis manifest records this restriction. Every-update scalar traces still supply exact event times, integrated signed forces, loss-increase counts, and endpoint windows. The raw full-state archive retains every 20th update regardless of this analysis option.

`python -m experiments.expD34_readout_race.plot --root <evidence-directory>` renders line plots from the exported evidence. Full rate and seed coverage appears in scale and endpoint-contrast plots; detailed signal and reference-error panels use the predefined ratios $10^{-4},1,100$. Every plot labels geometry time $\tau=\eta n$ or the readout/geometry rate ratio. No script authors the scientific report.

`python -m experiments.expD34_readout_race.curate --inputs <completed-analysis-directories> --output <review-directory>` combines analyses at the same endpoint, losslessly compresses plotting arrays, and reduces full diagnostic tables into per-trajectory audits. It retains all summary rows and all rate contrasts. Its reference-error threshold of 5% is a descriptive reporting threshold, not a scientific acceptance rule. Full diagnostic tables remain at the source paths recorded in `evidence_manifest.json`.

The completed extension uses `--stage continue --frontier 600000` for the original five width-177 seeds and `--stage refine --frontier 200000` for the equal-time check against primary 100k training. These commands continue the saved states without resetting parameters or rates. The final reports retain the 20k and 100k comparisons separately.

`python -m experiments.expD34_readout_race.curvature_check --root <run-root> --output <csv>` computes the exact empirical half-MSE Hessian at saved 20k, 100k, and 600k states for width 177, seeds 0–4, and ratios 1 and 100. It includes residual-weighted second derivatives. Its eigenvalues are for $D^{1/2}\nabla^2L D^{1/2}$, where $D$ contains the physical block learning rates. Eigenvector block fractions are measured in these step-scaled coordinates. This is an offline local stability diagnostic, never an optimizer or an intervention in training.

`compress_affine.py` losslessly compresses completed affine NPZ archives and verifies every array's dtype, shape, and bytes before replacing the original. It excludes mutable `state.npz`. Its worker partitions are disjoint; verification ledgers record each replacement. The runner also compresses new affine archives and checks successful writes before atomic replacement. Curation accepts either CSV or lossless gzip tables. These storage measures preserve numerical evidence after the shared volume stopped accepting complete writes during the long continuation.

## Transport and residual-basis extension

`transport_run.py` evolves exact tanh features using either the full residual (`--degree -1`) or its empirical polynomial projection. Degree 33 is the predefined primary forecast; degrees 9, 17, and 65 check projection error. Every forecast evolves independently from the original random initialization, without subsequent true states as inputs. The primary horizon is physical time 1200 (600,000 updates at $\eta=0.002$). Half-step comparisons use the same physical horizon.

With `--nodes 0`, particles are the actual width-$W$ neurons, each with mass $1/W$. With positive `--nodes q`, $q^3$ Gauss–Legendre nodes integrate the independent uniform initialization law. Physical width still controls the model output and initialization scale; quadrature count controls integration accuracy. Its characteristic velocity differs from the Euclidean gradient of quadrature coordinates by the factor $Ww_i$. Thus law quadrature and a larger neural network are distinct experiments. Orders 8, 12, and 16 check the law approximation; the full residual at order 12 checks the modal approximation independently.

The original five seeds and three targets establish numerical adequacy. Fresh seeds 10–14 assess the locked degree-33 forecast; secondary checks use readout ratios 0.1 and 10 on seeds 10–12 and widths 89 and 353 on seeds 0–2. Main outcomes are finite-time population fractions above $\gamma=1,3.2,16$, positive and negative scale travel, slope path and energy, layer-balance defect, omitted-mode force, and coarse/fine kernel coupling. A few escaping neurons do not invalidate a population barrier. Projection error at a true state and error of an independently evolved trajectory must be reported separately.

The extension has an eight aggregate GPU-hour ceiling, including compilation and failed attempts. Its jobs request one GPU and run serially through Slurm, leaving another GPU slot for concurrent work. The first two hours cover numerical adequacy, then up to three hours each cover fresh predictions and secondary controls. Source hashes, initialization/data hashes, GPU environment, status, cumulative motion, sparse curves, and selected full states are saved. A deadline produces an explicitly incomplete resumable run. No report is generated by code.

Adaptive verification adds law quadrature orders 24 and 32 after the order-12/16 tail discrepancy, a full-residual order-32 check after sharper law particles show measurable modal-force error, and a matched-time half-step check at order 12. Short order-8 law runs at widths 89 and 353 through physical time 40 separate finite-seed variation from the formal width-scaling prediction. These controls retain the frozen primary degree-33 predictor and the original eight-GPU-hour ceiling. The final report distinguishes these adaptive checks from the predeclared fresh-seed validation.
# Useful slope growth and mechanism diagnostics

`mechanism.py` analyzes compact actual-GD archives without modifying them.
Its force drivers distinguish evolving residuals from evolving sensitivities,
and its signed movement measurements distinguish available force from growth.
The optional frozen-geometry diagnostic trains raw readouts from a common zero
state by evaluating the exact linear GD recurrence through an SVD. This is a
geometry comparison; ordinary D34 training retains its random readout.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD34_readout_race.mechanism \
  --archives results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/compact_states.npz \
  --output results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/baseline_metrics
```

Add `--geometry` and use a separate output directory for frozen-readout curves
and capacity diagnostics. Curves use the original 2,048-point target map and an
8,192-point evaluation grid, physical step 0.002, and unchanged raw readout
coordinates. The capacity solves are separate from the finite-budget curves.

`replay_recovery --targets runge moment5` adds the two missing equal-rate
baselines and verifies their initialization, targets, and archived observations.
`mechanism_run` accepts these and the original compact archive through
`--archives`, one `--seed`, and `--fork-step 20000` or `100000`. Its six default
arms share exactly the same saved state. Frozen blocks remain bitwise unchanged.
The saved `steps` always use the original 0.002 reference clock; a half-step
refinement takes two updates per clock tick and records `training_eta=0.001`.
Positive/negative travel and path are accumulated at every actual update.
Completed archives can resume unchanged toward a larger `--end-step`.

GPU entry points must run through `mechanism.sbatch` in an isolated committed
checkout. Queue these jobs sequentially or with an array concurrency of one.

`mechanism_summarize --root <evidence> --archives <completed archives...>` joins
the `*_metrics` and `*_geometry` outputs into per-seed endpoints, paired
intervention contrasts, numerical identity checks, and figures. Keep half-step
verification outputs in a separate subdirectory rather than pooling them as
additional independent seeds. Reports are written directly after inspection.

`mechanism_verify --root <evidence> --primary <original compact archive>` checks
the completed continuations against original GD, every frozen block, all 60
matched half-step cases, degree-129 modal diagnostics, and doubled frozen-study
grids. `mechanism_curate --root <evidence> --archives <full archives...>` retains
all scalar tables losslessly, mode-0-through-9 residual coefficients,
mode-2-through-9 force curves, and verified states
at fixed milestones and sampled post-20k force maxima. The full compact archives
remain separate from this curated Git evidence package.

## Conditional stagnation near coarse balance

`stagnation_run` tests whether correction of generated lower modes sustains
weak scale acquisition in the retained degree-9 joint-GD states. It compares
the full tanh loss, exact-tanh residual losses retaining modes 0/1/2/3/9 or
0–9, and the full loss with penalties on modes 2–8 removed. Every arm trains
all four parameter blocks by simultaneous raw-coordinate GD. Coarse projection
is used to measure forces; it is never imposed on the training update.

First audit and deduplicate the retained states and prepare the shared inputs:

```bash
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m experiments.expD34_readout_race.stagnation_run audit \
  --curated results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/curated \
  --output results/checkpoint_D_optimizers/expD34_readout_race/coarse_balance
```

For each `--arm full`, `five_mode`, `ten_mode`, and `remove_lower`, submit
`stagnation_run run --inputs <inputs.npz> --output <unique run directory>
--arm <arm> --eta 0.002` through `mechanism.sbatch` in an isolated committed
checkout. Run the four arms sequentially. Repeat each with `--eta 0.001` in
a separate directory. The primary setting batches all five seeds at the 20k
and 100k starts; the half step uses seed 0 at both starts. The default horizon
is 20,000 reference updates (physical time 40), giving 40 primary continuations
and eight numerical controls.

The audit includes all parameter-block tracking forces, readout conditioning,
degree-65/129 diagnostic comparisons, individual signed slope forces, and
fixed-state loss-penalty and hard-target probes. Continuations sample their own
active-loss forces every 200 reference updates, with extra early samples after
intervention. Exact positive/negative scale travel, gradient path, layer-energy
identities, and full-complement coarse tracking travel are accumulated every
actual GD update. Sampled finite-degree tracking and omitted forces are reported
separately. Fixed-state effective-direction reversals do not by themselves
predict raw GD motion after the loss changes.

The five-mode choice uses prior checkpoint evidence; the continuations test
that fixed choice without fitting coefficients or importing future states.
These are training-grid optimization measurements. Partial error reduction
and small outward drift are not evidence of the intended scale or precision.

`stagnation_analyze --root <coarse_balance>` joins all eight run directories,
checks common starting states and the available first-step baseline replays,
compares force vectors and parameter trajectories with full GD and matched
half steps, and writes endpoint tables, numerical checks, and two figures.
It also reproduces the motivating signed-mode attribution from the existing
curated diagnostics. Reports are authored after inspecting those artifacts.
The scientific argument and interpretation belong in the
[conditional stagnation note](../../docs/d34_coarse_balance_stagnation.md).

Local CPU execution uses the same `stagnation_run.run` function with
`require_gpu=False` and `JAX_PLATFORMS=cpu`; the CLI retains the established
Slurm GPU allocation check. The evidence records the backend and source
revision, and the matched half-step comparisons use the same backend.

## Persistence and evolving sensitivities

`persistence` compares degree-nine GD with exact-tanh five- and ten-mode losses
and a model whose tanh features are linearized at the starting slopes and biases.
The latter continues to train the readouts, preserving their bilinear interaction
with geometry. It is distinct from freezing the entire tangent map. All four
parameter blocks use simultaneous updates and the original rate and grid.
The linearized-feature runner evaluates its gradient through the full fixed
feature Gram matrix, centered at the initial residual to reduce cancellation.
This changes evaluation cost without dropping ranks or training samples.

The source is the full `adam_force_extension/raw` archive. Default seeds are
0–4; `--start 100000 --end 600000` tests against retained trajectories, while
`--start 600000 --end 6000000` specifies the prospective continuation. Select
`--model full`, `five_mode`, `ten_mode`, or `linear_features`. Each output
directory has a fixed manifest, resumable sampled states, exact per-update
positive/negative gamma travel, and coarse-projection tracking accounting.
The projection diagnoses the active model and never changes its updates.

The curated `persistence/inputs.npz` also works as `--source` for the runner,
analysis, and bounds. It contains all 195 original degree-nine GD states,
their step and seed labels, and the unchanged empirical inputs and target.
Use a fresh output directory when replaying from this pack: its source hash
differs from the original larger archives even though the parameter arrays
are identical.

Run the module with `--source`, `--output`, and `--model`. The default GPU
backend verifies a Slurm allocation. `--backend cpu` with `JAX_PLATFORMS=cpu`
supports the same protocol locally when remote source transfer is unavailable;
CPU runs on Runpod still use CPU-only Slurm steps. Set `JAX_ENABLE_X64=true`.
Use `--eta .001 --seeds 0` for matched-time half-step controls. Wall-clock
limits retain completed prefixes; missing horizons are not marked complete.

These comparisons test how much evolving sensitivity is needed to predict
stagnation. A successful forecast alone is not a uniform error certificate or
evidence of useful geometry. The theory must distinguish measured travel from
a bound obtained without the future true trajectory.

`--quadrature 64` accelerates the tanh models using Gaussian quadrature for the
original discrete 2048-point measure. It preserves its polynomial moments
through degree 127 in real arithmetic; it does not replace them with continuum
moments. Tanh integrals remain numerical approximations. Direct-sum forecasts,
node doubling, and analytic truncation bounds separate this evaluation error
from modal-surrogate error and GD step-size effects. The default still sums
the original samples. The linearized-feature model uses its full Gram matrix.

`persistence_analyze` has four actions: `audit` resolves generated-error
relaxation and the force derivative; `predict` constructs frozen-tangent,
constant-gradient, and two-mode forecasts from their starting states; `bounds`
evaluates the analytical neighborhood enclosures without future true states;
`compare` joins the completed forecast prefixes. Each takes `--source` and
`--root`. The exact spectral remainder is separate from nonlinear prediction
error. Infinite-time two-mode budgets apply only to that restricted model.

The additional `persistence_reduction` calculation retains the physical
quadratic/cubic residuals and freezes their coarse-projected tangents. It tests
both zero hard forcing and a fixed ninth-degree residual. The resulting
two-by-two residual system has an explicit discrete solution: a decaying
generated-error contribution plus a weak persistent force. Its finite-time
path bound separates a bounded transient budget from a term linear in the
horizon. `physical_mode_budgets.csv` records both contributions. This reduction
was added after the two dominant spectral modes were identified; it is a
mechanistic check, not an independently selected model.

`persistence_verify --source ... --root ...` checks starting states, accumulated
travel, matched numerical controls, empirical quadrature, modal reconstruction,
and independent sampled errors against the predictive radii. Its JSON explicitly
lists incomplete runs. `persistence_figures --root ...` renders the curated
numerical comparisons; neither entry point writes the scientific report.
The bound formulas are analytical real-arithmetic statements evaluated in
ordinary FP64, not directed-rounding interval certificates.

`persistence_energy --source ... --root ...` evaluates a second predictive
bound. A Chebyshev approximation on complex ellipses bounds the hard output
coefficient throughout a parameter ball. The resulting local loss floor,
combined with discrete GD descent and a step-containment induction, bounds
the total available parameter travel. `energy_bounds.csv` retains the best
valid candidate from a fixed grid of ball radii; `energy_candidates.csv.gz`
retains every candidate. No future trajectory enters the bound. This refinement
was motivated by the persistence analysis rather than fixed before it.
