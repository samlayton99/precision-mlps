# Prospective function-transfer panel

This protocol fixes new function instances before their training. The question is
whether the existing coupled-force descriptions transfer beyond the original
thirteen targets. Outcomes are held out; the families were selected after studying
the earlier experiments. This is not a random sample of all functions.

## Functions and normalization

Use $x\in[-1,1]$. Define
$$
B(u)=\begin{cases}\exp(1-1/(1-u^2)),&|u|<1,\\0,&|u|\ge1.\end{cases}
$$

| Family | Identifier | Raw function |
|---|---|---|
| Exponential | `exp_right` | $\exp(2x)$ |
| Exponential | `exp_left` | $\exp(-4x)$ |
| Gaussian | `gauss_left` | $\exp(-((x+0.35)/0.22)^2)$ |
| Gaussian | `gauss_right` | $\exp(-((x-0.42)/0.09)^2)$ |
| Compact bump | `bump_left` | $B((x+0.30)/0.40)$ |
| Compact bump | `bump_right` | $B((x-0.35)/0.22)$ |
| Tanh step | `step_left` | $\tanh(6(x+0.27))$ |
| Tanh step | `step_right` | $\tanh(14(x-0.31))$ |
| Continuous kink | `kink_abs` | $\lvert x+0.23\rvert$ |
| Continuous kink | `kink_relu` | $\max(x-0.37,0)$ |

Divide each raw function by its RMS on the original 2048-point midpoint grid.
Reuse that same scalar on the 8192-point evaluation grid; do not renormalize on
evaluation points. Do not center the functions or add a prescribed coarse part.
This preserves their different natural coarse/fine proportions. Existing target
definitions and their normalizations remain unchanged.

The exponentials introduce asymmetric boundary concentration. Off-center Gaussian
peaks and compact smooth bumps separate analytic tails from compact support. The
tanh steps provide sharp, exactly realizable features. The kinks test continuous
functions with nonsmooth derivatives and slowly decaying modal tails. These are
ten instances in five families, not ten independent families.

## Paired experiment

Use seeds 22 and 23, the existing width-177 initialization, physical coordinates,
FP64, learning rate 0.002, original 2048 training points, and retained degree 65.
Train twenty ordinary-GD backbones and fork each at 100k, 400k, and 600k updates.
Sixty fork states each receive `joint`, `freeze_map`, and `clamp_residual`, giving
180 continuations. The interventions retain their existing slope-only definitions;
the other parameter blocks and omitted modal force remain unchanged.

Issue the existing fixed-full-map Schur and applied-field affine predictions at
each fork before continuing its backbone or launching its intervention branches.
Predictions use the same formulas and horizons as the previous campaign, with no
refitting. Pilot numerical execution, then advance all branches through common
10k, 50k, and 200k additional-update endpoints as the shared budget permits.
Incomplete cases remain visible; do not select completion by scientific outcome.
GPU launches require explicit reservations in the existing shared ledger.

Lock numerical controls for seed 22, all ten targets, fork 400k: repeat all three
arms with degree 129 at the original learning rate and with degree 65 at half the
learning rate. These are sixty additional branches. Compare matched physical
times at 10k and 50k original-step equivalents; extend to 200k equivalents within
the budget. The half-step runs therefore require twice as many actual updates.
Keep these as separately labeled numerical checks, preserving the original panel.
The total reservation for backbones, pilot, primary branches, and these controls
is 3600 GPU-seconds, subject to the parent campaign's remaining shared budget.

## Locked questions and interpretation

1. Does the affine applied-field forecast predict signed mean-slope displacement
   and modal residual evolution more accurately than the fixed-full-map Schur
   forecast on the same states and horizons? The hypothesis is that evolving
   sensitivities matter beyond residual evolution in a frozen map. Compare the
   pure and frozen-remainder Schur forecasts separately.
2. Do the issued affine predictions capture the signed contrasts
   `freeze_map` minus `joint` and `clamp_residual` minus `joint`? Retain their
   magnitudes and all cases, including small or numerically unsupported forecasts.
   An unsupported prediction is not evidence for either mechanism.
3. Do residual sign changes and map-orientation changes require the same coupled
   description across these families? Inspect all retained modal channels; cubic
   error is a named secondary diagnostic, not a universal privileged mode.

Use the existing campaign error definitions and report individual cases and
function- and family-equal summaries at every common endpoint. Additionally score
vector forecasts against the natural zero-motion baseline:
$$
\mathrm{skill}=1-\frac{\|\widehat{\Delta v}-\Delta v\|_2^2}{\|\Delta v\|_2^2}.
$$
Here $v$ is the named predicted quantity; arm contrasts use the zero-effect
baseline with the same definition. Negative skill means worse than predicting no
motion or no intervention effect. Mathematically zero or empirically unresolved
denominators receive an explicit unresolved label, not an arbitrary numerical
floor. Do not fit percentage-accuracy cutoffs. Multiple seeds, forks, and horizons
from one function are repeated observations, not independent function samples.
Do not choose a winning horizon, target subset, or new model after seeing these
outcomes. Two seeds probe
initialization variability; they do not establish population-level probability
bounds for an unrestricted function class.

Audit coarse disequilibrium, omitted force, conditioning, numerical validity,
actual training/evaluation losses, and signed travel for every case. A large
coarse or omitted force identifies departure from the proposed theorem's regime;
it cannot be discarded or presented as confirmation. Such a departure does not
excuse a failed full applied-field affine forecast: that predictor already
includes derivatives of the remainder. Degree and time-step controls must not
silently replace the primary intervention or its predictions.

## Provenance and implementation boundary

`effective_feedback_holdout.py` supplies raw functions, the fixed normalization,
CPU initialization packs, and fork export. Export copies training inputs and
labels from the original immutable pack rather than regenerating them through
the older target registry. Packs record source hashes. Existing GPU kernels,
runners, target registry, and launchers are unchanged so their active source
capsules remain immutable. Store this panel in separate input, prediction, and
raw-output directories; preserve all issued forecasts and failed-job logs.
