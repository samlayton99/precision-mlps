# What the coupled-feedback tests reveal

The experiments narrow the explanation of slow scale acquisition. Small
outward perturbations usually persist over the tested window; they do not
quickly return to the unperturbed geometry. Changing the relative learning
rates of geometry and readouts produces predictable differences, but readouts
do not universally dominate the removal of fine error. At independently
initialized larger widths, the features remain nearly affine and the
effective force is already small. In that regime, evolving the fine residual
with a fixed sensitivity map adds little to a constant-force prediction.

These results refine the mechanism rather than establish permanent trapping.
The central object remains the coupled effective force $F=Te$: both its error
load and the sensitivities that turn that load into parameter movement can
change. The useful approximation depends on the state and width.

This report records the intervention evidence. The
[main reading note](../../../../docs/d34_coarse_balance_stagnation.md)
explains the argument, the
[mechanism theorem note](../../../../docs/d34_mechanism_rate_theorems.md)
gives the conditional results, and the
[certified instance](../../../../docs/d34_certified_instance.md)
proves a finite exclusion window for one empirical GD problem.

**Table 1. Notation shared with the acquisition theorem.**

| Symbol | Meaning |
|---|---|
| $a,b,c,d$ | Signed slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=|a_j|$ | Physical slope magnitude. |
| $\lambda_j=h|a_j|$, $h=2/N_{\mathrm{ref}}$ | Normalized scale; $N_{\mathrm{ref}}$ is construction resolution. |
| $W$ | Actual neuron count, including halo neurons. |
| $F=Te$ | Effective fine gradient, including the balanced coarse contribution. |
| $r_C$ | Coarse disequilibrium gradient. |
| $R=r_C+g_\perp$ | Tracking and omitted-mode correction. |

## 1. Perturb geometry while matching its initial slope force

**Example and question.** A stable low-scale geometry should tend to restore
an outward perturbation. Simply increasing a slope also changes its initial
gradient, so that experiment confounds immediate forcing with delayed
feedback. We instead choose a direction that increases scale while leaving
the coarse output, coarse disequilibrium, and slope gradient unchanged to
first order. This tests whether the subsequent coupled response brings the
geometry back.

**Theory and design.** The direction lies in the common nullspace of the
three corresponding derivatives. The slope-gradient derivative is the actual
loss Hessian block, including residual curvature. We apply symmetric pulses
at three halved amplitudes and compare their difference with predictions
issued at the fork. A retained-offset baseline predicts that the original
perturbation simply survives. Frozen-map and full-Hessian local models test
whether their predicted feedback improves on that baseline.

**Result.** All 46 checkpoints admit an outward direction. The maximum
normalized linear constraint residual is $1.68\times10^{-13}$. Matching errors
mostly decrease by approximately four when amplitude is halved, as expected
for a quadratic remainder; the development right-Gaussian case is less
asymptotic at the tested amplitudes and is retained in the scorecard.

At the smallest amplitude, the fraction of the initial directional offset
retained after 20k additional updates is:

| Cohort | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Development, 23 functions | 0.99024 | 0.99994 | 1.00084 |
| Confirmation, 23 functions | 0.99511 | 1.00000 | 1.00256 |

There is little restoration in these matched directions over this window.
That is narrower than a claim about every direction or eventual stability.
The full-Hessian forecast improves on retaining the offset in only 15/23
development and 14/23 confirmation cases. A proposed two-observable closure
fails dramatically for degree three on both seeds and also fails on some
Gaussian and bump cases. Its conditional stability theorem remains valid;
the required closure is not supported as a general empirical model.

![Matched pulse responses across all functions](pulses/summary/pulse_responses.png)

*Figure 1. Paired responses preserve most of the imposed offset. The full
function panel is shown so that retention and model failures are not reduced
to the motivating degree-nine example. The
[summary](pulses/summary/summary.json) and
[development](pulses/analysis_development.csv) /
[confirmation](pulses/analysis_confirmation.csv) scorecards retain amplitudes,
matching errors and forecast comparisons.*

## 2. Separate geometry mobility from readout mobility

**Example and question.** Copy a neuron four times and divide its readout by
four. The represented function is identical. Under ordinary GD, each copy's
geometry now moves four times more slowly, while its aggregate readout moves
four times faster. If readouts consume the useful driving error before
geometry responds, correcting the readout rate should materially alter the
subsequent slope response.

**Theory and design.** Exact replica symmetry reduces the copied network to
the original variables with changed mobilities. For $k$ copies, uncorrected
geometry mobility is $1/k$ and aggregate-readout mobility is $k$. Compensating
either rate isolates its effect; compensating both recovers the original
trajectory. The effective map and coarse balance must be recomputed in each
branch's parameter metric. Clones retain the original normalization $h$.

**Result.** Both fully compensated branches reproduce the original trajectory
bit for bit. In development, fourfold copying reduces median slope-motion
magnitude to about 0.275 of the original. Readout-only compensation leaves
that median nearly unchanged, while geometry-only compensation raises it to
about 1.05. Readout feedback still matters for individual targets, including
changes of sign in small mean responses. The evidence does not support a
universal explanation in which readouts dominate error removal.

The effective residual model predicts the intervention contrasts well. The
strong comparison is with a constant effective-force forecast, because both
already know the initial change in mobility:

| At 20k additional updates | Development | Confirmation |
|---|---:|---:|
| Contrasts improving on no intervention effect | 136/138 | 134/138 |
| Contrasts improving on constant effective force | 111/138 | 109/138 |
| Median error / constant-force error | 0.401 | 0.454 |

The constant-force development comparison was added retrospectively; its
confirmation predictions were saved before the continuation. Evolving fine
errors adds predictive information at these late checkpoints. Adding the
larger feature-tangent model gives little additional benefit.

A separate positive-semidefinite decomposition allocates instantaneous
effective dissipation among parameter blocks. In the original mobility,
median slope shares are 66.3% and 69.3% across the two cohorts; median readout
shares are 19.7% and 14.9%. Individual readout shares range from about 4% to
96%, so the median is not a universal law. These are the nonnegative forms
$Q_\ell=T_\ell^TM_\ell^{-1}T_\ell$, which sum to the effective residual
matrix. The signed blocks $J_{H,\ell}T_\ell$ are a different decomposition
and must not be interpreted as nonnegative dissipation shares.

The [development](splitting/development_analysis.csv) and
[confirmation](splitting/confirmation_analysis.csv) tables retain all
contrasts. The [block audit](diagnostics/psd_blocks.csv) checks positivity and
the sum identity on 450 states; discrepancies are at floating-point roundoff.

## 3. Independent width changes reveal a different timescale

**Example and question.** Copying preserves a function and its geometry.
Increasing width with a fresh initialization is a different experiment. We
train six fixed targets at construction resolutions 128, 512 and 1024,
corresponding to 177, 705 and 1409 actual neurons. Two independent seeds give
36 cases. After 20k ordinary-GD updates, we issue predictions for the next
20k updates at the same learning rate, $\eta=0.002$.

**Theory.** While every slope, hidden bias and readout remains of order
$W^{-1/2}$, subtracting the affine part of tanh leaves a cubic remainder.
After coarse balancing, the effective force per neuron is at most of order
$W^{-3/2}$ under explicit uniform parameter and conditioning assumptions.
Since $h$ is of order $W^{-1}$, the normalized acquisition rate is at most
$O(\eta W^{-5/2})$. The fine residual changes on a slower scale than the
rescaled parameters. This is a conditional bound for exact tanh, not a
fitted power law or an assumption that the sensitivity map is frozen. The
[proof](../../../../docs/d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map)
states the constants and a first-exit condition that closes the parameter
regime for a finite interval.

**Prediction and result.** The checkpoint diagnostics agree with the proposed
scalings across these widths:

| Median across 12 cases at each width | $W=177$ | $W=705$ | $W=1409$ |
|---|---:|---:|---:|
| $\sqrt W\,\mathrm{RMS}(a)$ | 1.456 | 1.471 | 1.466 |
| $\sqrt W\,\mathrm{RMS}(c)$ | 1.479 | 1.491 | 1.472 |
| $W^{3/2}\,\mathrm{RMS}(F_a)$ | 0.628 | 0.695 | 0.661 |
| $W\|J_H\|_F$ | 2.621 | 2.932 | 2.845 |
| Constant-force motion error | 22.4% | 4.74% | 2.18% |
| Fixed-map, evolving-error motion error | 22.5% | 4.74% | 2.18% |

Errors refer to the full slope-displacement vector over the continuation.
The force projections through degree 65 and 129 agree to numerical precision.
The displayed Jacobian norm uses the retained basis; it is not a certified
norm for the entire complement in the theorem. Checkpoint measurements also
do not prove its interval-wide hypotheses.

Here evolving the error adds essentially no predictive gain. The estimated
residual clock over 20k updates shrinks strongly with width, consistent with
near-constant error load. Changing sensitivities can therefore be the more
important missing feedback. No neuron reaches $\lambda=0.25$ in any of these
36 trajectories through 40k updates. This measured window is not a universal
barrier, and the rate bound is not extrapolated beyond its parameter regime.

The [width scorecard](diagnostics/width_forecasts.csv),
[scaling audit](diagnostics/width_scaling.csv) and
[summary](diagnostics/width_scaling_summary.json) retain target-level results.
The targets are degree five, mixed sine, an off-center Gaussian, a compact
bump, a tanh step and an absolute-value kink. Thus this width comparison does
not depend on degree nine or a single oscillatory target.

## Protocol and cohort definition

The original force audit covers 23 function instances. The present development
cohort uses the 13 original functions at seed 0 and the 10 additional functions
at seed 22; confirmation uses seeds 20 and 23 respectively. All GD forks are at
600,000 updates. These functions have already been studied: confirmation refers
to new intervention outcomes, not previously unseen target functions.

The primary observations are per-neuron normalized-scale change, its rate,
positive and negative travel, and the fraction ever reaching $\lambda=0.25$.
Loss and readout magnitude provide context; neither substitutes for acquisition.
Training remains noiseless full-batch optimization in raw physical coordinates.

**Matched pulses.** Find an outward direction in the nullspace of the coarse
output Jacobian, coarse-disequilibrium derivative, and actual slope-gradient
derivative. The latter includes residual curvature. Use symmetric positive and
negative pulses at three halved amplitudes, with matching errors checked against
their predicted orders. A numerically unresolved outward direction is a design
limitation, not evidence for stagnation. Compare retained displacement, return,
and amplification with fork-issued frozen-map and full-Hessian local-response
predictions. Both derivative matrices are fixed at the fork; the second retains
the local derivative of the changing effective map.
Anchor both predictions at each pulsed state's actual initial gradient. Small
$R$ does not imply small $DR$: retain its derivative consistently when testing
the subsequent feedback response.

**Exact splitting.** Replace each neuron by two or four identical copies with
readout $c_j/k$. Compare ordinary rates, geometry-only compensation, readout-only
compensation, and both compensations. Geometry compensation multiplies its rate
by $k$; readout compensation divides its rate by $k$. Both together reproduce
the original trajectory exactly. Main runs may use the exact replica-symmetry
quotient after verification against explicitly expanded networks. The quotient
has hidden mobility $1/k$ and aggregate-readout mobility $k$ for ordinary split
GD. Compute coarse balance in each branch's actual parameter metric. Keep the
original $h$ for clones; genuine width changes require independent networks.

**Adam histories.** At the original 13 functions' 600k Adam checkpoints, seed 0
is development and seed 1 is confirmation. Independently attenuate future slope
tracking inputs to first and second moments by factors 1 or 0.9. Preserve all
incoming moments and the update count. Square the complete modified input to
the second moment, including cross terms. Estimate both channels' two phases
from the fork and one virtual ordinary Adam update; issue periodic-driver and
constant-driver forecasts before observing intervention continuations. These
are conditional local forecasts, not a claim that only tracking oscillates.

Initial common horizons end at 20k additional updates. No endpoint is a
theoretical barrier. Any later protocol revision must identify which results
informed it; confirmation outcomes are not used silently to retune predictions.

## What would refine the theorem?

Predicted return motivates a conditional response-stability bound; retained
offset with decaying drive motivates a finite-displacement bound. Splitting
tests whether relative readout and geometry timescales explain these responses.
Adam tests whether a cancelling tracking signal still changes effective
mobility through its second-moment history. A failed forecast identifies an
unresolved coupling; it is not repaired by fitting the observed future curve.

The GD premise is interval-wide small tracking, separately in direct slope force
and fine-residual forcing. The experiment does not prove entry into that regime
or persistence of this premise. Any empirical error envelope is distinguished
from a uniform, numerically certified theorem enclosure.

## Execution and evidence preservation

The eight-hour research window begins at 2026-09-23 08:18:25 UTC. New GPU use is
capped at four GPU-hours and must also fit the reconciled prior authorization.
The effective-feedback ledger records 5.204 GPU-hours; charging the older Adam
audit's 0.789 hours as well still leaves approximately four hours below ten.
Previously queued plateau GPU jobs 1025, 1036, 1038, 1040, and 1041 were cancelled
before allocation, as checked in Slurm accounting at the start of this study.
All numerical work uses Runpod Slurm; at most two GPUs may be allocated at once.

Keep input hashes, prediction files, sparse states, online motion accumulators,
verification results, and final analysis. Dense windows are short. Remove only
redundant campaign scratch after checking that unique evidence is retained.
The last part of the research window is reserved for analysis, theorem revision,
and a directly authored explanation of the results.
