# What keeps the effective slope force from acquiring scale?

The question is whether the coupled effective fine force explains **limited
scale acquisition over a useful training window, across different functions**.
The primary tests span 1k, 10k, 50k, and 200k additional GD updates. Predicting
millions of updates is not a requirement. The completed original panel contains
13 targets, seven seeds, three forks, and 819 matched branches. Fresh seeds
test initialization dependence; they do not test transfer to new functions.
A separate, prospectively locked panel adds ten functions in five new families.

The original panel gives a useful short-window result: a model that retains
local sensitivity and error feedback predicts ordinary slope displacement at
10k updates to a median relative vector error of about 0.03% in both seed
cohorts. Its worst cases are 24% and 12%; by 50k–200k the errors become much
more uneven. A fixed-map residual model often remains useful for slowly moving
targets. The effective force is the common object; the usable approximation
and its horizon depend on the state. Degree 9 and sine illustrate different
regimes, rather than defining the function class of the theory.

| Term | Meaning |
|---|---|
| $\theta=(a,b,c,d)$ | Slopes, hidden biases, readouts, and output bias of the width-177 tanh sum. |
| $\gamma_j=\lvert a_j\rvert$ | Magnitude of one slope; its sign remains in the update. |
| $e_H$ | Network-minus-target coefficients in fixed empirical polynomial modes 2–65. |
| $F_a=T_ae_H$ | Effective fine slope gradient after eliminating instantaneous coarse response. |
| $R_a=g_a-F_a$ | Coarse disequilibrium and omitted-basis corrections to that gradient. |
| `joint` | Ordinary simultaneous GD. |
| `freeze_map` | Supply $T_{a,s}e_H(\theta)$ to the slope update. |
| `clamp_residual` | Supply $T_a(\theta)e_H(\theta_s)$ to the slope update. |
| Affine forecast | Linearize the complete applied field at the fork, then run that fixed model forward. |
| Fixed-$T$ forecast | Freeze the full effective map, while evolving all retained fine errors through its Schur coupling. |
| Relative motion error | Euclidean error in the predicted slope-displacement vector divided by its actual norm. |

## The experiment gives identical initial signals different feedback

**Example.** A smaller late slope force could mean that its driving errors
have relaxed, that its sensitivities have weakened, or that their signed
contributions now cancel. Deleting a group of modes changes the starting
force and does not isolate these responses. Here every branch starts at the
same checkpoint and has the same first update in real arithmetic.

**Theory.** Only the effective slope-force term is changed. Every branch
retains its ordinary, state-dependent $R_a$ and ordinary gradients for the
readouts and both bias blocks. Its actual network residual keeps evolving,
including when the residual supplied to the effective slope force is clamped.
Thus the experiment compares coupled trajectories. It does not assume that
the small corrections measured in ordinary GD remain small after intervention.

**Prediction.** Forecast the signed, per-neuron response to each intervention
before continuing it. Error-only feedback requires fixing the map to preserve
the relevant motion; sensitivity-only feedback requires clamping the supplied
error to do so. Both claims must predict the departure of the other branch.
All reported forecast files were issued before their dependent continuations;
none uses future residuals or Hessians. There was no fitted acceptance
threshold. We report errors and wrong signs rather than choosing a favorable
tolerance afterward.

Training uses noiseless, full-batch GD, $\eta=0.002$, FP64, and 2,048 fixed
midpoints on $[-1,1]$. The loss is half the empirical MSE for
$f_\theta=d+\sum_jc_j\tanh(a_jx+b_j)$, with no width normalization.
Evaluation uses 8,192 midpoints and the original training-grid target
normalization. Evaluation does not choose checkpoints, seeds, endpoints, or
hyperparameters. It checks approximation between training points, rather
than claiming statistical generalization to a different task distribution.

The 13 targets are sine, Runge, moments 3, 5, and 9, mixed sine, localized
sine, chirp, moment 4, and the four prescribed blends. `moment9` means the
existing constant-plus-linear-plus-ninth-polynomial target, not the monomial
$x^9$. Existing seeds 0–4 are discovery cases. Seeds 20 and 21 provide fresh
initializations under the unchanged protocol; two new seeds are a limited
validation set, not a population-level statistical guarantee.

| Cohort | Starts | Branches | Completed common additional horizon |
|---|---:|---:|---:|
| Existing: 13 targets, 5 seeds, 3 forks | 195 | 585 | 200k |
| Fresh: 13 targets, 2 seeds, 3 forks | 78 | 234 | 200k |
| Degree-129 control: 5 targets, seed 0, 2 forks | 10 | 30 | 200k |
| Half-step control: same starts, matched physical time | 10 | 30 | 400k actual updates, equivalent to 200k primary |
| Posthoc checks of all three 10k sign misses | 3 per control | 18 total | 10k primary-equivalent |
| New functions: 10 targets, seeds 22 and 23, 3 forks | 60 | 180 | In progress |
| Optional stress test: 5 targets, seeds 0 and 20, 2 forks | 20 | 60 existing branches extended | 500k and 2m |

The three primary forks are 100k, 400k, and 600k. The long panel uses sine,
moment 9, moment 5, mixed sine, and chirp at 400k and 600k. It resumes each
branch's actual 200k state with its original fork and travel accumulators;
the 60 extensions are not counted as independent new branches. The complete
[protocol](../../../../experiments/expD34_readout_race/README.md#proposed-effective-force-perturbations)
records the locked matrix and common budget.

The [new-function protocol](heldout_protocol.md) specifies two exponentials,
two off-center Gaussians, two compact smooth bumps, two exactly representable
tanh steps, and two continuous kinks. Their natural coarse/fine proportions
are retained. These ten functions were fixed before their training outcomes
were inspected; the families were chosen after the earlier experiments.
They are a broader transfer test, not a random sample of all functions. All
cases remain in the assessment, including good learners and departures from
the small-remainder regime. Functions and then families receive equal weight;
repeated forks are not counted as independent target functions.

At this report revision, all twenty new-function backbones and all sixty
fork forecasts are complete. Numerical controls are running. The 180 primary
continuations await approval review's required confirmation for uploading
their combined synthetic checkpoint pack to the existing Runpod destination.
Prepared forecasts and selected targets have not changed in response to this
execution block; no primary new-function outcome is claimed here.

## What works within the primary training window

**Example.** At 10k additional updates, both issued ordinary-GD predictors
beat the baseline of predicting no slope movement in every one of the 273
original-function starts. The coupled affine forecast is substantially more
accurate in the typical case. At 200k, its ordinary-displacement forecast is
worse than no movement in 42 starts, while the simpler pure fixed-map model
loses to that baseline in four. Greater local detail does not guarantee a
better extrapolation over a longer interval.

**Theory.** The affine model retains the first variation of both factors in
$F_a=T_ae_H$, along with the remainder derivatives. The fixed-map model lets
the errors relax through a constant effective coupling. These approximations
need different controls on future change. A finite-time theorem should state
its interval and bound the omitted change on that interval; it need not
predict a complete future training history.

**Prediction and result.** The following comparison uses every ordinary start
in each original-function cohort, without selecting a favorable target or
fork. “Beats zero” compares squared vector error with predicting no movement;
it is a minimal usefulness benchmark, not a precision certificate. A correct
intervention sign alone is weaker evidence.

| Additional updates | Affine median relative error, existing / fresh | Affine worst relative error, existing / fresh | Affine beats zero, existing / fresh | Pure fixed-map beats zero, existing / fresh |
|---|---:|---:|---:|---:|
| 1k | 0.00115% / 0.00143% | 0.382% / 0.306% | 195/195 / 78/78 | 195/195 / 78/78 |
| 10k | 0.0291% / 0.0335% | 24.0% / 11.6% | 195/195 / 78/78 | 195/195 / 78/78 |
| 50k | 0.290% / 0.281% | 1,415% / 64.9% | 189/195 / 78/78 | 194/195 / 77/78 |
| 200k | 4.59% / 4.04% | $8.78\times10^7$% / 26,612% | 161/195 / 70/78 | 192/195 / 77/78 |

The large worst errors remain visible despite the much smaller medians. The
[target-level audit](analysis/primary_forecast_scope.json) retains all thirteen
functions at every horizon. These counts are descriptive repeated-case
measurements. The zero-motion score was added after the original outcomes
and fixed before the new-function assessment; it does not change or refit
any issued forecast.

Here is the same comparison resolved by function for the existing cohort.
Each cell gives **affine / pure fixed-map** median relative vector error in
percent, over five seeds and three forks. The table includes easy partial
learners and slowly moving targets. Pooling forks can hide a difficult late
state, so these medians must be read together with the worst cases above.

| Original function | +10k error (%) | +200k error (%) |
|---|---:|---:|
| `blend_m010` | 0.0241 / 1.03 | 2.51 / 22.0 |
| `blend_p001` | 0.000371 / 0.0703 | 0.0205 / 1.38 |
| `blend_p010` | 0.00524 / 0.527 | 0.393 / 10.6 |
| `blend_p030` | 0.197 / 5.41 | 27.5 / 66.8 |
| `chirp` | 0.0891 / 3.35 | 36.2 / 40.1 |
| `localized_sine` | 0.204 / 2.80 | 71.7 / 42.3 |
| `mixed_sine` | 0.115 / 2.62 | 79.7 / 46.2 |
| `moment3` | 0.442 / 7.40 | 147 / 76.9 |
| `moment4` | 0.00638 / 0.495 | 0.595 / 10.4 |
| `moment5` | 0.000626 / 0.0821 | 0.0164 / 1.65 |
| `moment9` | 0.000574 / 0.0170 | 0.0164 / 0.330 |
| `runge` | 0.148 / 1.34 | 20.1 / 14.2 |
| `sine` | 0.114 / 2.84 | 39.1 / 41.0 |

At 10k, every function's median favors retaining local sensitivity feedback;
the largest fresh-cohort target median is also small, 0.241%. At 200k, this
ordering no longer holds for several functions. This supports a coupled local
description with an explicit usable interval. It does not support selecting
degree 9's especially long persistence as the general standard.

All horizons are measured from the stated checkpoint. The 100k fork followed
for 200k ends at **300k total updates**; the 600k fork ends at **800k total**.
Thus even the primary campaign studies a late post-transient regime, not an
entry theorem from initialization. The optional
[500k/2m stress test](analysis/long_2m/README.md) is reported separately. Its
5.4m extension was stopped after the horizon clarification. Failure at those
much later times does not overturn a successful shorter-window forecast.

## Degree 9 has a useful reduced model; sine needs evolving coupling

**Example.** At the existing degree-9 600k forks, ordinary GD's median change
in mean gamma over the next 200k updates is $-3.239\times10^{-5}$.
Fixing the slope map changes this motion by only $-2.470\times10^{-7}$;
clamping its supplied residual changes it by $-4.254\times10^{-7}$.
Both responses are inward, as predicted, in every existing seed. The small
feedback corrections are resolved by the numerical controls.

**Theory.** Generated quadratic and cubic errors drive the degree-9
contraction. Their slow relaxation and modest sensitivity changes soften it.
Removing either response leaves slightly more contraction. This extends the
earlier two-error persistence picture while testing it against the same
interventions used on other targets.

**Prediction and result.** The full fixed-$T$ residual model predicts
ordinary degree-9 displacement well at 200k. The affine model, which retains
the local change of sensitivities, is more accurate. Sine provides a sharp
failure of extending that local approximation too far:

| Existing-seed case, +200k | Affine relative motion error | Pure fixed-$T$ relative motion error |
|---|---:|---:|
| Degree 9, 600k fork | 0.0151% | 0.315% |
| Sine, 600k fork | 1,118% | 46.7% |

These are medians of five individual vector errors. A numerically finite
matrix-power forecast is not necessarily accurate. At the sine 600k forks,
freezing the map reduces mean gamma by a median 0.0346 relative to ordinary
GD, while clamping the supplied residual increases it by 0.1053. The large
departures support studying both factors, but the failed magnitudes reject
the fixed local predictor over this interval. Fresh seed 20 also supplies a
counterexample to a universal claim that freezing the map suppresses mean
scale: its 600k fixed-map branch has a positive contrast at 200k.

The [200k comparison](analysis/existing_200k/README.md) gives both cohorts,
individual seed variation, and the correction measurements. The earlier
[10k report](analysis/existing_10k/README.md) shows where the local forecasts
are considerably more accurate. Across all targets, sign counts are:

| Additional updates | Existing correct contrasts | Fresh correct contrasts |
|---|---:|---:|
| 50k | 381/390 | 153/156 |
| 200k | 375/390 | 153/156 |

These are paired contrast counts, not independent statistical trials. High
sign accuracy does not rescue an inaccurate magnitude forecast.

## Sine's accessible errors can contract, expand, and oppose each other

**Example.** At 100k, cubic error drives contraction in every existing sine
seed. At 400k, the error retains its sign but its effective slope contribution
has become outward in every seed. At 600k, the cubic error has reversed sign
in three existing seeds and both fresh seeds. Its inward contribution then
competes with the outward fifth-mode contribution.

**Theory.** The signed velocity from mode $k$ is $A_ke_k$, where
$A_k=-W^{-1}\sum_j\operatorname{sign}(a_j)(T_a)_{jk}$. The sign of the
error alone does not determine expansion. A small-slope heterogeneous
calculation identifies the signed readout–slope moments controlling $A_3$;
coarse fitting constrains a different moment. It explains how orientation
can change while the coarse fit remains nearly unchanged. This is a
checkpoint explanation, not yet a prediction of the reversal time.

**Prediction and result.** Both preissued fixed-$T$ models retain all 64
fine modes, yet miss all five observed cubic overshoots from 400k to 600k.
Adding the starting remainder as a constant scarcely changes those forecasts.
Thus keeping many residual modes while freezing their coupling is insufficient
on this interval. The posthoc scalar diagnostic gets the orientation right
at all 30 sine/degree-9 checkpoints, but its late-sine vector error is large;
it must not be promoted to a quantitatively valid global surrogate.

The [signed-mode audit](analysis/cubic_sign_audit/README.md) gives the
measurements. The technical exposition and detailed companion develop the
coupled interpretation without treating coarse disequilibrium as a newly
discovered baseline driver.

## What the theorem gains, and what remains conditional

**Example.** A large unresolved error can occupy directions with very weak
sensitivity. It then fits slowly and generates little movement over a finite
interval. Conversely, a strong force concentrated on a few neurons can
increase the maximum gamma while leaving most of the population small.
Neither the residual norm nor the maximum slope is an acquisition certificate.

**Theory.** The fixed-map surrogate has
$\widehat e_{n+1}=(I-\eta S_s)\widehat e_n$ and
$\widehat\theta_{n+1}=\widehat\theta_n-\eta T_s\widehat e_n$, with
$S_s=T_s^TT_s$. If $S_sv_i=\lambda_i v_i$ and $b_i=v_i^Te_s$, its
displacement contribution is

$$
-b_i\frac{1-(1-\eta\lambda_i)^N}{\lambda_i}T_sv_i,
\qquad \lambda_i>0.
$$

This separates error loading, available sensitivity, and time for relaxation.
The [detailed results](../../../../docs/d34_coarse_balance_stagnation_details.md#12-matched-feedback-tests-and-a-target-general-movement-budget)
turn it into a finite-time displacement and positive-travel budget, then
state the explicit map-drift, residual-evolution, and remainder corrections
needed to bound ordinary GD. The exact decomposition alone does not control
those future corrections.

**Prediction and limit.** The separate ordinary-GD neighborhood/path bound closes
only over sampled horizons of 100–50k updates in this campaign. Every long
affine forecast must be kept separate from that shorter bound. The earlier,
stronger degree-9 loss-floor theorem still supplies its 10.2–17.1-million
update exclusions from its specified checkpoints; the present experiments do
not establish an analogous sine theorem. The spectral positive-travel
transfer is the general conditional framework, not an additional numerically
closed certificate in this package.

Every run accumulates each neuron's positive and negative gamma travel at
every update, with first hits, signed effective and correction contributions,
and the exact absolute-value crossing correction. Those measurements audit
what happened. A prospective acquisition exclusion requires a future envelope,
not a sum computed after the trajectory is complete. Gamma 1, 3.2, and 16 are
reference diagnostics, not universal conditions for precision approximation.

## Numerical checks and execution evidence

The independent first/second-step audit verifies the matched-force identities
below $5.6\times10^{-17}$; CPU/Runpod/Modal parameter discrepancies in the
pilot are below $7\times10^{-18}$. All 80 tested 50k/200k control-contrast
signs survive degree and step refinement. At 200k, the largest relative
vector-contrast changes are $2.64\times10^{-8}$ for degree 129 and
$1.79\times10^{-4}$ for the half-step run at matched physical time.

All three 10k forecast sign misses also survive their exact-case posthoc
controls. These are model misses. Refinement discrepancies are sensitivity
evidence, not certified numerical-error intervals or model-error tolerances.
The corresponding raw records remain in `analysis/controls_200k/` and
`analysis/miss_controls/`.

All 30 focused implementation tests pass. The repository's full nonslow
suite reports 783 passed, 17 failed, 9 skipped, and 4 deselected. None of the
failures is in D34: they involve unavailable dependencies/data and unchanged
precision, dtype, or generic-module import issues. The full log and failure
classification are retained in the verification package.

`campaign.json` records the matrix; `budget.json` reserves and reconciles the
shared 10-GPU-hour ceiling. Runpod uses exact allocated GPU elapsed time.
Modal uses a conservative submit-to-return accounting with startup/teardown
padding, including queue waits; it is not an exact billing statement.
The stopped long-panel extension, including its cancelled work, remains in
the accounting. Its last complete common stress-test tier is 2m updates;
partially completed later states are not substituted for the primary horizons.

Inputs, forecasts, snapshots, source hashes, receipts, and analysis tables
are retained beneath this directory. Large arrays are ignored by Git.
Execution uses source `dc2804b`; later commits add analysis and exposition
without changing the running kernel. Modal volume `d34-effective-feedback`
retains immutable run capsules and full 1k-spaced histories under
`d34-feedback-{existing,fresh,long}-{joint,freeze_map,clamp_residual}-dc2804b`.
Local headline snapshots retain each reported horizon. Runpod artifacts and
scheduler ledgers are under `raw/{fresh_backbone,controls,miss_controls}`.
The analysis entry point is
`experiments.expD34_readout_race.effective_feedback_analysis`; pass one cohort
at a time with its matching prediction manifest so repeated starts are not
pooled as new evidence.
