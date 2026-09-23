# How far the effective-force reference can enclose ordinary GD

The moving-reference calculation excludes acquisition of $\lambda=0.25$
through 200,000 additional ordinary-GD updates for both tested `moment9`
checkpoints. The earlier initial-ball calculation closes at 20,000 but not
50,000 updates for those same states. This is a useful extension of the
**FP64-evaluated sufficient enclosure**, not a rounding-controlled certificate
at 200,000 updates.

The broader result is mixed. At the 600,000-update development checkpoints,
the same calculation excludes every neuron through 1,000 further updates on
all 23 functions, through 10,000 on 20 functions, and through 20,000 on 13.
The other ten enclosures become uninformative. That is a failure of the bound,
not an observation that those networks acquire scale.

Fresh wider checkpoints give a more favorable short-window result. At
construction resolutions 512 and 1024, all six tested functions retain
exclusion of every neuron through 20,000 additional updates. At resolution
128, only two of those six enclosures reach that horizon. The comparison
holds the normalized threshold fixed and includes the four failed smaller
width bounds below.

A separate [Arb calculation](../../../../../docs/d34_certified_instance.md)
does certify the entire 20,000-update window for the development `moment9`
instance with rounding control. Its stronger numerical status must not be
transferred to the longer FP64 calculation or to the other targets.

The distant threshold alone is a weak test of mechanism. The
[elementary energy baseline](../../../../../docs/d34_energy_baseline.md)
already excludes it for 50,000 additional updates in all 18 fresh width-panel
states, using Arb-verified initial norm and loss bounds and no effective-force
assumption. Those initial conditions have not been checked for the separate
23-state late panel. The moving-reference gain is its much tighter trajectory
enclosure and the rate information it can support; a failed moving tube need
not imply failure of a generic threshold bound.

## Example: a longer bound for the same two starting states

These runs start at 600,000 updates, use physical width 177 and step size
$\eta=0.002$, and bound $\lambda_j=|a_j|/64$. Thus the requested threshold
$\lambda_*=0.25$ corresponds to $|a_j|=16$, not 64. No neuron initially
occupies this threshold. An additional 200,000 updates end at a nominal total
of 800,000 updates; the proof starts from the archived checkpoint and does
not certify its earlier training history.

| Starting state | Old initial-ball bound at +20k | Old bound at +50k | Moving-reference radius at +200k | Maximum prefix $\lambda$ bound at +200k | Neurons excluded through +200k |
|---|---:|---|---:|---:|---:|
| `moment9`, development seed 0 | Closes | Does not close | $4.5003\times10^{-5}$ | 0.00296908 | 177/177 |
| `moment9`, confirmation seed 20 | Closes | Does not close | 0.00254015 | 0.00334587 | 177/177 |

The old method also fails at the tested +100k, +200k and +1m horizons for
both states. This comparison uses the existing method's unchanged radius
grid; it is not an optimization over every possible initial-ball estimate.
The new radius measures error about a moving reference, whereas the old
path bound measures displacement from the initial state. Their numerical
radii therefore have different meanings. Their acquisition exclusions test
the same event.

The records are [development](development_moment9_200k/summary.json),
[confirmation](confirmation_moment9_200k/summary.json), and the
[matched old-bound audit](old_bound_comparison.json). The compact NPZ files
retain every step's scalar radius, defect, stability bound and exclusion
count, together with parameter and per-neuron envelope checkpoints.

## Theory: charge the full GD defect of a prescribed reference

The reference is the pure frozen effective model, generated from the
checkpoint alone:

$$
\bar e_{n+1}=(I-\eta T_0^TT_0)\bar e_n,\qquad
\bar\theta_{n+1}=\bar\theta_n-\eta T_0\bar e_n.
$$

Let $G(\theta)=\theta-\eta\nabla L(\theta)$ denote the ordinary-GD map.
The reference's one-step defect is

$$
d_n=\|G(\bar\theta_n)-\bar\theta_{n+1}\|
    =\eta\|\nabla L(\bar\theta_n)-T_0\bar e_n\|.
$$

This evaluates the full nonlinear gradient on a predicted state. It uses no
future ordinary-GD state and no observed forecast-error maximum. The
finite-step model is discrete throughout.

On the full-parameter ball $B(\bar\theta_n,r_n)$, the implementation bounds
the loss Hessian between $-\kappa_n I$ and $U_nI$. Its Gauss--Newton part is
positive semidefinite. The negative bound starts with the signed eigenvalues
of the $3\times3$ residual-curvature blocks, then adds analytic bounds for
their variation across the ball. Consequently

$$
\beta_n=\max\{1+\eta\kappa_n,|1-\eta U_n|\},\qquad
r_0=0,\qquad r_{n+1}=\beta_nr_n+d_n
$$

gives a sufficient error enclosure in exact arithmetic. To prove it, assume
the current true state lies in its ball. The segment joining it to the
reference lies in that convex ball; the Hessian bounds control the difference
of their next GD updates. Adding the reference defect places the next true
state within $r_{n+1}$. This is induction, not an assumption that GD follows
the reference.

The population bound uses
$\max_{k\le n}h(|\bar a_{j,k}|+r_k)$ for each neuron. It covers every
intervening update and every neuron that could ever cross, including crossings
at different times. The current implementation stops evaluating a case when
no neuron remains excluded or a bound becomes nonfinite; censored horizons
are reported explicitly, not as completed negative results.

The analytic inequalities are implemented in ordinary FP64, including the
reference generation, norms and eigenvalues. These evaluations do not enclose
all arithmetic roundoff. The separate Arb certificate instead treats stored
reference endpoints as exact dyadics and rigorously charges their actual
increments, so it does not need to trust the reference-generation arithmetic.

## Prediction checked across all 23 development functions

Every row uses its prescribed 600k checkpoint: seed 0 for the original
13 functions and seed 22 for the ten additional functions. The table retains
all functions, including the early failures. “Through 20k” is right-censored
at the requested horizon, not a claim that the bound remains useful afterward.

| Function | Additional update at all-allowed censor | Excluded at +10k | Excluded at +20k |
|---|---:|---:|---:|
| sine | 3,659 | 0 | 0 |
| runge | 16,342 | 177 | 0 |
| moment3 | 18,275 | 177 | 0 |
| moment5 | Through 20k | 177 | 177 |
| moment9 | Through 20k | 177 | 177 |
| mixed_sine | 12,843 | 177 | 0 |
| localized_sine | 12,967 | 177 | 0 |
| chirp | 5,412 | 0 | 0 |
| moment4 | Through 20k | 177 | 177 |
| blend_m010 | Through 20k | 177 | 177 |
| blend_p001 | Through 20k | 177 | 177 |
| blend_p010 | Through 20k | 177 | 177 |
| blend_p030 | 11,477 | 177 | 0 |
| exp_right | Through 20k | 177 | 177 |
| exp_left | Through 20k | 177 | 177 |
| gauss_left | 13,650 | 177 | 0 |
| gauss_right | 5,773 | 0 | 0 |
| bump_left | 16,077 | 177 | 0 |
| bump_right | Through 20k | 177 | 177 |
| step_left | Through 20k | 177 | 177 |
| step_right | Through 20k | 177 | 177 |
| kink_abs | Through 20k | 177 | 177 |
| kink_relu | Through 20k | 177 | 177 |

The [machine-readable table](development_panel_summary.csv) gives the exact
censor indices. This column records the first update when the bound allows
every neuron. A separate [full-count trace audit](development_panel_count_audit.json)
confirms that this panel has no intermediate partial exclusions: its ten
failed cases really change from 177 excluded neurons to zero in one update.
This also verifies the CSV's `last_all_excluded` field directly, rather than
inferring it from the censor index. Each case's [summary](development_all_20k/summary.json) and
NPZ distinguish finite evaluated bounds from later censored entries. These
are repeated parameter bounds on a fixed function panel, not estimates of a
probability over functions.

The calculation is informative beyond `moment9`, including exponentials,
steps and kinks. Its early failure on sine, chirp and the shifted Gaussian
also prevents a universal long-window conclusion. A large bound radius can
come from conservative neighborhood amplification even when the actual
reference prediction is accurate. No true continuation was used to decide
which cases to retain.

## Fresh wider states: the correction radius also becomes smaller

These are separately initialized seed-30 networks at 20,000 updates, followed
by a 20,000-update reference calculation. Their nominal endpoint is therefore
40,000 updates. Construction resolution $N_{\rm ref}$ and physical width $W$
are different: the three pairs are $(128,177)$, $(512,705)$ and $(1024,1409)$.
We use $h=2/N_{\rm ref}$ throughout. The fixed threshold $\lambda_*=0.25$
therefore means physical slopes $|a|=16$, $64$ and $128$, respectively.
No starting neuron already occupies its corresponding threshold.

Every neuron remains excluded through the full window in all six cases at
512 and all six at 1024. The bounds include the entire prefix, not just its
endpoint. The table gives the censor update at 128 and the
20,000-update results at the two larger resolutions.

| Target | Censor update at 128 | Radius at 512 | Radius at 1024 | Maximum prefix $\lambda$ bound at 512 | Maximum prefix $\lambda$ bound at 1024 |
|---|---:|---:|---:|---:|---:|
| `moment5` | Through 20k | $5.018\times10^{-9}$ | $3.708\times10^{-10}$ | 0.000422330 | 0.000146696 |
| `mixed_sine` | Through 20k | 0.000234940 | 0.0000636934 | 0.000406433 | 0.000149610 |
| `gauss_left` | 11,395 | 0.00496653 | 0.00105419 | 0.000487894 | 0.000155849 |
| `bump_right` | 11,437 | 0.00297110 | 0.000560635 | 0.000460692 | 0.000147302 |
| `step_right` | 8,413 | 0.0109267 | 0.00142510 | 0.000588744 | 0.000185063 |
| `kink_abs` | 11,253 | 0.00453385 | 0.00101618 | 0.000450077 | 0.000149068 |

The [128](width128_seed30_20k/summary.json),
[512](width512_seed30_20k/summary.json) and
[1024](width1024_seed30_20k/summary.json) records retain all cases and their
input hashes. At 128, the two successful maximum prefix bounds are 0.00353834
for `moment5` and 0.00296032 for `mixed_sine`. The other four calculations
stop when their bound permits every neuron; this is not observed acquisition.

The improvement is not just the smaller conversion factor $h$. The
full-parameter correction radii also decrease. For example, `mixed_sine`
has radii 0.00324694, 0.000234940 and 0.0000636934 at the same additional
20,000-update horizon. The calculation charges the actual reference defect
and bounds its amplification, so its success tests more than a small initial
normalized slope. All of these numbers still evaluate sufficient formulas
in FP64; this table supplies no directed-rounding certificate.

For these same 18 initial states, the verified energy baseline bounds total
parameter travel by eight through 50,000 additional updates, or nominal
update 70,000. It excludes the distant threshold even in the four cases whose
moving tube fails before 20,000. Its positions are much less precise. The
two methods therefore answer different questions: the baseline certifies
the threshold exclusion, while this table tests how tightly the
effective-model path can enclose ordinary GD.

The experiment is consistent with a slower small-parameter clock at larger
width. It does not determine an asymptotic rate: there is one independently
initialized seed per width, the initial states differ, and the fixed physical
time $\eta N=40$ corresponds to rescaled time $40/W$. Separate longer
calculations for `mixed_sine` and `step_right` at 512 and 1024 were requested
to find where the bound loses usefulness. The next section reports those
separate refinements; they do not alter this fixed-window comparison.

## Longer reference windows depend on the function

The four refinements use the same seed-30, update-20k checkpoints and request
200,000 additional updates, with the existing early-censor rule. All four
calculations stop before that requested endpoint. `mixed_sine` nevertheless
retains a tight enclosure through the +50k checkpoint at resolution 512 and
the +100k checkpoint at 1024. `step_right` loses this particular enclosure
substantially earlier at both widths.

| $N_{\rm ref}$ | Target | Largest retained test horizon | Radius there | Maximum prefix $\lambda$ bound there | Enclosure censor update |
|---:|---|---:|---:|---:|---:|
| 512 | `mixed_sine` | +50k | 0.00201718 | 0.000406433 | +72,072 |
| 512 | `step_right` | +20k | 0.0109267 | 0.000588744 | +25,095 |
| 1024 | `mixed_sine` | +100k | 0.00547027 | 0.000151910 | +110,042 |
| 1024 | `step_right` | +20k | 0.00142510 | 0.000185063 | +39,096 |

Every neuron is excluded at each retained test horizon: 705 neurons at 512
and 1409 at 1024. The +50k and +100k horizons end at nominal total updates
70k and 120k, respectively. “Largest retained test horizon” refers to the
prescribed checkpoint grid, not the exact last update with all-neuron
exclusion. The censor column records the helper's stop for loss of exclusion
or a nonfinite bound. Later entries are explicitly unevaluated, with null
radius and upper-bound fields; they are not completed predictions.

The individual records are
[512 mixed sine](width512_mixed_sine_limit/summary.json),
[512 step](width512_step_right_limit/summary.json),
[1024 mixed sine](width1024_mixed_sine_limit/summary.json), and
[1024 step](width1024_step_right_limit_parallel/summary.json).
Their input hashes and helper source match the corresponding 20k checks.
All numbers in this table are **FP64-evaluated sufficient enclosures**,
not directed-rounding certificates or observed acquisition times.

The two step cases illustrate why that distinction matters. Their moving
tubes stop before +50k, yet the independently verified energy baseline still
excludes the threshold through +50k in both. The failure concerns precision
around this frozen effective reference. It does not establish that ordinary
GD leaves the small-scale regime, or that the effective fine force ceases
to drive its motion.

## Which correction matters, and what this does not establish

The instantaneous reference defect can be split exactly into map change,
nonlinear residual mismatch, coarse tracking and omitted modes:

$$
\nabla L(\bar\theta)-T_0\bar e
=(T(\bar\theta)-T_0)e_H(\bar\theta)
+T_0(e_H(\bar\theta)-\bar e)
+r_C(\bar\theta)+g_\perp(\bar\theta).
$$

The code records these terms at the stated reference checkpoints. They are
point diagnostics, not uniform channel bounds over the enclosing balls.
For development `moment9` at +200k, their full-parameter norms are
$3.42\times10^{-8}$ for map change, $3.87\times10^{-10}$ for residual
mismatch, $3.11\times10^{-7}$ for tracking, and about
$2.88\times10^{-16}$ for omitted modes. The larger tracking term here is
coarse imbalance generated by the **frozen reference**. It does not overturn
the existing evidence that tracking is small along ordinary GD or identify
tracking as its scale-acquisition driver.

This calculation uses the effective model as the reference and the actual
loaded residual curvature for stability. It does not yet exploit every
cancellation in the [paired-response theorem](../../../../../docs/d34_mechanism_rate_theorems.md#7-a-loaded-response-condition-that-closes-the-ordinary-gd-envelope).
The next improvement can therefore target a concrete obstruction: reduce the
reference defect, tighten its neighborhood amplification, or establish the
loaded paired-response bounds. A weak frozen-model correction budget must
not be relabeled a proof for ordinary GD.

## Reproducibility and current scope

The [helper](../../../../../experiments/expD34_readout_race/mechanism_moving_tube.py)
and [four focused tests](../../../../../tests/test_expD34_mechanism_moving_tube.py)
are committed in `bb15526`. The tests independently check gradients and ball
constants, a small nonlinear actual-GD enclosure, both discrete spectral
endpoints, and explicit censor handling. They passed before both main CPU
runs. Slurm jobs 1244 and 1245 completed with exit code zero in 26m09s and
24m39s respectively, each on four CPUs and no GPU.

Each summary records its exact input and source hashes. The all-function
command uses `--target all --steps 20000`; the two longer checks use
`--target moment9 --steps 200000`. The current table covers the completed
600k-start panel. The independently initialized width panel uses the same
helper with the exact $h=2/N_{\rm ref}$ and completed in CPU jobs 1267 and
1277. The optional one-million-update `moment9` extension was stopped for
scientific prioritization after 30m07s in CPU job 1289. It had saved no
completed case, so it contributes no result or partial-prefix claim; this
was not a numerical failure. The independently preserved 200k results are
unaffected. Longer width-specific sufficient horizons are separately
identified checks. CPU job 1292 saved the first three results in the longer
table and was stopped after 49m48s to avoid duplicating the final case,
which was already running concurrently. That last case completed in CPU
job 1299 in 11m31s with exit code zero. Its `parallel` directory is the
retained record; the original duplicate directory contains no completed
evidence. No new training or GPU computation was used for these refinements.
