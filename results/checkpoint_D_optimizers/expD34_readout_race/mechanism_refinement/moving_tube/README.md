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

A separate [Arb calculation](../../../../../docs/d34_certified_instance.md)
does certify the entire 20,000-update window for the development `moment9`
instance with rounding control. Its stronger numerical status must not be
transferred to the longer FP64 calculation or to the other targets.

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

| Function | Last additional update excluding all 177 neurons | Excluded at +10k | Excluded at +20k |
|---|---:|---:|---:|
| sine | 3,658 | 0 | 0 |
| runge | 16,341 | 177 | 0 |
| moment3 | 18,274 | 177 | 0 |
| moment5 | Through 20k | 177 | 177 |
| moment9 | Through 20k | 177 | 177 |
| mixed_sine | 12,842 | 177 | 0 |
| localized_sine | 12,966 | 177 | 0 |
| chirp | 5,411 | 0 | 0 |
| moment4 | Through 20k | 177 | 177 |
| blend_m010 | Through 20k | 177 | 177 |
| blend_p001 | Through 20k | 177 | 177 |
| blend_p010 | Through 20k | 177 | 177 |
| blend_p030 | 11,476 | 177 | 0 |
| exp_right | Through 20k | 177 | 177 |
| exp_left | Through 20k | 177 | 177 |
| gauss_left | 13,649 | 177 | 0 |
| gauss_right | 5,772 | 0 | 0 |
| bump_left | 16,076 | 177 | 0 |
| bump_right | Through 20k | 177 | 177 |
| step_left | Through 20k | 177 | 177 |
| step_right | Through 20k | 177 | 177 |
| kink_abs | Through 20k | 177 | 177 |
| kink_relu | Through 20k | 177 | 177 |

The [machine-readable table](development_panel_summary.csv) gives the exact
censor indices. Each case's [summary](development_all_20k/summary.json) and
NPZ distinguish finite evaluated bounds from later censored entries. These
are repeated parameter bounds on a fixed function panel, not estimates of a
probability over functions.

The calculation is informative beyond `moment9`, including exponentials,
steps and kinks. Its early failure on sine, chirp and the shifted Gaussian
also prevents a universal long-window conclusion. A large bound radius can
come from conservative neighborhood amplification even when the actual
reference prediction is accurate. No true continuation was used to decide
which cases to retain.

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
600k-start panel. Independently initialized wider 20k-start states and longer
`moment9` sufficient horizons are subsequent, separately identified checks;
their pending outcomes are not included above.
