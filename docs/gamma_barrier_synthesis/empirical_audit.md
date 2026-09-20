# Empirical audit of the proposed GD plateau claim

Read-only review of expD24, D25, D26, D27, and D30, prepared September 14, 2026. This report does not cover the new D28/D29 gradient interventions or audit the construction theorem itself. No training or parameter sweep was repeated. One additional quantity—the note's drift budget—was calculated directly from complete saved trajectories. Other contributors own the synthesis and D28/D29 review; no existing experiment files were changed.

**Assessment:** The evidence supports a separation between choosing an accurate, numerically usable representation and optimizing coefficients in that representation. It also establishes restricted examples of useful geometry learning. It does not yet prove that ordinary joint GD fails to find the construction's width-dependent scale because readout training removes a useful geometry signal. That last causal connection is the central gap in the proposed paper statement.

## Seven established observations

### 1. Ordinary finite-budget training largely fits readouts on nearly unchanged geometry

The matched four-way D24 experiment compares joint GD, readout-only GD on frozen initial geometry, joint trajectories evaluated with new readout solves, and the initial geometry with a readout solve. Four targets, Xavier and gamma 1/4/16, identical finite-domain samples, 2,000 steps, and rate 0.002 make this the clean baseline. The readout-refitted geometry changes little in most cases, while actual fitting improves. “Little” should not become “never” or “exactly invariant.” Some refits improve, and some accurate initializations leave almost no useful approximation improvement to measure.

For the same training objective and exact projection, the identity

\[
L(\theta,v)=L_*(\theta)+\tfrac12\|A(\theta)(v-v_*(\theta))\|^2
\]

identifies an approximation component and an unfinished-readout component. A decrease in current loss with almost constant refit loss belongs predominantly to the latter component. It does **not** by itself prove that the readout matrix became better conditioned. Geometry can provide another prediction-space route to reducing error already expressible by readout changes.

Sources: [matched comparison figure](../../results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/comparison.png), [geometry benefit figure](../../results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/gains.png), [catalogue, movement 14](../expD24_session_catalogue.md), [independent D24 review](../expD24_gamma_review/independent_review.md).

### 2. Fourier sensitivity is demonstrably real, but the strongest training-time application remains unproved

The controlled D24 matched-residual probe fixes residual norm, center, and unit readout while varying scale. It shows a rise and then fall of the actual centered-scale gradient. At gamma 128–256, fitted log-log slopes are approximately -3.00, -2.99, and -2.93 for the three carrier frequencies. The whole-line band-pairing diagnostic shows that the band holding most residual energy need not supply most scale gradient: at gamma 4, the low band supplies the strongest contribution while containing only about 16% of residual energy. At gamma 64, high frequencies dominate a much smaller total centered gradient.

These results establish filtering and amplitude shrinkage. They do not establish a persistently empty low-frequency band throughout the stalled trajectory, negligible finite-boundary/sampling corrections relative to the putatively tiny signal, or a Fourier escape-time lower bound for raw slope/bias GD. Large-gamma inverse-cube decay is a conditional centered-gradient asymptotic already present in the note. It does not contradict the note or mean that increasing gamma always decreases actual motion.

The coordinate distinction is quantitative: at the gamma-64 final snapshots, mean absolute raw slope gradients were approximately 184, 52, 39, and 26 times the centered gradients for sine, mixture, Runge, and whole-line Gaussian. The exact relation is \(g_a=g_\gamma+zg_b\). The Fourier gradient plots show \(g_\gamma\); raw GD moves slopes with \(g_a\).

Sources: [matched-residual probe](../../results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/matched_residual.png), [band pairing](../../results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/frequency_pairing.png), [tangent sweep](../../results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/tangent_gamma_sweep.png), [independent audit with checked numbers](../expD24_gamma_review/independent_review.md).

### 3. Useful geometry gradients exist, and changing their treatment can recover excellent approximation

D25 supplies direct counterexamples to “GD geometry movement cannot improve approximation.” Holding centers fixed and changing only the geometry rate from 0.002 to 20, with readout GD still at 0.002, improves the numerical refit in 2,000 steps:

| Target | Initial refit relative error | Final refit relative error |
|---|---:|---:|
| Mixed sine | 0.206 | 0.000342 |
| Runge | 0.0124 | 0.00000843 |
| Gaussian envelope | 0.223 | 0.0390 |

No failures occurred in the D25 large-rate runs. A 10,000-fold larger rate is evidence of poor block scaling at the common baseline rate, but is not itself evidence that those executed trajectories were unstable.

The stronger restricted remedy fixes uniform centers and trains one shared scale with Adam at 0.002 while readout still uses GD at 0.002. Starting at gamma 1, after 10,000 steps the learned scales are 6.245, 13.880, 7.738, and 15.496. Offline refits achieve approximately 6.37e-14, 7.79e-13, 3.33e-13, and 2.03e-14, with modest solved coefficient norms 0.674, 10.708, 0.816, and 4.148. The actual trained models remain much less accurate: 0.0225, 0.3214, 0.0535, and 0.2817 relative error.

Thus usable geometry signal is present in these finite-domain controls. This does not remove the Fourier inequality, prove a general random-initialization optimizer, or show that ordinary readout GD realizes the available accuracy. Uniform centers and a shared scale are substantial structural prior information. Cutoff and evaluation-grid audits support the orders-of-magnitude improvement; exact floor-level digits are not invariant.

Sources: [D25 report](../../results/checkpoint_D_optimizers/expD25_scale_barrier/expD25_results.md), [fixed-center scale escape](../../results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_escape.png), [adaptive scale from gamma 1](../../results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_1.png).

### 4. Optimizing current loss does not protect the best approximation regime

The same shared-scale Adam method starting at gamma 16 moves mixed sine to 32.605 and Gaussian to 32.018. Their refit errors deteriorate from near numerical precision to 5.74e-8 and 4.55e-8 while their current trained-readout errors improve. This is a particularly informative result: insufficient movement and movement toward the wrong precision regime are different problems. A method that merely increases geometry mobility need not select or preserve the construction's useful scale.

Source: [D25 gamma-16 control](../../results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_16.png), [D25 report](../../results/checkpoint_D_optimizers/expD25_scale_barrier/expD25_results.md).

### 5. Fixed-geometry readout dynamics are the most quantitatively verified explanation so far

D26's 77-value lambda sweep confirms that increasing lambda greatly improves target-relevant singular directions and ordinary GD's moderate-accuracy convergence. At rate 0.002, predicted mixture steps to 1% relative error fall from 1.67e11 at lambda 0.1 to 1.92e6 at 0.25 and 75,102 at 2. Gaussian changes from 4.27e11 to 2.33e6 to 27,048. Explicit GD checks agree with the spectral recurrence to about 3.44e-15 in relative error.

Large lambda nevertheless sacrifices the dense-grid numerical approximation floor: at lambda 2, floors are approximately 0.001565, 0.004031, 0.0008956, and 0.004164 for the four targets. The readout matrix spectrum is not a whole-line Fourier spectrum, and a numerical SVD cutoff is not an exact mathematical nullspace. The tradeoff here concerns the raw tanh coefficient parameterization, sampling, and finite model. It is not a universal conditioning theorem for all equivalent readout bases.

Sources: [D26 report](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/expD26_results.md), [readout spectrum GIF](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_spectrum.gif), [convergence and floor](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_convergence.png).

### 6. Even removing the observed training approximation floor leaves a long readout optimization problem

D30 uses gamma 128, 129 samples at exact in-domain centers, 177 neurons plus output bias, and zero readout. The initial readout matrix has full row rank, condition number approximately 10,621, and modest minimum-norm solution norms 0.124–1.044. Least squares reaches losses around 1e-30–1e-29.

The best tested constant rate, 0.01971069, is 95% of the initial fixed-geometry GD stability limit. After 200,000 actual updates, fixed-geometry relative errors are 1.604e-4, 6.072e-4, 1.229e-6, and 3.938e-5. Joint training notably improves mixture to 3.225e-4, but does not improve its independent-grid error. Independent-grid errors remain roughly 1e-3–5e-3. Mean absolute gamma displacement in the joint runs ranges from 8.4e-7 to 8.9e-4; maximum individual movement is 0.0289.

For fixed geometry, the exact-arithmetic expression

\[
L_k=\tfrac12\sum_j\alpha_j^2(1-\eta\sigma_j^2)^{2k}
\]

matches executed relative errors within roughly 3e-14 and predicts 225–655 million steps to reach 1e-12 relative error at that rate. Those long horizons were not executed. The formula approaches zero; the experiment has not proved a nonzero asymptotic training plateau. The tested warmup/cosine schedule loses to the larger constant rate, but this does not exclude all GD schedules or accelerated methods.

This experiment has an additional logical implication: a full-row-rank underdetermined training matrix already permits zero training residual throughout a nearby open set of geometries. Exact profiled **training** loss cannot rank their different between-center approximation quality there. The continuous-function objective and interpolation objective must stay separate.

Sources: [D30 report](../../results/checkpoint_D_optimizers/expD30_center_sampled_gd/expD30_results.md), [training traces](../../results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/training_convergence.png), [spectrum and predictions](../../results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/spectrum_and_schedules.png), [between-center error](../../results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/between_centers.png).

### 7. Freezing does not establish the proposed lost-information mechanism

D26 freezes early Xavier readouts at 2, 10, 50, or 150 steps, continuing geometry GD for 500 updates. All 16 branches finish with worse actual error and lower mean scale than matched joint controls. D27's additional 48 scheduled frozen branches also all finish worse than their continuing references. An accurate QI readout transferred to gamma-1 geometry improves actual mixture error from 1.287 to 0.963, yet its refit changes only from 0.205962 to 0.205910 and mean gamma ends at 0.998998. Clean QI retains its good geometry; noisy QI does not repair its precision defect.

Solving Xavier readouts and then freezing creates destructive large-scale movement, but coefficient norms around 1e10–1e11 cause material arithmetic sensitivity. This is not clean evidence about exact constant-rank VarPro. At each freeze, the first geometry update matches the continuing reference bitwise: freezing changes subsequent dynamics, not the instantaneous geometry gradient.

These experiments refute the simple practical prediction that the tested freezing schedules reliably rescue scale learning. They do not isolate which term in the note's decomposition supplied useful approximation descent, show that readout fitting removed it, or exclude all freezing methods. A large shared gradient is not automatically a useful approximation gradient.

Sources: [D26 report](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/expD26_results.md), [D27 report](../../results/checkpoint_D_optimizers/expD27_readout_information/expD27_results.md), [QI coefficient transfer](../../results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_transfer/all_targets.png).

## New audit from existing complete trajectories: the drift budget is restrictive

The note's sampled raw-slope bound is directly applicable because |x| <= 1 and |tanh'| <= 1. With R_k = sqrt(2L_k), define

\[
B_j(T)=\sum_{k=0}^{T-1}\eta |c_j^k|R_k.
\]

Then total raw-slope travel is bounded by B_j(T), and |a_j^T| <= |a_j^0|+B_j(T). This requires no Fourier gap, population quadrature approximation, frozen geometry, or centered-coordinate conversion.

D26 saves every state of the 650-update ordinary joint reference, including coefficients and fresh train loss. Computing this budget from those files gives:

| Target | Largest budget B_j | Largest actual total slope travel | Largest allowed final gamma, max_j(|a_j^0|+B_j) |
|---|---:|---:|---:|
| Sine | 0.172170 | 0.034783 | 0.334892 |
| Mixed sine | 0.195676 | 0.038569 | 0.358611 |
| Runge | 0.083668 | 0.016385 | 0.254297 |
| Gaussian envelope | 0.119800 | 0.014266 | 0.288706 |

The largest initial gamma is 0.181630 in all four runs. Every neuron satisfies the travel inequality. Thus even the deliberately loose absolute bound excludes reaching gamma 16 during these executed trajectories by a large margin; phase cancellation or Fourier suppression is not needed for that finite-horizon certificate.

This is an **a posteriori** budget, conditional on recorded coefficient/residual histories. It is not yet an a priori theorem that controls those histories or a lower bound for all future time. It also does not show that gamma 16 is necessary for approximating these targets, or that small scale is the sole obstruction in Xavier geometry. It is a useful concrete link between the note and the actual trajectories.

Reproduction uses each `freeze_TARGET.npz` in [D26 data](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/data): `a=joint__a`, `c=joint__v[:-1,:-1]`, `R=sqrt(2*joint__train_loss[:-1])`, `B=.002*sum(abs(c)*R[:,None],axis=0)`. Actual total travel is `sum(abs(diff(a,axis=0)),axis=0)`. The endpoint and each step use the saved raw slopes; halo neurons are included. The D24/D25 long runs save coefficients sparsely, so their corresponding exact per-step budgets were not inferred by interpolation.

For a separate prospective energy-descent calculation, the same trajectories have 177 neurons, initial mean absolute slope 0.08804407411763386, and initial RMS slope 0.10197082474286355. Their initial and final half-MSE values are:

| Target | Initial loss | Loss at step 650 | Squared net raw-slope displacement |
|---|---:|---:|---:|
| Sine | 0.405667317909540 | 0.217242146082213 | 0.073307389615 |
| Mixed sine | 0.497219049229259 | 0.279645660715945 | 0.089938467262 |
| Runge | 0.263043958996733 | 0.041505764557469 | 0.016048638847 |
| Gaussian envelope | 0.198641505524623 | 0.116732576066996 | 0.012379246680 |

The squared Euclidean distance from the initial slopes to the nearest vector with every absolute slope equal to 16 is 44,815.1588188882. If only mean absolute slope at least 16 is required, a valid weaker lower bound is `177*(16-0.08804407411763386)**2`. An energy-based time lower bound using these distances still needs a sufficient-descent premise for the actual joint optimization and the selected parameter metric. Violation of a conservative sufficient-descent condition is not by itself a proof of instability.

## What a defensible claim can currently say

Subject to the independent construction-theory audit, a defensible empirical claim is:

> The construction supplies a width-dependent scale regime with high attainable precision. In the tested tanh networks, ordinary full-batch GD at a common stable rate largely reduces the readout optimization gap while making little useful scale progress. Sampled gradient bounds quantitatively limit the observed finite-horizon drift, and fixed-geometry spectral dynamics explain a separate, severe high-precision readout slowdown. Increasing scale improves moderate-accuracy readout convergence but can worsen continuous approximation. Useful geometry learning is possible under structured scale updates, while current training loss alone does not reliably preserve the best precision regime.

It is not yet justified to replace that with “we prove noiseless GD fails to determine the regime because readout fitting removes useful geometry information.” The following links would close that gap:

1. **A specified regime and success criterion.** Fix initialization, width scaling, sampling, coordinates, step-size class, coefficient constraints, horizon, and whether success means finite-sample loss or continuous approximation. Full-batch deterministic fp64 training is not an exact-arithmetic theorem.
2. **A necessary accuracy condition.** A sufficient QI construction with slopes proportional to width does not establish that every accurate unconstrained MLP must use that regime. Either prove necessity for a stated class, or frame the result as failure to recover this construction's bounded-coefficient regime.
3. **A priori control of the drift factors.** Bound coefficients and residual histories, with constants that force travel below the required scale displacement for a substantial width/time range. The observed-budget calculation is promising but only retrospective.
4. **A useful-signal measurement or theorem.** Compare the shared geometry term with the gradient of profiled approximation loss and show positive descent alignment before readout learning, followed by its removal before useful movement. Norm dominance alone does not supply this implication. Account for the current versus solved readout Jacobian and numerical rank stability.
5. **A scope-appropriate escape/control.** Any claimed universal inability must accommodate D25's useful scale learning and D30's eventually convergent fixed readout. A finite-horizon/common-rate limitation is more consistent with current evidence than absolute failure.

The project already has substantial evidence for two distinct training bottlenecks and a real approximation-versus-optimization tradeoff. The remaining work is to connect those facts into a causal, quantitatively nonvacuous theorem for joint training, rather than interpret each correct identity as having supplied that theorem.
