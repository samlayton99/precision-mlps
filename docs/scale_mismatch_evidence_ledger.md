# Evidence ledger: Readout–scale competition

Prepared 19 September 2026 from the saved expD24–expD34 figures, catalogue, and experiment writeups. No new training or numerical analysis was performed for this ledger.

This follows the six sections of *Readout–scale competition*, supplied as `scale_mismatch_optimization (1).pdf`. “Supported” means demonstrated in the stated experiments, not a universal theorem. A negative experimental result can reject an extrapolation of a conditional theorem without contradicting the theorem itself.

The [session catalogue](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/expD24_session_catalogue.md) contains the full experiment and figure inventory. This ledger selects evidence relevant to the note's claims.

## Measurement rules that affect the conclusions

- Distinguish actual trained-model error from error after an evaluation-only coefficient solve. A small refit error establishes an available approximation, not that GD trained the readout to that accuracy.
- Numerical refits generally use an SVD cutoff of 1e-13. Their floor is a specified numerical approximation diagnostic, not the exact infimum over arbitrarily large coefficients. Near-floor fluctuations and unresolved gradient directions cannot establish meaningful improvement or cancellation.
- Whole-line Gaussian experiments use cancelling tails. They are not the same objective/model as the matched finite-interval experiments. The corrected D24 four-way comparison uses the same finite interval and standard model for all four targets.
- A center-preserving gamma derivative differs from a raw slope update with bias held fixed. The Fourier probes measure the former; ordinary raw MLP training generally uses the latter.
- The loss split L=F+G and its gradient split are not the same two terms as the note's current-readout projection identity. The VarPro derivative uses the solved readout; the projection identity uses the current readout.

## 1. Residual refinement and Fourier sensitivity

**Claim:** High-frequency residuals can pair very weakly with a broad tanh scale tangent. As fitting proceeds, the remaining residual may become concentrated at frequencies poorly seen by the current scales.

### Supported

1. **The residual–tangent pairing calculates the scale derivative.** D24 compared the absolute value of the signed, coefficient-weighted spatial integral against automatic differentiation for four targets, three initializations, and 500 steps × 128 neurons per panel. All R² values exceed 0.999999999. Dense integration differs slightly from the training quadrature; using the same samples agrees to floating-point precision. [Exact pairing, linear axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gradient_correlation/exact_pairing.png).

2. **Instantaneous frequency mismatch is real.** With residual, center, and readout fixed, the gradient is weak at small gamma relative to the carrier frequency, rises, then falls at sufficiently large gamma. Increasing gamma broadens the tangent spectrum while reducing its amplitude. [Fixed-residual sensitivity and tangent spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/02_scale_sensitivity.png).

3. **Low-band error is preferentially removed in the successful regimes examined.** This is visible in absolute band errors, not merely rising high-band fractions. [Whole-line Gaussian frequency-learning GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/01_frequency_learning.gif).

4. **Error energy and gradient contribution are different quantities.** Phase-sensitive band pairings show that the band containing most residual energy need not supply most scale gradient. At larger gamma, high frequencies supply a larger fraction of the signal, while the total signal can be smaller. [Frequency-band energy and gradient contributions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/05_frequency_gradient_pairing.png).

### What the evidence rules out or qualifies

- The exact-pairing scatter validates the finite-window gradient identity. It does not validate a persistent whole-line spectral gap or the exponential escape-time bound.
- The band plot retains substantial low-frequency error at small gamma. Its low band can dominate the gradient even when it contains a minority of the energy. The clean high-frequency-only premise is not established along these trajectories.
- The magnitude-only overlap bound in that plot is not the note's exponential envelope. Their tightness must not be conflated.
- A conditional gamma^-3 asymptotic exists for fixed smooth residuals and readouts, with appropriate boundary separation. We did not establish that this power law dominates all observed training trajectories. Residuals, centers, and readouts also change.

**Verdict:** Strong evidence for instantaneous Fourier sensitivity and scale-dependent band selection. No demonstration that this is the dominant cause of a persistent training barrier.

## 2. Readout compensation and sensitivity squaring

**Claim:** Some geometry motion changes predictions in directions already attainable by changing the readout. Independent geometry sensitivity is the projected Jacobian Z. Near an exact profiled fit, GD rates depend on squared singular values of Z.

### Supported

1. **Baseline joint GD improves current fitting much more than it improves the available numerical approximation.** D24's corrected four-way test compares joint GD, fixed-geometry readout GD, refits of learned geometry, and the initial-geometry refit. It uses four functions, Xavier/gamma 1/4/16, matched samples, 2,000 steps, and independent-grid evaluation. Refit changes are small but not identically zero. [Four-way comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/comparison.png); [differences exposing geometry benefit](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/gains.png).

2. **The current-readout out-of-span gradient is extremely small in the measured baseline states.** D25 measures J^T r_perp versus J^T r, alongside block curvature and gradient sizes. Ratios around 1e-11 or smaller often approach numerical resolution. This is direct evidence about the current J, not the solved-readout J*. [Derivative and curvature diagnostics](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/derivatives.png).

3. **Approximation-gradient imbalance occurs; universal cancellation does not.** D28 measures the numerical F/G loss split, gradient norms, and cosine similarity. Scaled Xavier exhibits strong norm imbalance. Ordinary Xavier mixed sine develops opposition, whereas Runge develops alignment. Several near-floor gradient directions are unresolved and their cosines are omitted. [Xavier decomposition](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/xavier.png); [scaled Xavier decomposition](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/scaled_xavier.png); [QI decomposition](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/qi_zero.png).

### What remains untested

We have not measured the relevant Z* singular spectrum, residual loading on its directions, and subsequent geometry convergence together to test the predicted s² rates. Readout experiments verify the generic squared-sensitivity mechanism, but they do not establish its geometry-specific causal role.

Neither a tiny current-J projected gradient nor little baseline refit improvement proves that useful geometry gradients are absent. D25's rate interventions and D31's amplified profiled-gradient experiments produce real approximation gains. Also, DF/DG cancellation does not directly measure cancellation between the two terms in equation (4).

**Verdict:** Evidence supports readout-dominated baseline progress and weak measured independent signals. The proposed projected-sensitivity explanation is not yet quantitatively isolated.

## 3. Exact allocation model and the freezing prediction

**Claim:** In the positive, underfit bilinear toy model, faster readout fitting limits scale growth. Earlier readout freezing selects a larger eventual gamma, but may make convergence slower. Broad tanh has an amplitude-like leading scale derivative.

### Direct experimental test: early freezing did not produce the hoped-for growth

D26 branches Xavier training after 2, 10, 50, or 150 updates, freezes all readout coefficients, and continues geometry GD for 500 updates at the same rate. Across all 16 branches, final mean gamma is below its initial value and below the matched joint-training control; actual fitting is worse. These results reject the practical prediction that this early-freezing recipe releases useful scale growth in our tested networks.

[Sine freeze branches](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine.png); [mixed-sine branches](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine_mixture.png); [Runge branches](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_runge.png); [Gaussian branches](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_gaussian_envelope.png); [evaluation-only refits](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_refit.png).

### Useful supporting and contrasting checks

- An early small-slope expansion explains the observed direction of shrinkage. This supports amplitude coupling, without establishing the positive-underfit toy trajectory as a model of random Xavier. [Early drift diagnostic](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/linear_drift.png).
- Freezing a well-fitted nonlinear Runge model gives tiny positive scale growth, slightly greater than joint training. It still slows fitting and produces no meaningful escape. Initialization and training stage differ from the early-Xavier test. [Late nonlinear freeze](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/late_freeze_runge.png).
- A fixed-geometry readout-GD calculation predicts the fitted model's 5,000-step nudge pattern to about 0.03% relative discrepancy. This explains much of the apparent phase-plot structure through readout and residual evolution. [Observed versus predicted nudges](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_flow_prediction.png).
- The nudge scatters used one-step block probes from a joint-GD trajectory. They do not demonstrate compensation during sustained freezing. [Actual joint parameter histories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/joint_gd_parameter_evolution.png).
- D27's additional recipes do not recover the desired regime reliably. Solving before freezing introduces enormous coefficients and numerical sensitivity; transferring a QI readout to gamma 1 does not drive scales back to QI. [Solve-then-freeze refits](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/refitted_geometry.png); [QI-readout transfer](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_transfer/all_targets.png).

**Not tested:** The toy endpoint formula as a quantitative predictor across learning-rate ratios, or a controlled tanh test of the remainder bound and its useful sign. The toy has zero profiled approximation loss for every positive gamma, so it is an allocation example, not itself an approximation barrier.

**Verdict:** Amplitude coupling has support. The simple early-freezing remedy failed in the cases run; this does not contradict the toy's conditional mathematics.

## 4. Bounded slopes, conditioning, and coefficient cost

**Claim:** Bounded slopes force rapidly decaying feature singular values. If the target needs those weak directions, high accuracy becomes expensive in coefficient size and/or GD steps.

### Strong numerical support for the conditioning part

D26 sweeps uniform lambda from 0.1 to 2 at fixed centers, measures the feature singular values, and predicts target-specific fixed-readout GD convergence. Representative executed trajectories match the modal predictions to approximately 3.4e-15. Larger lambda speeds convergence to moderate accuracy substantially, but can worsen the final available approximation. [Readout spectrum GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_spectrum.gif); [target floors and predicted steps](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_convergence.png).

D30 then executes 200,000 GD steps at gamma 128/lambda 2, using center-only samples. The fixed-geometry traces match spectral predictions to about 3e-14. The best tested constant rate still predicts roughly 2.25e8–6.55e8 steps to training relative error 1e-12. Those long horizons are predictions, not executed trajectories. [Executed convergence](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/training_convergence.png); [spectrum and convergence predictions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/spectrum_and_schedules.png); [between-center accuracy](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/between_centers.png).

D25 and D27 also expose huge solved coefficients and sensitivity to numerical rank at poor geometries. [Scale landscape, coefficient norms, and rank](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/landscape.png); [readout numerical audit](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/readout_sensitivity.png).

### Limits

These experiments do not test the specific exponential-in-width/cap bound over a width-by-cap sweep, nor evaluate the target polynomial approximation term delta_k and coefficient-budget inequality quantitatively. The large-gamma center-sampled test also changes sampling; it is not a pure gamma ablation against dense-grid runs. Large gamma does not make raw tanh columns orthogonal automatically.

**Verdict:** Fixed-readout spectral conditioning is the most quantitatively verified mechanism in the session. The note's uniform bounded-slope and target-budget bounds are consistent with observations but have not been tested sharply.

## 5. Residual depletion and remaining scale travel

**Claim:** Ordinary slope travel is bounded by the accumulated residual times readout magnitude and learning rate. Persistent high-frequency mismatch gives a stronger conditional escape-time bound.

### Supported observation, but not yet a tested time bound

D24 documents small baseline scale movement, including less movement at larger initial gamma in the examined runs. Net displacement and total travel are distinguished. [Gamma sweep](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gamma_motion.png); [net movement and travel on alternate axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/gamma_motion_axes.png).

We have not compared the accumulated bound sum eta|c|R with the required scale displacement along the runs, established its tightness, tested the h^-1/h^-2 scaling, or demonstrated persistent spectral-gap assumptions until escape. Small observed movement is not a training-time lower bound.

### Positive interventions constrain the interpretation

D25 increases only the independent scale learning rate with centers fixed and readout GD unchanged. In 2,000 steps, mixed-sine refit error improves from 0.206 to 0.000342 and Runge from 0.0124 to 8.43e-6. Useful geometry motion is available in those states. [Fixed-center error comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_errors.png); [corresponding movement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_motion.png).

Shared-scale Adam at fixed uniform centers, starting from gamma 1, yields independent-grid refit errors of approximately 2e-14–8e-13 after 10,000 steps at the production cutoff, with modest solved coefficients. Actual GD-readout relative errors remain 0.0225–0.3214. This is successful structured geometry learning, not a machine-precision trained network. [Adaptive scale from gamma 1](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_1.png).

Starting the same method from good gamma-16 geometry can instead worsen mixed-sine/Gaussian refits while improving actual training. [Adaptive scale from gamma 16](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_16.png).

**Verdict:** Baseline immobility is observed and can be changed by rate/optimizer/structure interventions. No practical escape-time lower bound has been established for the tested joint trajectories. Altered optimizers do not refute bounds explicitly restricted to ordinary GD.

## 6. Full mechanism and practical route forward

The proposed chain is: readout removes coarse residual; useful independent scale sensitivity stays weak; geometry remains in a bounded-scale regime; the target needs poorly resolved dictionary directions; stable finite-budget training cannot attain the requested precision.

**We have evidence for several links separately, not for the complete causal chain along one controlled trajectory.**

- Reject the broad statement that geometry motion only helps the readout: D25 fixed-center interventions and D31 amplified profiled-gradient training improve refitted approximation. [D31 Xavier refit sweep](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/figures/xavier.png); [10,000-step cosine run](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/cosine.png).
- J at the current readout and J* at solved coefficients are operationally different. Their matched 10,000-step experiments give different behavior; they must not be treated as interchangeable geometry gradients. [Sine](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD33_current_readout_split/j_vs_jstar/figures/sine.png); [mixed sine](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD33_current_readout_split/j_vs_jstar/figures/sine_mixture.png); [Runge](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD33_current_readout_split/j_vs_jstar/figures/runge.png); [Gaussian](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD33_current_readout_split/j_vs_jstar/figures/gaussian_envelope.png).
- D34 establishes an implemented differentiable approximate-floor penalty without a training-time coefficient solve. It does not establish a general successful replacement for VarPro. The three-method figures compare mu inside one Adam for the new penalty against mu outside two Adam streams for J/J*; they are not an isolated test of projector accuracy. [Spectral approximation diagnostic](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD34_ns_residual_penalty/figures/spectral_response.png); [mixed-sine three-method comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD34_ns_residual_penalty/j_comparison/figures/sine_mixture.png); [Runge comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD34_ns_residual_penalty/j_comparison/figures/runge.png).

**Unestablished:** A universal impossibility result; a verified dominant cause of geometry stalling; a persistent Fourier gap under joint GD; a geometry-specific spectral prediction of observed convergence; or a stable, scalable Xavier-to-machine-precision trained-network recipe across the targets and widths.

## Current accounting

| Proposition | Evidence status |
|---|---|
| The signed residual–scale pairing gives the gradient | Numerically verified for the implemented objective and coordinates |
| Frequency mismatch can suppress instantaneous scale gradients | Demonstrated by controlled probes |
| Low-frequency residual is absent whenever small gamma stalls | Not established; low-band leakage is visible |
| Fourier suppression is the dominant cause of the observed stall | Not established |
| Fixed-readout singular values predict GD convergence | Strong quantitative verification |
| Projected geometry singular values predict our stalled trajectories | Not yet tested directly |
| Early freezing releases useful scale growth | Failed in the tested Xavier branches |
| Geometry motion never improves the available approximation | Contradicted by later interventions |
| Increasing gamma always improves approximation | Contradicted by refit floors and oversharpening controls |
| Standard baseline steps can underexploit useful geometry gradients | Supported by controlled rate interventions |
| A practical stable-GD lower bound has been established | No |
| A general reliable optimizer has been established | No; structured successes and imperfect alternatives exist |
