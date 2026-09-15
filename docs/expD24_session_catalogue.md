# Session catalogue — expD24 through expD33

This follows the session from the initial paper discussion through the matched four-way comparison, the scale-barrier investigation, an explanatory figure collection, the fixed-geometry spectrum sweep, the readout-information/freezing experiments, the loss-gradient decomposition, the approximation-gradient weighting test, extreme-gamma center sampling, the six-weight mu sweep, and separate Adam streams with outside weighting. There are **148 available plots: 138 PNGs and 10 GIFs**, all linked below. Deleted and superseded versions are recorded in their original place in the progression. This is an inventory, not a results writeup.

**Training-domain correction:** Earlier comparisons containing the Gaussian-envelope row used a whole-line objective and a tail-cancelling model for that row, while the other targets used finite-interval training. Those comparisons did not hold the training samples or model constraints fixed across functions. Movement 12 corrects the steps 0–5 comparison; movement 14 extends matched training to four functions and four initializations for 2,000 steps. The earlier whole-line results remain records of their different objective.

**0. Establish the research question**

**Asked:** Read the papers, checkpoint A–C results, and checkpoint D; then work through the gamma-barrier handoff.

**Reviewed:** The Fourier scale-gradient argument, readout/geometry decomposition, VarPro, local GD conditioning, and scale-travel bounds. We also discussed reparameterizations, alternative activations, and the existing PDE solvers. This was background analysis; no new activation, reparameterization, or PDE experiment was run.

**1. Initial GD dry runs — initialization and width**

**Asked:** Compare standard Xavier, center-preserving scaled Xavier, and QI geometry with zero readout, on sine, mixed sine, Runge, and |x|³. Start with loss curves, using widths 64, 128, and 256.

**Ran:** A 1,000-step matrix of 36 cases. The first pass used the older, larger halos; it was followed by a corrected 36-case pass with halo 24 per side. Training/evaluation losses and initial-versus-final readout-refit probes were produced.

**Historical plots:** `train_loss_4x3.png`, `eval_loss_4x3.png`, and `readout_probes_4x3.png`, with early PDF counterparts. These belonged to `dry_run_1000/` and its superseded large-halo version. They were deleted in the later cleanup, so their original figures cannot be linked.

**2. Add Fourier evolution — then refine and extend it**

**Asked:** Animate the residual spectrum in a 4×3 grid, retain live relative L₂ values, reuse the archived animation pacing, and then improve spectral detail and playback duration.

**Ran, in order:**

- A three-width, 500-step training/Fourier pass completed; its animation render was stopped when the request changed.
- An N=128, 1,000-step animation used the saved trajectories and the archived pacing.
- The replacement used N=128, 2,000 steps, denser frequency evaluation, and a 16-second animation. It retained the original four functions and three initializations.

**Available plots:**

- [Baseline GD loss curves — four functions × three initializations](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/loss.png)
- [Baseline residual-spectrum evolution — 2,000-step GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/spectrum.gif)

**Deleted earlier plots:** The 1,000-step version had `residual_spectrum_4x3.gif`, `train_loss_4x3.png`, `spectrum_step_0000.png`, and `spectrum_step_1000.png` under `fourier_N128_1000/figures/`. The two links above are the later replacement experiment, not copies of those deleted files.

**3. Verify the spectrum and locate the remaining error**

**Asked:** Check whether the bumps were an implementation error, whether low frequencies were actually learned first, whether spectral energy agreed with L₂, and whether the frequency window was too short.

**Ran:** Analyses of the saved trajectories: an exact sine-transform comparison, a Fourier-series alternative, frequency-band energy tracking, and a wider-frequency coverage check. A separate loss-only replay of all 12 current baseline cases checked that Fourier diagnostics had not changed training.

**Plots:**

- [Initial sine residual — numerical versus exact windowed transform](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/validation/sine_window_check.png)
- [Continuous spectrum versus discrete Fourier-series modes — sine and mixture](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/validation/fourier_modes_comparison.png)
- [Where mixed-frequency error remains — spatial residuals and band energies over training](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/frequency_learning.png)
- [Spectral coverage — residual energy beyond increasing frequency cutoffs](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/spectral_coverage.png)

The Fourier-series view was explored, but did not replace the continuous-spectrum experiment. The temporary `frequency_learning_audit.png` is byte-identical to the saved band-energy figure linked here.

**4. Isolate boundary, halo, cropping, and smoothing effects**

**Asked:** Demonstrate the source of the bumps, show where halo neurons sit, compare [-1,1] with [-0.8,0.8], and try smoothing only the residual's ends.

**Ran:** Closed-form transform controls; an endpoint-based prediction of a trained spectrum; interior-error checks on the saved QI runs; full-versus-cropped spectra; and local Gaussian end smoothing. These were diagnostic changes, without retraining the original models.

**Plots:**

- [Fourier controls — two flat cutoff residuals and a smooth Gaussian](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/validation/fourier_controls.png)
- [Endpoint prediction — trained QI sine residual and its upper-frequency spectrum](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/validation/endpoint_prediction.png)
- [Halo placement and interior error — all four original functions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/validation/halo_and_interior.png)
- [Full versus interior spectra — 4×3 grid with log spectral axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/interior_spectrum.png)
- [Full versus interior spectra — the same comparison with linear axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/interior_spectrum_linear.png)
- [Gaussian end smoothing — original and modified residuals/spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/end_smoothing_linear.png)

**5. Introduce a residual that decays on the whole line**

**Asked:** Build a cleaner example without the imposed finite-window cutoff, then compare four gamma regimes with more snapshots.

**Ran:** A Gaussian-envelope mixed-sine target and a tanh model whose tails cancel. The initial comparison used gamma 1 and 16; it was expanded to 1, 4, 16, and 64, keeping the same width and initial centers. Training lasted 2,000 steps; the static figure was expanded to ten early-clustered viridis snapshots.

**Plots:**

- [Whole-line gamma ladder — residual, spectrum, and loss at ten snapshots](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/whole_line/overview.png)
- [Whole-line gamma ladder — training animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/whole_line/spectrum.gif)
- [Whole-line gamma ladder — later signed-log residual version](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/log_axes/whole_line.png)

The original two-gamma, three-snapshot layout was superseded at the same output paths; the links show the expanded four-gamma versions.

**6. Separate readout fitting from geometry learning — Xavier versus QI**

**Asked:** Compare Xavier and QI on four functions, showing both spatial residuals and spectra. Add VarPro and evaluation-only readout refits to check whether unsolved readouts were confounding the picture.

**Ran:** Sine, mixed sine, Runge, and Gaussian envelope, each with Xavier and QI initialization. We compared ordinary GD; VarPro, which solves the readout before each geometry step; and ordinary GD evaluated with refitted readouts, without feeding those refits back into training. The Gaussian row retained its whole-line formulation.

The requested presentation progressed from three static 4×4 figures, to GIFs, to additional signed-log residual GIFs with a 10⁻¹⁶ display floor. These are views of the same comparison, not three separate training studies.

| Method | Static viridis PNG | GIF: linear residual, log spectrum | GIF: signed-log residual and lower spectral floor |
|---|---|---|---|
| Ordinary GD | [GD static comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/gd.png) | [GD animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/gd.gif) | [GD signed-log animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/log_axes/gd.gif) |
| VarPro | [VarPro static comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/varpro.png) | [VarPro animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/varpro.gif) | [VarPro signed-log animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/log_axes/varpro.gif) |
| GD with evaluation-only refits | [GD-refit static comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/gd_refit.png) | [GD-refit animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/gd_refit.gif) | [GD-refit signed-log animation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/log_axes/gd_refit.gif) |

**7. Expand the gamma ladder across functions and readout methods**

**Asked:** Repeat the four-function/four-gamma comparison with VarPro and evaluation-only readout solves, using static viridis figures. Also measure average gamma movement over a denser gamma sweep.

**Ran:** Uniform initial centers and zero readouts, with gamma 1, 4, 16, and 64, across the same four functions and three views. Each figure has gamma rows and columns for residual, spectrum, and relative L₂. Ordinary GD additionally covered 13 logarithmically spaced initial gammas. The completed matrix contains 52 GD cases and 16 VarPro cases, reusing identical earlier runs where available.

| Function | Ordinary GD | VarPro | GD with evaluation-only refits |
|---|---|---|---|
| Sine | [Sine — GD gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd/sine.png) | [Sine — VarPro gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/varpro/sine.png) | [Sine — GD-refit gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd_refit/sine.png) |
| Mixed sine | [Mixed sine — GD gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd/sine_mixture.png) | [Mixed sine — VarPro gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/varpro/sine_mixture.png) | [Mixed sine — GD-refit gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd_refit/sine_mixture.png) |
| Runge | [Runge — GD gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd/runge.png) | [Runge — VarPro gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/varpro/runge.png) | [Runge — GD-refit gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd_refit/runge.png) |
| Gaussian envelope | [Gaussian — GD gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd/gaussian_envelope.png) | [Gaussian — VarPro gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/varpro/gaussian_envelope.png) | [Gaussian — GD-refit gamma ladder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gd_refit/gaussian_envelope.png) |

- [Gamma movement across 13 initial scales — mean signed and mean absolute changes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gamma_comparison/gamma_motion.png)

**8. Direct tests of geometry's contribution and scale sensitivity**

**Asked:** Test the interpretation that improved fitting at larger initial gamma was mainly a readout effect, and determine what the small observed gamma movement actually establishes.

**Ran:** Two different controls. First, matched ordinary GD against readout-only GD with geometry frozen, across all 16 function/gamma combinations, tracking both net gamma displacement and total travel. Second, held a unit-L₂ Gaussian-sine residual, center, and unit readout fixed while varying gamma, measuring the instantaneous scale gradient for three carrier frequencies.

**Plots:**

- [Frozen-geometry control — loss curves, geometry benefit, and gamma travel](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/training_control.png)
- [Matched-residual probe — scale-gradient strength versus gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/matched_residual.png)

The matched-residual probe was relabeled from its saved data: the figure now defines the three normalized Gaussian-sine residuals, the exact spatial gradient pairing, and the per-curve normalization in the right panel. Its horizontal axes explicitly vary neuron scale, not training step or residual frequency. No training or integral calculation was repeated for this presentation change.

**Additional requested view:** Replot the control figure's right column with linear–linear and log–log axes side by side, retaining all four functions and gamma regimes. Solid curves are mean absolute net displacement; dashed curves are mean total travel, both over all 177 neurons including halo. Shared limits within each column. The two initial zero-motion states are omitted only on log axes. This uses the original saved runs, including the earlier whole-line Gaussian case; no training was repeated.

- [Gamma movement — linear–linear and log–log views of the same trajectories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/gamma_motion_axes.png)

**9. Compare tangent spectra with residual spectra and measure their pairing**

**Asked:** Plot arithmetic and geometric means of the neuron scale-tangent spectra, and examine their overlap with the residual.

**Ran:** Diagnostics at the saved GD snapshots. Each function's figure shows the residual spectrum, bare tangent means, readout-weighted tangent means, and normalized final overlap. A separate whole-line Gaussian analysis retains phases and measures gradient contributions from frequency bands.

**Plots:**

- [Sine — residual and arithmetic/geometric mean tangent spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/spectra/sine.png)
- [Mixed sine — residual and arithmetic/geometric mean tangent spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/spectra/sine_mixture.png)
- [Runge — residual and arithmetic/geometric mean tangent spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/spectra/runge.png)
- [Gaussian envelope — residual and arithmetic/geometric mean tangent spectra](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/spectra/gaussian_envelope.png)
- [Gaussian frequency pairing — band energy, band gradients, and magnitude-overlap bound](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/frequency_pairing.png)

**10. Sweep gamma while holding individual neuron snapshots fixed**

**Asked:** Show the scale-tangent spectrum across ten gamma choices, for three randomly selected neurons at three training times.

**Ran:** Selected the same three neurons from the whole-line Gaussian GD run initialized at gamma 16, at steps 2, 352, and 2,000. Within each of the nine panels, kept the saved center and readout fixed and swept gamma from 0.5 to 256. The analytic whole-line spectra use viridis, shared log axes, and no amplitude normalization. This is a hypothetical parameter sweep, without retraining.

- [Individual scale-tangent gamma sweep — three neurons × three snapshots, ten gamma curves each](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/tangent_gamma_sweep.png)

**Additional requested view:** Overlay the original saved residual spectrum on the same nine tangent sweeps with linear axes. The residual stays fixed during each hypothetical gamma sweep. Tangents use the left axis with panel-specific absolute limits; residuals use the right axis with common limits. The view focuses on frequencies from 0 to 32, without amplitude normalization.

- [Linear tangent–residual overlay — ten gamma choices against the saved error spectrum](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/tangent_residual_overlay_linear.png)

The overlay also has a [version with logarithmic magnitude axes and linear frequency](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/tangent_residual_overlay_log_y.png), rendered from the same saved spectra with a 10⁻¹⁶ display floor. All nine panels share the same axis limits, and residuals and tangents use a single common magnitude scale without normalization. Both earlier plots are preserved.

**11. Resolve the first updates of GD with evaluation-only readout solves**

**Asked:** Rerun three functions, including the Gaussian envelope, saving every step from 0 through 5. Plot residuals in a 2×3 grid with Xavier above and QI below, using viridis for the six steps, and check whether the unusual Gaussian step-zero curve came from a missing readout solve.

**Ran:** Sine, mixed sine, and Gaussian envelope, preserving the initializations, learning rate, and finite-interval/whole-line objectives of movement 6. Fresh GD trajectories matched the earlier saved states at steps 0, 2, and 4. Every residual was evaluated after a separate readout solve, including step 0. The figure uses shared signed-log residual limits with a 10⁻¹⁶ floor. Saved diagnostics include rank, slope changes, and an independent spatial check of the initial Gaussian/Xavier gradient and its contribution from outside [−2,2].

- [Earlier first-five-update comparison — different training domains, Xavier/QI rows](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/first_steps.png)

**12. Match training samples and models across all three functions**

**Asked:** Correct the comparison so Gaussian, sine, and mixed sine all train on the same samples.

**Ran:** Fresh ordinary GD runs for all six target/initialization combinations, recording every step from 0 through 5. All use the same 1,024 midpoints in [−1,1], half-mean-square loss, rate 0.002, standard tanh MLP with an unconstrained readout and output bias, and identical initial parameter arrays within each initialization arm. Each displayed residual uses a separate readout solve on those same training samples, including at step 0; relative L₂ is evaluated on 32,768 independent midpoints in [−1,1]. The six-panel figure has identical signed-log residual and spatial limits throughout. Every GD update was checked against an independent NumPy gradient calculation, and QI step-zero/step-one geometry and refitted residuals agree exactly. The previous figure and data are preserved with their different training domains labeled.

- [Matched first five GD updates — identical finite-interval samples and model across all functions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/readout_comparison/first_steps_matched.png)

**13. Correlate per-neuron spatial magnitude bounds with scale gradients**

**Asked:** A 4×3 scatter comparison for four functions and three initialization regimes, with 128 neurons at each of 500 steps, viridis step colors, best-fit lines and R². Sam clarified that the vertical axis should be gradient magnitude.

**Ran:** Sine, mixed sine, Runge and Gaussian envelope, using Xavier, center-preserving scaled Xavier and QI with zero readout. Every run uses the same 1,024 midpoint training samples in [−1,1], standard tanh model, half-mean-square loss and GD rate 0.002. Each panel contains 64,000 points from pre-update states 0–499. The original 177-neuron model is retained; 128 fixed reference-interior slots are displayed, excluding both halo blocks and the extra right-endpoint slot. Xavier centers remain random rather than being filtered by current position.

**Measured:** Horizontal values are `(1/2) integral |e(x)| |phi_k(x)| dx` on [−1,1], evaluated on 4,096 independent midpoint nodes; the factor 1/2 matches the training loss density. The tangent includes the actual readout and holds center and neuron orientation fixed. Vertical values are the magnitudes of the corresponding centered-scale derivatives calculated from automatic differentiation, not raw-slope SGD updates. Signed spatial pairings were checked against differentiation at every state, with separate direct centered-coordinate differentiation and doubled quadrature at selected states. Fits use all original linear values and a free intercept. A logarithmic display with a linear region around zero exposes the many small observations without dropping points; an earlier signed-gradient view is retained as a companion.

**Notation audit against the note:** The figures now write the same horizontal quantity as `(|c_k|/2) integral |e(x)| |psi_{gamma_k,z_k}(x)| dx`, with `psi = (x-z) sech²(gamma(x-z))`, to expose the readout coefficient previously absorbed into `phi`. This is the spatial absolute-overlap bound derived from (2.1), adapted to the finite uniform objective. It is neither the signed integral identity nor the exponential Fourier bound (2.5). Plot labels were clarified without changing data, fits, or training; the independent coefficient, normalization, and cancellation checks pass.

- [Magnitude correlation — logarithmic display, all points and fits in original units](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gradient_correlation/magnitude_log.png)
- [Magnitude correlation — linear axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gradient_correlation/magnitude.png)
- [Signed-gradient correlation — companion view](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gradient_correlation/signed.png)

**Corrected requested comparison — absolute value outside the integral:** The user clarified that the x-axis should be `abs((c_k/2) integral e(x) psi_{gamma_k,z_k}(x) dx)`, retaining cancellation. The additional linear–linear figure compares this signed pairing's magnitude against the automatically differentiated centered-scale gradient magnitude. It reuses all 500 × 128 saved observations per panel and the existing 4,096-point signed integrals, with the current coefficients included. No training was repeated. All 12 R² values exceed 0.999999999; slopes differ from one by less than 2e-6. The largest panel-level relative L₂ discrepancy is 2.73e-5, comparing dense integration with the 1,024-sample gradient. Using the training samples for the signed pairing agrees to floating-point precision. This checks the identity (2.1), adapted to the finite uniform objective; earlier bound plots are preserved as different diagnostics.

- [Exact scale pairing — absolute value outside the signed integral, linear axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/gradient_correlation/exact_pairing.png)

**14. Compare free and frozen geometry, with ordinary and solved-readout evaluation**

**Asked:** Put four curves together for Xavier and uniform gamma 1, 4, and 16 (QI): free geometry with GD; frozen initial geometry with readout-only GD; the same free-geometry GD trajectory evaluated with solved readouts; and the fixed initial geometry evaluated with solved readouts.

**Ran:** Fresh paired trajectories for sine, mixed sine, Runge, and Gaussian envelope, all using the same standard tanh model with output bias, 1,024 midpoint training samples in [−1,1], half-mean-square loss, rate 0.002, and 2,000 steps. The N=128 construction has 177 neurons, including halo 24 per side. Each free/frozen pair has identical initial parameters; Xavier keeps its random readout, while uniform-gamma runs start with zero readout. Free geometry allows both raw slopes and biases to move. All four error curves use the same independent 32,768 midpoint evaluation grid in [−1,1], including Gaussian.

**Evaluation:** Saved 101 shared states: every step 0–20 followed by 80 logarithmically spaced states through 2,000. Every displayed free-geometry state, including step 0, receives a separate readout solve on the training samples using the existing SVD cutoff 1e-13. All solves occur after the ordinary training trajectories are complete. Curve 4 repeats the initial readout solve and is constant. Two independent NumPy-gradient checks passed, frozen readout gradients agree with direct residual gradients within 1.1e-15 across the full runs, and the six previously matched trajectories agree through step 5. Data and configuration are together in one compressed file under this comparison's `data/` folder.

**Plots:**

- [Four-way comparison — four functions × Xavier and gamma 1, 4, 16; shared logarithmic error axes](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/comparison.png)
- [Geometry benefit — ordinary-GD and least-squares error reductions in the same units](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/four_way_comparison/gains.png)

The companion subtracts curve 1 from curve 2 and curve 3 from curve 4. Both reductions use the original relative L₂ units, without dividing by their different baselines; positive values mean improvement. Each panel has its own vertical range. This avoids inflating numerical fluctuations when the initial least-squares error is tiny. The main figure retains all least-squares errors, including those near numerical precision.

**15. Identify a useful scale-learning escape and its limitations**

**Asked:** Continue until there are serious results explaining the small gamma movement and a tested way around it; change one factor at a time and distinguish poor geometry from an unfinished readout fit.

**Ran:** A separate expD25 investigation, preserving all earlier outputs. All four functions now use the matched finite-domain model, samples, and loss. A saved-state scale landscape motivated a geometry-only learning-rate sweep (48 trajectories), followed by fixed-center controls (16 new), independent-versus-shared scale controls (16 new), continuation to 10,000 steps (8), and shared-scale Adam with ordinary readout GD (12 new). There are 100 unique trajectories; the pilot and matched controls reuse cases instead of counting them twice. Readout solves occur only after training, including for step zero. The adaptive control tests both a poor initial scale and an already accurate scale. Offline derivative, curvature, coefficient, SVD-cutoff, and evaluation-grid audits accompany the intervention plots.

**What this adds:** Geometry-rate changes can improve the refitted approximation with centers fixed. Shared-scale Adam from gamma 1 yields excellent refit accuracy and modest coefficients, while its gamma-16 control shows that continued scale motion can harm approximation even as ordinary fitting improves. These are structured 1D controls, not a general optimizer result. The [complete expD25 writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/expD25_results.md) separates observations, interpretation, and unresolved questions.

**Plots:**

- [Scale landscape — current/readout-refit errors, coefficients, and rank](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/landscape.png)
- [Initial mixed-sine rate pilot — current and refitted errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/pilot/errors.png)
- [Initial mixed-sine rate pilot — gamma displacement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/pilot/motion.png)
- [Geometry learning-rate sweep — four functions and four initializations](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/rate_errors.png)
- [Geometry learning-rate sweep — mean absolute gamma displacement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/rate_motion.png)
- [Freeze centers — ordinary and refitted errors at two scale learning rates](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_errors.png)
- [Freeze centers — gamma displacement at the same two rates](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_motion.png)
- [Independent versus shared scales — ordinary and refitted errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/common_scale_errors.png)
- [Independent versus shared scales — mean gamma and scale spread](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/common_scale_scales.png)
- [Baseline diagnostics — block curvature, gradient sizes, and out-of-span contribution](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/derivatives.png)
- [Shared-scale GD to 10,000 steps — gamma, approximation component, and readout gap](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/continuation.png)
- [Numerical readout audit — five cutoffs, two SVD algorithms, and denser evaluation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/readout_sensitivity.png)
- [Focused escape comparison — independent-scale rate change followed by tying scales](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_escape.png)
- [Shared-scale Adam versus SGD from gamma 1 — scale, refitted accuracy, and actual error](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_1.png)
- [Shared-scale Adam versus SGD from gamma 16 — preservation and oversharpening control](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_16.png)

**16. Assemble five figures explaining the Fourier and readout observations**

**Asked:** Improve and assemble the five most informative figures supporting the developing account, with distinct purposes, an animation showing frequency learning, and readable axes for the scale-sensitivity and frequency-pairing diagnostics.

**Ran:** Rebuilt five figures from saved trajectories and probes, without repeating training or readout solves. The 32-second GIF combines the analytic whole-line spectrum with absolute frequency-band error curves. Four PNGs separate fixed-residual sensitivity, frozen-geometry readout fitting, geometry movement and error reductions, and phase-sensitive band gradients. An [offline gallery with captions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/index.html) assembles them in that order. Earlier figures remain available.

**Presentation and interpretation:** The gallery distinguishes whole-line examples from matched finite-interval comparisons. Both geometry-benefit columns use identical signed-log axes, exposing early Xavier gains and small nonzero refit changes of either sign. This avoids claiming that all gains are marginal or that refitted accuracy is exactly constant. It retains the note's conditional assumptions and distinguishes centered-scale derivatives from raw-slope GD updates.

**Checks:** Animation and pairing plots use identical parameter states at shared snapshots. Independent whole-line quadrature agrees with the animation's normalized band energies within 4.7e-11 at the checked states. The band-gradient sum and magnitude bound were checked; all 77 GIF frames decode and total 32 seconds. A small data folder contains derived band energies and source hashes.

**Plots:**

- [1. Frequency learning — analytic residual spectrum and absolute band-error evolution, 32-second GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/01_frequency_learning.gif)
- [2. Scale sensitivity — tangent spectra, fixed-residual gradients, and normalized preferred-scale curves](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/02_scale_sensitivity.png)
- [3. Readout fitting versus approximation — frozen-geometry GD and initial least-squares errors across gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/03_readout_vs_approximation.png)
- [4. Geometry motion and benefit — gamma displacement, current-GD error reduction, and refitted error reduction](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/04_geometry_motion_and_benefit.png)
- [5. Frequency-gradient pairing — remaining error fractions, signed band contributions, and the magnitude bound](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/story_figures/05_frequency_gradient_pairing.png)

**17. Freeze the readout early; measure fixed-geometry GD convergence across lambda**

**Asked:** From Xavier, branch after 2, 10, 50, and 150 joint-GD updates, freeze the readout, and continue geometry GD for 500 updates at the unchanged rate. Show actual error and mean absolute gamma in a 2×4 plot for each of the four functions. Separately, increase uniform lambda linearly from 0.1 to 2 and animate the readout singular spectrum with predicted convergence times.

**Ran:** Four joint trajectories and 16 frozen-readout branches, including an exactly matched continued-joint control for every branch and separate offline refit diagnostics. All four targets use the same finite interval, raw tanh model, samples, and loss. The fixed-center spectrum sweep uses 77 lambda values, a zero initial readout, and target-weighted SVD predictions at the common rate 0.002 and a documented spectral reference rate. No coefficient solves enter training. The [expD26 writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/expD26_results.md) records the design, results, numerical cutoff, and remaining mechanism questions.

**Checks:** Seven focused tests pass. Branch prefixes match exactly and every readout coefficient, including bias, stays bitwise fixed. Explicit readout GD through 2,000 steps agrees with modal predictions within 3.44e-15 across the representative lambda/rate checks. The 77-frame GIF plays for 24 seconds at uniform lambda progression.

**Plots:**

- [Sine — early readout freezing, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine.png)
- [Mixed sine — early readout freezing, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine_mixture.png)
- [Runge — early readout freezing, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_runge.png)
- [Gaussian envelope — early readout freezing, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_gaussian_envelope.png)
- [Readout-refitted geometry — four functions and four freeze times](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_refit.png)
- [Readout spectrum versus lambda — singular values and per-direction learning times, GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_spectrum.gif)
- [Numerical approximation floors and predicted GD steps versus lambda](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_convergence.png)

**18. Compare the information supplied by four readout-freezing variants**

**Asked:** Repeat the freezing test with readout solves before freezing; a solved QI readout transferred to poor geometry and fixed throughout; clean QI geometry with a zero readout; and noisy QI geometry with a zero readout. Sam confirmed the same four targets, freeze times 2/10/50/150 with 500 further steps, gamma-1 reset at unchanged QI centers, and 10% multiplicative scale noise preserving centers initially. Additional proposed variations require approval and were not run.

**Ran:** expD27 contains 12 continued references, 48 frozen-readout branches, and four permanently frozen QI-readout transfers. All use the matched finite interval, constant GD rate 0.002, and the previous width/sample conventions. Every saved geometry also receives sparse evaluation-only readout refits. Variant 2 has one 2×4 figure with function columns because its readout is fixed from step zero. The other variants each have four primary 2×4 figures and a refitted-geometry companion. The [writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/expD27_results.md) explains the large-coefficient numerical sensitivity exposed by variant 1 and distinguishes actual fit improvement from recovery of the QI approximation regime.

**Checks:** Five focused tests pass. Saved data verify all exact branch prefixes, identical first post-freeze geometry updates, frozen coefficients including bias, and zero-readout first geometry steps. The strongest numerical caveat is recorded: independent full NumPy versus PyTorch reevaluation changes some large-coefficient first updates noticeably, while the analytic update formula agrees using identical forward values. No learning-rate retuning, clipping, or damping was introduced.

**Plots:**

- [1. Solve before freezing — Sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/sine.png)
- [1. Solve before freezing — Mixed sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/sine_mixture.png)
- [1. Solve before freezing — Runge, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/runge.png)
- [1. Solve before freezing — Gaussian envelope, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/gaussian_envelope.png)
- [1. Solve before freezing — refitted geometry across functions and freeze times](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/refitted_geometry.png)
- [2. Transfer the solved QI readout to gamma 1 — all four targets, fixed readout throughout](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_transfer/all_targets.png)
- [3. Clean QI with zero readout — Sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/sine.png)
- [3. Clean QI with zero readout — Mixed sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/sine_mixture.png)
- [3. Clean QI with zero readout — Runge, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/runge.png)
- [3. Clean QI with zero readout — Gaussian envelope, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/gaussian_envelope.png)
- [3. Clean QI with zero readout — refitted geometry across functions and freeze times](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/refitted_geometry.png)
- [4. Noisy QI with zero readout — Sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/sine.png)
- [4. Noisy QI with zero readout — Mixed sine, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/sine_mixture.png)
- [4. Noisy QI with zero readout — Runge, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/runge.png)
- [4. Noisy QI with zero readout — Gaussian envelope, actual error and mean gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/gaussian_envelope.png)
- [4. Noisy QI with zero readout — refitted geometry across functions and freeze times](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/refitted_geometry.png)

**19. Decompose the loss and its geometry gradients along ordinary GD**

**Asked:** Four function columns and three rows showing loss, the norms of DL/DF/DG, and cosine(DF,DG). Produce separate figures for Xavier, center-preserving scaled Xavier in the lambda-0.25 regime, and uniform QI gamma 16 with zero readout.

**Ran:** Twelve matched, ordinary-GD trajectories for 2,000 updates at rate 0.002, with all four functions trained on the same finite interval. At 150 saved states per run, compute the loss decomposition and derivatives in all raw slopes and biases. Readout solves are strictly offline. The figures differentiate the existing SVD-truncated numerical floor F_tau, including the motion of the retained singular subspace, and define G_tau=L-F_tau. They distinguish this numerical reference from unrestricted exact-arithmetic VarPro.

**Checks and interpretation:** Five focused tests pass, including independent gradient checks and proof that offline refits cannot enter training. Alternative SVD algorithms and tanh rounding screen unreliable gradient directions; those cosines are omitted. Scaled Xavier shows a strong magnitude imbalance between DF_tau and DG_tau. Ordinary Xavier mixed sine develops opposition, whereas Runge develops alignment. QI starts near the numerical approximation floor with unresolved DF_tau directions. The [expD28 writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/expD28_results.md) explains the axes, derivative, numerical limits, and saved-state finite-difference checks.

**Plots:**

- [Xavier — loss decomposition, geometry-gradient norms, and cosine similarity](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/xavier.png)
- [Scaled Xavier — loss decomposition, geometry-gradient norms, and cosine similarity](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/scaled_xavier.png)
- [QI with zero readout — loss decomposition, geometry-gradient norms, and cosine similarity](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/qi_zero.png)

**20. Weight the approximation gradient by 1,000 during training**

**Asked:** Test geometry updates proportional to 1000 DF + DG, with the readout continuing ordinary GD, to see whether amplifying the approximation gradient improves geometry.

**Ran:** expD29 compares weights 1 and 1000 over 500 updates for the same four targets and three initializations, giving 24 trajectories. Geometry uses the exact local derivative of the numerical F_tau already defined in expD28; the added objective term is 999 F_tau. Readout coefficients are never replaced by their solved values. Three figures show actual loss, numerical approximation loss, and change in mean gamma, with per-panel vertical ranges and matched F evaluation steps.

**Checks and limits:** Three focused tests pass; all 12 ordinary-GD trajectories reproduce expD28 through step 500. The largest refit gain is Xavier Runge, confirmed on an independent grid, but its magnitude depends on a newly retained nearly singular direction and much larger solved coefficients. Numerical rank jumps and gradient sensitivity qualify interpretation. The [expD29 writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/expD29_results.md) records the results and terminal cutoff audit.

**Plots:**

- [Xavier — ordinary GD versus 1,000× approximation-gradient weighting](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/xavier.png)
- [Scaled Xavier — ordinary GD versus 1,000× approximation-gradient weighting](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/scaled_xavier.png)
- [QI — ordinary GD versus 1,000× approximation-gradient weighting](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/qi_zero.png)

**21. Extreme gamma and center-only samples: fixed versus joint GD**

**Asked:** Set gamma to 128, initialize all readouts to zero, train only on exact centers, and compare learning-rate schedules. Sam confirmed both fixed geometry and joint geometry/readout training on the same four targets.

**Ran:** 24 target/policy/schedule trajectories, each for 200,000 actual GD updates. All start at uniform centers, with 129 in-domain training points and 24 halo neurons per side. Compare rate 0.002, rate 0.01971069 from the initial readout curvature, and warmup/cosine decay. Sparse evaluation uses 8,192 independent midpoint samples. Separate SVD calculations establish interpolation feasibility and predict the longer fixed-geometry horizon. The [writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/expD30_results.md) distinguishes executed steps from predictions and training accuracy from between-center accuracy.

**Checks:** Four focused tests pass. All trajectories remain finite; full fixed-GD traces match spectral predictions to about 3e-14 in relative error. No coefficient solves enter training. The larger constant rate wins these endpoint comparisons; joint training helps mixed-sine training most, while high-precision fixed-GD convergence still needs hundreds of millions of predicted updates.

**Plots:**

- [Actual training losses — four targets, fixed versus joint geometry, three learning-rate choices](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/training_convergence.png)
- [Center-sampled readout spectrum, weakest directions, predicted GD convergence, and schedules](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/spectrum_and_schedules.png)
- [Final residuals and training-center versus between-center accuracy](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/between_centers.png)

**22. Compare six approximation-gradient weights**

**Asked:** Extend the weighted-gradient test beyond mu=1000 with multiple sizes of mu.

**Ran:** mu=1, 10, 100, 1000, 10000, 100000 with the same 500 updates, rate 0.002, four functions, three initializations, model, samples, and seed. The previous 24 cases are reused and 48 new cases are added. Three viridis figures overlay mu while showing actual loss, numerical refitted loss, and change in mean gamma. The original two-curve figures remain available. New data and figures are neatly contained in the existing experiment's mu_sweep folder; the existing [expD29 writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/expD29_results.md) now includes the sweep.

**Checks and result:** All 72 cases complete. Six focused tests pass, matched initial states are verified, and independent-grid endpoint audits use three cutoffs. At mu=100000, all four Xavier refits improve, with the benefit persisting across those cutoffs; actual GD loss improves on sine/Gaussian but worsens on mixed sine/Runge. Larger weights can therefore reveal approximation improvement that the mu=1000 pilot understated, while numerical sensitivity, irregular trajectories, and large solved coefficients remain material limitations.

**Plots:**

- [Xavier — six-weight mu sweep, losses and gamma movement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/xavier.png)
- [Scaled Xavier — six-weight mu sweep, losses and gamma movement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/scaled_xavier.png)
- [QI — six-weight mu sweep, losses and gamma movement](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/qi_zero.png)

**23. Normalize DF and DG separately with Adam, then apply the multiplier**

**Asked:** Repeat the weighting test using Adam for both gradient components, with mu=1000, 10000, 100000 outside normalization.

**Ran:** expD31 runs 36 split-Adam trajectories and 12 ordinary-Adam controls for 500 updates at the same base rate, four functions, three initializations, and matched samples. DF and DG have independent moment histories; mu scales the DF direction afterward. The trained readout uses ordinary Adam and is never replaced by solved coefficients. The three trajectory figures reuse matching weighted-GD traces as dashed controls. The [writeup](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/expD31_results.md) records the exact update and its numerical limits.

**Checks and result:** Seven new Adam checks and five inherited profile-gradient checks pass. All 48 runs finish. Xavier/mu=1000 improves actual fitting on all four targets and final refitted approximation on three; Runge is the strongest case. Larger weights often damage geometry. Independent-grid checks confirm excellent transient refits that are later lost, and expose a large-scale Runge refit failure between training samples. Initial Xavier directions are numerically sensitive; this is not an established stable or scalable optimizer.

**Plots:**

- [Xavier — split Adam, matched weighted GD, ordinary Adam; losses and gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/xavier.png)
- [Scaled Xavier — split Adam, matched weighted GD, ordinary Adam; losses and gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/scaled_xavier.png)
- [QI — split Adam, matched weighted GD, ordinary Adam; losses and gamma](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/qi_zero.png)
- [Independent-grid verification — actual and refitted errors across all initializations and multipliers](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/independent_evaluation.png)

**24. Small check of readout-size and gamma trends**

**Asked and ran:** Check whether the early shrinkage follows the direction of readout-size changes, then try one fitted nonlinear model. Existing Xavier data support a small-slope explanation. One new Runge model at uniform gamma 16 reaches 1% error, then branches into joint and frozen-readout GD for 5000 steps. In this case readout size was growing, and frozen mean gamma grows slightly more than the joint control; both changes are tiny. This is a hypothesis check, not a sweep or general mechanism proof.

- [Primary diagnostic — nonlinear Runge, error, mean readout size and mean gamma before/after freezing](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/late_freeze_runge.png)
- [Supporting saved-data check — linear trend and predicted versus observed scale drift](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/linear_drift.png)

**25. Correlate paired geometry and readout nudges**

**Asked and ran:** A small comparison of early Xavier/Runge and fitted nonlinear Runge. Two one-step frozen-block probes from each of 200 shared joint-GD states generate 128 neuron pairs per state. Plot signed changes of magnitudes, rescaled by the opposite coefficient as requested, with linear axes and viridis time colors. Correlation is weak early and moderate later; this checks instantaneous alignment rather than subsequent redistribution during prolonged freezing.

- [Paired nudge correlation — early and fitted models, 25,600 points per panel](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation.png)

**Extension:** Repeat both windows for 5,000 steps (640,000 pairs per full panel), retaining the original figure. Add late-window zooms and a time-evolution figure to distinguish pooled correlation from changes over training. The early nudges contract toward a poor-fit plateau; the fitted model keeps improving with much weaker geometry than readout nudges. A small-slope expansion explains the early curved tracks but fails once tiny nonlinear forces become dominant.

- [Extended paired-nudge scatter — all 5,000 steps and last-1,000-step zooms](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000.png)
- [Time evolution — fit, correlation, growth/shrinkage agreement, and nudge strength](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000_evolution.png)

**Raw-coordinate follow-up:** Remove the coefficient multipliers from the same 5,000-step pairs, retaining signed growth/shrinkage, linear axes, viridis time colors, and the late-window zooms. The inward contraction persists in the actual deltas. No training repeated.

- [Raw geometry/readout deltas — all 5,000 steps and late-window zooms](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000_raw.png)

**Dynamical follow-up:** Derive the dynamics in update coordinates. An initial two-mode Jacobian model predicts early offset/slope decay rates and fails for late nonlinear nudges. An independent fixed-geometry readout-GD prediction reproduces the fitted model's full 5,000-step nudge pattern to about 0.03% relative discrepancy per coordinate, using only the initial state. No decay rates are fitted.

- [Observed phase pattern versus prediction from fixed geometry and readout GD](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_flow_prediction.png)

**Joint-GD clarification:** The nudge scatter's base trajectory already trained every parameter; frozen labels referred only to one-step probes, not sustained frozen branches. New views use actual successive-state differences and show parameter/error histories. All 4,999 overlapping updates agree with the old probes bitwise. These scatters cannot establish compensation after prolonged freezing.

- [Nothing frozen — actual simultaneous gamma/readout updates](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/joint_gd_raw_updates.png)
- [Nothing frozen — error, mean gamma, and mean readout over time](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/joint_gd_parameter_evolution.png)

**26. Amplify the projected signal at the current readout**

**Asked and ran:** Repeat expD31's Xavier setup using the current-readout signal $J^T(I-P_\tau)r$ instead of the VarPro derivative, with outside multipliers 100, 500, 1000, 5000, and 25000. Twenty 500-step runs, independent-grid refit checks, and matched ordinary-Adam/VarPro references. Sine, mixed sine, and Gaussian nearly follow ordinary Adam; Runge differs modestly without recovering the VarPro approximation improvement.

- [Current-readout split — actual loss, refit squared loss, and expanded linear gamma axes](../results/checkpoint_D_optimizers/expD33_current_readout_split/figures/xavier.png)

**27. Tune smaller multipliers in the original VarPro split**

**Asked and ran:** Xavier only, original expD31 numerical VarPro gradient, unchanged learning rate, and $\mu=100,500,1000,2000$. Sixteen 500-step runs, with every geometry refitted and evaluated on an independent grid. The four repeated 1000 runs match the old trajectories bitwise. Smaller multipliers improve the final approximation, particularly Runge/500, but none reaches the QI numerical floor. Best geometries are separately checked at three cutoffs and on a denser grid.

- [Original VarPro split — 500 steps, explicit relative-L2 refits, linear mean gamma](../results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/figures/xavier.png)
- [Original VarPro split — first 30 steps, with every early refit visible](../results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/figures/xavier_early.png)

**28. Extend the current-readout projected split to 10,000 steps**

**Asked and ran:** Repeat all four functions and the five current-J multipliers for 10,000 steps, keeping the learning rate, Adam epsilon, and outside scaling unchanged. Add four ordinary-Adam controls of the same duration. All 24 trajectories reproduce their original first 500 steps. Explicit refits and denser-grid checks show later separation: weight 100 improves final geometry relative to ordinary Adam, while larger multipliers often drive greater gamma movement and worse approximation. Ordinary Adam still gives the lowest final trained-model error on each function.

- [10,000-step current-J split — trained loss, refitted relative L2, and linear mean gamma](../results/checkpoint_D_optimizers/expD33_current_readout_split/long_run/figures/training.png)
- [Post-Adam update-strength ratio — when the amplified projected stream becomes significant](../results/checkpoint_D_optimizers/expD33_current_readout_split/long_run/figures/stream_strength.png)
- [Same gamma histories on shared logarithmic axes](../results/checkpoint_D_optimizers/expD33_current_readout_split/long_run/figures/gamma_log.png)

**29. Extend the original VarPro split to 10,000 steps with two learning-rate versions**

**Asked and ran:** Xavier initialization, all four functions, and outside multipliers $50,100,250,500$. One version keeps the common learning rate at $0.002$; the other uses cosine decay from the start to $0.000002$ over 10,000 steps. The same rate applies to both geometry streams and the readout, with continuous Adam moments. All 32 split runs and eight matched ordinary-Adam controls reach step 10,000; four constant controls reuse completed data. Middle rows show explicitly reconstructed relative $L_2$ after least-squares refitting, with independent-grid checks. Corresponding axes match across the two figures, and gamma axes are linear. No new interpretation is added.

- [Original VarPro split, 10,000 steps — constant learning rate](../results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/constant.png)
- [Original VarPro split, 10,000 steps — cosine learning-rate decay](../results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/cosine.png)

**Saved-data balance follow-up:** Plot $F_\tau/G_\tau$ and the post-Adam, post-multiplier geometry-update norm ratio for the same trajectories. Four function columns, viridis multipliers, matching axes across schedules, and a reference at ratio one. No training is repeated.

- [Constant-rate balance — scalar losses and the two update contributions](../results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/balance_constant.png)
- [Cosine-decay balance — scalar losses and the two update contributions](../results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/balance_cosine.png)

**30. Prescribe the post-Adam F/G update ratio dynamically**

**Asked and ran:** Replace fixed multipliers with $\mu_t=r\|u_G\|_2/\|u_F\|_2$, for $r=0.01,0.1,1,10,100$. All 40 Xavier trajectories finish 10,000 steps across the four functions and both common-rate policies. Adam histories and readout updates remain unchanged. Record actual loss, refitted relative error, mean gamma, effective multipliers, and achieved ratios. The two figures retain the preceding layout and matched axes. Seventeen implementation tests pass; every recorded applied-step ratio matches its target within floating-point tolerance.

- [Dynamic ratio — constant learning rate](../results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/figures/constant.png)
- [Dynamic ratio — cosine learning-rate decay](../results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/figures/cosine.png)

**Catalogue preservation notes**


The two early diagnostic figures in movement 3 were recovered from temporary storage. The three static comparison PNGs in movement 6 were regenerated from saved trajectories for this catalogue; they are not preserved original image files. The original inventory preparation did not rerun training; movement 15 records the later, separately requested investigation. Extracted animation preview frames repeat their parent plots and are represented by the animation links. Earlier deleted outputs are named above rather than given broken links.
