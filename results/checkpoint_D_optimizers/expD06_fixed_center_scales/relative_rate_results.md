# Changing the relative readout and geometry rates

Does slowing geometry let the readout continue learning, or does it remove a useful route for reducing the residual? **None of the three late rate interventions improves the mean training MSE in any of the 12 width/target/seed cases at 1.1 million updates.** Raising the readout rate often increases occasional error excursions while barely changing the median error. Primary training has completed its budgeted phase; frozen-dictionary solvers, early handoffs, and the feedback policy are still running or queued. Two seeds establish paired observations, not a population-level significance claim.

**Table 1. Terminology used throughout this study.**

| Term or symbol | Meaning |
|---|---|
| Physical readouts $c=(b,w)$ | Output bias and coefficients multiplying the tanh features. |
| Physical slopes $\gamma$ | Slopes attached to the fixed physical centers. |
| Bandwidth $\lambda=h\gamma$ | Signed dimensionless slope; reported localization quantiles use $\lvert\lambda\rvert$. Here $h=2/N$. |
| Scaled training | Train $a,\lambda$ with $c=Da$ and $\gamma=\lambda/h$. $D$ is fixed at the theoretical reference bandwidth $0.25$. |
| Shared rate | The same scalar rate for $a$ and $\lambda$. The prescribed physical scales still differ. |
| Slower geometry | Change only the scalar geometry rate to one tenth of the shared control. |
| Faster readout | Change only the scalar readout rate to ten times the shared control. |
| Both changes | Apply slower geometry and faster readout together. |
| MSE | Mean squared residual. Training differentiates half-MSE; every error in this report is MSE. |
| Detached refit | SVD least-squares diagnostic at a saved geometry. It never replaces the trained readout. |
| $\tau$ | Relative singular-value cutoff used in a detached refit. |
| DFT band | A band of discrete Fourier indices on the actual training grid, including both frequency signs. These describe the sampled residual; finite-window leakage can mix frequencies. |

## The comparison holds the theory scales fixed

Every joint-training branch uses the same fixed coordinate map $c=Da$, $\gamma=\lambda/h$, fixed centers, square-root halo radius, and FP64 arithmetic. The two acquisition histories use shared scalar rates $10^{-3}$ or $10^{-2}$ through 80k updates, followed by an 80k cosine decay to $10^{-6}$. Initialization is physical Xavier on slopes and the legacy signed reference-envelope draw on readouts. The latter is $w_j=\alpha_j\operatorname{sign}(\xi_j)$ before absorbing the initial slope signs into the corresponding readouts; it is **not** Gaussian Xavier on $a$.

The model is $\hat f(x)=b+\sum_jw_j\tanh(\gamma_j(x-z_j))$ on $[-1,1]$. Here $N$ denotes core grid resolution: with $R=\lceil\sqrt N\rceil$ halo centers on each side, the actual hidden-unit count is $N+1+2R$, or 559 and 1089. Targets are $\sqrt2\sin(2\pi x)$, $\sqrt5x^2$, and $[\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}$. Each update uses the entire fixed grid of $16N+1$ points, including endpoints. There is no minibatch noise. Validation uses 32,768 midpoint samples. Seeds change initialization, not the sampling grids.

At update 160k, each higher-acquisition-rate checkpoint is copied with its Adam moments intact. A 20k cosine transition reaches the following rates. The lower-acquisition-rate history supplies one additional shared-rate control. There are two widths ($N=512,1024$), three targets (sine, quadratic, mixed), and two seeds, giving 60 primary continuations. We compare complete 20k training windows, independently of the separately recorded validation endpoints. No held-out test evaluation or validation-driven schedule selection occurs.

**Table 2. Scalar rates after update 180k. All rows retain the same theoretical coordinate map.**

| Intervention | Readout $\eta_a$ | Geometry $\eta_\lambda$ | Compared with shared rate |
|---|---:|---:|---|
| Shared rate | $10^{-6}$ | $10^{-6}$ | Control |
| Slower geometry | $10^{-6}$ | $10^{-7}$ | Geometry divided by 10 |
| Faster readout | $10^{-5}$ | $10^{-6}$ | Readout multiplied by 10 |
| Both changes | $10^{-5}$ | $10^{-7}$ | Both scalar changes |

For Adam, the physical rate factors are $\eta_aD_j$ for each readout and $\eta_\lambda/h$ for each slope. These factors multiply normalized moment directions; they are not the actual parameter displacements. With the corresponding physical moments, epsilon becomes $\epsilon/D_j$ and $\epsilon h$, respectively. For GD the induced factors would instead be $\eta_aD_j^2$ and $\eta_\lambda/h^2$. This campaign changes the two scalar rates explicitly; it neither changes $D$ nor applies the length-scale factors twice.

**Table 3. Physical Adam rate factors in the shared $10^{-6}$ control. Multiply the readout columns by 10 for faster readout, or divide the slope column by 10 for slower geometry. Corrected halo slots have their own fixed scales.**

| $N$ | Ordinary readout | Corrected-halo readout range | Bias | Slope |
|---:|---:|---:|---:|---:|
| 512 | $9.31\times10^{-8}$ | $9.31\times10^{-8}$–$1.02\times10^{-6}$ | $3.28\times10^{-6}$ | $2.56\times10^{-4}$ |
| 1024 | $6.41\times10^{-8}$ | $6.41\times10^{-8}$–$7.28\times10^{-7}$ | $2.82\times10^{-6}$ | $5.12\times10^{-4}$ |

## What the matched primary comparison establishes

**Table 4. Mean training MSE over updates 1.08m–1.10m. Ranges span the two seeds. Intervention columns divide each seed's MSE by its own shared-rate control, then report the range; values above one are worse. All 60 primary branches reached this common horizon.**

| $N$, target | Shared-rate MSE | Slower geometry / shared | Faster readout / shared | Both / shared |
|---|---:|---:|---:|---:|
| 512, sine | $[3.56,5.65]\times10^{-11}$ | 1.032–1.039 | 3.072–4.256 | 3.099–4.296 |
| 512, quadratic | $[5.83,9.46]\times10^{-11}$ | 1.072–1.187 | 2.180–2.957 | 2.249–3.131 |
| 512, mixed | $[2.79,4.69]\times10^{-9}$ | 1.432–1.483 | 1.018–1.037 | 1.477–1.503 |
| 1024, sine | $[6.08,6.43]\times10^{-12}$ | 1.013–1.014 | 28.442–30.181 | 28.201–30.808 |
| 1024, quadratic | $[8.151,8.156]\times10^{-12}$ | 1.041–1.046 | 22.800–23.170 | 22.621–22.737 |
| 1024, mixed | $[2.22,3.31]\times10^{-10}$ | 1.205–1.248 | 1.538–1.783 | 1.741–2.064 |

The distinction between a typical step and occasional excursions matters. In the earlier 320k–340k window for sine at $N=1024$, seed 0, increasing the readout rate changes the median MSE from $5.25\times10^{-12}$ to $6.36\times10^{-12}$, but changes the mean from $6.52\times10^{-12}$ to $1.81\times10^{-10}$. Its 90th percentile rises from $8.20\times10^{-12}$ to $4.07\times10^{-10}$. An isolated endpoint can miss this behavior. The slower-geometry intervention gives essentially the same distribution as the shared control in this example; on the mixed target it is consistently worse at both widths.

The ordering is unchanged at 340k, 820k, and 1.1m. Slower geometry eventually raises mixed-target MSE by 43–48% at $N=512$ and 20–25% at $N=1024$. Between 340k and 1.1m, the shared mixed-target controls reduce their window means by 44–52% and 18–20%, respectively, so they are not described as converged. In contrast, the sine controls improve only 1–4% over the same interval, and faster readout retains its larger excursions.

This evidence does not support the simple explanation that geometry motion is the dominant obstruction and readout learning merely needs a larger rate. It does not exclude benefits from an earlier handoff or another rate schedule. The finite-update and modal measurements below narrow the mechanism behind the observed excursions.

<figure>
  <img src="ratio_primary_analysis/relative_effects_N1024.png" alt="Paired MSE ratios through 1.1 million updates for all three targets and both seeds at width 1024; raising the readout rate raises window MSE" style="max-width: 100%;">
  <figcaption>At N=1024, each curve divides a complete 20k-window MSE by its matched shared-rate control. The 160k–180k window includes the transition. Subsequent windows show persistent additional error from the larger readout rate; slowing geometry alone is close to neutral on sine and quadratic and detrimental on the mixed target.</figcaption>
</figure>

## The readouts move mostly in directions different from the remaining residual

The 340k diagnostics give a more specific explanation than lost plasticity. For the sine control at $N=1024$, seed 0, the RMS of accumulated absolute readout travel from 160k to 340k is $4.28\times10^{-4}$, whereas the RMS net displacement is only $1.21\times10^{-7}$. Increasing the readout rate tenfold raises accumulated travel to $4.37\times10^{-3}$ but leaves net displacement near $1.21\times10^{-7}$. The parameters keep moving; most of that motion cancels.

A fixed SVD basis at the start of the final 2048-update window distinguishes directions by their sensitivity. Define a relative singular value $s=\sigma/\sigma_{\max}$. In the two sine seeds, 76.3% and 80.9% of residual energy lies in directions with $s<10^{-4}$. Yet 99.8% of the readout update's projected function-space energy lies in directions with $s\geq0.1$. On the mixed target at $N=512$, more than 99.5% of residual energy lies below $10^{-4}$, while more than 99.8% of readout update energy lies above $0.1$. These thresholds summarize the spectra after observation; they are not acceptance criteria.

Fourier decomposition shows the corresponding mismatch. In the $N=512$ mixed control, seed 0, 81% of residual MSE lies in DFT indices 64–127 and another 12% in 128–255, while the largest readout-update descent terms are DC and the lowest frequencies. In the $N=1024$ sine case, raising the readout rate makes DC and the first frequency pair account for about 92% of the sampled residual energy. This is consistent with the increased excursions coming from low-frequency readout motion, rather than beneficially attacking the weak residual modes.

Suppressing boundary discontinuities with a Hann window does not remove the mixed target's high-frequency residual: at the 340k endpoint, indices 64–255 contain 95–99% of the tapered residual energy in the two $N=512$ seeds. Only 2–5% of their untapered residual energy lies in the outer 10% of the domain. The taper is a diagnostic only; it never changes the training objective. These checks support an interior spectral effect, without equating a finite-grid Fourier spectrum to the whole-line exponential attenuation formula.

The finite-update budget supports that interpretation. In the shared sine control, seed 0, the readout's sampled mean linear MSE change is $-6.32\times10^{-12}$ and its quadratic cost is $+6.23\times10^{-12}$; the geometry terms are orders of magnitude smaller. These are averages over 64 stratified states, so their small difference must not be treated as an accurate long-window drift estimate. Complete scalar traces determine improvement and oscillation.

Adam's second moment explains why the larger nominal rate does not yield a comparable increase in useful movement. Its update is $\Delta a_j=-G_j\hat m_j$, where $G_j=\eta_a/(\sqrt{\hat v_j}+\epsilon)$. In sine/1024, seed 0, the median $\sqrt{\hat v}/\epsilon$ rises from 4.62 to 53.95 when the readout rate rises tenfold. Evaluating $G$ at that median second-moment scale gives only 17.80 versus 18.20. Seed 1 changes from 18.11 to 17.93. The corresponding multipliers on mixed/512 change by only 3–5%. These are native-coordinate moment multipliers calculated from the saved optimizer state, distinct from the prescribed physical rate factors in Table 3. The larger second moments offset almost the entire scalar-rate increase. Together with the larger strong-mode excursions and nearly unchanged net displacement, this supports a feedback mechanism in which oscillation keeps readout normalization large and limits progress on weak modes. The frozen-solver and moment-reset controls will further test this interpretation.

A post-hoc stability check supplies an independent reason to expect difficulty settling at a constant Adam rate. For the frozen readout quadratic, let $L=\sigma_{\max}(AD)^2$. After gradients and second moments vanish, the linearized Adam recurrence is stable only if $\eta_aL<2(1+\beta_1)\epsilon/(1-\beta_1)$. With $\beta_1=0.9$ and $\epsilon=10^{-8}$, the sine/1024 dictionary gives a limiting rate near $3.12\times10^{-8}$, about 32 times smaller than the shared tail rate. This is a statement about the limiting stationary point, not a claim that the observed trajectory is epsilon dominated or a prediction of its MSE floor. The bound was checked against the eigenvalues of the linearized parameter/momentum recurrence.

The 1.1m diagnostics preserve the mechanism. For sine/1024, seed 0, accumulated readout travel since 160k is $2.31\times10^{-3}$ under the shared rate and $2.34\times10^{-2}$ under faster readout; net displacements are nearly identical at $5.60\times10^{-7}$ and $5.62\times10^{-7}$. The exact coordinatewise median of $G_j$ is 18.33 versus 18.25. In the shared controls, 81–83% of sine/1024 residual energy and over 99.7% of mixed/512 residual energy remains below relative singular value $10^{-4}$, while over 99.7% of readout-update energy lies above 0.1. Longer training preserves the mismatch between the residual directions and the dominant update directions.

Geometry still supplies a different update direction. Its motion is much less concentrated in the strongest singular modes than readout motion. The geometry gradient perpendicular to the retained readout span is also tiny relative to its parallel component: about $3.3\times10^{-5}$ to $5.5\times10^{-4}$ on the two sine/1024 seeds, and $8.6\times10^{-8}$ to $1.4\times10^{-6}$ on mixed/512. This uses the window-start projector and cutoff $10^{-12}$. The observations support a useful geometry contribution within the readout-accessible span; they do not establish that learning new out-of-span geometry is driving the late improvement. Slowing geometry removes some of this contribution and worsens mixed-target convergence at the measured horizon.

Regional readout accounting also shows that the effect is distributed. Across all 12 shared controls, core neurons supply 66–83% of sampled signed linearized readout descent; corrected halo neurons supply 9–20%, ordinary halo neurons 3–5%, and the bias 5–9%. Corrected halo neurons are influential, but this accounting does not identify them as the cause of the plateau. It partitions the gradient/update inner product, not the nonlinear finite-step improvement, whose quadratic terms include interactions between regions. These values reconstruct the saved signed descent to relative error below $7\times10^{-11}$ from consecutive physical parameter states.

<figure>
  <img src="ratio_340000_analysis/high_shared/sine_N1024_adam_both_envelope_s0_160a6008c214/mechanism.png" alt="Cumulative singular-mode energies separate the remaining residual from readout motion; finite-step readout descent nearly cancels its quadratic cost" style="max-width: 100%;">
  <figcaption>Sine, N=1024, seed 0, shared scalar rate 10^-6 at update 340k. The upper-left curves accumulate function-space energy from weak to strong singular directions in one fixed window-start basis. The other panels show signed Fourier-band descent, the exact finite-step loss budget, and trained versus detached-fit coefficients. All temporal averages use 64 stratified states from the final 2048 updates. Prediction-decomposition closure is below 10^-14 across all 84 analyzed acquisition and primary cases.</figcaption>
</figure>

## A small least-squares error does not establish easy readout optimization

The acquisition checkpoints already separate representability from iterative accessibility. At $N=512$, sine, seed 0, the higher acquisition rate produces median $|\lambda|=0.262$, live validation MSE $5.38\times10^{-11}$, and physical coefficient $\ell_1$ norm 10.74. A detached refit can lower that error much further, but its coefficient norm grows sharply as weaker directions are admitted.

**Table 5. Sine, N=512, seed 0: the learned dictionary at update 160k and a uniform reference dictionary on exactly the same centers. Coefficient norms include the output bias. These are detached diagnostics, not trained models.**

| Geometry | Relative cutoff $\tau$ | Retained rank | Refit validation MSE | Physical coefficient $\ell_1$ norm |
|---|---:|---:|---:|---:|
| Learned | $10^{-10}$ | 455 | $6.32\times10^{-15}$ | 101.6 |
| Learned | $10^{-12}$ | 496 | $7.52\times10^{-18}$ | 980.0 |
| Learned | $10^{-14}$ | 518 | $2.04\times10^{-19}$ | 13,104 |
| Uniform $\lambda=0.25$ | $10^{-12}$ | 520 | $6.86\times10^{-25}$ | 6.04 |

Doubling the learned dictionary's fitting-grid density gives closely matching refit errors, while the weakest coefficient norms remain more sensitive. Thus the learned dictionary can represent a much smaller residual, but reaching that residual can require large coefficients in weak singular directions. **A learned median bandwidth near 0.25 does not reproduce the uniform construction geometry:** the reference dictionary attains a much smaller residual with modest coefficients. This comparison supports examining the full learned slope distribution and target-loaded singular directions, rather than declaring the geometry correct from its median alone.

Nor is larger learned bandwidth uniformly better for representability. The lower-acquisition-rate sine dictionary at $N=512$, seed 1, has median $|\lambda|=0.060$ and attains refit MSE $8.00\times10^{-23}$ with coefficient norm 11.86 at the same cutoff. Its trained error is nevertheless higher than the higher-acquisition-rate run. Dictionary quality and the optimizer's route through parameter space are separate questions; the zero-start frozen comparisons are designed to distinguish them.

Localized tanh derivatives also do not make the tanh feature columns themselves disjoint: each feature still approaches constant tails. The frozen-dictionary comparison will test GD, momentum GD, and Adam in the prescribed coordinates and in fixed neighbor-difference coordinates that preserve the represented function space. It uses first-order updates and loss-only line searches, never a least-squares training update.

At this acquisition checkpoint, the median $\sqrt{\hat v}/\epsilon$ is about 54 for readouts and 0.0094 for geometry. Most geometry coordinates are already epsilon dominated; most readout coordinates are not. Consequently, multiplying a nominal Adam rate need not create proportionate useful motion in the two blocks. These moment measurements motivate inspecting actual updates; they do not by themselves explain the plateau.

## Earlier rate changes alter geometry without necessarily improving training

The shared acquisition rate matters even after both histories decay to the same $10^{-6}$ tail. At 1.1m, the $10^{-2}$ acquisition has lower mean training MSE than the $10^{-3}$ acquisition in all 12 cases. Their median absolute bandwidths span 0.105–0.303 and 0.014–0.059, respectively. The largest training gap is on mixed/512: the lower acquisition rate leaves 499–65,031 times more error. This comparison changes the complete joint optimization history, including readouts and moments; it does not isolate a causal effect of bandwidth.

The early-handoff experiment intervenes at 20k: it reduces only the geometry scalar rate tenfold over the next 20k updates, then retains the same scheduled decay at 80k–160k. All four runs reach 340k. Relative to the matched shared control over 320k–340k, this raises mean MSE by 2.75 and 6.23 times for sine/512, and 3.47 and 3.71 times for mixed/1024. Earlier geometry slowdown therefore does not rescue training in these cases either.

The resulting dictionaries reveal why one bandwidth statistic is insufficient. For sine/512, early slowdown lowers median $|\lambda|$ from 0.262–0.303 to 0.095–0.098. At cutoff $10^{-12}$, these new dictionaries attain detached-fit MSE $[1.63,3.84]\times10^{-22}$ with coefficient norms 11.8–14.7, compared with $[0.759,6.91]\times10^{-17}$ and norms 978–1198 for the shared controls at the same 340k checkpoint. They represent sine more economically under this diagnostic, yet their trained readouts perform worse. On mixed/1024, early slowdown worsens both the trained error and the detached-fit error and coefficient norm. Geometry quality is target dependent, and representability alone does not determine first-order optimization speed.

## Evidence and completion status

All 60 primary branches completed 1.1m updates; a subset reached 1.12m before the worker budgets expired. Comparisons use the common 1.1m horizon. All four early handoffs completed their planned 340k horizon. Detailed 340k spectral and motion measurements are retained with their explicit horizon and checked against the final primary export. The frozen-dictionary block, feedback rule, and historical higher-rate continuations remain pending in this interim report. No branch is declared converged merely because a budget limit or reporting horizon was reached.

Saved evidence includes complete per-step scalar traces; 20k checkpoints with parameters and Adam moments; dense physical parameters, signed gradients, and actual updates; Fourier-band residuals and forces; fixed-basis singular-mode projections; core/halo contributions; coefficient norms; and exact readout/geometry/interaction contributions to finite-step MSE changes. Detached fits use three cutoffs and a doubled sampling density. Per-seed movies place physical $w$ above physical $\gamma$ at fixed centers and slow the first 320k checkpoints.

**Table 6. Parameter movies through 1.1m, with separate seeds. Each page includes the full trajectory and a magnified view of the final 2048 updates. The first 320k checkpoints are held for one second each; actual update counts and both scalar rates remain visible.**

| Target and width | Intervention | Seed 0 | Seed 1 |
|---|---|---|---|
| Sine, 512 | Shared rate | [Movie](ratio_primary_analysis/animations/N512_sine_high_shared/seed_0.html) | [Movie](ratio_primary_analysis/animations/N512_sine_high_shared/seed_1.html) |
| Sine, 512 | Both changes | [Movie](ratio_primary_analysis/animations/N512_sine_high_both_changes/seed_0.html) | [Movie](ratio_primary_analysis/animations/N512_sine_high_both_changes/seed_1.html) |
| Mixed, 1024 | Shared rate | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_shared/seed_0.html) | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_shared/seed_1.html) |
| Mixed, 1024 | Both changes | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_both_changes/seed_0.html) | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_both_changes/seed_1.html) |

The full and magnified views have different vertical scales. Each view keeps its scales fixed through time and shared between its two seeds. Read the labeled physical units when comparing different interventions.

The eight-GPU-hour limit includes compilation, I/O, and the controlled cancellation used to replace slow compressed dense saves with lossless uncompressed saves. At most two GPUs are allocated concurrently. Analyses run in zero-GPU Slurm allocations. Actual allocation charges will be reconciled before the final report; reserved and completed allocations are distinguished in the ledger.

### Reproducibility

- [Protocol and commands](../../../experiments/expD06_fixed_center_scales/README.md#paired-continuation-at-the-learned-geometry).
- [Exact 340k window statistics](ratio_campaign/340k_windows.json), derived from all 60 complete 320k–340k traces in `ratio_campaign/340k_scalar_inputs/`.
- [Exact 820k window statistics](ratio_campaign/820k_windows.json), with complete 800k–820k traces and 820k checkpoints in `ratio_campaign/820k_scalar_inputs/`.
- [Final primary evidence](ratio_primary_analysis/evidence.json), including complete-window summaries through the common 1.1m horizon and endpoint modal, moment, spectral, and motion diagnostics.
- [Early-handoff evidence](ratio_early_analysis/evidence.json), with all four runs analyzed at 340k and compared against the same-horizon shared controls.
- [Acquisition evidence](ratio_acquisition_analysis/evidence.json), including endpoints, cutoff checks, sampling refinement, and finite-update audits.
- [340k mechanism evidence](ratio_340000_analysis/evidence.json), with per-case `dense_mechanism.npz` files for the modal and Fourier measurements. The original provenance's available common horizon is 360k; the explicit analysis cap and all primary records are 340k. Subsequent exports distinguish available and analyzed horizons explicitly.
- [Regional readout contributions](ratio_340000_analysis/regional_readout_summary.json), reconstructed from the consecutive physical states and saved band gradients.
- [Fourier boundary-sensitivity check](ratio_340000_analysis/boundary_sensitivity.json), using the separately identified 340k endpoints and an optional Hann taper.
- [Uniform-reference diagnostics](ratio_uniform_reference/evidence.json), with the same fixed centers, reference metric, and sampling grid. Table 5 uses the prescribed-coordinate projector; coordinate-dependent truncations are not pooled.
- [Acquisition animation, seed 0](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_0.html) and [seed 1](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_1.html), each showing $w$ above $\gamma$.
- [Allocation ledger](ratio_campaign/allocation_ledger.json). Remote source runs are under `/workspace/junmiaoh/experiments/precision-mlps/runs/ratios/`.

The complete repository test suite passes: **208 tests**, with seven existing deprecation warnings. Finite-update prediction closure and Fourier-gradient reconstruction were checked against direct calculations. Acquisition exports record maximum prediction-closure errors of order $10^{-15}$ in the example above. Final campaign validation and allocation accounting remain to be appended with the completed evidence.
