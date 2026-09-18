# Changing the relative readout and geometry rates

Does slowing geometry let the readout continue learning, or does it remove a useful route for reducing the residual? **None of the three late rate interventions improves the mean training MSE in any of the 12 width/target/seed cases at 1.1 million updates.** Readouts keep moving, mostly in sensitive directions that poorly match the remaining residual; small geometry updates still supply consistent improvement in weak directions. Frozen-geometry controls and the early and feedback interventions support that distinction. Neighbor-difference readout coordinates improve Adam on all three targets at both widths with uniform geometry; GD, momentum, and learned geometries show exceptions. The campaign and analysis are complete within 7.797 allocated GPU-hours; budget-limited trajectories are not declared converged. Two seeds establish paired observations, not a population-level significance claim.

**Table 1. Terminology used throughout this study.**

| Term or symbol | Meaning |
|---|---|
| Physical readouts $c=(b,w)$ | Output bias and coefficients multiplying the tanh features. |
| Physical slopes $\gamma$ | Slopes attached to the fixed physical centers. |
| Bandwidth $\lambda=h\gamma$ | Signed dimensionless slope; reported localization quantiles use $\lvert\lambda\rvert$. Here $h=2/N$. |
| Scaled training | Train $a,\lambda$ with $c=Da$ and $\gamma=\lambda/h$. $D$ is fixed at the theoretical reference bandwidth $0.25$. |
| Prescribed coordinates | The readout coordinates $c=Da$ used by scaled training, also retained in the frozen-geometry comparisons. |
| Neighbor-difference coordinates | An invertible readout map using differences of adjacent fixed tanh features and cumulative envelope scales; defined below. |
| Shared rate | The same scalar rate for $a$ and $\lambda$. The prescribed physical scales still differ. |
| Slower geometry | Change only the scalar geometry rate to one tenth of the shared control. |
| Faster readout | Change only the scalar readout rate to ten times the shared control. |
| Both changes | Apply slower geometry and faster readout together. |
| MSE | Mean squared residual. Training differentiates half-MSE; every error in this report is MSE. |
| Detached refit | SVD least-squares diagnostic at a saved geometry. It never replaces the trained readout. |
| $\tau$ | Relative singular-value cutoff used in a detached refit. |
| DFT band | A band of discrete Fourier indices on the actual training grid, including both frequency signs. These describe the sampled residual; finite-window leakage can mix frequencies. |
| DC | The zero-frequency Fourier component: a constant residual offset. Its MSE contribution is the squared mean residual. |
| Singular direction | A pattern on the training grid given by a left singular vector of the scaled readout matrix $AD$; generally not a single Fourier frequency. |

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

The common decay to $10^{-6}$ is an empirical schedule choice, not a consequence of the theory scales. The ratio changes occur after that decay: even the increased readout rate remains far below the original $10^{-2}$. These comparisons therefore address the late, low-rate regime. They do not test a sustained readout increase and geometry decrease starting from the original shared rate without first decaying both blocks.

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

The measurements below use 64 stratified states from updates 337,952–339,999, one from each block of 32 saved states with fixed random offsets. They describe this sampled late window, not the single 340k checkpoint or the full 20k window used for the primary MSE comparison. Let $A(\gamma)$ contain the bias and tanh features divided by $\sqrt M$, where $M$ is the training sample count, and let $r=(\hat f-y)/\sqrt M$. Then $\|r\|^2$ is MSE. We fix the SVD $B_0=A(\gamma_{337952})D=U\Sigma V^T$ at the beginning of the window. A left singular vector $u_i$ is an output pattern on the training grid; its singular value measures the output sensitivity to the corresponding direction in scaled readout parameters.

Define $s_i=\sigma_i/\sigma_{\max}$. Residual energy in a set of singular directions is $\sum_i|u_i^Tr_t|^2$. Readout-update energy uses $\sum_i|u_i^T\delta r_{w,t}|^2$, with the actual readout-only output change $\delta r_{w,t}=A(\gamma_t)\Delta c_t$. Each percentage divides the sampled mean band energy by the sampled mean total energy of that same quantity. Residual percentages and update percentages therefore have different denominators. They are neither percentages of parameters nor fractions of error reduction. Any energy outside the fixed thin $U$ basis is counted separately.

In the two sine/1024 seeds, 76.3% and 80.9% of residual energy lies at $s<10^{-4}$, while about 99.8% of readout-update energy lies at $s\geq0.1$. On mixed/512, more than 99.5% of residual energy lies below $10^{-4}$ and more than 99.8% of readout-update energy lies above 0.1. The mixed seed-0 figure below shows these quantities directly: 99.60% versus 99.87%. These thresholds summarize the spectra after observation; they are not acceptance criteria. Concentrated update energy alone does not prove zero progress in weak directions; the motion and finite-update measurements provide the additional evidence of cancellation.

Fourier analysis uses the same normalized residuals and sampled states, but a different basis. We apply an orthonormal DFT $F$ to each $r_t$, pair positive and negative indices, and sum $|(Fr_t)_k|^2$ within each band. Band energies add to MSE by Parseval's identity. The plotted percentage is

$$
100\,\frac{\operatorname{mean}_t E_b(t)}{\operatorname{mean}_t\|r_t\|^2},
\qquad E_b(t)=\sum_{k\in b}|(Fr_t)_k|^2.
$$

This is a ratio of means, not a mean of per-state percentages. **DC means zero frequency:** the constant offset of the residual, with energy $(\operatorname{mean}_x(\hat f-y))^2$. The index $k$ counts cycles across the DFT period, which is the sample count times the grid spacing; it is not a neuron index. The remaining bands include both frequency signs. No taper is used in these measurements.

For mixed/512, seed 0, sampled mean MSE is $4.9243\times10^{-9}$. The 64–127 band contributes $3.9927\times10^{-9}$, or **81.08%**, and 128–255 contributes $5.9717\times10^{-10}$, or **12.13%**. Yet the largest signed linear readout descent terms occur at DC and low frequencies. The figure shows absolute MSE and percentages side by side, using exactly the same samples as those numbers. Its lower-left panel is $-2\langle Q_b r,A\Delta c\rangle$, the signed linear contribution to MSE reduction in band $b$; it excludes the positive quadratic update cost and is not a net improvement measurement.

<figure>
  <img src="ratio_340000_analysis/high_shared/mixed_N512_adam_both_envelope_s0_770e6899dded/spectral_window.png" alt="Matched 340k-window diagnostics for mixed/512 seed 0: absolute Fourier-band MSE, percentages including 81.08 percent in 64–127 and 12.13 percent in 128–255, signed linear readout descent, and singular-mode energy percentages" style="max-width: 100%;">
  <figcaption>Mixed, N=512, seed 0, shared scalar rate 10^-6. All four panels use the same 64 sampled states from the final 2048 updates before 340k. The top panels express the same Fourier residual energies in absolute units and percentages. The lower-left panel shows signed linear readout descent, excluding quadratic costs. The lower-right panel uses the fixed window-start SVD and divides residual and update energies by their own totals. Fourier bands and singular directions are distinct decompositions.</figcaption>
</figure>

For sine/1024, seed 0, DC plus the $k=1$ pair accounts for 22.4% of sampled residual MSE under the shared rate and 92.1% under faster readout. The [shared-rate figure](ratio_340000_analysis/high_shared/sine_N1024_adam_both_envelope_s0_160a6008c214/spectral_window.png) and [faster-readout figure](ratio_340000_analysis/high_faster_readout/sine_N1024_adam_both_envelope_s0_160a6008c214/spectral_window.png) show both measurements. This supports low-frequency excursions under the larger readout rate. It does not identify individual Fourier bands with individual singular modes.

<figure>
  <img src="ratio_primary_analysis/high_shared/mixed_N512_adam_both_envelope_s0_770e6899dded/spectral_history.png" alt="Checkpoint history showing separate 64–127 and 128–255 curves in absolute MSE and percentage units, bandwidth quantiles, and both scalar learning rates" style="max-width: 100%;">
  <figcaption>Mixed, N=512, seed 0, full shared-rate history. The upper panels show the same checkpoint Fourier-band energies in MSE and percentage units; 64–127 and 128–255 are separate curves. Their combined share is 91% at 160k and 95% at 1.1m. The lower panels show bandwidth quantiles and the actual common scalar decay. These are individual checkpoint measurements, distinct from the sampled-window figure above and the complete-window means used to compare interventions.</figcaption>
</figure>

Suppressing boundary discontinuities with a Hann window does not remove the mixed target's high-frequency residual: at the 340k endpoint, indices 64–255 contain 95–99% of the tapered residual energy in the two $N=512$ seeds. Only 2–5% of their untapered residual energy lies in the outer 10% of the domain. The taper is a diagnostic only; it never changes the training objective. These checks support an interior spectral effect, without equating a finite-grid Fourier spectrum to the whole-line exponential attenuation formula.

The finite-update budget supports that interpretation. In the shared sine control, seed 0, the readout's sampled mean linear MSE change is $-6.32\times10^{-12}$ and its quadratic cost is $+6.23\times10^{-12}$; the geometry terms are orders of magnitude smaller. These are averages over 64 stratified states, so their small difference must not be treated as an accurate long-window drift estimate. Complete scalar traces determine improvement and oscillation.

Adam's second moment explains why the larger nominal rate does not yield a comparable increase in useful movement. Its update is $\Delta a_j=-G_j\hat m_j$, where $G_j=\eta_a/(\sqrt{\hat v_j}+\epsilon)$. In sine/1024, seed 0, the median $\sqrt{\hat v}/\epsilon$ rises from 4.62 to 53.95 when the readout rate rises tenfold. Evaluating $G$ at that median second-moment scale gives only 17.80 versus 18.20. Seed 1 changes from 18.11 to 17.93. The corresponding multipliers on mixed/512 change by only 3–5%. These are native-coordinate moment multipliers calculated from the saved optimizer state, distinct from the prescribed physical rate factors in Table 3. The larger second moments offset almost the entire scalar-rate increase. Together with the larger strong-mode excursions and nearly unchanged net displacement, this supports a feedback mechanism in which oscillation keeps readout normalization large and limits progress on weak modes. The moment-reset comparison below shows that simply discarding the old moment state does not remove the limitation.

A post-hoc stability check supplies an independent reason to expect difficulty settling at a constant Adam rate. For the frozen readout quadratic, let $L=\sigma_{\max}(AD)^2$. After gradients and second moments vanish, the linearized Adam recurrence is stable only if $\eta_aL<2(1+\beta_1)\epsilon/(1-\beta_1)$. With $\beta_1=0.9$ and $\epsilon=10^{-8}$, the sine/1024 dictionary gives a limiting rate near $3.12\times10^{-8}$, about 32 times smaller than the shared tail rate. This is a statement about the limiting stationary point, not a claim that the observed trajectory is epsilon dominated or a prediction of its MSE floor. The bound was checked against the eigenvalues of the linearized parameter/momentum recurrence.

The 1.1m diagnostics preserve the mechanism. For sine/1024, seed 0, accumulated readout travel since 160k is $2.31\times10^{-3}$ under the shared rate and $2.34\times10^{-2}$ under faster readout; net displacements are nearly identical at $5.60\times10^{-7}$ and $5.62\times10^{-7}$. The exact coordinatewise median of $G_j$ is 18.33 versus 18.25. In the shared controls, 81–83% of sine/1024 residual energy and over 99.7% of mixed/512 residual energy remains below relative singular value $10^{-4}$, while over 99.7% of readout-update energy lies above 0.1. Longer training preserves the mismatch between the residual directions and the dominant update directions.

Geometry's smaller motion makes more useful progress in the weak directions. Projecting the finite updates onto $s<10^{-4}$ at 1.1m, its sampled mean linear MSE reduction is 11–15 times the readout reduction on sine/1024 and 85–90 times on mixed/512. The projected geometry-only MSE change is negative in every one of the 64 sampled states in all 12 shared controls, with negligible quadratic cost in this band. Removing modes below relative singular value $10^{-10}$ changes the reported block means by less than 1%. An independent analytic-Jacobian calculation from consecutive parameter states reproduces the geometry means to relative error below $3\times10^{-7}$ in four representative cases, with descent in all 256 sampled states. These sampled budgets explain why small geometry motion can matter despite much larger readout travel; they are not complete-window drift estimates.

The geometry gradient perpendicular to the retained readout span is tiny relative to its parallel component: at 340k, about $3.3\times10^{-5}$ to $5.5\times10^{-4}$ on the two sine/1024 seeds, and $8.6\times10^{-8}$ to $1.4\times10^{-6}$ on mixed/512. This uses the window-start projector and cutoff $10^{-12}$. The observations support a useful geometry contribution within the readout-accessible span; they do not establish that learning new out-of-span geometry is driving the late improvement. Slowing geometry removes some of this contribution and worsens mixed-target convergence at the measured horizon.

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

The mixed target gives the complementary case in which geometry is a substantial representational obstacle. At 160k and $N=512$, the lower acquisition rate produces median bandwidths 0.021 and 0.048. Its $10^{-12}$ refits need coefficient norms $3.23\times10^6$ and $8.85\times10^4$, attaining MSE $2.98\times10^{-9}$ and $8.42\times10^{-15}$. The higher-rate dictionaries, with medians 0.204 and 0.179, attain $3.45\times10^{-18}$ and $7.02\times10^{-19}$ with norms 422 and 508. Doubling the fitting-grid density preserves these comparisons. This supports a role for geometry in making the higher-frequency target accessible with manageable coefficients. The two acquisition histories do not, by themselves, measure an exponential attenuation law or isolate geometry from the accompanying readout and moment changes.

Localized tanh derivatives also do not make the tanh feature columns themselves disjoint: each feature still approaches constant tails. The frozen-dictionary comparison below tests GD, momentum GD, and Adam in the prescribed coordinates and in fixed neighbor-difference coordinates that preserve the represented function space. It uses first-order updates and loss-only line searches, never a least-squares training update.

At this acquisition checkpoint, the median $\sqrt{\hat v}/\epsilon$ is about 54 for readouts and 0.0094 for geometry. Most geometry coordinates are already epsilon dominated; most readout coordinates are not. Consequently, multiplying a nominal Adam rate need not create proportionate useful motion in the two blocks. These moment measurements motivate inspecting actual updates; they do not by themselves explain the plateau.

## Freezing geometry and resetting Adam do not remove the late plateau

The frozen comparison uses all 24 learned dictionaries at 160k and six uniform $\lambda=0.25$ dictionaries, one per target and width. Each dictionary receives zero-start GD, momentum GD with momentum 0.9, and Adam in two coordinate systems, plus a zero-start Adam schedule that decays $10^{-3}$ to $10^{-6}$ over 80k. Each learned dictionary also receives two continuations from its trained readout at fixed $10^{-6}$: one preserves Adam's moments and one resets them. These are 258 logical solves, all compared after 140k additional updates. Methods sharing a dictionary execute together, so gradient and loss evaluation counts are meaningful costs; independently measured per-method wall times are unavailable.

Freezing the higher-acquisition-rate geometry and preserving its readout moments leaves sine's 280k–300k mean MSE within 0.2–0.7% above joint training. On mixed, freezing raises MSE by 27–32% at $N=512$ and 4.9–5.5% at $N=1024$. Every one of the 24 frozen continuations has higher window MSE than its matched joint control at this horizon. Resetting the readout moments changes frozen-window MSE by only $-0.335\%$ to $+0.060\%$. Thus neither geometry motion nor stale moment memory explains away the late readout limitation in these comparisons. Freezing geometry removes useful descent, especially on mixed.

## Neighbor-difference readouts improve Adam across uniform targets

**The uniform-reference plot below does not start from a learned geometry checkpoint.** It prescribes $\lambda_j=0.25$ for every neuron, hence $\gamma_j=0.25/h$, freezes all slopes and centers, and trains only the readouts from zero for 140k updates. Its curves are first-order training trajectories, not detached least-squares refits.

**Prescribed coordinates** retain $c=Da$. To define **neighbor-difference coordinates**, order the $W$ centers from left to right and write the frozen features and output as

$$
\phi_j(x)=\tanh\!\left(\frac{0.25}{h}(x-z_j)\right),
\qquad
f(x)=b+\sum_{j=1}^{W}w_j\phi_j(x).
$$

Introduce cumulative readouts

$$
q_0=0,
\qquad q_j=\sum_{k=1}^{j}w_k,
\qquad w_j=q_j-q_{j-1}.
$$

Substituting and collecting adjacent terms gives the exact identity

$$
\boxed{
f(x)=b+\sum_{j=1}^{W-1}q_j\bigl[\phi_j(x)-\phi_{j+1}(x)\bigr]
+q_W\phi_W(x).
}
$$

Thus the optimizer can adjust coefficients of neighboring-feature differences instead of coefficients of individual tanhs. For these uniform slopes, each difference cancels the constant tails and forms a localized bump. Keeping the final tanh and bias preserves exactly the original function space; the inverse $w_j=q_j-q_{j-1}$ recovers the physical readouts.

The implementation also scales the cumulative coefficients. If $\alpha_k$ is the fixed theoretical reference envelope for readout $k$, define

$$
S_j=\left(\sum_{k=1}^{j}\alpha_k\right)^{1/2},
\qquad q_j=S_j\theta_j,
\qquad b=D_0\theta_0.
$$

The trained variables are $\theta_0,\ldots,\theta_W$; the bias retains its original scale $D_0$ and is excluded from the cumulative sum. For learned dictionaries with signed slopes, the code first absorbs each negative slope's sign into its corresponding readout before applying the same construction to positive-slope features. This is an invertible, data-independent, non-diagonal readout map. It changes optimization coordinates while leaving geometry fixed, and is a separate experiment from changing the scalar readout/geometry LR ratio. It is not whitening.

### Why differencing acts as a preconditioner

There is an exact GD interpretation of this map and a simple limiting case that explains the benefit. Let $L$ be the lower bidiagonal difference matrix, so $(Lq)_j=q_j-q_{j-1}$ with $q_0=0$, and let $S=\operatorname{diag}(S_1,\ldots,S_W)$. For the positive slopes in the uniform-reference comparison, $w=LS\theta$. Applying ordinary GD to $\theta$ induces the physical update

$$
\Delta w=-\eta\,\underbrace{LS^2L^T}_{P}\,\nabla_w\mathcal L,
\qquad
\Delta b=-\eta D_0^2\,\partial_b\mathcal L.
$$

Thus $P$ is a fixed positive-definite preconditioner with a weighted discrete-Laplacian form, including its boundary terms. If $A_w$ is the matrix of tanh columns divided by the square root of the sample count, the hidden-readout block of the half-MSE Hessian in the new coordinates is

$$
H_\theta=S L^T A_w^T A_w L S.
$$

The columns of $A_wL$ are precisely neighboring tanh differences followed by the final tanh. This algebra follows directly from the implemented coordinate map. It is exact for GD; Adam adds coordinatewise moment normalization, so its update is not equivalent to GD with this one fixed matrix.

The sharp-step limit makes the reason for cancellation explicit. For equally spaced centers and $\gamma\to\infty$, ignoring values exactly at a center,

$$
\phi_j(x)-\phi_{j+1}(x)
\longrightarrow 2\,\mathbf 1_{(z_j,z_{j+1})}(x).
$$

These cell functions have disjoint interiors. With uniform probability measure on $[-1,1]$, their interior Gram matrix is

$$
\frac12\int_{-1}^{1}
[\phi_i(x)-\phi_{i+1}(x)]
[\phi_j(x)-\phi_{j+1}(x)]\,dx
\longrightarrow 2h\,\delta_{ij},
$$

for cells fully inside the domain. In unscaled $q$ coordinates this block is a multiple of the identity. In the implemented $\theta$ coordinates it becomes $2h\operatorname{diag}(S_j^2)$. This is an interior-block statement: it excludes the bias, final anchor feature, and halo cells. Differencing removes the nonlocal correlations of cumulative step features, while the additional scale choice still affects conditioning.

At finite slope the bumps overlap. Their exact form is a smoothed cell indicator,

$$
\phi_j(x)-\phi_{j+1}(x)
=\int_{z_j}^{z_{j+1}}\gamma\,\operatorname{sech}^2\!\bigl(\gamma(x-z)\bigr)\,dz.
$$

This explains why local corrections become easier to express in an individual coordinate, but does not remove the smoothing of very fine spatial patterns. Constructing localized kernels from differences of shifted sigmoids is established in approximation theory; see the density function in Section 2 of [Costarelli (2022)](https://journals.vilniustech.lt/index.php/MMA/article/download/15974/11323/67690). That result is not an Adam convergence theorem or a justification for our particular cumulative weights.

A finite-slope conditioning theorem would need upper and lower bounds on the Gram matrix of these overlapping bumps, then account for $S$, the bias/anchor, and the finite-domain halo. The corresponding framework for translates is the Gramian characterization of stable bases in [Aldroubi et al., Section 2](https://mate.dm.uba.ar/~hafg/papers/determining03.pdf). We have not established those bounds for the implemented system or proved that its cumulative scaling is optimal. The observed 32–37-fold increase in the median relative singular value describes the middle of the spectrum; it does not establish a well-conditioned full matrix, whose smallest computed singular values remain near or below floating-point resolution.

### What improves across targets

The six line-search methods use Armijo constant $10^{-4}$, halving, at most 40 trials, and initial trials 0.1 for GD/momentum or 0.001 for Adam. Subsequent trials double the last accepted rate, capped at one. An uphill momentum or Adam direction falls back to the negative gradient. These are empirical first-order comparisons, separate from the theory-scaled shared-rate joint baseline.

**Table 6. Uniform $\lambda=0.25$ dictionaries: mean training MSE over frozen updates 120k–140k, starting from zero readouts. Entries are prescribed coordinates / neighbor-difference coordinates. Several difference-coordinate Adam curves encounter numerical stalls; their values are achieved errors, not convergence floors.**

| $N$, target | GD | Momentum GD | Adam |
|---|---:|---:|---:|
| 512, sine | $1.29\times10^{-6}$ / $1.13\times10^{-7}$ | $6.00\times10^{-8}$ / $7.01\times10^{-9}$ | $1.06\times10^{-8}$ / $3.38\times10^{-11}$ |
| 512, quadratic | $4.73\times10^{-7}$ / $3.28\times10^{-6}$ | $2.04\times10^{-8}$ / $1.87\times10^{-7}$ | $3.01\times10^{-8}$ / $1.41\times10^{-12}$ |
| 512, mixed | $7.99\times10^{-6}$ / $3.37\times10^{-7}$ | $2.38\times10^{-7}$ / $1.98\times10^{-8}$ | $3.75\times10^{-8}$ / $3.11\times10^{-11}$ |
| 1024, sine | $5.97\times10^{-7}$ / $1.30\times10^{-7}$ | $3.07\times10^{-8}$ / $1.37\times10^{-8}$ | $4.71\times10^{-9}$ / $7.38\times10^{-12}$ |
| 1024, quadratic | $2.08\times10^{-7}$ / $4.49\times10^{-6}$ | $1.12\times10^{-8}$ / $4.77\times10^{-7}$ | $3.17\times10^{-8}$ / $2.49\times10^{-13}$ |
| 1024, mixed | $2.72\times10^{-6}$ / $2.41\times10^{-7}$ | $1.25\times10^{-7}$ / $2.62\times10^{-8}$ | $2.05\times10^{-8}$ / $7.34\times10^{-12}$ |

**For Adam, neighbor differences help every target at both widths in the uniform-reference plot.** The reductions in final-window MSE are 313 and 638 times for sine, 21,380 and 127,320 times for quadratic, and 1,206 and 2,796 times for mixed, respectively at $N=512$ and $1024$. This is substantial evidence of an optimization benefit at identical geometry. It does not require geometry learning to explain the improvement.

The exceptions concern other comparisons: GD and momentum improve on uniform sine and mixed but become worse on uniform quadratic. Quadratic MSE rises by 6.9–21.6 times for GD and 9.1–42.5 times for momentum. On the learned dictionaries, neighbor differences improve Adam in 16 of 24 cases. Thus the benefit is broad for Adam on uniform geometry, without being uniform across optimizers and learned geometries. Some Adam curves contain numerical stalls; these are achieved finite-budget errors rather than certified convergence floors, as the paired audit below shows.

<figure>
  <img src="ratio_solver_analysis/uniform_solver_comparison.png" alt="GD, momentum, and Adam training curves through 140k readout updates in both coordinate systems on the six uniform dictionaries" style="max-width: 100%;">
  <figcaption>Prescribed, frozen uniform bandwidth 0.25; no learned geometry checkpoint is used. Readouts start at zero. Solid lines use prescribed coordinates and dashed lines use neighbor-difference coordinates; blue is GD, orange is momentum, and green is Adam. Curves show endpoint MSE under the original Armijo implementation, while Table 6 uses complete-window means. Some flat curves contain failed numerical updates; the paired audit below distinguishes those failures from slow but nonzero progress.</figcaption>
</figure>

The target's position in the singular basis explains why a larger typical singular value is insufficient. In the uniform dictionaries, neighbor differences raise the median relative singular value by about 32–37 times, yet the quadratic target loads directions that remain slower under GD. A detached constant-step GD calculation gives, for dictionary $B=U\Sigma V^T$ and target vector $y$ both normalized by the square root of the sample count, zero initial readout, and step $1/\sigma_{\max}^2$,

$$
E_{\mathrm{retained}}(t)=\sum_i (u_i^T y)^2
\left(1-\frac{\sigma_i^2}{\sigma_{\max}^2}\right)^{2t}.
$$

This predicts improvement from the coordinate change for sine and mixed but deterioration for quadratic. It is a diagnostic of target-loaded conditioning, not a prediction of the actual Armijo trajectories or of Adam. It excludes the component orthogonal to the exported thin SVD and does not simulate roundoff. The finite-step formula was checked against direct GD on a small quadratic problem.

<figure>
  <img src="ratio_uniform_reference/target_loaded_gd.png" alt="Predicted constant-step GD residual decay for prescribed and neighbor-difference coordinates; the target-dependent comparison reverses for quadratic" style="max-width: 100%;">
  <figcaption>Detached uniform-dictionary calculation at each coordinate system's fixed GD step 1 divided by its largest squared singular value. The full target projection, not a single condition statistic, determines the predicted decay. These curves are diagnostics, not additional training runs.</figcaption>
</figure>

## Some frozen-solver plateaus were numerical stalls

The original line search sometimes stops changing coefficients despite a nonzero gradient. It compares nearly equal full losses, and an uphill Adam direction falls back to the raw negative gradient while inheriting a trial rate appropriate to the normalized Adam direction. Repeated shrinking can produce an unrepresentably small update or an absorbing zero rate. These cases are numerical failures, not evidence of convergence or a fundamental optimization floor. Joint training has no line search, so this failure does not account for its plateau.

A paired audit forks all six uniform dictionaries and one learned quadratic/512 dictionary with observed momentum stagnation at frozen update 140k. It copies the exact parameters, moments, counters, and accepted rates for each of the six line-search methods. One arm retains the original evaluation; the other evaluates the quadratic loss change as $r^TB\Delta z+\|B\Delta z\|^2/2$ using the actual rounded displacement, restarts an inherited zero rate, and gives gradient fallback the GD trial scale 0.1. This is a package of numerical guards: the experiment does not isolate each guard's effect. Every one of the 84 continuations completes at least 40k additional updates.

The four persistently stalled controls change no coefficients in those 40k updates. Their paired guarded runs have zero stagnant updates, as do all 42 guarded arms. Over the final 20k window, the guarded neighbor-difference Adam solves lower mean MSE by 0.42% on uniform quadratic/1024, 1.28% on quadratic/512, and 1.30% on sine/1024. The learned quadratic/512 momentum solve lowers its window mean by 26.3%; its validation endpoint improves by 33.3%. All fourteen GD pairs retain identical window MSE. On the zero-rate quadratic/1024 Adam case, the repair also reduces counted loss evaluations per gradient from 41 to 4.64.

The [paired curves](ratio_line_search_audit_analysis/numerical_audit.png) and [complete audit evidence](ratio_line_search_audit_analysis/evidence.json) preserve both outcomes. Restoring representable updates is necessary, but the modest Adam improvement shows that it does not eliminate the broader slow descent. Some other learned-dictionary stalls were not included in this bounded audit; the original solver table remains explicitly subject to that limitation.

## Earlier rate changes alter geometry without necessarily improving training

The shared acquisition rate matters even after both histories decay to the same $10^{-6}$ tail. At 1.1m, the $10^{-2}$ acquisition has lower mean training MSE than the $10^{-3}$ acquisition in all 12 cases. Their median absolute bandwidths span 0.105–0.303 and 0.014–0.059, respectively. The largest training gap is on mixed/512: the lower acquisition rate leaves 499–65,031 times more error. This comparison changes the complete joint optimization history, including readouts and moments; it does not isolate a causal effect of bandwidth.

A separate historical comparison keeps the acquisition rate fixed through 320k, decays it over 80k, and compares both histories at 1.68m on sine/512. The new higher-rate tails attain window means $2.61\times10^{-11}$ and $2.25\times10^{-11}$, versus $1.46\times10^{-10}$ and $9.34\times10^{-11}$ for the existing lower-rate tails. Their median absolute bandwidths are 0.3145 and 0.3170, versus 0.0337 and 0.0802. This confirms a better trained regime above median bandwidth 0.3 in these cases; it does not establish 0.3 as an optimum. The higher-rate dictionaries are heterogeneous: their 90th-percentile bandwidths are about 0.49–0.52, with some slopes nearly zero. A median near the reference value does not make them uniform reference dictionaries.

The early-handoff experiment intervenes at 20k: it reduces only the geometry scalar rate tenfold over the next 20k updates, then retains the same scheduled decay at 80k–160k. All four runs reach 340k. Relative to the matched shared control over 320k–340k, this raises mean MSE by 2.75 and 6.23 times for sine/512, and 3.47 and 3.71 times for mixed/1024. Earlier geometry slowdown therefore does not rescue training in these cases either.

The resulting dictionaries reveal why one bandwidth statistic is insufficient. For sine/512, early slowdown lowers median $|\lambda|$ from 0.262–0.303 to 0.095–0.098. At cutoff $10^{-12}$, these new dictionaries attain detached-fit MSE $[1.63,3.84]\times10^{-22}$ with coefficient norms 11.8–14.7, compared with $[0.759,6.91]\times10^{-17}$ and norms 978–1198 for the shared controls at the same 340k checkpoint. They represent sine more economically under this diagnostic, yet their trained readouts perform worse. On mixed/1024, early slowdown worsens both the trained error and the detached-fit error and coefficient norm. Geometry quality is target dependent, and representability alone does not determine first-order optimization speed.

## The feedback rule rejects larger readout steps but does not rescue convergence

The training-only rule starts from the same 160k checkpoints. Two successive window improvements below 5%, together with more than 95% of residual energy inside the readout span at both cutoffs $10^{-10}$ and $10^{-12}$, trigger a tenfold geometry slowdown over 20k. After 40k at the new rates, a persistent stall permits a threefold readout-rate increase only if its readout-only counterfactual decreases mean MSE and decreases MSE in at least 95% of 64 stratified states. These are declared operational thresholds, not constants derived from the theory.

Eleven cases complete a geometry slowdown beginning at 220k or 240k. At the common 380k–400k window, all eleven have higher mean MSE than their shared-rate controls: mixed/512 is 11–13% worse and mixed/1024 is 4–5% worse; sine and quadratic change by less than 4%. The remaining sine/1024 seed keeps the shared rates through 400k and differs in mean error by only 0.16%. It finally schedules a slowdown at its last checkpoint, 420k, leaving that intervention unexecuted at the budget stop. Seed 0 cases finish at 400k and seed 1 cases at 420k; the outcome comparison uses 400k for both.

All 81 recorded tests of a threefold readout step reject it. Every sampled mean MSE change is positive, between $6.48\times10^{-12}$ and $4.92\times10^{-11}$; at most 54.7% of the sampled states improve. Thus the rule avoids the harmful readout increase seen in the fixed schedules, but its geometry slowdown still fails to improve the measured outcomes. Being inside the readout span establishes representability at a chosen cutoff, not rapid accessibility to the current optimizer. That distinction is the failure in this particular feedback criterion.

## What this says about the bandwidth hypothesis

The evidence supports a **readout least-squares conditioning bottleneck at the current geometry**. For plain GD with frozen geometry and $B=AD$, each singular residual coefficient $e_i=u_i^Tr$ evolves exactly as

$$
e_i(t+1)=(1-\eta_a\sigma_i^2)e_i(t).
$$

The largest singular values limit a stable scalar step; small singular values then decay slowly. At the illustrative choice $\eta_a=1/\sigma_{\max}^2$, a mode with $s=10^{-4}$ contracts by only $1-10^{-8}$ per update. Multiplying one readout scalar rate does not change the ratio of these curvatures. Adam adds momentum and a changing diagonal scaling, so this GD recurrence is not a quantitative prediction of its trajectory. The measured concentration of actual Adam updates in strong directions, cancelling parameter motion, and continued plateau under frozen geometry supply the experimental evidence. They support slow weak-mode progress, not a claim that progress is mathematically impossible.

The least-squares system and geometry are linked: $B$ depends on every $\gamma_j$. Changing gamma can change its singular values and the target's projections. A larger median $\lambda$ is therefore not guaranteed to improve the relevant conditioning. Localized tanh derivatives still leave tanh columns with correlated constant tails, and even the uniform reference dictionary has slow first-order readout solves. Geometry can also reduce an error by moving features, offering another route through output space without having to create a new direction outside the readout span. This explains how its small late updates can remain useful when a detached readout fit already represents the target accurately.

The narrow hypothesis is well supported: **changing only the scalar readout/geometry rate ratio does not repair the fixed readout system's conditioning.** It can change which geometry is learned during joint training, a separate effect that the untested early opposing schedule could address. A change of readout coordinates is different: $c=Tz$ changes the first-order system from $AD$ to $AT$ even at identical gamma. Neighbor differences substantially improve Adam on every uniform target/width case, including quadratic; the quadratic counterexamples concern GD and momentum. Thus these results do not support the broader statement that all reparameterizations leave the bottleneck unchanged.

The prescribed scales put the updates in the intended units; they do not impose a loss whose minimizer must have uniform bandwidth 0.25 or small coefficients. The mixed-target acquisition comparison supports a geometry barrier, while sine shows that economical representation can coexist with slow readout optimization. The late measurements do not identify an exponential slowdown law or prove that out-of-span attenuation causes the plateau. They distinguish representation, target-dependent least-squares conditioning, and the optimizer's actual motion rather than treating one bandwidth statistic as a certificate that all three are satisfactory.

## Evidence and completion status

All 60 primary branches completed 1.1m updates; a subset reached 1.12m before the worker budgets expired. Comparisons use the common 1.1m horizon. All four early handoffs completed their planned 340k horizon, all 12 feedback cases completed the common 400k comparison, and both historical higher-rate tails reached 1.68m. All 258 original frozen solves completed the common 140k horizon; the 84 numerical-audit continuations completed another common 40k. Detailed 340k spectral and motion measurements are retained with their explicit horizon and checked against the final primary export. These are related continuations of shared ancestors, not independent random trials. No branch is declared converged merely because a budget limit or reporting horizon was reached.

Saved evidence includes complete per-step scalar traces; 20k checkpoints with parameters and Adam moments; dense physical parameters, signed gradients, and actual updates; Fourier-band residuals and forces; fixed-basis singular-mode projections; core/halo contributions; coefficient norms; and exact readout/geometry/interaction contributions to finite-step MSE changes. Detached fits use three cutoffs and a doubled sampling density. Per-seed movies place physical $w$ above physical $\gamma$ at fixed centers and slow the first 320k checkpoints.

**Table 7. Parameter movies through 1.1m, with separate seeds. Each page includes the full trajectory and a magnified view of the final 2048 updates. The first 320k checkpoints are held for one second each; actual update counts and both scalar rates remain visible.**

| Target and width | Intervention | Seed 0 | Seed 1 |
|---|---|---|---|
| Sine, 512 | Shared rate | [Movie](ratio_primary_analysis/animations/N512_sine_high_shared/seed_0.html) | [Movie](ratio_primary_analysis/animations/N512_sine_high_shared/seed_1.html) |
| Sine, 512 | Both changes | [Movie](ratio_primary_analysis/animations/N512_sine_high_both_changes/seed_0.html) | [Movie](ratio_primary_analysis/animations/N512_sine_high_both_changes/seed_1.html) |
| Mixed, 1024 | Shared rate | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_shared/seed_0.html) | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_shared/seed_1.html) |
| Mixed, 1024 | Both changes | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_both_changes/seed_0.html) | [Movie](ratio_primary_analysis/animations/N1024_mixed_high_both_changes/seed_1.html) |

The full and magnified views have different vertical scales. Each view keeps its scales fixed through time and shared between its two seeds. Read the labeled physical units when comparing different interventions.

Slurm accounting records **28,069 GPU-seconds, or 7.79694 GPU-hours**, including compilation, I/O, the controlled cancellation used to replace slow compressed dense saves, and the final numerical audit. Allocation start/end times verify a maximum of two concurrent GPUs. All campaign jobs have ended. Remote analyses used zero-GPU Slurm allocations; local checks and evidence curation used CPUs. The 731 remaining GPU-seconds are unused budget, not an unreported allocation. Existing historical training is excluded from the new charge. Raw dense windows remain on the remote host; checkpoints, scalar traces, selected dense analysis exports, figures, and movies are retained locally.

### Reproducibility

- [Protocol and commands](../../../experiments/expD06_fixed_center_scales/README.md#paired-continuation-at-the-learned-geometry).
- [Exact 340k window statistics](ratio_campaign/340k_windows.json), derived from all 60 complete 320k–340k traces in `ratio_campaign/340k_scalar_inputs/`.
- [Exact 820k window statistics](ratio_campaign/820k_windows.json), with complete 800k–820k traces and 820k checkpoints in `ratio_campaign/820k_scalar_inputs/`.
- [Final primary evidence](ratio_primary_analysis/evidence.json), including complete-window summaries through the common 1.1m horizon and endpoint modal, moment, spectral, and motion diagnostics.
- [Complete joint-training export](ratio_final_analysis/evidence.json), covering all 102 acquisition, primary, early, feedback, and historical records. Its 84 acquisition/primary window records agree exactly with the separate primary export.
- [Weak-mode finite-update budgets](ratio_primary_analysis/weak_mode_loss_budget.json), including signed descent, quadratic costs, sample indices, input hashes, and lower-cutoff sensitivity.
- [Independent geometry-Jacobian check](ratio_primary_analysis/weak_mode_tangent_check.json), reproducing the signed weak-mode geometry descent from consecutive parameter states.
- [Matched frozen-solver window comparison](ratio_campaign/solver_140k_summary.json), including all 258 solves and warm-frozen versus joint training at absolute update 300k.
- [Final frozen-solver analysis](ratio_solver_analysis/solver_evidence.json), with a common prescribed-coordinate modal projector, both coordinate spectra, refits, evaluation counts, and complete-window summaries. Its window means agree exactly with the independent local scalar-trace reconstruction for all 258 solves.
- [Numerical-audit evidence](ratio_line_search_audit_analysis/evidence.json), with exact paired initial-state checks, input hashes, full-window means, validation diagnostics, and measured parameter displacement.
- [Early-handoff evidence](ratio_early_analysis/evidence.json), with all four runs analyzed at 340k and compared against the same-horizon shared controls.
- [Feedback comparison](ratio_campaign/feedback_400k_windows.json), computed from complete 380k–400k traces. Per-case controller ledgers and sampled counterfactuals are in `ratio_campaign/progress/feedback/`.
- [Historical comparison at 1.68m](ratio_campaign/historical_1680k_windows.json), using complete windows and the existing lower-rate continuation evidence. Earlier lower-rate training is excluded from this campaign's new GPU charge.
- [Acquisition evidence](ratio_acquisition_analysis/evidence.json), including endpoints, cutoff checks, sampling refinement, and finite-update audits.
- [340k mechanism evidence](ratio_340000_analysis/evidence.json), with per-case `dense_mechanism.npz` files for the modal and Fourier measurements. The original provenance's available common horizon is 360k; the explicit analysis cap and all primary records are 340k. Subsequent exports distinguish available and analyzed horizons explicitly.
- [Cited spectral-window values](ratio_340000_analysis/high_shared/mixed_N512_adam_both_envelope_s0_770e6899dded/spectral_window.json), including exact sample indices, energy denominators, band labels, and the source-array hash. The companion figure uses the same samples for every panel.
- [Regional readout contributions](ratio_340000_analysis/regional_readout_summary.json), reconstructed from the consecutive physical states and saved band gradients.
- [Fourier boundary-sensitivity check](ratio_340000_analysis/boundary_sensitivity.json), using the separately identified 340k endpoints and an optional Hann taper.
- [Uniform-reference diagnostics](ratio_uniform_reference/evidence.json), with the same fixed centers, reference metric, and sampling grid. Table 5 uses the prescribed-coordinate projector; coordinate-dependent truncations are not pooled.
- [Preconditioner algebra check](ratio_uniform_reference/preconditioner_algebra_check.json), verifying the implemented GD map to $2.8\times10^{-15}$ maximum absolute error and the sharp-step interior Gram identity on cell-midpoint samples. These check the algebra and ideal limit, not finite-slope convergence.
- [Acquisition animation, seed 0](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_0.html) and [seed 1](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_1.html), each showing $w$ above $\gamma$.
- [Allocation ledger](ratio_campaign/allocation_ledger.json) and [final Slurm accounting](ratio_campaign/final_slurm_accounting.txt). Remote source runs are under `/workspace/junmiaoh/experiments/precision-mlps/runs/ratios/`; the numerical audit is in the sibling `ratio_line_search_audit/` directory.

The complete repository test suite passes: **211 tests**, with seven existing deprecation warnings. Finite-update prediction closure and Fourier-gradient reconstruction were checked against direct calculations. Acquisition exports record maximum prediction-closure errors of order $10^{-15}$ in the example above. Numerical-loss guards pass synthetic FP64 checks on each GPU worker; local integration also verifies paired initialization, interrupted continuation, and recovery from a zero rate. Parameter movies were checked for frame counts, timing, offline playback assets, and physical-center labels. The implementation and report are committed on `experiment/fixed-center-scale-rates`; no changes were pushed.
