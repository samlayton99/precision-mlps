# Changing the relative readout and geometry rates

Does slowing geometry let the readout continue learning, or does it remove a useful route for reducing the residual? **At the first common horizon, 340k updates, none of the three late rate interventions improves the mean training MSE in any of the 12 width/target/seed cases.** Raising the readout rate often increases occasional error excursions while barely changing the median error. These are intermediate results from an eight-GPU-hour campaign; continued training, frozen-dictionary solvers, and the feedback policy are still running or queued. Two seeds establish paired observations, not a population-level significance claim.

**Table 1. Terminology used throughout this study.**

| Term or symbol | Meaning |
|---|---|
| Physical readouts $c=(b,w)$ | Output bias and coefficients multiplying the tanh features. |
| Physical slopes $\gamma$ | Slopes attached to the fixed physical centers. |
| Bandwidth $\lambda=h\gamma$ | Signed dimensionless slope; reported localization quantiles use $|\lambda|$. Here $h=2/N$. |
| Scaled training | Train $a,\lambda$ with $c=Da$ and $\gamma=\lambda/h$. $D$ is fixed at the theoretical reference bandwidth $0.25$. |
| Shared rate | The same scalar rate for $a$ and $\lambda$. The prescribed physical scales still differ. |
| Slower geometry | Change only the scalar geometry rate to one tenth of the shared control. |
| Faster readout | Change only the scalar readout rate to ten times the shared control. |
| Both changes | Apply slower geometry and faster readout together. |
| MSE | Mean squared residual. Training differentiates half-MSE; every error in this report is MSE. |
| Detached refit | SVD least-squares diagnostic at a saved geometry. It never replaces the trained readout. |
| $\tau$ | Relative singular-value cutoff used in a detached refit. |

## The comparison holds the theory scales fixed

Every joint-training branch uses the same fixed coordinate map $c=Da$, $\gamma=\lambda/h$, fixed centers, square-root halo radius, and FP64 arithmetic. The two acquisition histories use shared scalar rates $10^{-3}$ or $10^{-2}$ through 80k updates, followed by an 80k cosine decay to $10^{-6}$. Initialization is physical Xavier on slopes and the legacy signed reference-envelope draw on readouts. The latter is $w_j=\alpha_j\operatorname{sign}(\xi_j)$; it is **not** Gaussian Xavier on $a$.

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

## What the first matched window establishes

**Table 4. Mean training MSE over updates 320k–340k. Ranges span the two seeds. Intervention columns divide each seed's MSE by its own shared-rate control, then report the range; values above one are worse.**

| $N$, target | Shared-rate MSE | Slower geometry / shared | Faster readout / shared | Both / shared |
|---|---:|---:|---:|---:|
| 512, sine | $3.69$–$5.82\times10^{-11}$ | 1.006–1.007 | 2.992–4.138 | 3.042–4.144 |
| 512, quadratic | $6.83\times10^{-11}$–$1.01\times10^{-10}$ | 1.014–1.038 | 2.100–2.675 | 2.111–2.699 |
| 512, mixed | $4.98$–$9.72\times10^{-9}$ | 1.150–1.167 | 1.005–1.024 | 1.159–1.192 |
| 1024, sine | $6.18$–$6.52\times10^{-12}$ | 1.002–1.004 | 27.704–30.085 | 27.978–29.948 |
| 1024, quadratic | $8.48$–$8.52\times10^{-12}$ | 1.008–1.009 | 21.628–21.685 | 21.542–22.248 |
| 1024, mixed | $2.77$–$4.04\times10^{-10}$ | 1.045–1.054 | 1.442–1.650 | 1.481–1.692 |

The distinction between a typical step and occasional excursions matters. For sine at $N=1024$, seed 0, increasing the readout rate changes the median MSE from $5.25\times10^{-12}$ to $6.36\times10^{-12}$, but changes the mean from $6.52\times10^{-12}$ to $1.81\times10^{-10}$. Its 90th percentile rises from $8.20\times10^{-12}$ to $4.07\times10^{-10}$. An isolated endpoint can miss this behavior. The slower-geometry intervention gives essentially the same distribution as the shared control in this example; on the mixed target it is consistently worse at both widths.

This evidence does not support the simple explanation that geometry motion is the dominant obstruction and readout learning merely needs a larger rate. It does not yet identify which residual modes produce the excursions, or exclude benefits from an earlier handoff or another rate schedule. Those questions require the saved finite-update and modal measurements, and the remaining interventions.

## A small least-squares error does not establish easy readout optimization

The acquisition checkpoints already separate representability from iterative accessibility. At $N=512$, sine, seed 0, the higher acquisition rate produces median $|\lambda|=0.262$ and live validation MSE $5.38\times10^{-11}$. A detached refit can lower that error much further, but its coefficient norm grows sharply as weaker directions are admitted.

**Table 5. The same learned dictionary at update 160k, refitted with three singular-value cutoffs. Coefficient norms include the output bias. These are detached diagnostics, not trained models.**

| Relative cutoff $\tau$ | Retained rank | Refit validation MSE | Physical coefficient $\ell_1$ norm |
|---:|---:|---:|---:|
| $10^{-10}$ | 455 | $6.32\times10^{-15}$ | 101.6 |
| $10^{-12}$ | 496 | $7.52\times10^{-18}$ | 980.0 |
| $10^{-14}$ | 518 | $2.04\times10^{-19}$ | 13,104 |

Doubling the fitting-grid density gives closely matching refit errors, while the weakest coefficient norms remain more sensitive. Thus the dictionary can represent a much smaller residual, but reaching that residual can require large coefficients in weak singular directions. Also, localized tanh derivatives do not make the tanh feature columns themselves disjoint: each feature still approaches constant tails. The frozen-dictionary comparison will test GD, momentum GD, and Adam in the prescribed coordinates and in fixed neighbor-difference coordinates that preserve the represented function space. It uses first-order updates and loss-only line searches, never a least-squares training update.

At this acquisition checkpoint, the median $\sqrt{\hat v}/\epsilon$ is about 54 for readouts and 0.0094 for geometry. Most geometry coordinates are already epsilon dominated; most readout coordinates are not. Consequently, multiplying a nominal Adam rate need not create proportionate useful motion in the two blocks. These moment measurements motivate inspecting actual updates; they do not by themselves explain the plateau.

## Evidence and completion status

All 60 primary branches completed the 340k comparison and continue beyond it. The frozen-dictionary block, earlier geometry handoffs, feedback rule, and historical higher-rate continuations remain pending in this interim report. No branch is declared converged merely because a budget limit or reporting horizon was reached.

Saved evidence includes complete per-step scalar traces; 20k checkpoints with parameters and Adam moments; dense physical parameters, signed gradients, and actual updates; Fourier-band residuals and forces; fixed-basis singular-mode projections; core/halo contributions; coefficient norms; and exact readout/geometry/interaction contributions to finite-step MSE changes. Detached fits use three cutoffs and a doubled sampling density. Per-seed movies place physical $w$ above physical $\gamma$ at fixed centers and slow the first 320k checkpoints.

The eight-GPU-hour limit includes compilation, I/O, and the controlled cancellation used to replace slow compressed dense saves with lossless uncompressed saves. At most two GPUs are allocated concurrently. Analyses run in zero-GPU Slurm allocations. Actual allocation charges will be reconciled before the final report; reserved and completed allocations are distinguished in the ledger.

### Reproducibility

- [Protocol and commands](../../../experiments/expD06_fixed_center_scales/README.md#paired-continuation-at-the-learned-geometry).
- [Exact 340k window statistics](ratio_campaign/340k_windows.json), derived from all 60 complete 320k–340k traces in `ratio_campaign/340k_scalar_inputs/`.
- [Acquisition evidence](ratio_acquisition_analysis/evidence.json), including endpoints, cutoff checks, sampling refinement, and finite-update audits.
- [Acquisition animation, seed 0](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_0.html) and [seed 1](ratio_acquisition_analysis/animations/N512_sine_acquire_0.01/seed_1.html), each showing $w$ above $\gamma$.
- [Allocation ledger](ratio_campaign/allocation_ledger.json). Remote source runs are under `/workspace/junmiaoh/experiments/precision-mlps/runs/ratios/`.

The complete repository test suite passes: **208 tests**, with seven existing deprecation warnings. Finite-update prediction closure and Fourier-gradient reconstruction were checked against direct calculations. Acquisition exports record maximum prediction-closure errors of order $10^{-15}$ in the example above. Final campaign validation and allocation accounting remain to be appended with the completed evidence.
