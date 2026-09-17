# Readout scales, bandwidth motion, and sustained error

Xavier initialization on $a$, with $w=\sqrt h\,a$ and physical Xavier slopes, does acquire median bandwidths near the proposed regime: 0.303 and 0.278 after 320k updates in two fresh seeds. It does not achieve sustained high precision: training-window RMS remains about $6.2\times10^{-4}$, with persistent oscillation. Changing only Adam's epsilon convention gives smaller bandwidths but vastly better detached readout fits at essentially the same trained error. The completed 88-run crossed comparison and 544-case pilot therefore support bandwidth acquisition through scale-aware rates, while distinguishing it from useful geometry and successful joint optimization. These are fixed-center results on one sine target at $N=512$; width transfer and an exponential spectral law remain unestablished.

**Table 1. Definitions and normalization used throughout this report.**

| Quantity | Meaning |
|---|---|
| $N,h,R,W$ | Core resolution 512, spacing $h=2/N$, halo radius $R=\lceil\sqrt N\rceil=23$, and neuron count $W=559$ |
| $\lambda=h\gamma$ | Signed slope in grid units; reported quantiles use $\lvert\lambda\rvert$ |
| Raw / uniform / reference map | Train physical $(w,\gamma)$ / $(a,\lambda)$ with $w=\sqrt h\,a$ / $(a,\lambda)$ with $w=D_{\rm ref}a$ |
| Window RMS | Square root of the mean squared residual over samples and a stated training-time window |
| Sampled validation window | RMS over the independent validation grid and 20 saved iterates, spaced 1,000 updates apart, in a 20k window |
| Detached refit | Readout least-squares solve at saved slopes; it never changes the training trajectory |
| $P_\tau$, $r_\perp$ | Retained feature-space projector at relative SVD cutoff $\tau$, and $(I-P_\tau)r$ |
| Net / accumulated motion | RMS per-neuron displacement / RMS per-neuron sum of absolute updates over the same interval |
| Signed outward force | Negative gradient dotted with a unit direction that increases slope magnitudes in the stated region |

## Model and the coordinate identity

The model is $f(x)=b+\sum_j w_j\tanh(\gamma_j(x-x_j))$. Every center, including every halo center, is fixed. Readouts and signed slopes always train together on one half mean-squared-error objective. Training uses 8,193 endpoint-inclusive samples; validation uses an independent 32,768-point midpoint grid. The 65,536-point test grid remains unused. Targets, geometry, reference allowances, and diagnostic conventions follow the [experiment protocol](../../../experiments/expD06_fixed_center_scales/README.md) and the supplied [scale note](../../../docs/correcting_scales.md), [gamma-barrier handoff](../../../docs/gamma_barrier_handoff_v2.pdf), and [readout-scale note](../../../docs/readout_scale_collaborator_note.pdf).

For a positive, fixed coordinate scale $w=sa$, gradients transform as $g_a=sg_w$. Adam's moments consequently satisfy $m_a=sm_w$ and $v_a=s^2v_w$ when initialized consistently. Its physical update is

$$
\Delta w=-\eta_a s\,\frac{\widehat m_w}{\sqrt{\widehat v_w}+\epsilon_a/s}.
$$

Thus $w=\sqrt h\,a$ and $\gamma=\lambda/h$ give physical Adam rate factors $\eta_a\sqrt h$ and $\eta_\lambda/h$, together with physical epsilons $\epsilon_a/\sqrt h$ and $\epsilon_\lambda h$. This is the precise sense in which the proposed rates follow from reparametrization. GD instead gives factors $\eta_a h$ and $\eta_\lambda/h^2$. A coordinate map by itself does not introduce a distinct optimizer when the physical rates, epsilons, initial state, and moment state are matched. Multi-step tests verify this identity with nonzero Adam moments.

The four readout families use the same Gaussian draws within a seed. Physical Xavier draws $w_j=\sqrt{2/(W+1)}\,\xi_j$. Uniform-coordinate Xavier multiplies this draw by $\sqrt h$; reference-coordinate Xavier multiplies it by $d_j=\sqrt{\alpha_j}$. The legacy envelope draw is $\alpha_j\operatorname{sign}(\xi_j)$, which is a different distribution. All four use physical Xavier slopes with standard deviation $(5/3)\sqrt{2/(W+1)}$. Initial slope signs are absorbed into readouts without changing the function; later slopes remain free to cross zero. Bias starts at zero.

## What the sustained pilot changes

The pilot contains 544 configurations, not 544 independent replications: 292 GD and 252 Adam cases, with seeds 0 and 1. There are 59 nonfinite GD failures, 483 finite trajectories with a 320k checkpoint, and two poor-fit stationary GD trajectories ending at 160k. Of the 483, 188 also reached 640k. The complete finite traces are retained. The numerical failures are included in rate-landscape accounting rather than discarded as missing data.

The following rates minimize the paired geometric mean of training RMS over updates 300k–320k within each listed group. These are descriptive pilot rankings. The shared-rate Raw controls had a smaller declared grid than the tuned methods, and two seeds do not support a reliable uncertainty estimate.

**Table 2. Sustained pilot performance at 320k; seed values are shown separately. The two Raw rows with distinct rates use tuned physical-coordinate updates.**

| Optimizer / map / readout initialization | Readout LR, slope LR | Late training-window RMS, seeds 0 / 1 | Median $\lvert\lambda\rvert$, seeds 0 / 1 | Detached validation refit, seeds 0 / 1 |
|---|---|---|---|---|
| Adam / reference / physical Xavier | $10^{-4},10^{-2}$ | $7.03\times10^{-4}$ / $9.64\times10^{-4}$ | 0.203 / 0.198 | $2.86\times10^{-10}$ / $2.34\times10^{-10}$ |
| Adam / Raw / physical Xavier | $9.31\times10^{-6},2.56$ | $1.34\times10^{-3}$ / $4.78\times10^{-4}$ | 0.149 / 0.180 | $3.43\times10^{-11}$ / $2.03\times10^{-11}$ |
| Adam / reference / envelope | $10^{-4},10^{-2}$ | $1.48\times10^{-4}$ / $1.52\times10^{-4}$ | 0.322 / 0.351 | $2.25\times10^{-7}$ / $1.97\times10^{-7}$ |
| Adam / Raw / envelope | $9.31\times10^{-6},2.56$ | $9.94\times10^{-5}$ / $1.03\times10^{-4}$ | 0.115 / 0.110 | $2.39\times10^{-11}$ / $1.87\times10^{-9}$ |
| Adam / Raw shared LR / physical Xavier | $10^{-3},10^{-3}$ | 0.0122 / 0.0120 | $2.72\times10^{-4}$ / $4.69\times10^{-4}$ | $1.22\times10^{-6}$ / $5.40\times10^{-6}$ |
| GD / reference / physical Xavier | $0.1,0.01$ | $3.80\times10^{-3}$ / $3.31\times10^{-3}$ | 0.0409 / 0.0476 | $1.64\times10^{-11}$ / $1.55\times10^{-11}$ |

The reference-map Adam/Xavier endpoint-selected pair, $(0.01,0.01)$, reaches validation RMS around $2\times10^{-4}$ at 320k but has late-window RMS around $0.011$. Its late-window error remains near that level at 640k. The sustained ranking instead selects readout LR $10^{-4}$ at the same slope LR. For envelope initialization, the endpoint-selected reference pair uses readout LR $10^{-3}$ and has late-window RMS around $0.0011$; the sustained choice again lowers readout LR to $10^{-4}$ and reaches approximately $0.00015$. Endpoint rank and sustained rank answer different questions.

<figure>
  <img src="pilot_consolidation/figures/lr_windows.png" alt="Sustained training error across the pilot readout and slope learning-rate search, with numerical failures marked" style="max-width: 100%;">
  <figcaption>Paired geometric mean of training RMS over updates 300k–320k, with separate optimizer, map, and initialization panels. Axes use each map's trained coordinates; Raw's large slope rates include the declared scale matching. Red crosses mark numerical failures in either seed. The plot shows the full search, including poor and unstable regions, rather than only endpoint-selected trajectories.</figcaption>
</figure>

The legacy reference map also rescales the bias. Its readout Hessian has largest eigenvalue at least $\alpha_b=10.76391$, giving the necessary fixed-geometry GD condition $\eta_r<2/\alpha_b\approx0.1858$. The added rates 0.3 and 1 violate this condition before considering joint geometry dynamics. The focused comparison therefore keeps the physical bias update identical across maps; otherwise a reported halo or coordinate effect could include this bias confound.

The current geometry frequently admits a far more accurate readout fit than joint training attains. In the sustained reference-map Xavier example, detached RMS is around $10^{-10}$ while live window RMS is around $10^{-3}$. Its median bandwidth is about 0.2. The envelope example grows beyond the reference 0.25 but has a worse detached fit, around $2\times10^{-7}$. The fixed-$\lambda=0.25$ reference checks achieve roughly $10^{-13}$ across the nine planned width/target combinations, but those fixed geometries are not learned trajectories. A median bandwidth near 0.25 is therefore insufficient evidence that training has acquired the reference geometry.

Refitted coefficient size matters as well. The seed-0 reference/Xavier solve has physical readout $\ell_1$ norm about 49, while the reference/envelope solve requires about 27,474. A successful detached fit is an unconstrained diagnostic of the retained feature space; it is not evidence that the live optimizer can reach that readout at a useful scale.

<figure>
  <img src="pilot_consolidation/figures/fit_geometry_motion.png" alt="Pilot endpoint error versus window error, detached refit error versus median bandwidth, and net versus accumulated bandwidth motion" style="max-width: 100%;">
  <figcaption>All finite pilot cases with a 320k checkpoint. Endpoint error can be substantially below the sustained training error; detached fit quality is not monotone in median bandwidth. Accumulated motion can greatly exceed net displacement. These are descriptive comparisons across rates, initializations, and optimizers.</figcaption>
</figure>

Between 160k and 320k, the sustained reference-map Adam/Xavier seed-0 trajectory accumulates bandwidth-motion RMS 50.9 but moves only 0.0436 net; the envelope counterpart accumulates 36.2 and moves 0.219 net. These quantities have the same per-neuron RMS convention. Persistent updates do not necessarily represent productive travel. Conversely, GD/Xavier median bandwidth changes little while its late-window error improves from about 0.0035 at 320k to 0.0021 at 640k. Neither a moving slope distribution nor a nearly fixed one alone determines whether further training is useful.

## Focused comparison and measurement policy

The focused manifest contains 88 Adam trajectories with fresh seeds 2 and 3. Its 48 primary conditions cross four readout initializations with the uniform and reference training maps at uniform-coordinate rate pairs $(10^{-4},10^{-3})$, $(10^{-3},10^{-3})$, and $(10^{-3},10^{-2})$. Reference readout rates are multiplied by $\sqrt{h/\alpha_{\rm ordinary}}$ so ordinary physical update factors match. Bias is trained directly in physical coordinates, with identical rate $\eta_a\sqrt h$ across the two maps. The resulting primary-map differences are confined to corrected-halo physical rate factors and epsilons.

The primary epsilon convention anchors the uniform map at $10^{-8}$ and adjusts reference readout epsilon to match ordinary physical epsilon. At the faster geometry pair $(10^{-3},10^{-2})$, 16 additional trajectories instead fix every physical epsilon at $10^{-8}$. The remaining 24 trajectories train in physical coordinates with shared numerical rates $10^{-4},10^{-3},10^{-2}$, across all four initialization families. These are fixed-center null controls; a conventional MLP with trainable hidden biases/centers is not part of this comparison.

The two Gaussian coordinate initializers change both the ordinary readout scale and the halo profile. Their difference alone does not identify a halo-initialization effect. In contrast, comparing training maps at identical physical initialization and matched ordinary/bias settings isolates the corrected-halo optimizer settings within this experiment.

Every finite focused run targets the same 320k horizon, with constant rates and saved frontiers at 20k, 80k, 160k, and 320k. It records every training-step loss and 61 validation samples from 260k through 320k. Results distinguish endpoint error, each of the three final 20k training windows, sampled validation windows, feature-fit quality, and net geometry movement. A completed allocation or common training horizon is not a convergence claim. Sampling every 1,000 iterations can alias oscillations; the complete training trace remains the check against a favorable validation phase.

## Focused results: acquiring bandwidth versus acquiring a useful fit

All 88 trajectories completed 320k with finite, complete traces and all 61 validation samples. All remain labeled `continuing` by the settling criteria. The following comparison holds physical initialization fixed at Gaussian Xavier on $a$ with $w=\sqrt h\,a$, and physical Xavier on $\gamma$. The primary epsilon convention is used except in the explicit physical-epsilon control.

**Table 3. Exact proposed initialization, seeds 2 / 3, at 320k. Uniform-map rates are $(\eta_a,\eta_\lambda)$; Raw uses $(\eta_w,\eta_\gamma)$. Window RMS uses every update in 300k–320k.**

| Training map / epsilon | Rates | Training-window RMS | Median $\lvert\lambda\rvert$ | Detached validation refit RMS |
|---|---|---|---|---|
| Uniform / native | $10^{-4},10^{-3}$ | 0.00767 / 0.00710 | 0.00999 / 0.0114 | $1.01\times10^{-10}$ / $1.15\times10^{-10}$ |
| Uniform / native | $10^{-3},10^{-3}$ | $6.65\times10^{-4}$ / $6.41\times10^{-4}$ | 0.0434 / 0.0491 | $1.34\times10^{-11}$ / $3.69\times10^{-11}$ |
| Uniform / native | $10^{-3},10^{-2}$ | $6.20\times10^{-4}$ / $6.17\times10^{-4}$ | 0.303 / 0.278 | $8.66\times10^{-7}$ / $6.96\times10^{-7}$ |
| Uniform / physical $\epsilon=10^{-8}$ | $10^{-3},10^{-2}$ | $6.20\times10^{-4}$ / $6.25\times10^{-4}$ | 0.140 / 0.132 | $4.85\times10^{-12}$ / $2.59\times10^{-12}$ |
| Raw / shared LR | $10^{-4},10^{-4}$ | 0.0190 / 0.0171 | 0.000903 / 0.000937 | $1.26\times10^{-7}$ / $1.90\times10^{-7}$ |

The displayed Raw pair has the lowest paired complete-window error among its three declared shared rates. Its physical update factors, including the bias rate, differ from the mapped runs. It is a null for the full learning-rate prescription, not an isolated test of coordinate representation; the coordinate-equivalence test supplies that separate check.

The faster-geometry primary pair continues moving well beyond 20k: median bandwidth increases from 0.129 / 0.149 at 20k to 0.259 / 0.247 at 160k and 0.303 / 0.278 at 320k. This supports the proposed acquisition mechanism in the limited sense of continued scale movement from Xavier slopes. It does not establish a uniform $\lambda=0.25$ geometry: the final largest bandwidths are 1.89 / 1.49. Between 160k and 320k, net bandwidth-motion RMS is 0.101 / 0.0745, while accumulated motion is 17.7 / 22.1. Most traveled distance is therefore not retained displacement.

The learning-rate comparison is not monotone in readout slowdown. At fixed $\eta_\lambda=10^{-3}$, reducing $\eta_a$ from $10^{-3}$ to $10^{-4}$ produces smaller bandwidths and about eleven times worse sustained error at this horizon. Increasing geometry LR tenfold at fixed $\eta_a=10^{-3}$ produces much larger bandwidths, with only a small change in sustained error and a much worse detached fit. The slowest and fastest pairs have the same numerical rate ratio but very different outcomes. Absolute rates and the finite training horizon matter alongside their ratio.

<figure>
  <img src="focused_consolidation/figures/focused_xavier_a_trajectories.png" alt="Bandwidth and complete-window training-error trajectories for the proposed Xavier initialization under different rates and epsilon conventions" style="max-width: 100%;">
  <figcaption>Physical initialization is matched within each seed. Lines show paired geometric means; shading spans the two seed values and is not a confidence interval. The native fast-geometry run keeps acquiring bandwidth after its sustained error has flattened. The physical-epsilon control has almost the same error curve with much less bandwidth growth. All displayed cases reached 320k; none is certified converged.</figcaption>
</figure>

The epsilon control keeps the physical rate factors unchanged. For the uniform map it changes neuron-readout physical epsilon from $1.6\times10^{-7}$ to $10^{-8}$ and physical-slope epsilon from $3.90625\times10^{-11}$ to $10^{-8}$. Its final refits need physical readout $\ell_1$ norms only 9.07 / 7.59, compared with 20,078 / 28,197 in the native fast-geometry runs. This substantial difference in geometry and coefficient size survives despite nearly identical sustained trained error. Both blocks' epsilons change in this control; their individual causal contributions have not been separated.

Initialization also trades off these metrics. At the fast primary pair, physical-Xavier readouts under the uniform map have sustained RMS $1.01\times10^{-3}$ / $9.34\times10^{-4}$, median bandwidth 0.177 / 0.192, and detached errors around $2\times10^{-10}$. Both Gaussian scaled-readout initializers and the signed-envelope initializer reach sustained error around $6.2\times10^{-4}$, median bandwidth around 0.28–0.33, and worse detached errors around $3\times10^{-7}$–$9\times10^{-7}$. No single initialization dominates trained error and feature-fit quality in these settings.

<figure>
  <img src="focused_consolidation/figures/focused_conditions.png" alt="All primary initialization and coordinate-map conditions compared by complete-window training RMS, median bandwidth, and detached refit quality" style="max-width: 100%;">
  <figcaption>All 48 primary trajectories, aggregated as two-seed geometric means, with no selection by refitted error. Each map has the same three ordinary-neuron physical rate factors and the same physical bias settings. Figure label “both” denotes the reference map. The first panel deliberately uses complete training windows because 1,000-step validation sampling can alias oscillations; sampled validation results remain in the data files.</figcaption>
</figure>

For the three smaller-readout families at the fast pair, the reference halo metric raises sustained RMS from about $0.00062$ to $0.00069$, roughly 11%, while median bandwidth remains around 0.3. Its effect changes with the rate pair and initialization elsewhere in the matrix. There is no general benefit from the reference halo metric in this comparison. The radius itself is fixed throughout, so this is evidence about halo optimizer settings, not halo-radius optimality.

The sampled validation windows demonstrate why no final LR lock is justified. For the fast uniform/Xavier-on-$a$ seed-2 run, sampled validation RMS falls from 0.00116 to 0.000669 to 0.000363 across the last three windows, while complete training-window RMS stays at 0.000619, 0.000620, and 0.000620. These validation samples do not establish that the sustained error improved. The [saved rankings](focused_consolidation/choices.json) retain endpoint, complete-training-window, and sampled-validation choices separately; the latter are provisional optimization evidence, not held-out evaluation.

## Residuals, Fourier forces, and the mechanism

The diagnostics use sample-normalized residuals and feature/Jacobian columns. A single truncated SVD of $AD_{\rm ref}$ defines the detached refit and retained projector, with primary cutoff $\tau=10^{-12}$. The reference metric used for diagnostics is identical across training maps. For each orthogonal Fourier band $Q_b$, the saved gradients are

$$
g_b=J_\lambda^TQ_b r,\qquad
g_{\parallel,b}=(P_\tau J_\lambda)^TQ_b r,\qquad
g_{\perp,b}=((I-P_\tau)J_\lambda)^TQ_b r.
$$

These reconstruct both within each band and across bands without assuming that $P_\tau$ commutes with $Q_b$. In particular, an individual band's perpendicular force may be much larger than the signed sum over all bands. A norm of band magnitudes would hide this cancellation. The actual Adam update is retained separately: preconditioning and momentum mean it need not point along the instantaneous collective negative gradient. Its per-band first-order loss reduction is $-g_b^T\Delta\lambda$, with a separate readout term. These are linearized contributions, not an exact partition of nonlinear loss reduction; the measured prediction change includes the readout–geometry interaction.

In the pilot's sustained reference-map Adam/Xavier seed-0 example, live residual RMS at 320k is $3.89\times10^{-4}$, whereas the residual outside the retained readout space is $2.85\times10^{-10}$. Parallel geometry-gradient norm is $6.42\times10^{-5}$; perpendicular norm is $3.94\times10^{-16}$. The current geometry can therefore remove almost all of the live error through readout adjustment at this cutoff. The largest perpendicular residual band moves from DFT indices 4–7 at initialization to 128–255 at 20k and remains there at 320k; its amplitude first falls and then increases. The live residual has a different dominant band. There is no single monotone “residual frequency” that summarizes both objects.

<figure>
  <img src="pilot_consolidation/figures/spectral_sine_N512_adam_both_xavier_s0_e4ac78923453.png" alt="Time evolution of live, perpendicular, and refitted residual spectra, with signed band forces and update contributions" style="max-width: 100%;">
  <figcaption>Pilot Adam/reference/physical-Xavier, seed 0, at the rate pair selected by both seeds' 300k–320k training windows. The top row uses a common log-scale band RMS. The lower row shows signed forces and actual-update linearized descent at 320k. Perpendicular bands cancel strongly, and their tiny signed total is cutoff- and roundoff-sensitive. Region forces each use a unit direction within that region, so their magnitudes are not additive regional totals.</figcaption>
</figure>

The decomposition supports a weak out-of-span driving signal in these trajectories. It also shows that most ongoing geometry motion is associated with the much larger residual still accessible to readouts, with substantial oscillation. It does not identify exponential spectral attenuation as the causal bottleneck. Unit-RMS sine/cosine probes, raw/projected tangent norms, signed regional responses, and Hann-window spectra are retained to examine that hypothesis separately from residual amplitude. Finite-interval boundary leakage, heterogeneous slopes, cutoff dependence, and oscillatory Adam updates prevent an exponential-law claim from these plots alone.

Adam epsilon is measurably active in the pilot. At the sustained reference/envelope seed-0 endpoint, 52.2% of slope coordinates have $\sqrt{\widehat v}<\epsilon$; the tuned Raw/envelope counterpart has 100%. The reference/Xavier counterpart has 11.4%. These fractions use each run's stored-coordinate moments and epsilon. This is why the focused physical-epsilon controls are necessary: a late-time slowdown can reflect the denominator convention as well as residual depletion.

At the Xavier endpoint above, changing the diagnostic SVD cutoff from $10^{-10}$ to $10^{-14}$ changes perpendicular-gradient norm from $6.43\times10^{-14}$ to $7.39\times10^{-18}$ and detached validation RMS from $5.32\times10^{-9}$ to $2.27\times10^{-11}$. The large live/refitted gap survives; the tiny perpendicular magnitude and sign do not have the same robustness. The [cutoff records](pilot_consolidation/cutoff_sensitivity.json) preserve these distinctions.

The focused experiment shows the same distinction over time. In the native fast-geometry Xavier-on-$a$ seed-2 trajectory, perpendicular residual RMS rises from $3.64\times10^{-12}$ at 20k to $8.66\times10^{-7}$ at 320k while bandwidth keeps growing. In its physical-epsilon control, it stays around $5\times10^{-12}$. At 320k, 70.5% versus 100% of slope coordinates are epsilon-dominated. The native trajectory's instantaneous collective outward gradient force is positive, $2.35\times10^{-8}$, while its actual Adam update in that direction is negative, $-4.76\times10^{-5}$. This is one observed phase, not a systematic sign claim; it demonstrates why raw gradient norms alone do not explain the trajectory.

The focused native-versus-physical-epsilon refit gap also survives the outer SVD cutoffs. For seed 2, native refitted RMS ranges from $1.08\times10^{-6}$ at $\tau=10^{-10}$ to $6.85\times10^{-7}$ at $10^{-14}$; the control ranges from $3.47\times10^{-10}$ to $8.97\times10^{-14}$. The exact accuracy remains cutoff-dependent, but the relative geometry-quality difference is substantial throughout. The [focused spectral histories](focused_consolidation/spectral_history.json), [regional mechanism records](focused_consolidation/mechanism.json), and [cutoff checks](focused_consolidation/cutoff_sensitivity.json) preserve both seeds.

## Verification, provenance, and limits

The full repository suite passes: **188 tests**, including the slow test, with seven pre-existing deprecation warnings. The 26 experiment checks present at launch also passed in the remote environment before Slurm released the GPU dependency. Subsequent verification covers signed spectral exports as well. Tests check initialization pairing, fixed centers, GD/Adam coordinate identities, nonzero-moment Adam equivalence, exact checkpoint resume, batch independence, FP64 states, Fourier reconstruction, and the minimum scientific duration. Diagnostic code never replaces live readouts.

The data audit verifies complete finite traces for all 485 finite pilot cases and all 88 focused cases. Every floating array in the 483 pilot and 88 focused 320k checkpoints is FP64, including saved moments, gradients, predictions, and proposed updates. All focused cases contain exactly the 61 planned late validation samples. Across the saved pilot and focused diagnostic records, maximum CPU/GPU prediction discrepancy is $1.19\times10^{-13}$; maximum full-gradient Fourier reconstruction error is $2.26\times10^{-12}$; residual-energy reconstruction error is at most $8.53\times10^{-14}$; measured prediction-update closure error is at most $5.50\times10^{-14}$. These are absolute errors, not universal relative tolerances for tiny forces. Their detailed values are retained in the [pilot audit](pilot_consolidation/numerical_audit.json) and [focused audit](focused_consolidation/numerical_audit.json).

Training used remote JAX 0.10.2 and Optax 0.2.8 with FP64 enabled. Slurm array 300 ran two independent one-GPU workers; each runtime saw one allocated CUDA device and preserved the scheduler mask. Both jobs exited successfully after 17 minutes 36 seconds, totaling **0.5867 GPU-hours**, below the focused block's two-GPU-hour cap. CPU-only allocations performed verification and detached analysis; the last analysis completed successfully as job 303. No experiment job remains active. The pilot worker ledgers total 5.927 GPU-hours; with allocation overhead, this focused block, and a conservative 0.2-hour setup reserve, campaign accounting is approximately 6.72 of the authorized 24 GPU-hours.

Training implementation is recorded in commit `1698c20`; sustained-loss consolidation began in `e0bc90e`, and full-trace/spectral auditing in `12dd9ca`. The final reporting slice adds inspected display refinements without changing trained trajectories. Local [Slurm accounting](run_provenance/slurm_accounting.txt), [seed-2 environment](run_provenance/environment_focused_301.json), [seed-3 environment](run_provenance/environment_focused_300.json), manifests, logs, and budget ledgers preserve execution provenance. The saved training-module hashes match the local implementation. Full optimizer states and detailed per-checkpoint arrays remain under `/workspace/junmiaoh/experiments/precision-mlps/runs/{pilot,focused}`. Locally, the [pilot summary](pilot_consolidation/summary.csv), [focused summary](focused_consolidation/summary.csv), [window records](focused_consolidation/windows.json), and [epsilon contrasts](focused_consolidation/epsilon_contrasts.json) support the report's numbers. Reproduction commands and metric conventions are in the [experiment README](../../../experiments/expD06_fixed_center_scales/README.md).

This completes the bounded consolidation and crossed experiment, not the broader scientific question. Width and target transfer, a matched trainable-center MLP null, independent readout-versus-slope epsilon controls, and a causal exponential-barrier test have not been run. No test-grid result, converged precision solution, or unique advantage of coordinate changes is claimed. The strongest supported conclusion is that the proposed scaled Xavier setup acquires the desired central bandwidth range, while epsilon and sustained optimizer oscillations determine whether that movement preserves useful geometry or improves the trained function.
