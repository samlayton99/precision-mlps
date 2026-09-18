# Shared scale prescription: geometry, oscillation, and the remaining readout error

**The prescribed scaling acquires features that represent this target accurately, but the readout problem remains ill-conditioned. Decaying the one shared Adam learning rate reduces sustained MSE from about $10^{-6}$ to $10^{-10}$. Freezing geometry alone does not remove the oscillation floor.** With decay, joint training outperforms frozen geometry, and its remaining residual concentrates in weak readout singular directions. Both step size and slow fitting of those directions matter. Continued geometry movement is mostly driven by readout-accessible error; these results do not establish sustained learning driven by the out-of-reach residual. This evidence covers one sine target, one width, and two seeds. The decayed runs are still improving.

**Table 1. Terminology and normalization used throughout this report.**

| Term | Meaning |
|---|---|
| Unscaled training | Update physical readouts, output bias, and slopes directly with one shared base rate $\eta$ |
| Scaled training | Update $\mathbf a=D^{-1}\mathbf c$ and $\boldsymbol\lambda=h\boldsymbol\gamma$, both with the same $\eta$; $D$ and $h$ supply the prescribed scales |
| $\mathbf c=(b,w_1,\ldots,w_W)$ | Physical output bias and neuron readout coefficients |
| $\gamma_j$, $\lambda_j=h\gamma_j$ | Physical slope and dimensionless bandwidth; distribution plots show $\lvert\lambda\rvert$ |
| MSE | Mean squared prediction error; the optimizer minimizes **half-MSE**, and RMS is $\sqrt{\mathrm{MSE}}$ |
| Window MSE | Mean training MSE over **every update** in the stated 20,000-update interval |
| Detached refit | Truncated-SVD readout solve at saved geometry; its coefficients are never fed into training |
| Readout-accessible / out-of-span | Components under the retained SVD projector $P_\tau$ and its complement; this is a numerical span at cutoff $\tau$ |

Older source files call scaled training `both` or “reparametrized,” and unscaled training `raw`. Those are source aliases, not additional methods.

## The experiment and its single shared learning rate

The network is

$$
f(x)=b+\sum_{j=1}^{W}w_j\tanh\bigl(\gamma_j(x-x_j)\bigr),\qquad
\mathbf c=D\mathbf a,\quad \boldsymbol\gamma=\boldsymbol\lambda/h,
\qquad \boxed{\eta_a=\eta_\lambda=\eta}.
$$

Centers are fixed. The target is $\sqrt2\sin(2\pi x)$ on $[-1,1]$, with $N=512$, spacing $h=1/256$, halo radius $R=\lceil\sqrt N\rceil=23$, and $W=559$ neurons. Full-batch FP64 training uses 8,193 endpoint-inclusive samples. The 32,768-point midpoint grid is diagnostic validation; the final test grid remains unused. These are optimization results, not a final held-out evaluation.

The [scale note](../../../docs/correcting_scales.md) defines fixed reference allowances $\alpha_j$ and $d_j=\sqrt{\alpha_j}$ for $D$, including bias and corrected halo slots, evaluated at $\lambda_{\rm ref}=0.25$. Physical slopes start from tanh Xavier with standard deviation $(5/3)\sqrt{2/(W+1)}$. Physical readouts start at $w_j=\alpha_j\operatorname{sign}(\xi_j)$ with independent Gaussian draws; bias starts at zero. Initial slope signs are absorbed into readouts without changing the function. This **reference-envelope initialization is not Gaussian Xavier on $a$**. Initialization is held fixed here: scaled and unscaled training start from identical physical networks for each seed.

Adam uses moments $(0.9,0.999)$ and trained-coordinate epsilon $10^{-8}$. GD is full-batch gradient descent without momentum. For coordinate scale $s$, GD supplies physical rate $\eta s^2$; Adam supplies physical prefactor $\eta s$ and physical epsilon $10^{-8}/s$. Adam prefactors multiply normalized moments and are not observed displacements.

**Table 2. Physical rates induced by the shared prescription at this width, including bias and halo.**

| Physical parameter | Unscaled Adam, $\eta=10^{-3}$ | Scaled Adam, $\eta=10^{-3}$ | Unscaled GD, $\eta=10^{-2}$ | Scaled GD, $\eta=10^{-2}$ |
|---|---:|---:|---:|---:|
| Ordinary readout, including uncorrected halo | $10^{-3}$ | $9.3075\times10^{-5}$ | $10^{-2}$ | $8.6630\times10^{-5}$ |
| Corrected halo readout | $10^{-3}$ | $9.3075\times10^{-5}$–$1.0224\times10^{-3}$ | $10^{-2}$ | $8.6630\times10^{-5}$–$1.0452\times10^{-2}$ |
| Output bias | $10^{-3}$ | $3.2808\times10^{-3}$ | $10^{-2}$ | $0.10764$ |
| Slope $\gamma$ | $10^{-3}$ | $0.256$ | $10^{-2}$ | $655.36$ |

Scaled Adam's physical epsilons are $1.0744\times10^{-7}$ on ordinary readouts, $3.0480\times10^{-9}$ on bias, and $3.90625\times10^{-11}$ on slopes, versus $10^{-8}$ throughout unscaled Adam. The decay below ends at $\eta=10^{-6}$, dividing all scaled Adam prefactors by 1,000 while keeping their ratios and epsilons fixed. [Per-coordinate records](stall_analysis/evidence.json) retain all 560 readout/bias entries.

The earlier $(10^{-4},10^{-2})$ run **did use $D$**: those were unequal coordinate rates $(\eta_a,\eta_\lambda)$. Its ordinary-readout, bias, and slope Adam prefactors were $9.3075\times10^{-6}$, $3.2808\times10^{-4}$, and $2.56$. A separate historical unscaled run used physical rates $9.3075\times10^{-6}$ and $2.56$ without $D$. Neither implements the one-shared-rate prescription.

## Adam and GD baselines at 320,000 updates

**Table 3. Complete training-window MSE over updates 300k–320k, seed 0 / seed 1. No endpoint selection or detached fitting is included.**

| Optimizer and training method | Shared $\eta$ | Window MSE |
|---|---:|---:|
| Adam, unscaled training | $10^{-3}$ | $9.70\times10^{-4}$ / $1.70\times10^{-4}$ |
| Adam, scaled training | $10^{-3}$ | $4.50\times10^{-6}$ / $1.23\times10^{-6}$ |
| Adam, scaled training | $10^{-2}$ | $1.20\times10^{-4}$ / $1.21\times10^{-4}$ |
| GD, unscaled training | $10^{-2}$ | $1.23\times10^{-1}$ / $6.59\times10^{-2}$ |
| GD, scaled training | $10^{-2}$ | $2.14\times10^{-5}$ / $6.45\times10^{-6}$ |
| GD, scaled training | $10^{-1}$ | $1.11\times10^{-6}$ / $9.65\times10^{-7}$ |

The GD baselines show smooth progress. Scaled GD at $\eta=0.1$ reaches about $10^{-6}$ MSE without Adam's large oscillations. Over 160k–320k, its bandwidth net displacement and accumulated travel are both about $0.0032$ RMS. Scaled Adam at $\eta=10^{-3}$ travels 4.83 / 4.09 in bandwidth RMS over that interval, despite net displacement of only 0.0624 / 0.0700. Much of its motion therefore revisits parameter space rather than producing net change.

These are selected comparisons from the [complete 36-case shared-rate grid](stall_analysis/shared_rate_grid.json), not equally extensive optimizer tuning. Unscaled GD was searched only through $\eta=0.01$; scaled GD at $0.3$ and $1$ failed numerically. All finite displayed runs completed 320k updates and remained nonstationary. Scaled Adam's $10^{-3}$ shared rate was selected descriptively from the existing decade grid using late-window error, before the continuation intervention.

<figure>
  <img src="stall_analysis/figures/optimization.png" alt="Adam and GD training MSE, individual-step ranges, validation checkpoints, and detached refit errors for scaled and unscaled training across two seeds" style="max-width: 100%;">
  <figcaption>Figure 1. Baselines through 320k updates. Curves average training MSE in 1k bins; shading shows the 10th–90th percentiles of individual-step MSE. Crosses are validation checkpoints; dotted curves are detached validation refits. Adam endpoints can be far below sustained error, whereas GD progresses smoothly.</figcaption>
</figure>

## What geometry and coefficients were learned?

At scaled Adam's shared $\eta=10^{-3}$, median $|\lambda|$ reaches 0.0329 / 0.0808 and maxima reach 0.849 / 0.637. Slopes remain heterogeneous. A uniform construction's $\lambda=0.25$ is a reference, not a requirement that every learned slope or the median equal 0.25.

The final detached refit has validation MSE $3.99\times10^{-23}$ / $2.65\times10^{-21}$, with physical coefficient $\ell_1$ norms 16.6 / 16.2. Live coefficient norms are 24.0 / 13.8. Accurate representation is available with moderate coefficients, but only 204 / 313 of the 560 readout directions survive relative SVD cutoff $10^{-12}$. Calling this “good geometry conditioning” would be incorrect: it is good representability in an ill-conditioned dictionary.

<figure>
  <video controls preload="metadata" poster="animations/readouts_overview_last.png" style="max-width: 100%;" aria-label="Physical readout weights moving at their fixed centers through training, seeds zero and one">
    <source src="animations/readouts_overview.mp4" type="video/mp4">
    Readout animation: open the linked MP4 or interactive player below.
  </video>
  <figcaption>Figure 2a. Physical readout weights at all 559 fixed centers, seeds 0 and 1. Playback follows the shared-rate scaled Adam trajectory from initialization through 320k, then the joint-decay continuation to 1.68m. Blue circles mark core neurons; orange triangles mark halo neurons. Bias is displayed separately. The vertical scale stays fixed and is logarithmic outside a small linear region around zero. Frames use actual common checkpoints with unequal training-update gaps; no intermediate states are invented.</figcaption>
</figure>

<figure>
  <video controls preload="metadata" poster="animations/gammas_overview_last.png" style="max-width: 100%;" aria-label="Signed physical slopes moving at their fixed centers through training, seeds zero and one">
    <source src="animations/gammas_overview.mp4" type="video/mp4">
    Gamma animation: open the linked MP4 or interactive player below.
  </video>
  <figcaption>Figure 2b. Signed physical slopes gamma at the same centers and checkpoints as Figure 2a. Horizontal positions never move, and signs are retained as trained. The displayed update counter, rather than playback time, identifies training progress. Geometry continues to move after accurate detached fits are available, but its late movement is small on this full-trajectory scale.</figcaption>
</figure>

Open the interactive players for [readouts](animations/readouts.html) and [gammas](animations/gammas.html) to pause, step through individual frames, scrub the timeline, or change playback speed. Direct videos are also available: [readout overview](animations/readouts_overview.mp4) and [gamma overview](animations/gammas_overview.mp4). Each interactive player includes a second, magnified movie of changes during total updates 1,677,952–1,679,999: [late readout movement](animations/readouts_late.mp4), [late gamma movement](animations/gammas_late.mp4). These subtract the window's initial values, keep a fixed vertical scale, and show every eighth recorded state plus the final state. They illustrate motion; the complete per-update traces remain the evidence for oscillation statistics.

For a target-specific coefficient reference, we evaluated the boundary-corrected construction from [the theorem note](../../../theorem_for_sam.pdf), using the same $N=512$, radius-23 halo, and uniform $\lambda=0.25$. Its validation MSE is $1.48\times10^{-31}$ after FP64 export, with coefficient $\ell_1$ norm 6.75. Coefficients stabilize when increasing arithmetic precision from 50 to 80 decimal digits and quadrature degree from 7 to 9. The reference allowances $\pm\alpha_j$ define $D$; they are **not** the target-specific construction coefficients or a proved bound for the unit-RMS sine's complex extension.

<figure>
  <img src="stall_analysis/figures/adam_scaled_0.001_coefficients.png" alt="Learned coefficients versus a refit on learned slopes and boundary-corrected construction coefficients, with core and halo panels" style="max-width: 100%;">
  <figcaption>Figure 3. Final coefficients at 320k, with slope signs canonicalized for display. The detached fit uses learned slopes; the construction uses uniform slope 64. Coefficient differences cannot be interpreted as error in a common basis. Bias values are printed separately, and halo slots are expanded for visibility.</figcaption>
</figure>

Corrected-halo weight RMS is 0.676 / 0.399, versus core RMS 0.0531 / 0.0223. Halo coefficients and bias cannot be omitted from optimizer accounting. These observations measure participation, not an isolated causal halo benefit. [GD parameter histories](stall_analysis/figures/gd_scaled_0.1_parameters.png) and [GD coefficients](stall_analysis/figures/gd_scaled_0.1_coefficients.png) provide the smooth-training comparison. GD's median bandwidths are 0.0906 / 0.0648, but maxima reach 16.6 / 6.48; smooth error reduction does not mean every slope approaches the uniform construction's bandwidth.

## Residual frequencies, gradient directions, and actual movement

Let $r=(f-y)/\sqrt M$, and let $A$ include bias and the same sample normalization. The detached SVD uses $AD=U\Sigma V^T$. With $P_\tau$ projecting onto retained left singular vectors, record $P_\tau r$ and $(I-P_\tau)r$ separately. For the bandwidth tangent $J_\lambda$,

$$
g_\lambda=J_\lambda^Tr
=(P_\tau J_\lambda)^Tr+((I-P_\tau)J_\lambda)^Tr.
$$

For each orthogonal Fourier-band projector $Q_b$, the stored signed terms are $(P_\tau J_\lambda)^TQ_br$ and $((I-P_\tau)J_\lambda)^TQ_br$. Summing bands reconstructs the gradient without assuming that Fourier and readout projectors commute. Unit-RMS sine/cosine probes separately measure tangent sensitivity, distinguishing small residual amplitude from weak sensitivity.

<figure>
  <img src="stall_analysis/figures/adam_scaled_0.001_spectra.png" alt="Fourier-band MSE of full, readout-accessible, and out-of-span residuals over Adam checkpoints" style="max-width: 100%;">
  <figcaption>Figure 4. Scaled Adam, shared rate 0.001, both seeds. Out-of-span residual falls to about 1e-11 RMS while readout-accessible error remains. Axes show DFT indices on the endpoint-inclusive training vector; the common color scale exposes the magnitude gap.</figcaption>
</figure>

At 320k, bandwidth-gradient norms are $2.32\times10^{-3}$ / $1.47\times10^{-6}$, while out-of-span components are $1.26\times10^{-16}$ / $2.15\times10^{-17}$. Geometry predominantly responds to error the readout can represent. The dominant out-of-span Fourier band shifts from indices 4–7 initially to 32–63 / 128–255. This supports depletion of coarse residual; it does not verify an exponential frequency-dependent gradient law. Tiny projected forces require cutoff and boundary-sensitivity checks, and their exact signs are not treated as reliable evidence.

<figure>
  <img src="stall_analysis/figures/adam_scaled_0.001_fourier_gradients.png" alt="Per-band readout and geometry gradients, signed predicted descent under actual updates, and Fourier probe sensitivity" style="max-width: 100%;">
  <figcaption>Figure 5. Fourier diagnostics at 320k. Positive signed descent predicts linearized loss reduction; bias is shown separately and is also included in the full readout term. Individual band norms can exceed the norm of their signed sum because bands cancel.</figcaption>
</figure>

Neither block has stopped moving: readout-update RMS is $3.05\times10^{-6}$ / $1.32\times10^{-8}$ and bandwidth-update RMS is $1.29\times10^{-5}$ / $3.23\times10^{-6}$. Readout and geometry function-step cosines are 0.291 / $-0.156$, so uniform cancellation is unsupported. Readout moment denominators exceed Adam's epsilon at all coordinates; 0.9% / 7.7% of slope coordinates are epsilon-dominated. [Gradient and actual-step histories](stall_analysis/figures/adam_scaled_0.001_gradients.png) retain these measurements over time. The two endpoint phases differ substantially, motivating dense continuation traces.

## Separate step-size effects from geometry motion

Each seed's scaled Adam state at 320k is forked into four continuations, preserving all parameters and Adam moments. Cross **joint / frozen geometry** with **constant / decaying shared LR**. Constant means $\eta=10^{-3}$. Decay means one cosine schedule from $10^{-3}$ to $10^{-6}$ over 80k additional updates, then a constant tail. Frozen geometry leaves every slope exactly unchanged while training readout and bias. No readout is frozen or replaced by a solve.

The common reporting horizon is 1,360,000 additional updates, or **1,680,000 total**. This is a saved snapshot; training continues beyond it.

**Table 4. Complete training-window MSE over total updates 1.66m–1.68m, seed 0 / seed 1. All branches use scaled training and the same source state within each seed.**

| Geometry | Shared LR | Window MSE |
|---|---|---:|
| Joint training | Constant $10^{-3}$ | $1.17\times10^{-6}$ / $1.10\times10^{-6}$ |
| Joint training | Decay to $10^{-6}$ | $1.46\times10^{-10}$ / $9.34\times10^{-11}$ |
| Frozen geometry | Constant $10^{-3}$ | $1.16\times10^{-6}$ / $1.11\times10^{-6}$ |
| Frozen geometry | Decay to $10^{-6}$ | $9.56\times10^{-8}$ / $3.17\times10^{-9}$ |

Decay lowers joint-training MSE by about 8,050 / 11,750 times. Freezing geometry alone leaves the sustained floor almost unchanged. With decay, joint training is 656 / 34 times more accurate than frozen geometry. This controlled comparison supports a step-size contribution to the floor and a benefit from continued geometry adjustment under the smaller shared rate. “Geometry is simply too fast for readout” is therefore insufficient. The joint-decay detached refits remain near $2.6\times10^{-22}$ MSE, so the remaining live error is not a representation floor.

<figure>
  <img src="stall_continuation_analysis/figures/optimization.png" alt="Adam continuation MSE for joint or frozen geometry crossed with constant or decaying shared learning rate" style="max-width: 100%;">
  <figcaption>Figure 6. Eight continuations from two 320k source states at the same total update count. Shared LR decay suppresses the sustained oscillation floor. Joint training with decay fits more accurately than frozen geometry with decay, although both admit very accurate detached fits.</figcaption>
</figure>

<figure>
  <img src="stall_continuation_analysis/figures/joint_decay_dense.png" alt="Every-step loss, physical readout and slope movement, and gradient-step alignment over 2048 consecutive updates of joint training with decay" style="max-width: 100%;">
  <figcaption>Figure 7. Joint training with decay: 2,048 consecutive states ending at the reporting horizon. Readout curves include bias, three core centers, and both outer halo centers. Small loss bursts and parameter drift remain; sparse checkpoints alone would conceal them.</figcaption>
</figure>

For fixed geometry, the readout loss is convex. A very accurate detached least-squares solution establishes representability but does not guarantee rapid convergence of Adam with inherited moments and a chosen schedule. If $r_i=u_i^Tr$, the scaled readout gradient in singular direction $i$ is $\sigma_i r_i$: appreciable residual can produce a very small gradient when $\sigma_i$ is small. Squared singular values control fixed-step GD's linear convergence factors; they are not an exact model of Adam's time-varying preconditioner.

At this horizon, 87.6% / 84.9% of joint-decay residual energy lies at singular values below $10^{-3}$, and 82.7% / 78.0% below $10^{-5}$. The corresponding below-$10^{-3}$ fractions for joint constant-rate training are 0.86% / 0.26%. Frozen geometry with decay has over 99.95% below $10^{-3}$. These descriptive thresholds refer to the sample-normalized $AD$ spectrum, not a preselected success criterion. Fractions depend on oscillation phase: at the earlier 1.16m-total checkpoint, joint-decay fractions exceeded 99.9%. Weak-direction residual and small remaining oscillations coexist.

Across the continuation, joint constant-rate bandwidth travel is 35.5 / 29.8 RMS, versus net displacement 0.176 / 0.147. With decay those become 1.12 / 0.897 travel and 0.0224 / 0.0215 net displacement; most travel occurred during the initial decay. In the final dense 2,048-step window, the maximum physical slope change is still 0.00657 / 0.00621, with maximum readout/bias change $5.89\times10^{-7}$ / $9.65\times10^{-7}$. Slopes in both frozen branches remain exactly unchanged. Gradients and updates are measurable; the evidence does not support complete signal loss in either trained block. About 1–2% of the joint-decay steps point uphill within each block, and a downhill first-order direction need not reduce loss at finite step size.

<figure>
  <img src="stall_continuation_analysis/figures/joint_decay_singular.png" alt="Readout singular spectra and remaining residual MSE by singular direction at the common continuation horizon" style="max-width: 100%;">
  <figcaption>Figure 8. Joint training with decay, using the reference-scaled, sample-normalized feature matrix including bias. Residual energy concentrates in weak retained singular directions. The cutoff line distinguishes truncated directions from slowly fitted directions retained by the detached solve.</figcaption>
</figure>

## Best Adam results and verification

The historical audit separates a fortunate checkpoint from sustained accuracy. Among recorded D06 pilot/focused runs, the best validation checkpoint is MSE $1.46\times10^{-11}$ at 160k, with unequal physical readout/slope rates $9.3075\times10^{-5}$ and $25.6$. The best complete historical 20k training window is MSE $8.50\times10^{-9}$ over 280k–300k, using unequal physical rates $9.3075\times10^{-6}$ and $25.6$. These correspond to RMS $3.83\times10^{-6}$ and $9.22\times10^{-5}$. The new shared-decay runs improve on that sustained result, while not establishing a new best single validation checkpoint. [The audit](stall_analysis/historical_adam_audit.json) records exact cases. Older D02/D05 reports use different initializations, precisions, or targets and lack full raw results locally; this is not a certified global ranking of every repository experiment.

All **194 repository tests pass**, including Adam-state restoration, equivalence of constant-rate continuation to the original update, exact frozen slopes, independent batched branches, schedule endpoints, convergence-window eligibility, construction refinement, and singular-mode identities. Analysis verifies complete finite traces. Baseline Parseval error is at most $4.5\times10^{-16}$; Fourier-gradient and measured function-update closure are checked separately. Detached fits are repeated at relative SVD cutoffs $10^{-10},10^{-12},10^{-14}$; the large live/refit gap survives. These checks establish numerical consistency, not convergence or a frequency-decay theorem.

The [baseline evidence](stall_analysis/evidence.json), [continuation evidence](stall_continuation_analysis/evidence.json), [dense motion records](stall_continuation_analysis/dense_evidence.json), [construction check](stall_analysis/construction_verification.json), and cutoff audits ([baseline](stall_analysis/cutoff_and_numerical_audit.json), [continuation](stall_continuation_analysis/cutoff_and_numerical_audit.json)) preserve results. Each analysis directory also contains physical parameter histories, signed per-neuron Fourier arrays, and source hashes in `analysis_provenance.json`.

The implementation is in [`continue_stall.py`](../../../experiments/expD06_fixed_center_scales/continue_stall.py) and [`stall_analysis.py`](../../../experiments/expD06_fixed_center_scales/stall_analysis.py), with commands in the [experiment README](../../../experiments/expD06_fixed_center_scales/README.md). Slurm array 308 runs four branches per seed on two H200s; saved environments identify JAX 0.10.2, Optax 0.2.8, and matching training-source hashes ([seed 0](stall_run_provenance/environment_stall_308.json), [seed 1](stall_run_provenance/environment_stall_309.json)). CPU-only jobs 320/321 finalize the baseline and common-horizon exports. The two-hour cap per GPU worker is a resource limit, not convergence: any case still improving remains explicitly unconverged and resumable. Stationarity requires stable errors, spectra, predictions, and small parameter travel over consecutive doubled windows after the schedule ends; stable oscillation has a separate classification. The report horizon does not stop training.
