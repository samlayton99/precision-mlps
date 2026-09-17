# Fixed-center scale learning: protocol and interim pilot

The experiment is implemented. Across the expanded grid and its boundary extension, 485 finite trials have completed at least 20,000 updates, and 59 larger-rate GD trials failed numerically. The clearest result so far is that trained error, bandwidth growth, and the quality of the current feature geometry can move differently. Small trained error does not establish convergence, and larger bandwidth does not consistently improve the geometry. These are interim, validation-selected observations on one target and width; remote worker termination has interrupted the convergence study, and trained width/target comparisons are unfinished.

| Term | Meaning |
|---|---|
| $N,h,R,W$ | Core resolution, spacing $h=2/N$, halo radius $R=\lceil\sqrt N\rceil$, and neuron count $W=N+2R+1$ |
| $\lambda_j=h\gamma_j$ | Signed bandwidth in grid units; localization plots use $\lvert\lambda_j\rvert$ |
| Raw | Train physical readout $\mathbf c=(b,\mathbf w)$ and physical slopes $\boldsymbol\gamma$ |
| Both | Train $\mathbf a=D^{-1}\mathbf c$ and $\boldsymbol\lambda$ with fixed reference $D$ |
| Trained RMS | Error of the live jointly trained network on the validation grid |
| Refitted RMS | Validation error after a detached, truncated-SVD readout solve at the current slopes |
| $P_\tau$ | Projector onto the retained left singular vectors of the scaled feature matrix, at relative cutoff $\tau$ |
| GD / Adam | Full-batch gradient descent without momentum / Adam with $(0.9,0.999)$ moments and trained-coordinate $\epsilon=10^{-8}$ |

## Question and controlled comparison

The proposed mechanism is that readout adaptation removes residual components that drive useful hidden-geometry motion, leaving weak bandwidth gradients. Rescaling coordinates and changing the relative rates may change that competition. A successful intervention must improve the trained function and the numerically retained geometry, rather than merely increasing slopes. The theoretical drift bounds and projection identities motivate diagnostics; they do not guarantee that every slope should approach $\lambda=0.25$.

The implementation follows the reference scales in [correcting_scales.md](../../../docs/correcting_scales.md), interpreted with the supplied [gamma-barrier handoff](../../../docs/gamma_barrier_handoff_v2.pdf) and [readout-scale note](../../../docs/readout_scale_collaborator_note.pdf). Later user decisions supersede the note's shorter screens and readout-freezing branch: every finite trial receives at least 20,000 updates, readouts and slopes always train together, and learning rates stay constant throughout a trajectory. The two parameter blocks minimize the same half mean-squared error; their rates control relative update sizes.

All centers, including halo centers, are fixed. Hidden biases are enforced through $\gamma_j(x-x_j)$, and slopes remain independently signed. The reference metric includes the bias and corrected halo allowances, is computed once at $\lambda_{\rm ref}=1/4$, and is never refreshed during training. Raw and Both start from the same physical function within each seed and initialization family. Xavier and reference-envelope readouts are separate families; physical slopes use the prescribed tanh Xavier scale in both.

The first target is $\sqrt2\sin(2\pi x)$ at $N=512$, giving $R=23$ and $W=559$. Training uses 8,193 equally spaced, endpoint-inclusive samples. The independent 32,768-point midpoint grid is used for validation and all rate/display choices. The 65,536-point test grid has not been used. Follow-up targets are unit-continuous-RMS $x^2$ and $\sin(2\pi x)+0.1\sin(20\pi x)$.

## Rate tuning and duration

The expanded pilot crosses five base rates $10^{-5},\ldots,10^{-1}$ with five bandwidth/readout ratios $0.01,0.1,1,10,100$, two arms, two readout initializations, and two paired seeds. Both uses the base rate directly for the readout. Raw's tuned grid multiplies that rate by $\alpha_{\rm ordinary}$ for GD or $\sqrt{\alpha_{\rm ordinary}}$ for Adam, and multiplies the bandwidth rate by $h^{-2}$ or $h^{-1}$ respectively. These match ordinary-slot update scales, while allowing the bias, halo metric, and Adam epsilon to differ. Three equal-numerical-rate Raw controls per initialization and seed bring the count to 212 trials per optimizer.

The initial 84-trial-per-optimizer grid is a subset of this expanded grid. Completed trajectories are resumed, not restarted when the grid expands. The boundary follow-up adds GD base rates $0.3,1$ and Adam ratio $1000$, equally in Raw and Both, because the common-20k rankings reached those search boundaries. Its full manifest contains 292 GD and 252 Adam configurations. A nonfinite update is recorded as a numerical failure; meaningless nonfinite updates are not padded to 20k.

Twenty thousand updates is a minimum observation horizon, not a stopping rule. Continuation uses 40k, 80k, 160k, and successive doublings. Stationarity requires three consecutive doubled windows with stable validation error and predictions, stable residual-band energy, and small accumulated parameter travel. Relative error/prediction tolerance is $10^{-4}$, band-energy tolerance is $10^{-3}$, bandwidth-travel RMS tolerance is $2.5\times10^{-5}$, and readout travel in reference-envelope units must be below $10^{-4}$. Precision floors depend on FP64 epsilon and readout norm. Persistent oscillation is recorded separately using stable window loss statistics and bandwidth quantiles. Neither an interrupted worker nor a budget limit constitutes convergence.

## Evidence retained

Every run saves its configuration, reference scales and masks, full parameter/optimizer checkpoints, complete compact per-update traces, sign crossings, accumulated travel, and GD drift bounds. Checkpoints are dense near initialization and logarithmic thereafter, with additional exact first-crossing checkpoints when residual RMS drops by powers of ten. Adam moments and moment-to-epsilon ratios are retained.

Detached analysis saves the live residual, $P_\tau r$, $(I-P_\tau)r$, the refitted residual, singular values, retained rank, and exported readout norms. Refits never enter the live trajectory. Fourier bands partition the actual training vector using an orthonormal DFT; the endpoint convention and implied period are recorded. A Hann-window spectrum provides a boundary-sensitivity check.

For the live bandwidth Jacobian $J_\lambda$, each band records $J_\lambda^TQ_b r$, $(P_\tau J_\lambda)^TQ_b r$, and $((I-P_\tau)J_\lambda)^TQ_b r$. This order avoids assuming that Fourier and feature-space projections commute. The perpendicular term is evaluated directly, with the error from subtracting larger gradients also saved. Signed collective forces and equal-RMS sine/cosine probes separate residual size, tangent strength, and alignment, for core and halo regions separately. Proposed readout and slope updates are decomposed into their linearized and measured prediction changes, including the interaction term.

## What the complete 424-trial, 20k snapshot shows

The table shows the rate pair with lowest mean log trained validation RMS across seeds 0 and 1 within each group. It is a descriptive selection at this common horizon, not a final rate lock. Refitted error did not influence selection. The two seed values remain separate because two repetitions do not establish a reliable uncertainty estimate.

| Optimizer / arm | Initialization | $\eta_{\rm r},\eta_{\rm g}$ | Trained RMS, seeds 0 / 1 | Median $\lvert\lambda\rvert$, seeds 0 / 1 | Refitted RMS, seed 0 |
|---|---|---|---|---|---|
| Adam / Both | Xavier | $0.01,0.001$ | $3.50\times10^{-4}$ / $7.21\times10^{-4}$ | $0.0171$ / $0.0166$ | $4.45\times10^{-12}$ |
| Adam / Raw | Xavier | $9.31\times10^{-4},0.256$ | $3.26\times10^{-4}$ / $4.20\times10^{-4}$ | $0.0162$ / $0.0171$ | $5.37\times10^{-12}$ |
| Adam / Both | Envelope | $0.01,0.1$ | $9.44\times10^{-3}$ / $1.44\times10^{-5}$ | $0.594$ / $0.595$ | $1.35\times10^{-7}$ |
| Adam / Raw | Envelope | $9.31\times10^{-5},25.6$ | $2.50\times10^{-3}$ / $1.54\times10^{-5}$ | $0.598$ / $0.602$ | $8.78\times10^{-6}$ |
| GD / Both | Xavier | $0.1,0.01$ | $1.59\times10^{-2}$ / $1.64\times10^{-2}$ | $0.0411$ / $0.0477$ | $1.48\times10^{-11}$ |
| GD / Raw | Xavier | $8.66\times10^{-4},655.36$ | $2.51\times10^{-2}$ / $1.76\times10^{-2}$ | $0.0387$ / $0.0450$ | $1.18\times10^{-11}$ |
| GD / Both | Envelope | $0.1,1$ | $3.81\times10^{-3}$ / $5.15\times10^{-3}$ | $0.720$ / $0.453$ | $2.04\times10^{-6}$ |
| GD / Raw | Envelope | $8.66\times10^{-4},65536$ | $1.59\times10^{-2}$ / $5.15\times10^{-3}$ | $0.770$ / $0.453$ | $2.08\times10^{-5}$ |

The Xavier examples already have features permitting approximately $10^{-11}$–$10^{-12}$ validation refits while their trained errors remain much larger. Their present optimization error therefore cannot be attributed wholly to an inability of the current features to represent this sine target. Tuned Raw is competitive with Both in these interim Adam/Xavier observations; they do not establish a distinct algorithmic advantage for changing coordinates.

The selected envelope examples grow larger slopes but have worse refitted errors than those Xavier examples. Their refits can also require large readout norms: seed-0 Adam/Both has $\|\mathbf c_*\|_1\approx367$, whereas Adam/Raw has approximately $3.47\times10^5$, compared with about 14 for the Xavier examples. Thus localization alone is not an adequate success metric, and retained rank alone is not sufficient either. These comparisons are observations across tuned settings, not a causal attribution to initialization or halos.

The Adam/envelope seed spread also makes the limitation of a single endpoint conspicuous. At the displayed Both rate pair, trained error differs by more than two orders of magnitude between seeds despite very similar median bandwidths. Per-update loss traces, actual momentum updates, and the longer convergence windows are needed before ranking these rates.

The [last-5,000-update traces](evidence_expanded424_000020000/tail_training_20k.json) demonstrate this directly. Seed-0 Adam/Both/Xavier ends at validation RMS $3.50\times10^{-4}$, but training RMS during updates 15,000–19,999 ranges from $3.51\times10^{-4}$ to $0.107$, with RMS over samples and time $9.22\times10^{-3}$. This is not a stationary error floor. Endpoint rankings therefore remain provisional even after satisfying the 20k minimum.

The [regional first-step measurements](evidence_expanded424_000020000/halo_motion_20k.json) also justify the halo control. For the displayed GD/Both/envelope trial, the first update gives median $|\lambda|=0.684$ in the core and $1.51$ in corrected halo slots; the largest corrected-halo value is $166$. These are immediate large-step effects. They must be separated from a later residual-depletion mechanism and do not, on their own, demonstrate useful localization.

![Trained error, bandwidth, accumulated travel, and detached refit](evidence_expanded424_000020000/figures/pilot_trajectories.png)

*Seed 0 at each pair selected using both seeds at the same 20k horizon. Curves describe unfinished trajectories. The dashed bandwidth line is the reference $1/4$, not a claim that every neuron must reach it. Refits are diagnostics on the same validation grid.*

![Learning-rate map](evidence_expanded424_000020000/figures/pilot_rate_map.png)

*All 20k seed-0 endpoint errors in the expanded grid, with initialization families separated. Both arms receive the same number of tuned rate pairs; Raw also has the declared equal-numerical-rate controls. Rates are expressed in each arm's trained coordinates.*

![Residual and gradient frequency decomposition](evidence_expanded424_000020000/figures/pilot_frequency_split.png)

*Envelope representatives at 20k. The residual remaining outside the retained readout space is much smaller than the live residual in these examples. Band-gradient panels display norms after signed contributions within each band have been summed. The last column uses the actual proposed optimizer update and can show opposite contributions from readout and geometry in different bands. Tiny projected quantities require the cutoff and cancellation checks; these finite-window spectra do not by themselves establish an exponential-decay law.*

## Boundary extension

The complete extension contains 544 configurations. The [trace audit](trace_audit.json) confirms a finite, complete first-20k loss trace for all 485 finite trials, with none below the required minimum. The 59 numerical failures are all GD cases: 40 Both and 19 Raw. The added Adam ratio-1000 trials do not change the best paired endpoint choices in the table.

The larger base rate does improve Raw GD's best 20k pairs. At base $0.3$, Raw/Xavier has trained validation RMS $0.01424$ and $0.009371$, and Raw/envelope has $0.007928$ and $0.002200$. Their trained readout rate is $0.002598896$; bandwidth rates are $196.608$ and $196608$, respectively. Thus even the GD comparison changes when the rate boundary is expanded. The [544-configuration snapshot](evidence_000020000/summary.csv) and its [paired choices](evidence_000020000/paired_best.json) preserve this extension separately from the 424-trial figures above.

There is a useful scale explanation for the Both/GD instability. With the bias-inclusive normalized feature matrix $A$, the readout Hessian is $D A^T A D$, whose largest eigenvalue is at least the bias diagonal $\alpha_{\rm b}=10.76391$. Fixed-geometry readout GD therefore requires

$$
\eta_{\rm r}<\frac{2}{\lambda_{\max}(D A^T A D)}\leq\frac{2}{\alpha_{\rm b}}=0.185806.
$$

The added Both base rates $0.3$ and $1$ already exceed this necessary readout-block bound. This does not characterize every joint trajectory, but it identifies a concrete readout stability constraint that must accompany attempts to increase bandwidth motion.

## Reference and numerical checks

A single fixed-geometry check at $\lambda=0.25$ reproduces a useful regime across all planned widths and targets. It is an anchor, not another lambda sweep. Using the same $10^{-12}$ relative SVD cutoff, [validation RMS](reference_metrics.json) ranges from $9.02\times10^{-14}$ to $8.27\times10^{-13}$ across the nine width/target combinations. This confirms that the implemented centers, halo construction, targets, and detached solver can realize highly accurate reference functions. It does not show that training acquires that geometry.

The cutoff study covers eight representatives at steps 0, 1, and 20,000, at $\tau=10^{-10},10^{-12},10^{-14}$: 72 checkpoint/cutoff combinations. Absolute projected-gradient magnitudes are sensitive. For example, Adam/Both/Xavier at 20k changes from $\|g_\perp\|=2.77\times10^{-15}$ to $3.41\times10^{-19}$ across the outer cutoffs, while refitted RMS changes from $2.17\times10^{-10}$ to $3.68\times10^{-14}$. The large trained/refitted gap survives this sensitivity, but the smallest projected forces should not be interpreted as cutoff-independent physical measurements. The largest CPU/GPU prediction difference in these checks is $1.78\times10^{-14}$. Full and perpendicular Fourier reconstruction errors are also saved in the [cutoff results](evidence_expanded424_000020000/cutoff_sensitivity.json).

Focused checks cover fixed centers, paired initialization, finite-difference gradients, one-step GD/Adam coordinate identities, stable saturated-tanh derivatives, the GD drift bound, Fourier reconstruction, detached refits, exact optimizer-state resume, batching independence, minimum duration, nonfinite failures, and complete convergence windows across interrupted sessions. Final validation: **177 fast repository tests passed**, one slow test deselected, with seven pre-existing deprecation warnings; the experiment-specific suite contains 17 passing tests.

## Continuation and remaining scope

The implementation is JAX/Optax in FP64. At most two H200 GPUs have been used, with CPU-only SVD diagnostics. The remote experiment lives under `/workspace/junmiaoh/experiments/precision-mlps`; manifests, source hashes, dependency versions, per-case states, and cumulative GPU-worker budget ledgers are preserved there. The existing remote checkout and unrelated GPU workloads were left untouched. A cumulative allocation of three GPU-hours per optimizer was set for pilot continuation, reserving the rest of the 24-GPU-hour campaign budget for follow-ups. Allocation exhaustion leaves unconverged states explicitly unfinished; it was not the cause of the current interruption.

Long-running workers have repeatedly disappeared without a Python exception, including detached and tmux launches; inspected OOM counters did not show OOM kills. SSH loss, SIGTERM exits, and subsequently SIGKILL exits were observed. Short sessions successfully advanced and resumed checkpoints. Automatic reconnection with an exclusive lock per optimizer also made progress, then stopped on SIGKILL. No campaign worker remains active. The cause of termination and the host's required launch/cleanup policy remain unresolved, so unattended convergence work is paused rather than repeatedly relaunching killed jobs. No run is being called converged on that basis. Interrupted output is preserved, and partial traces from the earliest runner version were archived before those affected trials were rerun. The [progress record](progress_snapshot.json) contains closed ledgers charging approximately 2.02 GPU-hours, excluding brief setup checks; charging through verified termination overcounts some idle time.

After rate selection, the prepared confirmation matrix transfers fixed rates to $N=512,1024,2048$, all three targets, and new seeds 2–6. The halo control crosses full versus ordinary corrected-halo initialization with full versus ordinary halo metric, keeping the bias and diagnostic metric fixed. The sampling control doubles training density. These trained comparisons have not yet been executed, and no test-grid result or converged winner is claimed.

The [experiment README](../../../experiments/expD06_fixed_center_scales/README.md) gives runnable commands and diagnostic conventions. The saved 424-trial snapshot contains [per-case summaries](evidence_expanded424_000020000/summary.csv), [paired choices](evidence_expanded424_000020000/paired_best.json), and figures; full states and diagnostic arrays remain in the remote case directories. Subsequent boundary trials and continuations are separate from this fixed-horizon evidence snapshot.
