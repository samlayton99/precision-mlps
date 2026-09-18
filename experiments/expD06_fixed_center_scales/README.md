# Fixed-center bandwidth acquisition

This experiment asks whether one shared learning rate, with the prescribed readout and slope scales, sustains useful geometry learning. The current answer is in the [shared-rate report](../../results/checkpoint_D_optimizers/expD06_fixed_center_scales/scale_learning_consolidation.md). **Unscaled training** updates the network's readouts and slopes directly. **Scaled training** updates $\mathbf a$ and $\boldsymbol\lambda$, with $\mathbf c=D\mathbf a$ and $\boldsymbol\gamma=\boldsymbol\lambda/h$. Each method uses one shared rate for both blocks. Their historical code labels are `raw` and `both`, respectively. The implementation uses JAX and Optax in FP64, fixed grid and halo centers, independent signed slopes, and identical initial physical networks within a seed. The reference bandwidth is $\lambda=0.25$ and the halo radius is $R=\lceil\sqrt N\rceil$.

Install the optional `jax-experiments` dependency group for CPU development. On the remote H200 host, use an isolated environment with the CUDA 12 JAX extra and record its exact installed versions. This experiment does not change the repository's existing PyTorch runners.

The scientific baseline trains $\mathbf c=D\mathbf a$ and $\boldsymbol\gamma=\boldsymbol\lambda/h$ with **one shared base rate**, $\eta_a=\eta_\lambda=\eta$, including the output bias. The fixed reference metric supplies the relative scales automatically. Tune only $\eta$; independent block-rate ratios are outside this baseline. The two blocks minimize one half mean squared error objective. Every finite scientific run must receive at least 20,000 updates. Baseline runs keep the rate fixed; the paired continuation below explicitly tests a shared decay schedule. Shorter executions are implementation checks or throughput benchmarks, never scientific comparisons or rate-selection evidence.

`shared_rate_analysis.py` filters the existing pilot to scaled training and paired unscaled training with the shared-rate Adam/reference-envelope setup, then analyzes their saved checkpoints without new training. Run it with `--root /workspace/junmiaoh/experiments/precision-mlps/runs/pilot` inside an eight-CPU Slurm allocation, with GPU execution disabled and two BLAS threads. Its assertions verify the 16 expected cases and complete 320k traces. The historical launchers below reproduce earlier work; their independent-rate and crossed matrices are not the current baseline.

`core.py` defines the reference envelopes, paired initialization, coordinate maps, and compiled training chunks. The tanh derivative uses the equivalent stable $4e^{-2|u|}/(1+e^{-2|u|})^2$ expression to avoid cancellation in $1-\tanh^2 u$.

Focused verification: `python -m pytest -q tests/test_expD06_fixed_center_scales.py`.

## Diagnostic conventions

The training residual and feature matrix include the $1/\sqrt M$ normalization. Diagnostic refits use `scipy.linalg.svd` with the `gesdd` driver on $AD_{\rm ref}$ and retain singular values above $10^{-12}\sigma_{\max}$. The fitted coefficients are exported back to physical readout units. The retained projector describes numerically accessible directions, not the exact full feature span.

Fourier bands use an orthonormal DFT on the actual training sample vector. Its implied period is $M\Delta x$, including the endpoint convention. DC and paired positive/negative dyadic bands partition the vector exactly. The band gradients are $J_\lambda^T Q_b r$, $(P_\tau J_\lambda)^TQ_b r$, and $((I-P_\tau)J_\lambda)^TQ_b r$. This order makes both the frequency sum and the within-band parallel/perpendicular sum reconstruct the original gradient without assuming the two projectors commute.

Detached analyses preserve the live residual, its two projected components, the actual refitted residual, complex spectra, per-neuron band gradients, unit-RMS sine/cosine sensitivity probes, and signed forces in the collective magnitude-increase direction. The readout correction from a full truncated refit is recorded separately from $P_\tau r$: discarding a live coefficient component can make them differ. A Hann-window residual spectrum is a boundary-sensitivity diagnostic and does not change the objective.

## Training and continuation

`python -m experiments.expD06_fixed_center_scales.run --cases cases.json --output results/checkpoint_D_optimizers/expD06_fixed_center_scales --frontier 20000` advances a JSON list of case configurations to a saved frontier. A case specifies the target, core resolution, optimizer, coordinate arm, initialization, seed, two constant rates, and optional halo ablations. Each compiled batch shares the resolution, target, optimizer, and sample grids. GPU launches additionally use `--require-gpu` and restrict visible devices before Python starts.

A frontier is a continuation checkpoint, not a claim of convergence. Repeat with frontiers 40,000, 80,000, 160,000, and successive doublings. A finite run is never eligible to terminate before 20,000 updates. Stationarity requires three consecutive doubled windows with stable validation error, predictions, Fourier band energy, and small accumulated readout and bandwidth travel. Persistent oscillation has a separate label based on stable window loss statistics and bandwidth quantiles. Nonfinite updates are recorded as failures; advancing meaningless nonfinite states to 20,000 is not a valid scientific run.

Each case preserves its configuration and reference geometry, full parameter/Optax checkpoints, complete compact per-step traces, predictions, per-neuron gradients, proposed next updates, and residual-reduction event checkpoints. Loading a checkpoint restores parameters, moments, counters, and accumulated travel. The full-batch deterministic training phase consumes no new random draws after initialization.

## Paired continuation at the learned geometry

The new relative-rate study uses `ratio.py` and `ratio.sbatch`. It keeps the
theory coordinates $c=Da$, $\gamma=\lambda/h$, the reference $D$ at
$\lambda_{\rm ref}=0.25$, fixed centers, and the existing square-root halo.
Only the scalar rates $\eta_a(t)$ and $\eta_\lambda(t)$ change. Initialization
is physical Xavier on slopes and the legacy signed reference-envelope draw
on physical readouts; the latter is not Gaussian Xavier on $a$. Training
minimizes half-MSE. Comparisons use MSE. Validation is diagnostic; test data
are not evaluated.

The core matrix is $N=512,1024$, sine/quadratic/mixed targets, seeds 0 and 1.
Both shared acquisition rates $10^{-3},10^{-2}$ run through 80k, then decay
over 80k to $10^{-6}$. Exact existing checkpoints are reused with hashes.
At 160k the high-rate history branches into a $2\times2$ change: readout rates
$10^{-6},10^{-5}$ and geometry rates $10^{-6},10^{-7}$, reached by a 20k
cosine transition. The low-rate history retains the shared control. This
gives 60 primary continuations. The initial common reporting horizon is
340k; subsequent matched advances remain eligible for continuation.
Two historical high-rate sine/512 runs also decay from 320k to compare with
the existing low-rate tails. Four early handoffs lower geometry tenfold
at 20k (sine/512 and mixed/1024, both seeds).

Folders store explicit schedule knots, absolute update indices, source hashes,
native rates, physical Adam rate factors $\eta_aD,\eta_\lambda/h$, optimizer
state, complete scalar traces, and checkpoints every 20k. Dense parameters,
signed gradients, and actual updates cover 2048 consecutive states before
and after each 20k frontier. Forks preserve moments. Interrupted groups
resume at their common saved state. A budget stop is unfinished; oscillation
is a diagnosis rather than convergence. Each finite new branch must complete
at least 20k updates before scientific comparison.

The authorized new allocation budget is eight GPU-hours, at most two GPUs
concurrently, including compilation, I/O, and unsuccessful allocations.
Phase allowances are 0.3h verification, 4.5h primary/historical training,
0.4h early handoffs, 0.8h feedback, 1.3h frozen-dictionary first-order solves,
and 0.7h reserve. Reconcile Slurm allocation elapsed times before submission;
Python runtime alone is not the charge. CPU analyses request zero GPUs.
These limits supersede the historical pilot allowance below.

`feedback.py` adds one training-only policy from the same high-acquisition
160k checkpoints. It checks every 20k: improvement below 5% in each of two
successive window means, and more than 95% of residual energy in the readout
span at both cutoffs $10^{-10},10^{-12}$. The first intervention lowers the
geometry rate tenfold over 20k. After another 40k at fixed rates, a persistent
stall permits a threefold readout-rate increase only if readout-only
counterfactual steps lower mean MSE and lower MSE in at least 95% of 64
systematically sampled states from the latest 2048-state window. There are
at most two interventions. Thresholds are operational choices. Every check,
rejection, source window, cutoff, counterfactual, and schedule extension is
saved in `feedback.json`; no validation observation controls this policy.

`continue_stall.py` resumes the shared-rate scaled Adam/envelope runs, seeds 0 and 1, from update 320,000. Each seed has four branches: joint or frozen geometry, crossed with constant $\eta=10^{-3}$ or a shared cosine decay to $10^{-6}$ over 80,000 additional updates followed by a constant tail. All branches preserve the source parameters and Adam moments. Freezing leaves every slope unchanged; its unused geometry moments continue evolving independently of the readout moments. No readout is frozen or replaced by a detached solve.

Submit `stall.sbatch` through Slurm. Each one-GPU worker advances all four branches for its seed, with a cumulative two-hour worker cap and at most two concurrent GPUs. Reconcile the existing campaign budget before submission. Checkpoint and trace steps under `runs/stall/<branch>/<source-case>/` count **additional** updates; add 320,000 for the full trajectory. Full checkpoints occur every 1,000 updates; dense parameters, gradients, and actual updates cover the first 2,048 steps and the last 2,048 steps of every 20,000-step window. Complete scalar traces cover every update. Convergence checks use doubled windows after the schedule reaches its constant tail at 80k, so the first possible stationary/oscillatory classification is at 240k additional updates. Budget interruptions remain unconverged and resumable. The four branches continue to a common horizon until all have a terminal classification or the worker budget is exhausted.

## Readout, parameter, and spectral analysis

`stall_analysis.py --root <experiment-root>` audits twelve existing shared-rate runs: scaled Adam at $10^{-3},10^{-2}$, unscaled Adam at $10^{-3}$, scaled GD at $10^{-2},10^{-1}$, and unscaled GD at $10^{-2}$, each with seeds 0 and 1. It preserves the complete shared-rate search table, including numerical failures. The entry point emits data and figures only: training-window MSE, individual-step error ranges, validation/refit MSE, coefficient and slope histories, construction comparisons, residual spectra, signed band gradients and updates, Fourier sensitivity probes, and residual loading in singular directions of $AD$. All feature matrices include the bias and use the sample normalization $1/\sqrt M$. The optimizer still minimizes half-MSE; diagnostics never replace a trained readout.

`construction_reference.py` evaluates the target-specific sine construction from `theorem_for_sam.pdf`, equations A.2, A.26, B.3–B.6, and B.16–B.17. It uses the same 559 slots and radius-23 halo as training. Compare 50 versus 80 decimal digits and quadrature degrees 7 versus 9 before export to FP64. This is separate from the magnitude envelope defining $D$ and from the older Toeplitz QI construction. The learned dictionary has heterogeneous signed slopes; coefficient plots canonicalize signs for comparison without changing training. Coefficient differences are not a basis-independent error measure.

Submit `stall_analysis.sbatch` for an eight-CPU, zero-GPU analysis allocation. Add `--continuations --additional-steps <completed-common-horizon>` to analyze the paired branches and their final 2,048 consecutive parameter states. Outputs are `runs/stall_analysis` and `runs/stall_continuation_analysis`; original checkpoints remain in their owning run folders. Diagnostic cache revision 3 adds singular residual/target coefficients, modal readout-gradient coefficients, bias update contributions, and explicit MSE. The historical Adam audit distinguishes best saved validation checkpoints from complete 20k training windows and reports their actual configurations. It cannot certify missing raw results from older studies.

## Parameter animations

Run `python -m experiments.expD06_fixed_center_scales.animate_parameters --root results/checkpoint_D_optimizers/expD06_fixed_center_scales` locally after downloading the two analysis exports. Each seed has its own animation, with physical readouts $w$ on the top row and physical slopes $\gamma$ on the bottom row, synchronized at the same update. The overview joins actual common checkpoints from scaled Adam at shared rate $10^{-3}$ to the joint-decay continuation, ending at 1.68m total updates. Signed parameters stay at their original centers; the output bias is displayed separately. Each parameter's vertical scale stays fixed during playback and matches across seeds, with a linear region around zero and logarithmic tails.

The overview holds each checkpoint through 320k updates for one second, then displays later checkpoints at six per second. This slows the first 300k updates, including the next saved checkpoint at 320k, by a factor of six. Playback time is not proportional to training updates: checkpoint gaps are unequal, and every frame displays its actual update count.

The late view requires the three `dense_001357000_001358000.npz`, `dense_001358000_001359000.npz`, and `dense_001359000_001360000.npz` files from each seed's `runs/stall/joint_decay/<source-case>/` directory, copied into `animations/input/joint_decay_s0/` and `animations/input/joint_decay_s1/` beneath the local result root. It validates all 2,048 consecutive saved states, then displays every eighth state plus the final state as changes from the window's start. No temporal interpolation, sign canonicalization, training, or detached refitting occurs.

Matplotlib and the local `ffmpeg` executable produce four MP4s, two standalone HTML pages (`animations/seed_0.html` and `animations/seed_1.html`) with overview and late players, and still frames for inspection. Players provide pause/step/scrub/speed controls and embed their images and controls without network dependencies. `animations/provenance.json` records input hashes, physical centers, row order, playback cadence, exact displayed steps including early holds, and the source hash. This export uses the existing report horizon rather than silently extending the scientific comparison to newer training states.

## Historical base-rate and relative-rate search

`campaign.py` runs one optimizer per GPU worker. Its initial grid uses base rates $10^{-4},10^{-3},10^{-2}$ and bandwidth/readout ratios $0.1,1,10$. The expanded grid uses five base rates from $10^{-5}$ through $10^{-1}$ and five ratios from $0.01$ through $100$. Two physical initializations and two paired seeds are evaluated in both coordinate arms. Each finite trial receives the full 20,000-step minimum, followed by constant-rate continuation.

In Both, the base rate is $\eta_{\rm r}$ and the ratio gives $\eta_{\rm g}/\eta_{\rm r}$. Raw's tuned grid matches the ordinary-slot physical update scales at the pilot width: multiply the readout rate by $\alpha_{\rm ordinary}$ for GD or $\sqrt{\alpha_{\rm ordinary}}$ for Adam, and the bandwidth rate by $h^{-2}$ or $h^{-1}$ respectively. This is an initial search parameterization, not a claim of whole-network equivalence: the bias/halo metric and Adam's physical epsilon differ. Three equal numerical-rate Raw controls remain separate. The expanded search has 25 tuned rate pairs per arm plus these Raw controls; all trial counts and costs are reported.

The common-20k pilot reached the upper base-rate boundary for GD and the upper ratio boundary for one Adam configuration. The follow-up flag `--boundary` therefore adds GD bases $0.3,1$ at all five ratios and Adam ratio $1000$ at all five bases. Both coordinate arms and both initializations receive these extensions with two seeds, giving 292 GD and 252 Adam configurations in total. This is an evidence-driven expansion of the search, not a rate change within any trajectory.

The H200 host requires Slurm allocations. From the remote experiment's `code` directory, submit `sbatch experiments/expD06_fixed_center_scales/pilot.sbatch`. This array requests two independent one-GPU jobs in partition `gpu`, account `lab`, QOS `lab-gpu2`: task 0 resumes GD and task 1 resumes Adam. Each receives four CPUs and 16 GiB of memory. `srun` launches the existing JAX environment inside the allocation, preserving Slurm's `CUDA_VISIBLE_DEVICES`; the application rejects CPU fallback or visibility of more than one GPU. Startup logs and environment records include Slurm job/step identifiers and the actual JAX device.

The pilot submitter sets a cumulative three-GPU-hour cap per optimizer, including approximately one hour already charged to each worker before the scheduler correction. Its 2h15m walltime therefore covers the remaining roughly two hours plus checkpoint/setup margin. Inspect the saved ledgers before reusing or changing that walltime. The full campaign is limited to 24 GPU-hours and two concurrent GPUs; reconcile worker ledgers with Slurm allocation elapsed times, charging allocation overhead as well as setup checks. Reserve the remaining budget for width, target, halo, and sampling follow-ups. `--max-frontier` pauses for intermediate analysis and leaves unfinished cases labeled `continuing`.

Inspect `squeue -j JOB_ID`, `scontrol show job JOB_ID`, and `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES`. Logs are written to `/workspace/junmiaoh/experiments/precision-mlps/logs/slurm-ARRAY_ID_TASK_ID.log`. A scheduler `COMPLETED` state means the worker exited successfully; scientific convergence is recorded separately in each case's `latest.json`.

After rate selection, `--mode confirmation --selected selected.json` transfers the unchanged two rates to $N=512,1024,2048$, all three targets, and five new seeds (2–6). `--mode halo` runs the $2\times2$ comparison of full versus ordinary corrected-halo initialization and metric, using Both/envelope at $N=512,1024$ with seeds 2–4; bias scaling and the diagnostic reference metric remain fixed. `--mode sampling` compares 16 and 32 training subintervals per cell at both smaller widths on sine and the mixed-frequency target, with seed 2. All modes share the same worker budget ledger and minimum duration. These commands prepare follow-up matrices; an unexecuted matrix is not evidence of width transfer or halo causality.

## Detached analysis

On the cluster, run this CPU work inside a Slurm allocation as well. Request CPUs and memory without a GPU; retain the experiment's existing virtual environment and run the following command through `srun`.

Run `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 python -m experiments.expD06_fixed_center_scales.analyze --root /workspace/junmiaoh/experiments/precision-mlps/runs/pilot --step 20000 --workers 4` to compare a common saved horizon while GPU workers continue. The default analyzes early checkpoints and each logarithmic decade on seed 0 at each group's best two-seed rate pair, chosen by mean log trained validation RMS. These representatives are descriptive examples, not final rate selections. `--mode latest` analyzes every case at that horizon; `--mode all` analyzes all saved checkpoints. Saved parameters also allow further diagnostics without rerunning training.

Each analysis exports arrays and JSON metrics: the SVD/refit, full and projected residual spectra, signed frequency-band gradients, signed collective sine/cosine responses, update contributions, retained rank, and readout norms. The CPU/GPU prediction difference and frequency-sum reconstruction errors are recorded as numerical checks. Perpendicular band gradients are computed directly with $(I-P_\tau)J_\lambda$; the difference from subtracting the two larger band gradients is also saved to expose cancellation. CSV summaries and four figure groups cover trained/refitted error and bandwidth trajectories, the rate map with separate initialization panels, the residual/gradient mechanism, and the frequency decomposition of the next readout and geometry updates. Repeat representative analyses with `--cutoff 1e-10` and `--cutoff 1e-14` before interpreting tiny projected forces. Test-grid evaluation requires an explicit saved selection record and is excluded from the pilot.

The earlier direct SSH, nohup, and tmux workers were terminated because they bypassed Slurm. Use the submitter above for continuation. `--session-seconds` remains an optional controlled checkpoint deadline, with exit code 75; it is not a substitute for an allocation and is not used by the submitter. Resume preserves parameters, optimizer state, and constant rates. Convergence-window statistics are reconstructed from the complete saved trace across interruptions.

## Consolidating the pilot

Run `python -m experiments.expD06_fixed_center_scales.consolidate --root /workspace/junmiaoh/experiments/precision-mlps/runs/pilot --workers 4` inside an eight-CPU Slurm allocation with GPU execution disabled and two BLAS threads per process. This reads complete saved traces, retains failed and earlier-terminal cases in the accounting, and compares common horizons through 320k. The final three 20k training windows expose oscillation and drift; training-window rankings are descriptive optimization diagnostics, separate from validation endpoint rankings and untouched test data.

The consolidation refits every available 320k geometry and analyzes both seeds at each group's best paired endpoint and late-window rate pair, plus shared-LR controls. Representative histories retain projected and full residuals, signed forces, actual updates, core/halo probes, and Adam moment-to-epsilon ratios; cutoff checks use $10^{-10},10^{-12},10^{-14}$. Output is data and figures under `consolidation/`. The scientific report is authored separately after inspecting those artifacts. No new training or readout replacement occurs during consolidation.

## Historical initialization and coordinate comparison

`focused.py` specifies 88 new Adam trajectories at $N=512$ on the unit-RMS sine target, with fresh paired seeds 2 and 3. Every trajectory retains physical Xavier slopes, fixed centers, trainable readouts, and constant rates through the common 320k horizon. The physical slope draw has standard deviation $(5/3)\sqrt{2/(W+1)}$; canonicalizing its initial sign into the readout preserves the signed-Xavier function, and later slopes may cross zero. Bias starts at zero.

Four physical readout initializations are crossed independently with the training map: Gaussian Xavier on physical $w$; Gaussian Xavier on $a$ with $w=\sqrt h\,a$; Gaussian Xavier on $a$ with $w=D_{\rm ref}a$; and the legacy signed-envelope draw $w_j=\alpha_j\operatorname{sign}(\xi_j)$. The last is not Gaussian Xavier. Each seed uses the same Gaussian draws across all four families and both maps.

The uniform map trains $w=\sqrt h\,a,\ \gamma=\lambda/h$. The reference map trains $w=D_{\rm ref}a,\ \gamma=\lambda/h$. The three uniform-map rate pairs $(\eta_a,\eta_\lambda)$ are $(10^{-4},10^{-3})$, $(10^{-3},10^{-3})$, and $(10^{-3},10^{-2})$. Reference-map readout rates are multiplied by $\sqrt{h/\alpha_{\rm ordinary}}$ so ordinary-neuron physical Adam rate factors match. Both maps train the physical bias with rate $\eta_a\sqrt h$ and epsilon $10^{-8}$. This gives 48 primary trajectories. The reference metric differs from the uniform metric only on corrected halo slots after this matching.

For a positive coordinate scale $s$, Adam in mapped coordinates induces physical rate factor $\eta s$ and physical epsilon $\epsilon/s$. GD instead induces factor $\eta s^2$. The primary epsilon convention anchors the uniform map at $10^{-8}$, adjusting the reference readout epsilon by $\sqrt{\alpha_{\rm ordinary}/h}$ to match ordinary physical epsilon. Reference halo epsilons still vary with their scale. At $(10^{-3},10^{-2})$, 16 additional runs set every physical epsilon to $10^{-8}$; the stored-coordinate epsilon is then $s\,10^{-8}$. These controls measure the effect of epsilon while leaving the physical rate factors fixed. They do not establish optimizer-independent coordinate advantages.

The remaining 24 runs are fixed-center physical-coordinate controls at shared readout/slope rates $10^{-4},10^{-3},10^{-2}$ for all four initializations and both seeds. Their bias uses the same shared rate. They are nulls for this fixed-center model, not a trainable-center MLP baseline.

Submit `sbatch experiments/expD06_fixed_center_scales/focused.sbatch` from the remote `code` directory. The two one-GPU workers split by seed, verify their active Slurm step and runtime device count, and have separate cumulative 55-minute worker caps within one-hour allocations. The two-hour combined allocation limit covers this focused block only; check its ledgers and allocation overhead before any resubmission. Frontiers are 20k, 80k, 160k, and 320k; finite trajectories continue to the common horizon regardless of provisional settling labels. A budget interruption leaves a resumable incomplete experiment, never a convergence claim.

In addition to existing checkpoints and complete per-step traces, these runs save validation RMS every 1,000 updates from 260k through 320k. Consolidation reports all three final 20k windows, sampled validation RMS, net versus accumulated bandwidth movement, endpoint error, and detached fit quality separately. Test-grid evaluation remains disabled. Saved reference arrays include the actual per-coordinate physical rate factors and epsilons; environment records include source hashes, installed versions, allocation details, and the runtime GPU mask.

Consolidation also verifies the full finite trace, all 61 late validation samples in completed focused runs, and FP64 floating arrays in each 320k checkpoint. `spectral_history.json` preserves band RMS and signed regional forces over time, actual-update linearized descent by band, Parseval residual-energy closure, and measured update closure. `numerical_audit.json` and `cutoff_sensitivity.json` retain the numerical limits of these measurements. Per-case spectral figures use paired late-window selections; `focused_conditions.png` instead shows every primary crossed condition, and `epsilon_contrasts.json` pairs each epsilon control to the identical seed, initialization, map, and rates.
