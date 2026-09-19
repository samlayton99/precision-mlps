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

## Constant shared-rate sweep with neighbor differences

`difference_training.py` tests the [combined scale and difference prescription](../../docs/neighbor_difference_conditioning.md#combining-the-reference-scales-with-differences). Both arms evaluate the same physical tanh sum and differentiate it before pulling the readout gradient back through their respective maps. This isolates the coordinate metric from changes in forward evaluation. Checkpoints also compare the algebraically equivalent difference-form evaluation to detect cancellation.

The [constant-rate conditioning report](../../results/checkpoint_D_optimizers/expD06_fixed_center_scales/conditioning_results.md) consolidates the 100k search, transfer checks, and longer selected continuations. It keeps this study separate from the historical optimizer and relative-rate campaigns below.

The completed campaign contains 32 distinct trials: 24 finite 100k trajectories and eight recorded nonfinite failures. Four selected $N=512$ trajectories continued at unchanged scalar rates to a common 13 million updates. Slurm charged 6787 GPU-seconds against the separate 7200-second cap. This resource-limited horizon is not a convergence claim. The report links full-window summaries, spectra, projection audits, and separate seed animations; the raw training records remain under remote `runs/conditioning/`.

Start with full-batch FP64 GD on normalized sine, $N=512$, seed 0, the existing $16N+1$ training grid, and $R=\lceil\sqrt N\rceil$ halo. All centers stay fixed; both readouts and slopes learn. Compare **scaled training** ($w=D_wa$, $b=d_ba_b$, $\gamma=\lambda/h$) with **scaled neighbor differences** ($w=LS\theta$, $b=d_b\theta_b$, $\gamma=\lambda/h$). Keep the fixed reference construction at $\lambda_{\rm ref}=0.25$. Initialize both arms from the same `xavier_a_reference` physical network: Xavier on $a$, physical-slope Xavier, and zero bias. Convert that same network to cumulative coordinates; do not draw new independent $\theta$ values.

Use the following initial grid of **constant shared scalar rates**, matched across the two coordinate arms:

$$
\eta\in\{10^{-5},\ 3\times10^{-5},\ 10^{-4},\ 3\times10^{-4},\ 10^{-3},\ 3\times10^{-3},\ 10^{-2},\ 3\times10^{-2}\}.
$$

This gives 16 runs of **100,000 updates each**. The grid is an empirical scalar-rate search, not an additional theoretical scale prescription or a claimed optimum. Both trained blocks receive the same constant scalar throughout each run. No backtracking, schedules, clipping, separate block rates, or in-training least-squares solves are used. Every finite run reaches 100k; a nonfinite update is recorded as a numerical failure without silently lowering its LR. A best rate at the grid boundary calls for extending the grid, with the same 100k horizon and matched arms. A poor early loss is not grounds for screening a run out.

Rank rates by mean training MSE over updates 80k–100k, and report window variability and trends alongside that mean. Compare coordinates at each matched rate as well as each arm's best tested rate. Preserve per-step MSE, gradient norms and physical-update norms; parameter checkpoints with physical $w$, $\gamma$, and $\lambda$; residual Fourier spectra; and a final 2048-update dense window for gradient/update decomposition. A detached readout refit at the saved geometry diagnoses the remaining optimization gap and never changes training. The 32,768-point midpoint validation grid is diagnostic; the rate ranking uses training-window error. No held-out test selection or generalization claim is made.

The 100k comparison measures finite-horizon optimization performance, not a certified floor. After the initial sweep, extend a boundary optimum with matched rates if the remaining budget permits. Transfer the union of the two selected scalar rates, unchanged, to $N=512$, seed 1; then $N=1024$, seeds 0 and 1, in that order. Both arms receive each transferred rate. Remaining resources go to paired 100k continuations of the selected cases. Selection uses training-window means; numerical failures remain in the accounting. No claim of convergence follows from simply reaching a frontier.

The separately authorized budget is **two GPU-hours**, including compilation, checks, I/O, and failed allocations, with at most two GPUs concurrently. Reconcile actual Slurm allocation times and reserve the full requested walltime before each submission. This budget is separate from the completed eight-hour relative-rate campaign. All remote computation, including detached CPU diagnostics, runs through Slurm.

Submit `difference_training.sbatch --cases <absolute-matrix.json> --seconds <worker-deadline>` from the remote code directory, overriding its walltime only after checking the budget. The initial matrix is `difference_pilot.json`. Array worker 0 advances scaled training; worker 1 advances scaled neighbor differences. Each requests one GPU. A deadline checkpoints and pauses unfinished runs; resume uses the same matrix and frontier. It does not change the rate or turn a partial trajectory into an eligible 100k result.

Every update records half-MSE, native and physical gradient norms, physical movement, sign crossings, constant scalar rate, and zero-motion flags. Full parameter checkpoints occur at 0, 1, 10, 100, 1k, and every 2k. Dense windows retain all physical states, gradients, and rounded updates for the final 2048 updates before 20k, 60k, 100k, and later 100k frontiers. Numerical failure freezes only that batch member and records its first failing update. The first-order trajectories never receive detached refit coefficients.

`difference_analysis.sbatch` requests eight CPUs and no GPUs. With no extra arguments it exports complete-100k training-window rankings and parameter histories. `--cases <selected.json>` analyzes selected trajectories at initialization, 20k, 60k, and the requested endpoint in **both** coordinate maps at each identical saved geometry. It retains all singular values but reports numerical rank and refits at relative cutoffs $10^{-10},10^{-12},10^{-14}$; its resolved condition number excludes unresolved directions and is not the full condition number. It repeats the endpoint fit on a doubled grid. In each dense window, 64 fixed stratified samples receive exact finite readout, geometry, and interaction attribution; a fixed window-start SVD separates residual occupancy from actual function-space motion. Fourier bands include DC (the spatially constant component) and both signs of each frequency. Band energies sum to MSE; signed linear MSE contributions and the positive quadratic cost are reported separately.

`--uniform` evaluates detached uniform-bandwidth references at $N=512,1024$ and $\lambda=0.125,0.20,0.25,0.35,0.50,1.0$, with the actual bias, anchor, and halo scales. These are diagnostic matrices, not training runs or the ideal whole-line difference block from the proof. `--figures` plots the exported data; `--animations` locally renders separate seed movies with physical readouts above physical slopes, attached to fixed centers, at one saved checkpoint per second. These entry points write evidence artifacts only. The report is written separately after inspection.

For longer movies, the first 300k are shown slowly and later checkpoints at six frames per second; `--animations --late` magnifies physical changes over the final 256 consecutive states of the saved 2048-update window, at 20 frames per second. Consecutive sampling exposes short-period oscillations that a regular stride can hide. The offline HTML export retains every displayed frame. `--projection-only --cases <matrix.json>` refreshes endpoint forces without repeating the dense-window analysis. Perpendicular residuals are projected a second time to suppress leakage from the much larger in-span residual; the removed leakage and gradient-split closure are recorded. An additional analytic bias/outer-anchor bound explains why full halo-matrix conditioning can remain poor even when the interior difference block improves. It is evaluated separately from the cutoff-dependent retained condition numbers.

To reproduce the final comparison, pass `--end 13000000` to both the summary and selected-case analysis commands, with a separate output directory. Training resumes with `--frontier <update>` and the same case configuration; it preserves the constant rate. The final local export is `results/checkpoint_D_optimizers/expD06_fixed_center_scales/conditioning_analysis/continuation_final/`, alongside the initial search, 1-million-update summary, allocation ledger, source hashes, and selected initial/final physical states.

## Paired continuation at the learned geometry

The [relative-rate report](../../results/checkpoint_D_optimizers/expD06_fixed_center_scales/relative_rate_results.md)
records the matched comparisons, numerical limitations, and allocation accounting.

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

Dense NPZ archives retain every FP64 value without compression. A CPU-only
benchmark on an actual 512-width dense window took 4.53 seconds to compress
110 MB to 95 MB, versus 0.18 seconds to serialize the same arrays directly.
Both archive forms are readable by the same analysis. This avoids spending
the GPU allocation on compression; smaller diagnostic exports remain compressed.

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
stratified sampled states from the latest 2048-state window. One state is
selected per block of 32 using fixed random offsets (seed 391), to avoid
aliasing periodic optimizer motion. There are
at most two interventions. Thresholds are operational choices. Every check,
rejection, source window, cutoff, counterfactual, and schedule extension is
saved in `feedback.json`; no validation observation controls this policy.

`readout_solvers.py` compares frozen dictionaries at the 160k checkpoints
from both acquisition histories, plus a uniform $\lambda=0.25$ dictionary
at each width and target. GD, momentum GD (0.9), and Adam (0.9/0.999,
$\epsilon=10^{-8}$) use loss-only Armijo backtracking, separately from an
ordinary scheduled-Adam control. All zero-start comparisons have identical
initial predictions. Warm-start prescribed-coordinate Adam additionally
compares saved moments against reset moments, at rate $10^{-6}$.

The second coordinate map canonicalizes the frozen slope signs, then uses
neighbor differences $\phi_j-\phi_{j+1}$ and the final feature as an anchor,
with the bias unchanged. If $v_j$ are canonical physical weights, then
$q_j=\sum_{k\leq j}v_k$ and $v_j=q_j-q_{j-1}$. The fixed scales on $q_j$
are $\sqrt{\sum_{k\leq j}\alpha_k}$; the bias keeps its original scale.
This invertible, data-independent map preserves the represented functions.
It is not singular-vector whitening. The optimizer accesses cached feature
matrices through matrix-vector products, never normal equations or a
curvature inverse. Detached SVD diagnostics do not enter training.

Line-search trials start at 0.1 for GD/momentum or 0.001 for Adam. Later
trials double the previous accepted rate up to one, then halve until the
Armijo condition with coefficient $10^{-4}$ holds, with at most 40 trials.
An uphill momentum/Adam proposal falls back to the negative gradient for
that step. Failed searches and unchanged floating-point states are counted
as numerical stagnations, separately from convergence. Saved evidence
includes gradient/loss evaluation counts, accepted rates, fallbacks, timing,
coefficient paths, and 2048-state dense windows. Each frozen solve receives
at least 20k updates before comparison, subject to explicit budget reporting.
On resume, a completed frontier is skipped before rebuilding its dictionaries;
unfinished frontiers reload the saved parameters and optimizer state.

Recorded numerical stagnation prompted a separate paired audit,
`line_search_audit.py`, within the remaining allocation. It forks the six
uniform dictionaries and the learned quadratic/512 dictionary with observed
momentum stagnation at one common saved horizon. For every GD, momentum, and
Adam arm in both coordinate maps, it preserves parameters and moments and
compares the original Armijo evaluation with the algebraically identical
loss change $r^TB\Delta z+\|B\Delta z\|^2/2$. Here $B\Delta z$ uses the
rounded parameter displacement. This avoids comparing nearly equal losses;
an inherited zero trial rate restarts from the existing advertised initial
rate. A gradient fallback uses the advertised GD trial rate 0.1, rather than
inheriting a potentially tiny rate from an Adam-normalized direction. These
guards are reported together as a numerical repair, not an isolated test of
loss evaluation alone. There is no curvature inverse or least-squares update.
The original runs remain separate controls. The audit records its source
hashes, complete traces, dense windows, counters, and validation diagnostics
under `runs/ratio_line_search_audit/`. Its `--seed` argument partitions work
between two GPUs; it does not introduce new random initializations.

`analyze_line_search_audit.py --root <audit-directory> --output <export-directory>`
verifies identical initial parameters and optimizer state in each pair and
compares complete 20k windows at the common additional horizon. It exports
MSE, validation error, actual net physical displacement, accepted rates,
gradient/loss evaluation counts, and numerical-stagnation counts for all 84
arms. The figure selects the four persistent stalls identified before the
audit. The guards are interpreted as a combined intervention.

Use `--export=ALL,D06_MODULE=feedback` or
`--export=ALL,D06_MODULE=readout_solvers` with `ratio.sbatch` to run those
phases. Pass `--seconds` below the allocation walltime and reconcile the
eight-hour allocation ledger before submitting any phase.

`ratio_analysis.py` exports evidence and plots from a completed common primary
horizon, including ancestors' actual checkpoints. It checks three SVD cutoffs
and doubles the training-grid density for detached fits. Each final dense
window retains all 2048 physical parameter states; 64 stratified samples
receive Fourier gradient decomposition, actual-update attribution, and
readout/geometry/interaction loss budgets. A single SVD basis, fixed at the
window's first state, tracks residual and update directions across that
window. Frozen solvers use the same prescribed-coordinate reference SVD
projector for both coordinate maps, while separately reporting both singular
spectra. Thus changing numerical rank under a coordinate change cannot
silently redefine the residual attributed to the dictionary.
Frozen-solver exports use the same completed update horizon across all
available dictionaries and report coverage against the 30-dictionary matrix.
Checkpoint histories also retain residual MSE in every dyadic Fourier band;
line plots show its evolution beside bandwidth quantiles. These sampled
checkpoint errors are distinct from complete-window MSE statistics.

The analysis also records a limiting stability diagnostic for a frozen
readout quadratic. With $L=\sigma_{\max}(AD)^2$, constant-rate GD requires
$\eta L<2$. Momentum as implemented here requires $\eta L<2(1+\beta_1)$.
After Adam's second-moment square root decays below epsilon at a stationary solution,
its first-moment linearization requires
$\eta L<2(1+\beta_1)\epsilon/(1-\beta_1)$. This is an asymptotic local bound,
not a claim that the current trajectory is epsilon dominated, nor a
prediction of its observed error floor. Captured moment-to-epsilon ratios
and measured finite-update loss budgets provide that separate evidence.

Submit `ratio_analysis.sbatch --output <new-export-directory> --horizon 340000`
after that common horizon is available, and repeat into a separate directory
after all allocated continuations finish. The export records incomplete
matrix coverage and excludes branches with fewer than 20k new updates.
`--solvers-only` exports the frozen-dictionary block without repeating joint
diagnostics. Figures use MSE and label native rates explicitly. Reports are
authored directly after inspecting these outputs.

After downloading an export, run `animate_ratios --analysis <export> --n 512
--target sine` as a D06 Python module. It reuses the per-seed movie renderer:
physical $w$ above physical $\gamma$, fixed physical centers, and the first
320k shown slowly. Both actual scalar rates appear in every frame. The
default compares the shared and both-changes branches; `--labels` selects
other recorded branches. Repeat with `--n 1024 --target mixed` for the second
focused view. Every seed gets its own HTML and MP4 files, including a
magnified late-window view.

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

## Joint conditioning: individual scales and neighboring differences

The new `joint_conditioning` campaign compares individual parameter scales with
and without neighboring differences under GD, Adam, damped Gauss–Newton, and
SSBroyden. It has a fresh cap of **28,800 allocated GPU-seconds**, at most two
concurrent GPUs, and uses Slurm for all remote computation. The sine target,
FP64 half-MSE objective, fixed centers, reference allowances, and paired physical
Xavier initialization are unchanged.

For `parameter_scale`, $c_j=\alpha_j u_j$. For `parameter_differences`, set
$A_j=\sum_{k\le j}\alpha_k$, train $q_j=A_jv_j$, and recover
$w_j=q_j-q_{j-1}$ with $q_0=0$. Both train $b=\alpha_bu_b$ and
$\gamma=\lambda/h$. The last feature remains an anchor. The map stays fixed
after sign crossings; do not multiply by the individual scales again after
differencing. The physical initialization is paired by exact coordinate
conversion, not an independent draw on cumulative coefficients.

GD and Adam use one constant shared scalar rate for both blocks. This campaign
uses an explicitly specified native Adam epsilon, initially $10^{-12}$ outside
the square root, with no inside-square-root floor. Unlike the preceding diagonal
comparison, it does not claim matching physical epsilons across a nondiagonal map.
Historical optimizer defaults and saved configurations remain unchanged.

Every finite first-order trial receives at least 100k updates; higher-order
trials require 20k accepted physical updates unless they blow up or their search
fails. Rejected trials and unchanged states do not count as accepted updates.
Numerical failure, budget interruption, and convergence are distinct outcomes.
The main matrix uses width 512, seed 0, with seed 1 and width 1024 transferred
only when complete comparison blocks fit the remaining budget.

`joint_conditioning.py --prepare` writes the 44-case pilot: ten constant rates
$\{1,3\}10^k$, $k=-5,\ldots,-1$, for each first-order optimizer and map,
plus one default higher-order run per map. `--select` ranks completed runs by
mean MSE over the final fifth of their mandatory horizon, retaining median and
maximum MSE. It writes the guard and transfer manifests. If a higher-order
default fails before its mandatory horizon, its unchanged configuration remains
the confirmation candidate; this is not evidence that it was optimal.

Damped Gauss–Newton solves the augmented least-squares system by QR:

$$
\min_\delta\;\frac12\|r+J_z\delta\|^2+\frac\mu2\|\delta\|^2.
$$

Initially $\mu=10^{-3}\max_j\|(J_z)_j\|^2$. A trial is accepted when actual
over predicted reduction exceeds $10^{-4}$; damping decreases by three when
that ratio exceeds $0.75$, and increases tenfold on rejection. Forty rejected
trials terminate the run. The relative damping floor is initially $10^{-24}$.
There is no additional block learning-rate multiplier after this solve.
Undamped, uniquely solved Gauss–Newton is invariant to these invertible linear
maps; native-coordinate damping changes the physical regularization metric.
Rank-deficient minimum-norm solutions need not have identical parameter steps.

SSBroyden uses the [specified library](https://github.com/IvanBioli/ssbroyden_optimistix)
at commit `4c87785c68f0fec6b09000f474daef76fb181eea`, with its Optimistix
submodule at `8cd4931713658f8dfe4423ead6f11b348b675540`. The adapter checks
the source hash and exposes the explicit $s^Ty$ threshold, initially
$10^{-24}$. Zoom starts each search at one, uses $c_1=10^{-4}$, $c_2=0.9$,
the library's approximate-Wolfe $c_3=10^{-6}$, and at most 64 evaluations.
Its minimum step and interval thresholds initially equal $10^{-15}$.
An accepted state with no representable physical change is recorded as a
terminal numerical stall, not counted toward the horizon. Initial evaluations
and rejected line-search points likewise do not count.

An integration audit found that this pinned source passed Zoom's **next proposed
step** into the SSBroyden scaling calculation. That calculation requires the
**accepted step** $\alpha_k$, since

$$
b_k=\frac{s_k^TB_ks_k}{s_k^Ty_k}
=-\alpha_k\frac{s_k^Tg_k}{s_k^Ty_k},\qquad s_k=-\alpha_kH_kg_k.
$$

The `accepted_step` adapter corrects this one argument using
`search_state.stepsize`; the search and scaling formulas otherwise remain those
of the pinned library. A non-unit-step quadratic check verifies the resulting
inverse-Hessian update against the formula in the
[authors' technical note](https://arxiv.org/html/2603.10599v1#S2).
Cases explicitly record `ssb_integration=accepted_step` in their identities.
Earlier source-matching runs remain audit evidence and are excluded from
scientific SSBroyden selection; their early search failures do not establish
a limitation of the correctly integrated method.

Guard comparisons change one setting at a time: Adam epsilon outside the square
root ($10^{-8},10^{-12},10^{-15}$, with `eps_root=0`); SSBroyden curvature
threshold and GN relative damping floor (machine epsilon, $10^{-24},10^{-30}$);
and SSBroyden step/interval thresholds ($10^{-6},10^{-12},10^{-15}$).
The upstream SSBroyden checks on finite inverse-curvature/scaling factors remain
in place; the adapter does not replace every floating-point guard indiscriminately.

`joint_conditioning.sbatch` launches one map on each of two Slurm array tasks.
`--benchmark` pauses the actual pilot trajectories after 2k first-order or 100
higher-order updates for throughput measurement. Resume these checkpoints before
making scientific comparisons. Higher-order checkpoints reconstruct all dynamic
solver arrays against the original static structure, preserving Lineax tag
identity. Both methods retain failed proposals and the last accepted state.
The GPU budget includes all benchmark/compile/allocation overhead; reconcile
Slurm accounting before admitting further complete comparison blocks.

`joint_analysis.py` reads saved states without advancing them. At each selected
checkpoint it compares both readout maps and both joint Jacobians at the same
physical geometry. Detached least-squares refits use a common individual-scale
readout basis and relative cutoffs $10^{-10},10^{-12},10^{-14}$; they are never
inserted into training. Undamped joint shadow solves use the same explicit
cutoffs. Differences between their minimum-norm steps at rank deficiency are
recorded, not interpreted as a violation of full-rank GN invariance.

The early and final dense windows retain actual finite readout and geometry
updates. Sixteen reproducibly sampled updates per window receive Fourier,
singular-direction, gradient-projection, and exact finite-update budget audits.
Fourier bands include DC (the constant component), both signs of each frequency,
and every DFT index through Nyquist. Figures show both absolute band MSE and its
percentage of the same sampled mean MSE. Singular occupancy uses a fixed
window-start basis, including an explicit outside-basis remainder. Endpoint
checks include a doubled training grid, held-out midpoint grid, and 80-digit
spot evaluations of both stored physical parameters and the native-coordinate
decode. These checks distinguish measured numerical error from convergence.

The overnight follow-up adds two **post-hoc detached diagnostics**, without
changing the training matrix, selected settings, or eight-GPU-hour cap. First,
saved SVDs predict frozen-readout GD residual decay at the stable spectral step
$1/\sigma_{\max}^2$, repeating relative cutoffs $10^{-10},10^{-12},10^{-14}$.
This is an analytic fixed-dictionary prediction, not additional joint training
or an Adam prediction. Second, the difficult width-1024, seed-0 GN checkpoint
receives a damping sweep of trial steps in both coordinate maps, recording
predicted/actual reduction and separate readout/geometry motion. An augmented-QR
control transforms the damping penalty to the same physical metric in both
maps. Agreement of those physical steps verifies damped coordinate invariance
when the metric itself is held fixed. These probes use CPU-only Slurm jobs;
neither their trial parameters nor their readout fits enter training. Their
purpose is to distinguish spectral attenuation, nonlinear step rejection, and
changes in the optimizer's metric; they do not select new hyperparameters.
The same saved dictionaries also receive detached readout fits to normalized
sines with 2, 4, 8, 16, 32, and 64 cycles across the domain. A common
individual-scale SVD basis, three cutoffs, midpoint errors, and physical
coefficient norms distinguish target-specific fitting from reusable localization.
Uniform slopes at reference bandwidth $0.25$ provide a geometry comparison.
These changed right-hand sides are diagnostic probes, not newly trained targets;
the construction allowances and all training trajectories remain fixed.
For learned states, the frequency probes also project each new residual outside
the retained readout space and measure its instantaneous geometry force with
the **stored trained readouts**. This tests the gamma-signal hypothesis without
substituting the detached fitted coefficients into the geometry Jacobian;
core and halo contributions and all three projection cutoffs are retained.

## Parameter-scale normalization: paired GD and Adam

The [parameter-scale report](../../results/checkpoint_D_optimizers/expD06_fixed_center_scales/parameter_scale_results.md)
contains the completed 64-trial search/transfer study and 16 selected trajectories
through 2.3 million updates. It reports matched-rate bandwidths, physical update
scales, loss distributions, Fourier/SVD diagnostics, and separate seed movies.
Slurm charged 7075 GPU-seconds against this campaign's separate 7200-second cap;
the endpoint is resource-limited, not a convergence claim.

This campaign tests whether normalizing individual parameters by their construction
allowances sustains useful geometry learning. Collective normalization trains
$c=Da$; parameter-scale normalization trains $c=D^2u$, including the bias and
corrected halos. Both train $\lambda$ with $\gamma=\lambda/h$. The same reference
$\alpha_j=D_{jj}^2$ at $\lambda_{\rm ref}=0.25$ stays fixed throughout training.
Ordinary and corrected-halo readouts have $O(h)$ construction bounds at fixed
lambda, with different constants; the bias is $O(1)$. These are reference
allowances, not constraints on the training trajectory.

The physical initialization is identical across arms: draw the existing Xavier
$a$ and physical gamma once, set $c_0=Da_0$, and convert $u_0=D^{-1}a_0$.
Bias starts at zero, initial slope signs are absorbed into the readouts, and
subsequent signed slopes are unrestricted. There is no independent redraw of $u$.

Full-batch FP64 minimizes half-MSE on the normalized sine target and $16N+1$
points. Report MSE. Each trial uses one constant shared scalar $\eta$ for
readout and geometry. GD has no momentum; Adam uses 0.9/0.999. Native readout
epsilon is $10^{-8}$ for the collective control and $10^{-8}D_{jj}$ for
parameter-scale normalization, preserving the physical threshold $10^{-8}/D_{jj}$.
Native geometry epsilon remains $10^{-8}$ in both. There are no schedules,
frozen parameter blocks, neighbor differences, or in-training readout solves.

`parameter_scale.py --prepare --root <run-root>` creates the 40-trial pilot:
both optimizers, both maps, and rates $\{1,3\}10^k$ for $k=-5,-4,-3,-2,-1$
at $N=512$, seed 0. Every finite pilot receives 100k updates. Select by mean
training MSE over updates 80k–100k, breaking ties toward smaller rates.
`--select` writes the selected, boundary-extension, confirmation, and continuation
manifests. Extend a selected boundary by at most one decade in both maps for
that optimizer, subject to reserving the mandatory confirmation budget.
Transfer the union of each optimizer's two selected rates to $(N,\mathrm{seed})$
equal to $(512,1),(1024,0),(1024,1)$, without retuning. Continue each map's
selected rate across both widths and seeds to 300k, then common 100k blocks
within the budget. A budget interruption or oscillation is not convergence.

Submit `parameter_scale.sbatch --manifest <absolute-manifest> --frontier 100000
--seconds <worker-deadline>` through Slurm. Worker 0 runs GD, worker 1 Adam.
Use at most two allocated GPUs and preserve the scheduler's device mask.
The **new, separate cap is 7200 allocated GPU-seconds**, including setup,
compilation, verification, and failed runs. Reconcile accounting before every
submission. Detached analysis uses CPU-only Slurm jobs.

The runner saves complete scalar traces, resumable optimizer states, early
checkpoints, and consecutive dense records over the first 2048 updates and
late 2048-update windows. Diagnostics compare the actual parameter motion,
signed scale-growth forces, residual Fourier bands, singular residual/update
occupancy, and detached fits using a common reference projector rule. Repeat
projection cutoffs and sampling density before interpreting tiny forces. The
32,768 midpoint points are diagnostic evaluation, not rate selection or an
independent generalization test. Larger lambda alone is not a success criterion.

`parameter_scale_analysis.sbatch --output <export> --cases <selected-manifest>
--end <common-horizon>` performs detached CPU analysis. It exports both native
readout spectra and a common-reference projection/refit using $AD/\sqrt M$ in
both arms. Three SVD cutoffs and a doubled sampling grid expose numerical
sensitivity. The early and late dense windows use 16 stratified states for
Fourier and finite-update budgets; all 2048 consecutive states contribute to
the adjacent-step motion audit. Adam panels use its actual moment-dependent
updates, not a GD modal-rate prediction. `--movies` exports one comparison per
optimizer and seed at width 512, with physical readouts above physical gamma,
slow first-300k playback, and separate consecutive-update late close-ups.
Movie rendering requires an `ffmpeg` executable. It can run locally with
`python -m experiments.expD06_fixed_center_scales.parameter_scale_analysis
--root <run-root> --output <downloaded-export> --movies` after downloading the
summary, histories, and dense parameter archives; this mode does not read or
modify the training root.

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
