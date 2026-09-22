# Ordinary-GD readout competition

## Adam force attribution and additional targets

`frozen_scale_gap --root <Adam-evidence> --output <comparison-directory>`
compares learned Adam geometry with common frozen gammas through 256 and with
multiples of its own learned slopes. The latter multiplies both $a$ and $b$,
preserving feature centers and relative slope magnitudes. Every geometry gets
the same zero-initialized readout-GD assay at 0.002 through 600k steps, evaluated
by the verified linear recurrence. Scale selection uses training-grid error;
independent-grid error and doubled-grid checks are reported separately. This
measures the best tested scale under the stated assay, not a universal optimal
gamma. Run the numerical work and figure generation in CPU-only Runpod Slurm.

The [implemented plan](../../docs/d34_adam_force_plan.md) specifies the target
definitions, comparisons, measurements, and interpretation gates. The
[completed report](../../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/README.md)
collects all 184 cases and the distinction between force activity and net growth.

`adam_forces.py` defines thirteen targets and an exact full-residual split into
effective force and coarse-tracking correction. The higher-mode residual is
the entire complement of the constant/linear projector, with no activation or
polynomial truncation. All parameter blocks enter the coarse kernel, including
the output bias. The five original targets retain their exact definitions;
new mixed-sine, localized-sine, and chirp targets use unit training-grid RMS.

Auxiliary first-moment buffers propagate the force components through ordinary
Adam using its actual shared second-moment denominator. Their steps add to the
actual optimizer step. They are measurements, not independent Adam optimizers
or interventions in training. A separate diagonal-mobility balance can be
evaluated with the actual Adam preconditioner; it is not a claim that Adam
obeys the GD tracking theorem. Unresolved coarse solves retain the full gradient
in an explicitly unresolved channel rather than changing the training update.

Focused checks compare gradients and coarse Jacobians with PyTorch autodiff,
reconcile the full-complement split with the retained-basis split, and check
ordinary Adam and component histories against PyTorch under cancellation,
tiny gradients, and zero-gradient intervals:

```bash
JAX_ENABLE_X64=true .venv/bin/python -m pytest -q tests/test_expD34_adam_forces.py
```

`adam_run.py` runs five paired primary seed bundles, two Adam rate-sensitivity
bundles, three control bundles, and one epsilon-sensitivity bundle: 184 cases
through 600k updates. Primary GD and Adam use 0.002; Adam rates 0.0002 and 0.001
are separate sensitivity cases. Adam uses betas (0.9, 0.999), epsilon 1e-8,
and zero initial moments. Controls use bias-corrected EMA momentum without
adaptive scaling, or Adam with beta1 zero. Epsilon 1e-12 is a separate control.
The existing raw physical coordinates and half-MSE are unchanged.

Every-update accumulators retain raw-force allowances, actual component path
lengths, signed outward contributions, actual positive/negative travel, and
reconstruction errors. Trace rows describe the gradient state before the last
update of their interval; interval minima and maxima retain intervening
excursions. Snapshots and the resumable state retain all moment buffers. A
nonfinite training case is frozen and explicitly marked failed; unresolved
decomposition alone does not alter training. Loss increases are recorded, not
used for selecting a checkpoint or stopping a finite run.

Use `adam.sbatch <campaign-root> <stage> [end-step]` with arrays `0-4%2` for
`primary`, `0-1%2` for `rates`, `0-2%2` for `controls`, and index 0 for
`epsilon`. Queue dependent stages to keep at most two campaign GPUs active;
each array task requests one GPU and preserves Slurm's allocation mask. The
new extension has a six allocated GPU-hour ceiling within the earlier total
eight-hour allowance. Include compilation and unsuccessful allocations.

`adam_analyze --source <completed-bundle> --output <analysis-directory>` exports
state diagnostics, exact motion-window accounting, modal refinement, and
frozen-geometry curves. It requires the full 600k horizon. `--construction`
instead exports the common-gamma references. The Adam balance check uses the
virtual next update's bias-corrected moments and denominator, retaining both
the momentum-lag forcing and actual finite-step coarse-residual defect.
The virtual update at the terminal state is not part of the trained trajectory;
actual-step attribution comes from online traces and cumulative measurements.
These measurements do not assume that the GD balance is Adam's equilibrium.
Analysis tables retain all cases and failure flags; scientific interpretation
and figure captions are written after inspecting the outputs.

`adam_analysis.sbatch <campaign-root>` runs the numerical analysis, verification,
curation, and figures in a CPU-only Slurm allocation on Runpod. It explicitly
disables GPUs. `adam_summarize --root <evidence> --figures-only` redraws the
figures from merged tables and curated traces. Curated traces retain selected
last-state values and exact minimum/maximum envelopes over each combined
interval; cumulative movement comes from every-update accumulators, not
quadrature of these plotting samples. `adam_verify` checks the original GD
replays and fixed-state grid refinement; `--curate` exports selected complete
optimizer states and the coarsened traces with source hashes.

## Signal-recovery audit

`recovery.py` analyzes whether renewed slope signal produces signed movement
and population-scale acquisition. It reads the existing evidence packages,
including exact cumulative signed-force summaries; it never integrates sparse
plotting traces. The primary targets are sine, degree three, and degree nine.
The 20k state is the fixed comparison baseline, not a selected signal minimum.

```bash
MPLCONFIGDIR=/tmp/race-mpl python -m experiments.expD34_readout_race.recovery \
  --evidence results/checkpoint_D_optimizers/expD34_readout_race \
  --output results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery
```

The original raw archive was retired after curation. To recover neuronwise
diagnostics, `replay_recovery.py` replays only width 177, seeds 0–4, those three
targets, and equal physical rates 0.002. It uses the unchanged D34 vector field,
checks original initial/data hashes, and checks archived scalar observations
at 20k, 100k, and 600k. Run it in a GPU Slurm step with `JAX_ENABLE_X64=true`,
`--evidence <curated-evidence> --output <compact-replay-directory>`.
The executable verifies the allocation and GPU mask. It stores selected states
and per-neuron upward/downward travel accumulated at every update, rather than
recreating the retired dense archive. Pass its `compact_states.npz` to the
analysis command with `--replay`.

New measurements separate signed alignment, concentration of positive travel,
endpoint net movement, residual filtering, coefficient amplification, and
output-bias effects. Hessian-block attribution is an instantaneous flow
diagnostic at a GD state. Reference population bounds are conditional on a
justified error radius; observed prediction errors are not certificates.
Raw slope thresholds and construction bandwidths remain diagnostics, not
universal necessary conditions for approximation.

Focused checks: `JAX_ENABLE_X64=true python -m pytest -q tests/test_expD34_recovery.py`.

The completed [focused report](../../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md) provides the compact replay, per-seed tables, population budgets, and verification records. Its [theory companion](../../docs/d34_scale_acquisition_theory.md) states the actual-initialization scope. Early integrated attribution uses 20-update trapezoids through update 2,000, checks a 40-update coarsening, and reports its discrepancy from the observed GD log decline. It is an approximate integration of flow diagnostics, not an exact finite-step attribution. The generic energy-budget comparison checks the final update in addition to the archived 599,999 descent ratios.

This experiment tests whether readout adaptation depletes residual moments before a substantial population of hidden slopes can grow. The evidence sought is an initialization-only prediction of signed scale trajectories and their changes across readout rates. Successful execution does not require the hypothesis to hold.

Measured findings and limitations are in the [results report](../../results/checkpoint_D_optimizers/expD34_readout_race/REPORT.md), with [curated evidence](../../results/checkpoint_D_optimizers/expD34_readout_race/README.md) that regenerates the figures without training.

**Notation.** These symbols refer to physical raw coordinates throughout.

| Symbol | Meaning |
|---|---|
| $f=d+\sum_j c_j\tanh(a_jx+b_j)$ | Network with independently trained slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=\lvert a_j\rvert$ | Reported slope magnitude; neuron indices and live signs are preserved. |
| $\lambda_j=(2/N)\gamma_j$ | Reporting scale tied to the reference budget, not to sample or evolving-center spacing. |
| $\eta$, $\kappa$ | Geometry rate and readout/geometry rate ratio. Output bias uses the readout rate. |
| Polynomial reference | Independently evolved degree-1/3/5/7 activation, using only its own state and fixed target moments. |

## Locked protocol

All four parameter blocks update simultaneously from the old state, using the loss $L=\tfrac12\operatorname{mean}(f-y)^2$. There is no momentum, normalization of training residuals, refit, clipping, freeze, parameter remapping, or schedule. Reported MSE is $2L$.

The main matrix uses $(N,H,W)=(64,12,89),(128,24,177),(256,48,353)$, seeds 0–4, and $\kappa\in\{10^{-4},10^{-3},10^{-2},0.1,1,10,100\}$. The geometry rate is $0.002$. All finite scientific trajectories receive at least 20,000 updates. Training has 2,048 midpoints on $[-1,1]$ at every width; evaluation uses 8,192 different midpoints. Two additional seed-0, width-177 baseline comparisons retain D28's 1,024 training samples for sine and Runge. Independent evaluation diagnoses interpolation error and is not used to choose a winning optimizer, rate, or checkpoint.

Initialization calls the unchanged D28 helper. Slopes and hidden biases are independent uniform Xavier draws with bound $\sqrt{6/(W+1)}$ from NumPy's stream seeded by `[seed, N]`. Readouts use a separate `[seed, N, 24]` stream and the same bound. Output bias is zero. There is no tanh gain, readout norm rescaling, reference-envelope multiplier, or sign canonicalization. Copies of these physical arrays initialize every paired rate and reference.

Experiment A uses the upstream sine and Runge target amplitudes. Experiment B uses empirical orthonormal Legendre polynomials fitted once on the training grid:

$$
y_k=0.3\phi_0+0.4\phi_1+\sqrt{0.75}\phi_k,\qquad k=3,5,9.
$$

Every target has training RMS one and the same first two target coefficients. The saved polynomial coefficient map evaluates the same function on independent grids. The affine references must agree across all three targets; the cubic references must agree for $k=5,9$. Actual tanh trajectories can differ immediately. The main matrix contains 525 tanh and 2,100 reference trajectories.

## Mathematical verification

`references.py` computes polynomial coefficients and the exact moment-gradient pullback. Its separate affine Gram recurrence retains the quadratic finite-step term and reconstructs individual states with a common $3\times3$ map. Higher-order references retain all parameters. `core.py` implements simultaneous GD and analytic tanh gradients; its stable expression for $\operatorname{sech}^2$ avoids saturation cancellation. FP64 is enabled by the experiment entry point or test environment, not as an import side effect of the kernels.

Tests compare the analytic gradients with independent sample-space differentiation, the affine recurrence with explicit polynomial GD at extreme rates, the initialization and early trajectory with the unchanged PyTorch D28 trainer, and the state-conditioned moment remainder bound. A reference loss computed from moments can suffer cancellation; it must be checked against direct sample evaluation at saved states and never silently clipped as a training objective.

## Execution and evidence

GPU computation must run inside Slurm on the allocated devices, with at most two H200s concurrently. The initial allocation budget is two GPU-hours. Core comparisons precede half-step, equal-geometry-time refinements. Remaining time funds balanced width-177 continuations toward 100k, seeds 5–9, then further common-horizon continuations. A finite trajectory interrupted by the allocation remains incomplete, not converged. Polynomial breakdown does not terminate its independently trained tanh counterpart.

The diagnostic record includes signed mean and median scale changes, quantiles, population thresholds, residual moments, coarse and remainder signed slope forces, parameter gradients and actual changes, readout norms, full states, reference prediction errors, and independent-grid errors. Hypothesis assessment must use paired rate contrasts and reference-validity checks, not just declining loss or a few large slopes. Numerical findings belong in the results report after execution.

`JAX_ENABLE_X64=true python -m experiments.expD34_readout_race.run --root <run-root> --stage core --frontier 20000` runs the core matrix. Stages `baseline`, `refine`, `continue`, and `replicate` select the two historical-grid checks, width-177 seed-0 half-step comparisons, width-177 common continuations, and seeds 5–9 respectively. Use `--frontier 40000` for the equal-time refinement. `--worker 0 --workers 2` and `--worker 1 --workers 2` split complete seed/width bundles between the two workers. Reinvocation automatically resumes the saved GD state without changing its rate or initialization.

Each bundle stores its immutable protocol and physical arrays once. Subdirectories `p0,p1,p3,p5,p7` contain independent resumable states, full every-update scalar traces, gradients and states at updates 0–20 and every 20 updates thereafter, and the first coarse-threshold crossing states. `p0` is tanh. Snapshot step $n$ and trace step $n$ refer to the state before update $n\to n+1$. The final state is also retained in `state.npz`. Normalized residual moments and sensitivities can be reconstructed from the stored unnormalized moments, gradient norms and half-MSE; undefined zero-residual ratios must remain undefined.

Submit `run.sbatch` only after deploying a committed source snapshot into the isolated remote `race-code` directory and recording its commit in `.source_commit`. The launcher preserves Slurm's device mask and the runner checks an active GPU job/step and exactly one visible GPU. The deadline writes a resumable incomplete trajectory; it never promotes a partial run into the scientific comparison. Raw NPZ evidence stays outside Git; curated numerical summaries and figures accompany the final report.

`python -m experiments.expD34_readout_race.analyze --root <run-root> --output <evidence-directory> --end 20000` exports tables and compact plotting arrays. Pass `--bundles` to analyze named bundles independently. Use `analyze.sbatch` for remote CPU analysis. Common-horizon checkpoints at each 20k boundary support comparisons after training continues. Incomplete and failed trajectories are identified explicitly and excluded from endpoint rate contrasts.

Sample-space checks use updates 0–20 and the saved states nearest 61 logarithmically spaced times. They record direct versus moment-computed loss, exact readout and geometry contributions to coarse-residual velocity, and state-conditioned slope-gradient remainder intervals. Reference trajectory errors and matched-target differences use every saved state. The interval is a pointwise statement at the observed state, not a guarantee that an independently evolving reference remains nearby.

For long continuations, `--sparse-snapshots` restricts the analyzed full states to updates 0–20 and 61 logarithmic times; the analysis manifest records this restriction. Every-update scalar traces still supply exact event times, integrated signed forces, loss-increase counts, and endpoint windows. The raw full-state archive retains every 20th update regardless of this analysis option.

`python -m experiments.expD34_readout_race.plot --root <evidence-directory>` renders line plots from the exported evidence. Full rate and seed coverage appears in scale and endpoint-contrast plots; detailed signal and reference-error panels use the predefined ratios $10^{-4},1,100$. Every plot labels geometry time $\tau=\eta n$ or the readout/geometry rate ratio. No script authors the scientific report.

`python -m experiments.expD34_readout_race.curate --inputs <completed-analysis-directories> --output <review-directory>` combines analyses at the same endpoint, losslessly compresses plotting arrays, and reduces full diagnostic tables into per-trajectory audits. It retains all summary rows and all rate contrasts. Its reference-error threshold of 5% is a descriptive reporting threshold, not a scientific acceptance rule. Full diagnostic tables remain at the source paths recorded in `evidence_manifest.json`.

The completed extension uses `--stage continue --frontier 600000` for the original five width-177 seeds and `--stage refine --frontier 200000` for the equal-time check against primary 100k training. These commands continue the saved states without resetting parameters or rates. The final reports retain the 20k and 100k comparisons separately.

`python -m experiments.expD34_readout_race.curvature_check --root <run-root> --output <csv>` computes the exact empirical half-MSE Hessian at saved 20k, 100k, and 600k states for width 177, seeds 0–4, and ratios 1 and 100. It includes residual-weighted second derivatives. Its eigenvalues are for $D^{1/2}\nabla^2L D^{1/2}$, where $D$ contains the physical block learning rates. Eigenvector block fractions are measured in these step-scaled coordinates. This is an offline local stability diagnostic, never an optimizer or an intervention in training.

`compress_affine.py` losslessly compresses completed affine NPZ archives and verifies every array's dtype, shape, and bytes before replacing the original. It excludes mutable `state.npz`. Its worker partitions are disjoint; verification ledgers record each replacement. The runner also compresses new affine archives and checks successful writes before atomic replacement. Curation accepts either CSV or lossless gzip tables. These storage measures preserve numerical evidence after the shared volume stopped accepting complete writes during the long continuation.

## Transport and residual-basis extension

`transport_run.py` evolves exact tanh features using either the full residual (`--degree -1`) or its empirical polynomial projection. Degree 33 is the predefined primary forecast; degrees 9, 17, and 65 check projection error. Every forecast evolves independently from the original random initialization, without subsequent true states as inputs. The primary horizon is physical time 1200 (600,000 updates at $\eta=0.002$). Half-step comparisons use the same physical horizon.

With `--nodes 0`, particles are the actual width-$W$ neurons, each with mass $1/W$. With positive `--nodes q`, $q^3$ Gauss–Legendre nodes integrate the independent uniform initialization law. Physical width still controls the model output and initialization scale; quadrature count controls integration accuracy. Its characteristic velocity differs from the Euclidean gradient of quadrature coordinates by the factor $Ww_i$. Thus law quadrature and a larger neural network are distinct experiments. Orders 8, 12, and 16 check the law approximation; the full residual at order 12 checks the modal approximation independently.

The original five seeds and three targets establish numerical adequacy. Fresh seeds 10–14 assess the locked degree-33 forecast; secondary checks use readout ratios 0.1 and 10 on seeds 10–12 and widths 89 and 353 on seeds 0–2. Main outcomes are finite-time population fractions above $\gamma=1,3.2,16$, positive and negative scale travel, slope path and energy, layer-balance defect, omitted-mode force, and coarse/fine kernel coupling. A few escaping neurons do not invalidate a population barrier. Projection error at a true state and error of an independently evolved trajectory must be reported separately.

The extension has an eight aggregate GPU-hour ceiling, including compilation and failed attempts. Its jobs request one GPU and run serially through Slurm, leaving another GPU slot for concurrent work. The first two hours cover numerical adequacy, then up to three hours each cover fresh predictions and secondary controls. Source hashes, initialization/data hashes, GPU environment, status, cumulative motion, sparse curves, and selected full states are saved. A deadline produces an explicitly incomplete resumable run. No report is generated by code.

Adaptive verification adds law quadrature orders 24 and 32 after the order-12/16 tail discrepancy, a full-residual order-32 check after sharper law particles show measurable modal-force error, and a matched-time half-step check at order 12. Short order-8 law runs at widths 89 and 353 through physical time 40 separate finite-seed variation from the formal width-scaling prediction. These controls retain the frozen primary degree-33 predictor and the original eight-GPU-hour ceiling. The final report distinguishes these adaptive checks from the predeclared fresh-seed validation.
# Useful slope growth and mechanism diagnostics

`mechanism.py` analyzes compact actual-GD archives without modifying them.
Its force drivers distinguish evolving residuals from evolving sensitivities,
and its signed movement measurements distinguish available force from growth.
The optional frozen-geometry diagnostic trains raw readouts from a common zero
state by evaluating the exact linear GD recurrence through an SVD. This is a
geometry comparison; ordinary D34 training retains its random readout.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD34_readout_race.mechanism \
  --archives results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/compact_states.npz \
  --output results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/baseline_metrics
```

Add `--geometry` and use a separate output directory for frozen-readout curves
and capacity diagnostics. Curves use the original 2,048-point target map and an
8,192-point evaluation grid, physical step 0.002, and unchanged raw readout
coordinates. The capacity solves are separate from the finite-budget curves.

`replay_recovery --targets runge moment5` adds the two missing equal-rate
baselines and verifies their initialization, targets, and archived observations.
`mechanism_run` accepts these and the original compact archive through
`--archives`, one `--seed`, and `--fork-step 20000` or `100000`. Its six default
arms share exactly the same saved state. Frozen blocks remain bitwise unchanged.
The saved `steps` always use the original 0.002 reference clock; a half-step
refinement takes two updates per clock tick and records `training_eta=0.001`.
Positive/negative travel and path are accumulated at every actual update.
Completed archives can resume unchanged toward a larger `--end-step`.

GPU entry points must run through `mechanism.sbatch` in an isolated committed
checkout. Queue these jobs sequentially or with an array concurrency of one.

`mechanism_summarize --root <evidence> --archives <completed archives...>` joins
the `*_metrics` and `*_geometry` outputs into per-seed endpoints, paired
intervention contrasts, numerical identity checks, and figures. Keep half-step
verification outputs in a separate subdirectory rather than pooling them as
additional independent seeds. Reports are written directly after inspection.

`mechanism_verify --root <evidence> --primary <original compact archive>` checks
the completed continuations against original GD, every frozen block, all 60
matched half-step cases, degree-129 modal diagnostics, and doubled frozen-study
grids. `mechanism_curate --root <evidence> --archives <full archives...>` retains
all scalar tables losslessly, mode-0-through-9 residual coefficients,
mode-2-through-9 force curves, and verified states
at fixed milestones and sampled post-20k force maxima. The full compact archives
remain separate from this curated Git evidence package.
