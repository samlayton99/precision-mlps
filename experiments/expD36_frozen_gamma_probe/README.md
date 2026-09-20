# Frozen-gamma theorem probe

This experiment measures whether the necessary-polynomial-tail theorem explains
readout learning times. The first matrix contains two targets, four frozen
slopes, and raw/collective coordinates at core resolution $N=512$ and hidden
width $W=559$. The configuration is frozen before execution. Zero readout
initialization makes the target the initial correction in every cell.

The sine mixture is $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(10\pi x)$; the
control is $\sqrt{5}x^2$. These are not D06's differently defined `mixed` target.
D06 supplies the corrected-halo geometry and reference allowances, held fixed
as gamma varies. All training uses half empirical MSE in FP64.

## Evidence roles

The endpoint grid supplies training and theorem measurements. The offset
validation grid selects between the two declared Adam learning rates. The
independent midpoint evaluation grid measures fixed-budget precision and never
selects a recipe. All features are frozen; only readouts are trainable.

Dense QR, SVD, refits, and high-precision calculations are offline diagnostics.
They do not enter optimizer updates or claim scalable-optimizer performance.
The older optimizer-program promotion gates do not govern this theorem audit.

## Diagnostic conventions

Unpivoted Householder transforms retain every sample-space complement row.
An independent discrete orthogonal-polynomial recurrence checks the projection.
The code saves directional access, sampled subspace access, Frobenius norms,
and analytic bounds separately. SVD cutoff floors describe retained numerical
models, not mathematical rank. Predicted large horizons remain extrapolations.

FP64 access estimates carry a heuristic roundoff monitor, not a rigorous error
enclosure. High-precision agreement is reported separately and does not create
an interval certificate. The nominal real tanh dictionary and exact arithmetic
on a stored FP64 matrix are distinct numerical objects.

The quadratic has exactly zero polynomial tail above degree two. Its measured
roundoff tail is retained in artifacts but cannot generate a time certificate.
This control need not have gamma-independent optimization dynamics.

## Execution contract

The authorized limit is two allocated H200 GPU-hours, at most two GPUs at once,
and a CPU diagnostic job capped at 60 minutes. All remote computation uses
Slurm. Initial GPU allocations request at most 55 minutes each, with the
remaining ten GPU-minutes reserved for necessary checks/retries. Compilation,
idle allocated time, and failures count. No automatic scope or budget expansion.

Implementation checks cover projection/complement identities, target definitions,
access bounds, the single-mode discrete certificate, mixed-mode slack, gradients,
coordinate maps, and actual GD versus spectral evolution. Scientific success
does not require tight bounds or a favorable gamma intervention.

The [completed results report](../../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/REPORT.md)
distinguishes trained precision, attainable tolerances, theorem slack,
arithmetic limitations, and unexecuted horizons. All 16 GD cases and 16 Adam
trials completed using 0.0575 allocated GPU-hours. The principal signal is
capacity-sufficient but very slow GD at gamma 4; directional lower bounds still
understate the two observed sine-mixture hitting times by about 44–52 times.

## Entry points

From a source snapshot, run `python -m experiments.expD36_frozen_gamma_probe.screen
--root <output>` for the detached screen and `python -m
experiments.expD36_frozen_gamma_probe.train --root <output> --map raw
--require-gpu` for a GPU worker. A second worker uses `--map collective`.
The Slurm scripts take `PROBE_CODE`, `PROBE_OUTPUT`, and, for training,
`PROBE_MAP`. The existing Runpod Python environment is reused.

Each worker advances all its gamma/target GD cases together to 20k and then
100k updates. It then runs the two declared 50k Adam recipes on the sine mixture.
Saved traces measure state $n$ before update $n+1$; saved checkpoints include
the endpoint. Adam checkpoints retain both moments and the bias-correction
counter. Failed cells retain their first failure index and are not reset.

CPU work is split into a screen allocation of at most 20 minutes and a later
precision/analysis allocation of at most 40 minutes, preserving the total
60-minute CPU walltime allowance while allowing training to start promptly.

Run `python -m experiments.expD36_frozen_gamma_probe.precision --root <output>`
in the CPU allocation to reconstruct the selected witnesses at 80 and 120
decimal digits and compare two SVD implementations. This also exports compact
diagnostic arrays for plotting.

After downloading the artifacts, run `python -m
experiments.expD36_frozen_gamma_probe.analyze --root <output>`. It writes
`summary.json` and PNG, SVG, and PDF figures for access, executed GD curves,
hitting-time comparisons, and equal-budget trained precision. It requires the
saved training traces; it does not launch or extend training. The summary keeps
executed hits, budget-censored cases, and spectral extrapolations separate.
Scientific prose and result tables are authored directly in the results report.

## Full sweep

`full_config.yaml` freezes the expanded campaign: five targets, five exact
readout maps, eleven slopes, nonzero paired initializations, width and coordinate
controls, polynomial probes, and a separate end-to-end baseline. The original
probe artifacts remain unchanged. Full-sweep outputs use a distinct directory.

`full_core.py` implements the expanded maps and formulas. `full_kernels.py`
batches independent targets, recipes, and initializations as matrix columns;
no diagnostic solve enters an update. Neighboring coefficients are scaled
before differencing and retain their final anchor. The Adam pilot schedule
holds its terminal rate after 50k, including when resumed.

Before training, `full_benchmark.py` measures actual frozen and end-to-end batch
shapes. Launch `full_benchmark.sbatch` once with one GPU and once with
`--gres=none` and `PROBE_CPU=1`; both require `PROBE_CODE` and `PROBE_OUTPUT`.
Benchmark compilation, allocation overhead, and failures count against the
existing total limits of 7,200 GPU-seconds and 3,600 CPU-only allocation seconds.
The first probe consumed 207 and 193 seconds respectively. Forecast the complete
matrix with a 20% reserve before committing the remaining compute; an over-budget
forecast requires a revised resource decision, not silent removal of cases.

`full_screen.py` preserves 82 dictionaries and their endpoint-grid transforms,
capacity diagnostics, all tolerance certificates, and independent evaluation
grids. It extends a resolved boundary maximizer from degree 256 to 512.
`full_train.py` executes ordinary GD, the complete Adam pilot and selected/common
continuations, paired nonzero initializations, width controls, coordinate and
learning-rate controls, and unit discrete-polynomial probes. Selection uses only
the five declared validation checkpoints; the continuation retains moments,
counter, first hits, and failure state. Chunked traces record every update.

`full_worker.py` partitions the frozen matrix across two workers and includes
the new all-parameter affine-Xavier baseline at all four widths. Submit
`full_run.sbatch` first with `PROBE_STAGE=screen`, `--gres=none`, and a 15-minute
limit, then twice with `PROBE_STAGE=train` and `PROBE_WORKER=0` or `1`.
The two GPU allocations are capped at 55 minutes each; the worker checkpoint
deadline is 100 seconds earlier. Check completed Slurm accounting before any
retry so the cumulative allocation caps remain binding. Full results are only
complete when every worker, diagnostic, precision check, and report is complete.

After training, `full_diagnostics.py` recomputes certificates from each actual
initial residual, converts target-relative tolerances, and records the
effective-generator estimate separately from slope and directional bounds.
It measures native and physical norm requirements and the selected ridge paths.
Its damping ladder always starts from the same normalized probe; direct residuals
are checked against stable spectral filtering, with disagreement marked unresolved.
