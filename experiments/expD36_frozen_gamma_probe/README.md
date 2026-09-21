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
The same offline job audits every saved update for first and sustained hits,
joins selected pilot histories to their continuations, and compares the resulting
first hits with the counters saved by the optimizer kernels.

`full_precision.py` independently reconstructs the twelve declared witnesses
at 80 and 120 digits, including exact-grid polynomials and nominal reference
allowances for the neighboring map. `full_reference.py` calls the existing
`construct_qi` at every declared width with the same halo, $\lambda=0.25$, and
$K_c=160$. It compares 40- and 80-digit construction, ordinary and extended
evaluation, and an independent 80-digit evaluation check. The constructor still
samples target derivatives and returns coefficients in FP64; its measured
recovery accuracy is not asserted to be an exact approximation floor.

The [full-sweep report](../../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/REPORT.md)
contains both three-panel figures, all controls, measured theorem slack, and
complete allocation accounting. All 4,095 prescribed optimizer trajectories
completed within the two-GPU-hour budget. `full_analyze.py --root <output>`
regenerates the compact summary and thirteen figure families from saved
evidence after checking every campaign completion flag. Run it as a module,
`python -m experiments.expD36_frozen_gamma_probe.full_analyze`.
The tracked compact artifacts suffice for figure regeneration; full matrices,
optimizer checkpoints, and every-update traces are archived on Runpod under
`/workspace/junmiaoh/experiments/precision-mlps/runs/frozen_gamma_full_v1`.

The subsequent CPU analysis `slope_spectrum_analysis.py` tests the
[slope-distribution spectral theorem](../../docs/slope_distribution_spectrum.md)
against saved raw-map spectra and executed first hits, and constructs eight
small heterogeneous dictionaries. It runs no campaign training. Regenerate its
JSON evidence, curve arrays, and three-panel figure with:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m experiments.expD36_frozen_gamma_probe.slope_spectrum_analysis \
  --root results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep
```

`slope_spectrum.py` contains the distribution tail bound, signed pole expansion
with explicit remainders, exceptional-feature target projection, and the
combined spectral-CDF error bound. The analysis records analytic predictions,
measured spectral diagnostics, and executed hits separately. Its report
discusses the remaining gap between polynomial-tail access and target-relevant
spectral mass; close measured spectral forecasts are not labeled a sharp
gamma-only theorem.

The capped-kernel follow-up adds independent periodic synthesis in
`fourier_law.py` and the finite common-slope Gram identity in
`finite_gamma_gram.py`. These are offline diagnostics; they do not change the
optimizer. Run their normalization, coherent-alias, null-mode, finite-Gram,
and ordinary-GD checks with `pytest -q tests/test_expD36_fourier_law.py`.
The Gram forecast reports unresolved spectral mass explicitly because forming
a Gram matrix loses accuracy in small singular directions. Its forecast is
not a certified bound for every heterogeneous dictionary under a slope cap.

`cap_certificate.py` implements the
[uniform capped-kernel theorem](../../docs/capped_gamma_kernel_certificate.md).
The convex solver produces candidates; the independent interval checker covers
the continuous slope interval and reports the trace cost of any bias repair.
Install the experiment-only `cap_requirements.txt` dependencies and run
`pytest -q tests/test_expD36_cap_certificate.py`. The two-sample control has a
known exact optimum, so it checks tightness as well as validity.

`cap_campaign.py` prepares the capped heterogeneous families, selects fast
admissible candidates, executes ordinary frozen-readout GD, and checks selected
trajectories against independent rectangular factorizations. Its output root
is separate from the completed full sweep. `prepare` and `reference` run in
CPU-only Slurm allocations. `train` requires one allocated GPU and verifies the
actual job-step mask. `cap_run.sbatch` accepts these stages through
`PROBE_STAGE`, with `PROBE_CODE` and `PROBE_OUTPUT` identifying the immutable
source snapshot and persistent results root. Slopes and targets are hashed;
resumed batches retain coefficients, update counts, and every-iterate hits.
Training defaults to one worker covering the entire supplied case list;
multi-GPU submissions explicitly specify both `--workers` and `--worker`.
Completion records identify their assigned cases so a completed shard cannot
be mistaken for a completed full sweep.
Only sampled scalar curves and restart states are archived. The new campaign
has an additional ten-GPU-hour cap, including compilation and allocated idle
time; reconcile Slurm accounting before submitting further allocations.

`cap_refine.py` searches target and spectral-tail witnesses plus windowed sine
directions, ranks finite-grid candidates, and independently certifies the
selected factors. It reuses archived common-slope left singular vectors only
to propose witnesses; the continuous-cap checker establishes validity for all
admissible heterogeneous slopes. A local end-to-end check uses `--n 64 --caps
16 --rank 4 --intervals 16 --verify-count 1`. Large interval repairs are retained
in the evidence and direct the next round toward tighter scalar enclosures.
Proposal ranking retains log-scale forecasts above FP64's exact-integer range.
Final integer bounds use adaptive precision and an explicit $10^{32}$ update
cap; a result above that cap is a certified obstruction, not an executed hit.
`--order 4 --reuse-candidates` rechecks saved proposals with higher-order
interval Taylor enclosures in a separate output directory. This directly
measures numerical enclosure slack without changing the proposed witnesses.
`--proposal-case <saved-case-id>` instead proposes directions from a searched
admissible dictionary's left singular vectors. Width, samples, centers, matrix
hash, and target values must match. This changes only the witness search; the
subsequent certificate still covers every dictionary under the declared cap.

`fourier_validate.py --archive <completed-full-sweep> --output <new-output>`
executes the periodic controls, checks independent finite-Gram predictions
across the saved matrix, and measures finite-window, anchor, halo, and native
coordinate effects. It preserves the actual training clock. A modified Gram
block without an output realization is labeled a curvature diagnostic and
receives no borrowed target weights or learning-time claim. Invalid contraction
clocks and unresolved small eigenvalues remain explicit in the JSON evidence.

`cap_search.py` searches slopes under the hard cap using differentiable binary
powering of the affine GD recurrence as a selection surrogate. These powered
evaluations are never counted as executed optimizer updates. Each selected
dictionary is saved under a new case ID and must pass an independent spectral
check and ordinary GD before contributing an upper witness. CPU requests for
subsequent GPU jobs are limited to two cores so CPU certification can progress
within Runpod's per-user CPU limit while both allocated GPUs are active.
The `--method lbfgs` refinement uses a bounded line search after the projected
Adam search has plateaued, with the same powered objective and subsequent
ordinary-GD verification requirement.
The `--explore` follow-up retains four fast starts and adds target-phase,
sparse-lattice, and bimodal initial slopes. This checks whether selecting only
initially fast dictionaries misses better admissible solutions. The primary
target remains fixed, and all final learning times require frozen-slope GD.
`cap_dual.py` uses the certificate solver's dual slope weights as another
source of adversarial dictionaries. It rounds and samples the mixture into
actual capped slope vectors. The mixture itself is a diagnostic relaxation;
only the rounded dictionaries can become executed upper witnesses.
Its `--method overlap` alternative takes dual weights from the joint
target-overlap problem, preserving the target's role in the stationarity
conditions instead of challenging only a fixed witness direction.
`cap_capacity.py` independently encloses the residual of an archived detached
readout under real tanh evaluation. It proves attainability of the declared
training tolerance for that dictionary; it makes no optimizer-timing claim.

`cap_resolvent.py --root <capped-campaign>` tightens the conversion from a
certified Rayleigh bound to a necessary learning time. The resolvent inequality
is proved in the theorem note; a second interval checker encloses its scalar
spectral minimum. Its output is a separate refinement artifact, preserving the
original CDF calculation and the certificate from which both follow.

`cap_joint.py` tests a stronger witness-selection strategy by jointly optimizing
the witness and PSD certificate for a resolvent shift. It records unsuccessful
conic solves, polishes successful directions with the fixed-witness solver, and
then verifies them independently. Symmetric target witnesses and reflection-
averaged certificates allow reuse of center-pair interval calculations without
restricting the independently heterogeneous slope family.
Its `--method overlap` variant maximizes target overlap at a fixed curvature
budget with scaled variables, addressing the poor conditioning observed in
small-shift joint solves. Both methods polish and independently verify their
proposed directions; an improved solver objective is not itself a theorem.
`--budget-count 1` checks only the smallest budget, in a separate directory.
The four-budget $N=128$ controls found the strongest cap-16 and cap-64 bounds
at that budget, motivating the cheaper primary-width confirmation.

`cap_analyze.py --root <campaign-json-snapshot> --output <curated-results>`
collects executed hits, independent forecasts, and certified bounds without
mixing their evidence roles. Larger-cap certificates apply to smaller caps;
smaller-cap executed dictionaries supply admissible upper witnesses for larger
caps. The script checks these comparisons and emits JSON plus a focused
three-panel figure. With `--archive <completed-full-sweep>`, it also creates a
new paper banner preserving the existing learned-slope and precision panels.
It does not generate report text or change the completed archive.
With the Fourier validation JSON present, it also plots periodic spectra and
packet decay, finite-Gram predicted versus executed hits, matched gamma-speedup
ratios, and finite learning curves. Width-control columns are matched through
their saved target identifiers, since those runs contain a target subset.
