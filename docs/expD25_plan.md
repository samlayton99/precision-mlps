# Scale barrier investigation

Coordinator: current Codex task. Status: completed — 100 trajectories, numerical audits, independent implementation review, and 15 figures. This is a sequence of causal diagnostics, not a promoted optimizer.

Question: Is limited useful scale motion caused by an unhelpful local direction, a small update in poorly scaled coordinates, a flat approximation objective, or a combination? Separate the attained numerical least-squares error from the current readout optimization gap. Every function uses the matched finite-domain model and samples from expD24; no whole-line/finite-domain mixing.

## First measurement: shared-scale landscape

Take the saved final gamma-1 GD state and multiply every slope and bias by the same positive factor, preserving each center exactly. Keep the readout fixed for one loss curve; independently solve it for the other. This does not retrain. Plot four function rows with (1) both errors versus scale multiplier, (2) solved coefficient norm, (3) retained SVD rank. Mark the actual starting state. Record the current signed directional derivative and verify it against a centered finite difference. Do not treat the differentiated truncated-SVD solution as an exact envelope derivative without a separate numerical check.

Discrimination: a large improvement in refitted error with little improvement or initial worsening at fixed coefficients indicates coupling to the readout. If both improve in the same direction, a weak GD step is a candidate barrier. Rank and coefficient norms expose numerical or unbounded-coefficient explanations for the apparent approximation landscape.

## First intervention: geometry learning rate only

Use identical initial parameters, samples, loss, ordinary SGD, and readout learning rate. Multiply only the raw slope/bias learning rate by 1, 100, or 10000. Start with mixed sine at gamma 1, then widen the controlled matrix if informative. No readout solves enter training. Record losses, initial-versus-current readout-refit accuracy, slopes, biases, coefficients, gradient path length, and sign changes. Save failures rather than silently retuning individual cases.

Plots are specified before running: matched error trajectories with solid ordinary error and dashed refit error, and a separate plot of actual scale displacement. The question is whether increased motion improves the refit floor, rather than merely improving the current readout fit. Use common limits where comparisons require them; show failure/stopping explicitly. Gamma 16 is a preservation control, not a case in which a floor-quality approximation can improve significantly.

## Requirements section 8 check

1. Passes: the intervention remains one forward/backward per GD step; a scalar multiplier changes no pass count. Sparse offline evaluations use an SVD and dense grids as measurement oracles, not optimizer operations. Count their cost separately.
2. State: plain SGD has no persistent moment arrays; the multiplier is a scalar. Saved trajectories and evaluation matrices are experiment data, not optimizer state. No new production per-sample state.
3. k/d: no history, block-memory parameter, or rank-dependent optimizer storage.
4. Reductions: unchanged from ordinary GD; offline diagnostics perform extra norms/SVD reductions.
5. Exact versus small: zero readout implies exactly zero first geometry update. Small gradients are measured, not thresholded away.
6. Precision: multiplication and SGD are dtype-compatible. This experiment uses fp64, matching the established runs; no claim of bf16 floor performance. The diagnostic SVD cutoff is explicitly recorded.
7. Control: fixed learning rates; no loss-based acceptance or noise-floor gauge. Stop only nonfinite or explosively large parameters/loss, and record the failure.
8. Kill list: this does not rebuild a solver/preconditioner or claim to solve the spectral tail. A large learning rate may fail, and that is a diagnostic outcome.
9. Classical baseline: ordinary GD with a parameter-group learning rate. Adam is an existing classical comparison if the fixed-rate sweep isolates a scaling problem.
10. Litmus tests: no general optimizer promotion from these 1D results. This diagnostic lacks architecture-blind grouping and does not independently solve least squares to its floor; the full requirements are not claimed. Any later candidate must meet the production gate before promotion.
11. Falsification: if gamma moves substantially while refit quality stays unchanged/worsens, increased mobility alone is not the cure. If all larger fixed steps become unstable before useful motion, coordinate/directional conditioning rather than a simple scalar rate is implicated. If successful, repeat a common setting across all targets and starts, then seeds, without tuning per case.

## Second intervention: fixed centers

The mixed-sine gamma-1 pilot with the raw geometry rate multiplied by 10000 showed materially improved refit accuracy. This involved both centers and scales. Freeze centers exactly by writing the same model as tanh(a*(x-z0)) and training only a and the readout. Compare scale learning-rate multipliers 1 and 10000, with all other settings identical, across four targets and initial gamma 1 and 4. Save raw-coordinate derivatives as diagnostics but update the actual center-preserving scale derivative. Plot error and gamma displacement separately. A positive result identifies a contribution available through scales alone; a failure would show the earlier result depended on centers or their coupling. This remains a causal parameter-group SGD control with the same compute/state budget, not a promoted architecture-blind optimizer.

## Third intervention: preserve a common scale

The larger rate improves off-floor gamma-1 approximation quality but degrades several already good gamma-4/16 geometries. Test whether loss of the uniform-scale structure contributes. Keep centers fixed and compare independently learned scales with one common scale. Project the same per-neuron centered-scale GD gradient onto the all-ones direction (replace it by its mean); keep the geometry rate at 20 and readout rate at 0.002. This is equivalent to scalar-gamma GD at rate 20/177, avoiding an unreported 177-fold speedup from tying parameters. Test initial gamma 1, 4, 16 on all four targets. Plot current error, refit error, and scale motion with matching colors. This changes only the permitted scale directions, not the target, samples, readout solve cadence, or loss. The method uses one additional reduction and constant state; the geometry prior is explicit and is not an architecture-blind optimizer claim.

## Fourth measurement: continue the common-scale control

The common-scale constraint substantially improves the mixed and Gaussian gamma-1 refit quality and preserves all gamma-16 floors in the 2,000-step test. It is not uniformly better than independent scales (Runge is worse). Continue the same gamma-1 common-scale GD to 10,000 steps, comparing geometry rates 0.002 and 20 with identical readout rate 0.002. Do not change rates mid-run or introduce solves. Mark step 2,000 and verify that the longer fast run reproduces its earlier trajectory at that step. Plot mean gamma, numerical approximation error, and the separate readout optimization gap, all through training. This tests continued escape versus a new plateau, rather than selecting a favorable terminal snapshot. Also check initial/final numerical readout accuracy with independent SVD drivers, five cutoffs, and a 32,768-point evaluation grid before interpreting small floors.

## Fifth intervention: classical adaptive scale step

Replace only shared-scale SGD by Adam, using its standard betas/epsilon and learning rate 0.002; keep readout SGD at 0.002 and fixed centers. Test gamma 1 and gamma 16 on all four functions to 10,000 steps. The shared gradient is still the mean of the individual scale gradients. This asks whether a conventional adaptive step removes the need for the ad hoc 10000 multiplier, and whether it preserves an already good regime. Plot actual GD error, independent-grid refit error, and gamma, and compare against the same shared-scale SGD rate 0.002 baseline. No LS solve is fed into either optimizer, so this does not recreate the known failure of unthrottled Adam immediately after exact solves. Adam adds its usual two moment buffers (identical across tied scales and compressible to two scalars), no extra model passes, and the standard shared-gradient reduction. This is still an explicitly structured 1D control, not a claim to satisfy the architecture-blind optimizer gate.

The current authorized scope is local investigation and controlled experiments. Existing expD24 results remain untouched. Data-obvious findings and remaining questions are recorded in the expD25 results writeup; broader optimizer claims remain unestablished.
