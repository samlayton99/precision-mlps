# expD30 — Extreme gamma, samples at centers

Status: complete; all 24 trajectories, three figures, data, and writeup finished. Sam requested gamma 128, zero readout, center-only samples, learning-rate schedules; confirmed comparing fixed and jointly trained geometry. Previous freezing/decomposition follow-ups remain on hold.

Design: four existing targets, N=128 intervals, 129 exact centers inside [-1,1] as training samples, 24 halo neurons per side (177 neurons total), fp64, all readout coefficients including bias initially zero. Gamma=128, lambda=2. Samples remain fixed if geometry moves. Joint training updates raw slopes and biases; fixed geometry updates readout only. Compare eta=0.002, constant eta=1.9/sigma_max(A0)^2, and 1,000-step linear warmup from 0.002 to that rate followed by cosine decay to 1% of the peak. Same rate applies to every trainable parameter. Run 200,000 actual GD updates per arm, with full training-loss traces and sparse independent-grid evaluations. Fixed-geometry exact-arithmetic spectral predictions extend the horizon without representing them as executed training.

Outputs: one writeup, figures/, data/. Main figure separates fixed/joint geometry and plots actual half-MSE; numerical readout floors, target-specific convergence predictions, schedules, and between-center errors explain what training accuracy means. No LS coefficients enter GD. Center-only sampling is expected to remove the previous overdetermined approximation floor, not necessarily all conditioning difficulty.

Verification before full runs: samples exactly equal initial in-domain centers; zero readout; analytic PyTorch GD against independent autograd for all trainable blocks; frozen-geometry invariance; independent target batching; exact spectral prediction against actual fixed GD at both constant rates; schedule endpoints and normalization; direct readout residual and held-out evaluation. Check finite trajectories and report any failure without silently adjusting rates.

Practicality checklist (diagnostic experiment, not a new optimizer proposal):
1. Each update is one model forward and one analytic backward in PyTorch; no solves or probes in the update. One offline SVD sets the diagnostic rate and predicts fixed-geometry convergence. Sparse final/evaluation solves are offline.
2. GD optimizer state is O(P), with no momentum. Stored trajectories and a cached small fixed feature matrix are experiment diagnostics, not a deployable optimizer requirement.
3. No memory-window k/d parameter.
4. Ordinary gradient reductions; no Krylov iteration or collective loop.
5. No inferred exact-zero classifier; numerical rank is separately reported using a specified cutoff.
6. fp64 experiment as requested by repo; no claim of validated bf16 behavior.
7. Prescribed rates do not branch on loss comparisons. Nonfinite checks detect failure, not acceptance/backtracking.
8. No proposed cure for the killed QI solver designs; this deliberately changes sampling and gamma to test an easier interpolation problem. Does not imply a general solver.
9. Constant GD and classical warmup/cosine are the requested schedule comparison. No momentum or preconditioning changes.
10. This bounded diagnostic is not being promoted to dl_test/batching/general optimizer use.
11. Falsification: if initial LS cannot fit samples, feasibility premise fails; if fixed GD disagrees with the exact recurrence beyond roundoff, implementation/prediction is wrong; if joint GD fits faster, the fixed spectrum is insufficient to describe joint convergence.

Completion: four focused tests pass; all 24 trajectories stay finite. Full executed fixed-GD traces agree with spectral predictions within 2.86e-14 relative-error discrepancy. The larger constant rate beats the two other tested choices at all endpoints. Joint training helps mixed-sine sample fitting most; continuous-domain error remains around 1e-3. No held experiments resumed or shifted-sample control run.

Numbering: moved this new experiment from D29 to D30 because a separate expD29_weighted_profile appeared in the shared repo during execution. Its files and catalogue entry were preserved.
