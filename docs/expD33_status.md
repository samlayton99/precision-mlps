# expD33 — Current-readout projected signal with split Adam

Status: complete; coordinator main task. All 20 requested runs completed 500 updates. Six new tests and seven inherited Adam tests passed; initial states, first readout updates, independent-grid refits, and alternative SVD calculations checked. The final figure was inspected with expanded linear gamma axes. Sam requests the expD31 Xavier protocol with the amplified stream replaced by the current-readout signal, at mu=100,500,1000,5000,25000. Four targets, 500 updates, unchanged initialization, samples, rate, Adam settings, and numerical cutoff. No readout replacement or freezing.

Write A=[tanh(ax+b),1]/sqrt(n), r=Av-y/sqrt(n), and P_tau=U_retained U_retained^T. The amplified stream is h=J_current^T(I-P_tau)r, treating the residual seed as fixed in the VJP. The remainder is grad_theta L-h. These are not DF_tau and DG_tau. Use two independent Adam histories, multiply the h direction by mu after normalization, and update the trained readout with ordinary Adam. Evaluate F_tau=.5||(I-P_tau)y/sqrt(n)||^2 at every state. Unlike an exact projector onto all of range(A), a retained projector need not annihilate Av outside its retained range; use the literal projected current residual and record that difference from the refitted residual.

Pre-build requirements section-8 checklist:

1. One ordinary forward/backward plus a feature SVD and two gradient-cost contractions per update; sparse numerical checks add two SVDs. SVD is O(n m^2+m^3), outside the production budget.
2. Two geometry Adam streams and one readout stream: O(P) persistent optimizer state. SVD transient storage is O(n m+m^2); saved scientific snapshots are not optimizer state.
3. No adjustable Krylov history; dense SVD is the bottleneck.
4. No Krylov reductions. SVD performs matrix reductions whose count is not constant in matrix size; no GPU-scaling claim.
5. Numerical projector and small signals have floating-point uncertainty. Record alternative SVD/backend effects without gating the requested update.
6. Same fp64 and relative cutoff 1e-13 as expD31; not a precision-agnostic recipe.
7. No acceptance controller, clipping, or loss-based stopping. Report nonfinite states or SVD failures.
8. Explicitly requested mechanism diagnostic, not promotion of normalized tiny gradients or a dense-solve production optimizer.
9. Reuse ordinary Adam as a control and expD31's VarPro-weighted mu=1000 trajectory as a directly relevant reference.
10. Fails current production/architecture gates; no dl_test or batching claim, and no new unrelated sweep.
11. Falsification: assess actual loss, numerical refit loss, mean gamma, and independent-grid refits. Compare the current-J result with the saved VarPro reference. Large movement without better approximation is not success.

Verification: independently differentiate prediction paired with detached projected residual; check component sum and current rather than solved readout weighting; test a zero-readout state; compare several full updates with independent PyTorch Adam histories; verify matched initial states/readout updates and absence of coefficient replacement. Retain the existing Adam-stream tests. Inspect the 3x4 figure, with log loss rows and readable linear gamma axes, before reporting results.

10,000-step extension complete: rerun all four targets and the original five outside multipliers from the same Xavier initializations. Extend ordinary Adam controls to the same duration. Learning rate, epsilon, moment settings, loss, samples, width, cutoff, and update formula are unchanged. Check every first-500-step loss/gamma history against the saved short runs (also F for split runs). Add explicit reconstructed training-refit relative error each step and independent-grid checks at saved states, plus post-Adam stream norm ratios. Results remain under expD33/long_run.

Extension verification: all 24 trajectories completed; all first-500-step loss and mean-gamma histories agree bitwise with the originals (and F for the 20 split runs). Six projector/independent-Adam tests pass after adding diagnostics. Endpoint independent-grid errors change by at most 0.90% on an eight-times-denser scoring grid. Three figures inspected; smaller gamma curves also have a common logarithmic view. No rate or epsilon change, clipping, readout replacement, or training rerun to fix labels.

Matched J/J* comparison requested: four separate function figures, columns mu=100,250,500,1000, two method lines per panel. Both use the same full-run cosine rate from .002 to .000002 over 10,000 steps. Reuse twelve J* trajectories; fill sixteen current-J trajectories plus four J*/1000 trajectories. The current-J trainer receives only the already-used common schedule, applied to both geometry streams and the readout; no direction or moment changes. This is the same authorized dense-SVD diagnostic and cost assessment, with no new optimization mechanism. Top and middle rows use relative L2 on identical training samples; top keeps trained coefficients, middle uses explicitly reconstructed least-squares coefficients. Bottom is mean gamma on linear axes. No extra interpretation requested.

Matched comparison complete: all 20 additional trajectories reach 10,000 steps; 12 existing J* trajectories reused. Schedule equality, pairwise initial-state/settings equality, and relative-L2 conversion are verified. Eleven selected implementation tests pass. Four requested figures are rendered, visually checked, and catalogued.
