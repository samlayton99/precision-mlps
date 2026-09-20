# expD29 — Multiply the approximation gradient by 1,000

Status: complete, including the six-weight sweep; 72 unique runs and six PNGs ready for Sam; coordinator main task.

Authorized by Sam: theta <- theta - eta(1000 DF + DG), with ordinary readout GD. Implement the same cutoff-based F_tau as expD28, so the modified objective is H=L+999 F_tau. Compare weight 1 with weight 1000 across the four targets and three initializations. Same width, initialization seeds, samples, eta=0.002. No clipping, retuning, coefficient replacement, or numerical-threshold gating of the requested update. Record numerical sensitivity explicitly.

Practicality checklist, before implementation:

1. One ordinary forward/backward per step; weight 1000 additionally computes one full feature-matrix SVD and its projector derivative, sequentially before the update. Offline checks add two independent profile evaluations at selected states. This is expensive: O(n m^2 + m^3) work per SVD, not gradient-class cost.
2. SGD has no momentum state. Geometry/readout parameters are O(m); the diagnostic materializes transient n-by-m feature, singular-vector, and derivative arrays. Saved scientific trajectories are separate from optimizer state. The solve violates the intended production scaling budget.
3. No Krylov memory or tunable k/d ratio; dense SVD is the limiting cost.
4. No Krylov loop/collectives. CPU LAPACK performs width-dependent reductions inside the SVD; the design has no O(1) reduction-count scaling claim.
5. The measured profile gradient can be merely small or unresolved, not exact zero. Alternative SVD/tanh checks record this; they do not alter the user-specified update.
6. This is an fp64 diagnostic using the existing 1e-13 cutoff, not a precision-agnostic candidate.
7. No loss-based decisions or feedback controller. Stop only if the requested trajectory becomes nonfinite or its required SVD fails; report rather than silently repair it.
8. No new claim that tiny projected gradients are uninformative. The explicit intervention intentionally amplifies them and records numerical sensitivity. This is a mechanism test, not promotion of a dense solver.
9. The classical baseline is ordinary GD with weight 1. No ad hoc optimizer is added.
10. Not eligible for production litmus tests or promotion: it already fails architecture-blind/compute requirements. Sam explicitly requested this small diagnostic.
11. Falsification: if 1000 DF does not improve the profiled approximation, or only changes unreliable numerical values, magnitude imbalance alone is insufficient at this rate. Track actual loss, F_tau, mean gamma, coefficient norms, rank, and update diagnostics.

Implementation checks: independently differentiate the weighted scalar loss on a resolved tanh example; confirm the readout's first update is ordinary GD and never a coefficient solve; recover the existing ordinary-GD trajectory at weight 1. Plot only measured quantities, with separate matched curves and per-panel vertical limits.

Completed: all four targets and three initializations at weights 1 and 1000, 500 updates each. Three focused tests pass; all full ordinary-GD controls exactly reproduce expD28 through step 500. No nonfinite runs. Strongest gain is Xavier Runge, with independent-grid refit relative L2 0.1473 -> 0.0975, while actual loss barely changes and mean gamma remains near 0.088. The gain depends on a new retained singular direction and large solved coefficients; with cutoff 1e-12 it shrinks substantially. Mixed sine exhibits hard-cutoff rank crossings. Saved terminal cutoff audit and numerical derivative sensitivity accompany the writeup. Loss axes use per-panel limits; F curves use matched saved steps; gamma row shows change from the initial mean.

Mu-sweep extension in progress: Sam requested multiple approximation-gradient weights. Sweep mu=1,10,100,1000,10000,100000 at the same base rate and 500 updates. Reuse the 24 existing cases and run only the 48 new ones. Same requirements checklist applies: controlled dense-SVD diagnostic, not a scalable optimizer proposal. Existing figures remain; put the extension in mu_sweep/ with data/ and figures/. Preserve failed-run endpoints and show any nonfinite termination; do not clip or silently change the update.

Sweep completed: 48 new cases plus 24 reused, all 72 finite through step 500. Six implementation tests pass. At mu=100000, refitted Xavier approximation improves on every target at all three audited cutoffs; actual losses improve for sine/Gaussian and worsen for mixed sine/Runge. Xavier mixed sine reaches mean gamma 1.7587, Gaussian 0.7574, Runge 0.1395, sine 0.09094. The larger-weight Runge refit gain survives the stricter cutoff and uses smaller solved-coefficient norm than the baseline. No generic stability or production-recipe claim: several trajectories jump, some gradient estimates are unresolved, coefficients remain large, and only one seed is tested.
