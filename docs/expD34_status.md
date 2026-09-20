# expD34 — Differentiable five-step Newton–Schulz residual penalty

Coordinator: main task. Status: complete, including the matched J/J* comparison.

Sam asks whether ordinary Adam preserves the effect of mu in L+mu F, and authorizes implementing the proposed differentiable Muon-style approximation. Test the literal five-iteration polynomial with coefficients (3.4445,-4.7750,2.0315), Frobenius normalization, and full differentiation through normalization and all iterations. Call the resulting penalty Fhat, not the exact VarPro floor. One ordinary Adam receives the combined gradient; no split moment streams or outside multiplier.

Sam's follow-up authorizes practical improvements rather than insisting on an exact implementation of the draft. Add four standard cubic Newton–Schulz refinement steps, X <- 1.5 X - .5 X(X^T X), after the five aggressive steps. These bring resolved singular values closer to one while preserving the matrix-product implementation and smooth treatment of small singular values. Compare the literal and refined versions; do not substitute one silently.

Bounded pilot: mixed sine, Xavier seed 0, the expD31 geometry/samples, 500 steps, mu=1,100,1000 for both penalties, plus one shared mu=0 control, constant learning rate .002. Solve readouts only for offline evaluation at saved states; never feed a solve into training. Preserve existing experiments. Two PNGs: the scalar spectral responses, and actual error / least-squares evaluation / surrogate penalty / mean gamma through training.

Requirements section-8 pre-build checklist:

1. Passes: one feature evaluation and one combined backward per update; five sequential matrix-polynomial stages in the auxiliary loss, plus four cheaper cubic stages for the refined version. For a tall n-by-m feature matrix each quintic stage costs O(n m^2+m^3), including an m-by-m Gram product; each cubic stage O(n m^2). No n-by-n projector is formed. Offline evaluation uses SVD and is excluded from training timings.
2. State: ordinary Adam's two parameter-sized moment arrays. Backprop through k unrolled stages retains O(k n m+k m^2) transient storage, k=5 or 9. Scientific trajectory files are separate from optimizer state.
3. k/d: k=5 fixed for this diagnostic. That cannot resolve arbitrarily small singular values; no width-independent projector accuracy is claimed.
4. Reductions: one Frobenius norm plus loss reductions; five dependent Gram/matrix-product stages and their backwards. No Krylov loop, line search, or distributed scaling claim.
5. Exact or small: a zero normalization norm is an exact invalid-input check. Spectral suppression is a continuous numerical weighting, not an exact rank test. No numerical signal threshold controls updates.
6. Precision: fp64 to match prior experiments. Coefficients are those in the supplied note; dtype is inherited by the implementation. No bf16 floor claim; this does not copy Muon's bf16 cast or detach.
7. Controls: fixed steps/rate/mu, no loss comparisons or acceptance gates; stop on nonfinite values. Offline SVD cutoff remains 1e-13 for comparability.
8. Kill list: removing SVD does not remove the dense feature-Gram cost or architecture dependence. This is a user-requested mechanism diagnostic, not a production candidate that passes the cost gate.
9. Baseline: ordinary Adam with mu=0, identical initialization and samples. The exact spectral floor is an offline diagnostic.
10. Litmus: no dl_test, batching, precision-floor, or production claim in this bounded pilot. Those remain necessary before promoting the mechanism.
11. Falsification first: scalar response and an exactly representable target test whether Fhat equals a floor/cutoff projector; finite-difference gradient checks and independently implemented Adam steps verify the literal proposal. Final readout evaluations use a separate grid and adjacent cutoffs.

Known issue before implementation: p(1)=.701, so the chosen polynomial does not fix the unit singular value. Muon's official implementation explicitly describes a factor with nonunit singular values. For s_i=sigma_i/||A||_F and d_i=p composed five times at s_i, Fhat=F+0.5 sum_i (1-d_i^2)^2 (u_i^T y)^2 in exact arithmetic. The extra in-span penalty is part of this experiment, not silently replaced by an exact solve.

Verification plan: spectral identity for tall/wide/rank-deficient matrices; finite-difference autograd check; nonzero penalty and derivative on a fixed-span family with exact F=0; unchanged readout partial derivative; multi-step agreement with the literal polynomial and independently maintained Adam moments; mu=0 baseline agreement. Inspect both rendered figures and report numerical results without promoting a 500-step pilot to a precision claim.

Pilot complete: seven trajectories finish 500 updates; eleven tests pass. The original penalty takes about 3.4 s per run, the refined penalty 5.6 s, ordinary Adam .21 s, excluding offline solves. All refined final training-refit errors are .420-.422 versus .448 for ordinary Adam; actual errors .881-.896 versus .913. The scalar response plot makes clear why refinement helps the projector approximation. Both figures inspected. The figure backend was changed to Agg after a display-backend abort; cached data were plotted without repeating training.

Sam requests adding the new method to the existing J/J* figures. Extend only the refined penalty to the exact matched setup: four targets, mu=100,250,500,1000, 10,000 steps, cosine learning rate .002 to .000002, same initialization/samples/Adam settings. Reuse all existing J/J* trajectories. Label the mathematical distinction: mu weights a scalar penalty inside one Adam here, whereas it scales one independently normalized stream outside Adam in J/J*. Save four three-method figures separately; retain the original two-method figures. The cost/precision assessment above remains unchanged. Schedule implementation is checked against independently calculated multi-step Adam updates before starting the runs.

Extension complete: all sixteen new trajectories finish 10,000 updates. Fourteen tests pass. Shared settings, complete step grids, exact initial arrays, and cosine rates are verified against the reused J/J* cases. Direct initial/final prediction checks and endpoint refits pass; dense-grid and adjacent-cutoff results are saved. Four three-method PNGs rendered and inspected. The original two-method figures are preserved. The new method uses mu inside a single Adam and a finite-iteration spectral surrogate, so the comparison is not a one-factor isolation of the projector approximation.
