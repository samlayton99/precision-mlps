# expD34 — Differentiate a Newton–Schulz residual penalty

## TL;DR

Implemented a scalar auxiliary loss whose training gradient uses matrix products and ordinary PyTorch autograd, with no SVD or least-squares solve. Completed the seven-run pilot and sixteen matched 10,000-step runs. The new method is included beside the existing J/J* trajectories in four figures. Fourteen implementation checks pass.

## Question

Can an inexpensive approximation to the geometry floor replace the full VarPro calculation in an altered loss? Does mu still affect training when it weights the loss before one Adam normalization?

## Method

Use the existing normalized feature matrix A=[tanh(ax+b),1]/sqrt(n), normalized target y, and L=0.5||Av-y||². Initialize X=A/||A||_F. Apply five iterations X <- 3.4445 X - 4.775 X(X^T X) + 2.0315 X(X^T X)^2. The refined version adds four steps X <- 1.5 X - .5 X(X^T X). Train on L+mu Fhat, where Fhat=.5||y-X(X^T y)||². Autograd includes the normalization and every iteration. One Adam updates slopes, biases, and trained readouts. No solved readout enters training.

The original coefficients produce nonunit singular values, as documented in the [official Muon implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py). The refinement reduces this error for resolved modes. Neither fixed iteration count computes the exact full-span projector. With normalized singular value s_i and resulting singular value d_i, the exact-arithmetic identity is Fhat=F+.5 sum_i (1-d_i²)²(u_i^T y)². This is a smooth spectral penalty, not an exact hard cutoff.

Mu changes the mixture of gradient contributions entering Adam. It does not guarantee mu-fold displacement. This differs from the independently normalized streams in expD31/33.

## Experiment design

Pilot: mixed sine, Xavier seed 0, 500 steps, ordinary Adam plus original/refined penalties at mu=1,100,1000, constant learning rate .002.

Matched comparison: sine, mixed sine, Runge, Gaussian envelope; mu=100,250,500,1000; 10,000 steps; cosine rate .002 to .000002. Reuse all 32 saved J/J* trajectories and run only the sixteen refined-penalty cases. Resolution 128, 177 neurons including 24 halo neurons per side, 1,024 midpoint training samples in [-1,1], float64, Adam betas (.9,.999), epsilon 1e-8. Both error rows use identical training samples. New-method readout refits are offline at roughly 200 saved states, with cutoff 1e-13. Independent-grid checks use 8,192 and 65,536 samples; endpoint checks also use cutoffs 1e-12 and 1e-14.

## Figures and results

- [Spectral response](figures/spectral_response.png): horizontal coordinate is sigma/||A||_F. Left shows the response of XX^T; right shows the remaining squared-error weight on a representable component. Five aggressive steps are purple; four added refinements are teal.
- [Mixed-sine pilot](figures/mixed_sine.png): trained relative L2, least-squares relative L2, the method's surrogate residual, and mean gamma. Colors encode mu; dashed is the original penalty, solid the refined penalty; gray is ordinary Adam.
- [Matched sine](j_comparison/figures/sine.png).
- [Matched mixed sine](j_comparison/figures/sine_mixture.png).
- [Matched Runge](j_comparison/figures/runge.png).
- [Matched Gaussian envelope](j_comparison/figures/gaussian_envelope.png).

Each matched figure has four mu columns and three rows: trained relative L2, explicit least-squares relative L2, and mean |a|. Purple is J*, dashed teal is J, orange is the refined Newton–Schulz penalty. Axes match across columns. The footer states that mu is outside split Adam for J/J* and inside the scalar loss for the new method. This compares the implemented methods; it does not isolate the projector approximation from the optimizer change. No new performance interpretation is requested for this comparison.

Complete numbers are in the [pilot summary](data/summary.json) and [three-method summary](j_comparison/data/summary.json). Initial states, shared configuration fields, schedules, and durations match. Direct prediction reconstruction verifies the new initial/final relative errors. The [endpoint checks](j_comparison/data/endpoint_checks.json) retain independent-grid and cutoff sensitivity.

## Verification and practical limits

Fourteen tests cover spectral identities for tall/wide/rank-deficient matrices, finite-difference gradients, a fixed-span counterexample, unchanged readout partial gradients, and multiple constant/cosine Adam steps against an independent implementation. All six figures were inspected. Plotting uses Agg; a pilot display-backend failure was repaired using cached trajectories, without repeating training.

Training avoids factorizations but uses dense feature-Gram products: O(n m²+m³) per quintic stage and O(n m²) per cubic stage. It does not meet the repo's architecture-blind, ordinary-Adam cost target. In the serial pilot, 500 updates took about .21 s for ordinary Adam, 3.4 s for the original penalty, and 5.6 s for the refined penalty, excluding offline solves. These timings are a small local measurement, not a scaling benchmark or a timing comparison with VarPro.

## Reproduction

[Implementation](../../../experiments/expD34_ns_residual_penalty/run.py), [pilot configuration](../../../experiments/expD34_ns_residual_penalty/config.yaml), [matched comparison](../../../experiments/expD34_ns_residual_penalty/comparison.py), [tests](../../../tests/test_expD34_ns_residual_penalty.py), [status and feasibility](../../../docs/expD34_status.md).

Run the pilot with run.py; --plot-only reuses its data. For the matched extension use comparison.py --target TARGET for each of the four targets, then comparison.py --plot. Existing complete extension files are reused with configuration checks. The [source manifest](j_comparison/data/sources.json) identifies every reused J/J* trajectory. Data remain in this experiment's data and j_comparison/data directories.

## Open question

Whether a different iteration budget or optimizer treatment makes this a useful replacement for VarPro remains open. The current figures compare the specific implementations and settings above.
