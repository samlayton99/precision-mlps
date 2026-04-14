# Experiments

Central question: can we find a training/optimization strategy that learns QI-like solutions, closing the gap between construction (~10^-15) and training (~10^-10)?

The paper identifies three violations in trained networks: (1) gamma stays O(1) instead of growing with N, (2) outer weights blow up instead of staying O(1), (3) features exhibit rank saturation instead of uniform utilization. Every experiment below should be evaluated against whether it explains or fixes these violations.

Metrics to log for every experiment:
- train L_inf, eval L_inf, eval relative L2
- gamma, lambda = gamma*h (mean/median/max)
- max absolute outer weight
- feature rank diagnostics (singular values, stable rank)
- seed-to-seed variance (3-5 seeds minimum)

Target families to include across the roadmap:
- low-frequency analytic
- high-frequency analytic
- boundary-layer / steep-transition analytic
- mixed-scale analytic
- polynomial / entire-function type
- one slightly rough but still smooth target

Success criterion:
- A method counts as working if, across widths N in {32, 64, 128, 256}, on the target-family matrix above, over 3-5 seeds, it reaches eval relative L2 <= 1e-13 and eval L_inf consistent with construction-level precision, without initialization from the exact constructive solution.

---

## 0. Numerics Sanity Checks

**Hypothesis:** Some of the observed precision floor may be numerical rather than optimization-limited. Before attributing failures to training, we need to verify that linear solves, function evaluation, and tolerance choices are not the bottleneck.

**Core:**
- Compare exact-readout solves using QR, SVD, and ridge-stabilized least squares.
- Check whether the same "QI + exact readout" result is reproducible across solver backends and tolerances.
- Track residual norms of the linear solves explicitly.
- Verify whether evaluating on a denser grid changes the claimed eval L_inf materially.

**Additional:**
- Test whether using extended precision only for the linear solve or only for evaluation changes the observed floor.
- Measure tanh evaluation stability at large arguments and compare cond(Phi) vs cond(Phi^T Phi) in the same sweep.

---

## 1. Lambda Tradeoff Verification

**Hypothesis:** The theory predicts a U-shaped error curve in lambda (aliasing at large lambda, ill-conditioning at small lambda). If this tradeoff is numerically real and sharp, it explains why unconstrained training -- which lets lambda drift to ~0 -- cannot reach high precision.

**Core:**
- Construct QI MLPs across widths N in {16, 32, 64, 128, 256} and sweep gamma to trace the error-vs-lambda curve for each width. Plot L_inf error (y) vs gamma (x) with one curve per width. Expect U-shaped curves with a shared optimal lambda* ~ 0.25-0.30.

**Additional:**
- Repeat with fixed QI geometry (gamma, x_k) but solve the readout via least squares instead of the full construction. Does the U-shape persist when the outer weights are learned rather than constructed? Isolates whether the tradeoff is purely geometric or also depends on exact coefficient computation.
- Overlay the lambda values that trained networks actually converge to on the same plot. Visualize how far off the trained solutions are from the viable regime.

---

## 2. QI Basin Stability and Path Experiments

**Hypothesis:** The QI solution sits in a basin that is narrow in certain parameter directions. Optimizers starting at or near the QI solution may drift away, and the construction and trained solutions may or may not share a basin.

Additionally, what can we learn about the Basin? what are its properties?

**Core:**
- **Low learning rate from construction:** Construct the QI MLP, then train with a small learning rate. Monitor whether the solution drifts away and in which parameter subspace. Compare SGD vs Adam. Train on the full dataset (no stochastic noise). Track lambda, gamma, outer weight norms, and loss throughout training.
- **Directional perturbation profiles:** From the QI solution, perturb along isolated directions (global gamma, per-neuron gamma, center shifts, readout weights). Sweep perturbation magnitude, measure loss increase. Then re-optimize from each perturbed point and record recovery probability and final error.
- **Path interpolation:** Interpolate between matched QI construction and trained-MLP solutions in parameter space. Track loss along the path. If a barrier exists, the construction and trained solutions are in different basins. If the path is flat, the difference is parameterization, not landscape.

**Additional:**
- Match neurons before parameter interpolation, and compare to function-space interpolation as a control.
- If needed, compare feature-subspace interpolation rather than raw-parameter interpolation to avoid permutation artifacts.
- Hessian-eigenvector perturbations: compute exact Hessian at the QI solution, perturb along the most negative, smallest positive, largest positive, and near-zero eigenvectors. Retrain. Does instability concentrate in specific eigen-directions?
- Repeat the low-LR experiment with different frozen subsets (freeze gamma only, freeze centers only, freeze readout only) to identify which parameter group is responsible for drift.
- Compare recovery probability from isotropic noise vs. Hessian-aligned noise at matched perturbation magnitude.

---

## 3. Geometry Ladder Cascade

**Hypothesis:** The inner-layer geometry (gamma, centers) is the hard part. Starting from the full construction and progressively relaxing constraints reveals exactly where precision is lost.

**Core:** Run a ladder from most constrained to least constrained. At each level, measure the best achievable error:
1. **Full construction:** QI geometry and readout weights all from construction. Baseline precision.
2. **Fixed geometry, exact readout solve:** Fix gamma and x_k from construction. Solve readout via least squares. Should match or nearly match step 1.
3. **Fixed geometry, trained readout:** Fix gamma and x_k. Train readout with Adam -> SSBroyden. How much precision is lost by training vs. exact solve?
4. **Fixed gamma, free centers, exact readout:** Fix gamma = lambda*/h, let centers float, solve readout exactly.
5. **Fixed gamma, free centers, trained readout:** Fix gamma, train both centers and readout.
6. **Free gamma, free centers, exact readout:** Everything free except readout is solved exactly.
7. **Fully free:** Standard end-to-end training (Adam -> SSBroyden).

**Additional:**
- Run the ladder at multiple widths to see if the precision-loss pattern changes with scale.
- For each level, log the three violation diagnostics: does lambda stay in the viable regime? Are outer weights O(1)? What is the feature rank?
- At level 4, compare starting centers on-grid vs. random initialization to test whether the grid structure matters or just the gamma scaling.

---

## 4. Hessian Landscape

**Hypothesis:** The full-parameter Hessian at the QI solution is ill-conditioned, but the reduced Hessian (over nonlinear geometry params only, with readout eliminated) is much better conditioned.

**Core:**
- Compute the Hessian eigenspectrum at the QI solution with only v_j (readout weights) free -- the most constrained training case from the geometry ladder (level 3). This is the Hessian of the MSE loss with respect to the readout weights only, with Phi fixed. Compare to the full-parameter Hessian at the same point. If the readout-only Hessian is well-conditioned (it should be -- it's Phi^T Phi), the optimization difficulty is entirely in the inner layer.
- Compare Hessian spectra at three matched solutions: QI construction, trained-MLP (Adam -> SSBroyden), and geometry-ladder solutions from Experiment 3. Does the construction sit in a more fragile region (more negative eigenvalues, more near-zero directions, worse conditioning)?

**Additional:**
- Compute the Gauss-Newton approximation J^T J at the QI solution and compare to the full Hessian. If they differ significantly, the loss surface is not well-approximated by a quadratic near the solution.
- Expand the residual on a Chebyshev basis for QI, trained-MLP, and SSBroyden solutions. Compare coefficient decay. If the gap is concentrated in high-frequency modes, the precision barrier is spectral.
- Track how the Hessian spectrum evolves along the training trajectory to see if conditioning degrades as the optimizer approaches the solution.

---

## 5. Phi Conditioning

**Hypothesis:** The feature matrix Phi (entries Phi_{i,m} = tanh(gamma * (x_i - x_m))) may be ill-conditioned, amplifying small errors in the readout weights or data into large output errors. This is separate from the loss Hessian -- it's about the forward map from weights to predictions.

**Core:**
- Compute cond(Phi) as a function of width N and lambda. Sweep N in {16, 32, 64, 128, 256} and lambda across a range including the viable regime, e.g. [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0].
- Compute the singular value spectrum of Phi at the QI solution. How many singular values are near zero? Does the effective rank match the width?
- Measure how perturbations to the readout weights propagate through Phi to output errors: if v_perturbed = v_true + epsilon * noise, how does ||Phi * (v_perturbed - v_true)|| / (epsilon * ||noise||) compare to cond(Phi)?

**Additional:**
- Compare cond(Phi) between QI geometry (uniform grid, gamma ~ N) and trained geometry (wherever the optimizer lands). Are trained solutions accidentally better conditioned?
- Measure how floating-point roundoff in the evaluation of Phi itself (computing tanh(gamma * (x - x_m)) at large gamma in fp64) contributes to output error.

- other thoughts: think about and interpret the rows/columns. how do they relate to gamma, should we be

---

## 6. Objective Mismatch

**Hypothesis:** Part of the training gap may be due to mismatch between the training objective (sampled MSE) and the evaluation criteria (eval L_inf, eval relative L2). Objective shaping may partially recover QI-like geometry.

**Core:**
- Compare training with uniform-grid MSE, denser-grid MSE, large-p losses approximating L_inf, and hybrid losses with boundary weighting.
- Check whether the optimizer still drives lambda -> 0 under all of these objectives.

**Additional:**
- Compare Chebyshev-weighted MSE or nonuniform collocation against uniform sampling.
- If derivatives are available analytically, test MSE + derivative matching.
- Measure whether the remaining error is concentrated near boundaries or in high-frequency regions.

---

## 7. Noise Sensitivity

**Hypothesis:** The construction is sensitive to noise because Phi amplifies perturbations. Understanding this sensitivity reveals whether training noise (gradient noise, floating point errors) is a fundamental barrier to high precision.

**Core:**
- **Y-noise:** Add Gaussian noise of varying magnitude to function values *before* constructing the QI MLP. Plot construction error vs noise level at different widths N. Does the error degrade gracefully or catastrophically?
- **X-noise:** Perturb the grid points x_k via Gaussian noise (breaking the uniform grid assumption). Same analysis. The construction assumes uniform spacing; how sensitive is it to this assumption?

**Additional:**
- For the trained-MLP solutions, add the same Y-noise to the training data and retrain. Compare the noise-sensitivity curve of trained vs. constructed solutions. If trained solutions are more noise-robust, they may be finding a more regularized region of the landscape.
- Measure how noise in gradient estimates (simulated by adding Gaussian noise to gradients during training) affects the final precision. This connects to whether SGD-style noise is fundamentally incompatible with high-precision convergence.

---

## 8. Reparameterization

**Hypothesis:** Raw coordinates (gamma_k, c_k) are the wrong optimization variables. The gradient d(tanh(gamma*x))/d(gamma) = x*sech^2(gamma*x) vanishes for large gamma*x, so gradient-based methods cannot efficiently increase gamma to the O(N) scale needed. Reparameterizing to natural coordinates should fix the scaling pathology.

**Core:** Compare these parameterizations head-to-head on the same widths, targets, and optimizer:
- **Raw:** standard weight matrix W, bias b
- **Log-scale gamma:** gamma = exp(eta), so d(phi)/d(eta) = gamma * d(phi)/d(gamma) = O(1) instead of O(1/N)
- **Global bandwidth:** single learnable lambda, gamma = lambda/h, centers on grid
- **Dimensionless centers:** c_k = -1 + h*(k + delta_k), learn delta_k instead of c_k directly

For each: train at widths N in {16, 32, 64, 128} with Adam -> SSBroyden. Log final error, learned lambda, outer weight norms, and convergence speed.

**Additional:**
- Alpha readout: learn alpha = a * gamma instead of a directly, so the learned parameter stays O(1) even when the effective readout weight a = alpha/gamma is O(1/gamma).
- Combined reparameterization: log-gamma + dimensionless centers + alpha readout + per-group LR scaling.
- Test whether reparameterization alone is sufficient, or whether it must be combined with exact readout solve / constrained geometry to reach high precision.

---

## 9. Variable Projection (VarPro) / Reduced Objective

**Hypothesis:** Eliminating the readout exactly and optimizing only the nonlinear geometry is not just a better method; it is a diagnostic that isolates whether the real failure sits in the geometry block.

**Core:**
- Nonlinear params: theta = (lambda, delta_k)
- Linear params v(theta) solved exactly via least squares at each iteration
- Optimize the reduced loss with Gauss-Newton, LM, or SSBroyden
- Compare reduced-objective training directly to the matched end-to-end run from the geometry ladder

**Additional:**
- Log reduced Jacobian / Hessian conditioning and compare to the full-parameter objective.
- If VarPro works dramatically better, that is evidence that raw end-to-end coordinates are the wrong optimization variables.

---

## Other thoughts
1. how does it work with regularization?
2. where does the mse objective come into this?
3. 

---

## Future Methods

These are methods to test *after* the diagnostic experiments above have clarified the landscape. Don't invest in implementation until the diagnostics tell us where to push.

### Deeper Networks

Delay until the 1-hidden-layer case is understood. Otherwise depth becomes a confound.

### Progressive Unfreezing / Width Continuation

Train small, prolongate to 2N by inserting intermediate centers, solve readout exactly, progressively unfreeze (lambda first, then delta_k). A homotopy that keeps the optimizer near the QI manifold at every scale. Requires the geometry ladder (Experiment 3) to first establish which constraints matter.

### Residual Stacking / Multilevel Refinement

Train stage 1 (optionally constrained), freeze, fit residual with stage 2, repeat. Each stage adds a fresh feature subspace and avoids rank saturation. Each stage solves a lower-dynamic-range problem. Requires noise sensitivity results (Experiment 6) to understand whether residual targets are too small for the optimizer's noise floor.

### PINN-style Regularization

Delay until there is a reliable clean-data solver. Then test whether constrained training survives extra loss terms.

---

## Thoughts

Increasing width gives more tanh functions to work with. Increasing gamma steepens the tanh slope, which the construction requires (gamma ~ N/2 * lambda). But gradient-based optimization cannot efficiently increase gamma because the gradient signal vanishes as gamma grows (vanishing sech^2 envelope). This is likely the central obstacle: the optimizer is blind to the direction it most needs to move in.

The most promising path is not a better unconstrained optimizer, but reduced-coordinate training: fix the geometry (or parameterize it in natural coordinates), eliminate the linear readout exactly, and optimize only a small nonlinear block. If this works, the story is "theoretical constructions as optimization guides" -- the QI theory reveals the right coordinate system for training.


