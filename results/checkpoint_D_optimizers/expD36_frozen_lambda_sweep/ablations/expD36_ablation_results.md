# Frozen-readout coordinate ablations — two sequential runs complete

Status: completed, one arm at a time, without combining interventions. Each arm contains 64 runs through 20,000 updates. Baseline runs and figures were preserved.

**21 September extension:** lambda1 (gamma128) was added to each arm and the raw baseline, again sequentially and through 20k steps. Each condition now has 72 runs over nine lambdas. All eight ablation figures include the addition and matched baseline points. The original histories, coefficients, and moments were preserved exactly; the reference-lambda table below is unchanged.

## 1. TL;DR

- Unscaled neighboring features and PR square-root allowance coordinates were tested separately against the original raw-coordinate baseline.
- Each arm has early/middle/final/best 4×4 figures. Rows 1–3 show that arm; row 4 overlays the raw baseline at the same budget.
- Both arms preserve geometry and the exact feature span. The baseline numerical least-squares reference is reused unchanged. No optimizer retuning or nonzero initialization was introduced.

## 2. Question

How much does the readout-coordinate choice change the observed lambda dependence before changing schedules, initial physical coefficients, geometry, or target definitions? Neighboring tests a localized basis; square-root scaling tests the PR's relative coefficient scales without neighboring.

## 3. Experiment design

The parent baseline supplies all four targets, eight lambda values, N256 uniform geometry with 24 halos per side, zero readout, FP64, 1,021 training midpoints, 4,093 evaluation midpoints, and 20,000-step horizon. Each arm starts afresh from this common zero prediction. The second arm is not initialized from the first.

Write the physical readout as $c=M\theta$, where $c$ contains the actual coefficients, $\theta$ contains the optimized coordinates, and $M$ is a fixed invertible map, including the bias coordinate. Let $A$ be the original physical training feature matrix, and $m$ the number of samples. Training uses $B=AM/\sqrt m$ and fresh products $B^\top(B\theta-y/\sqrt m)$.

**Arm 1: unscaled neighboring.** If $\phi_j$ is hidden feature $j$ ordered by center, use $\phi_1-\phi_2,\ldots,\phi_{W-1}-\phi_W,\phi_W$, plus the unchanged constant feature. The final anchor makes the transform square and invertible. There is no additional coefficient scaling.

**Arm 2: square-root allowance scaling.** Use $M=\operatorname{diag}(\sqrt\alpha)$, with the PR's reference allowance formula, evaluated on this campaign's existing 24-halo geometry. It includes the PR's special bias and outer-halo allowances. Reference lambda remains .25 across all actual swept lambdas. At N256, the ordinary-neuron allowance is .0194425, the largest halo allowance is 2.09254, and the bias allowance is 16.7726. These are allowances; their square roots are the coordinate multipliers. No neighboring is applied. The helper was checked against the independent numerical values recorded in the PR review at its N512/H23 setting.

GD retains the same rule, $\eta=1/\sigma_{\max}(B)^2$, so its numerical rate changes when the map changes. At lambda .25, rates are approximately .00680247 for raw coordinates, .496184 for neighboring, and .0341972 for square-root scaling. This is a comparison under the same curvature-normalized rate policy, not the same numerical learning rate.

Adam retains native learning rate .001, betas (.9,.999), and epsilon $10^{-8}$. There is no compensation to keep physical steps or physical epsilon thresholds identical: those changes are part of the coordinate intervention. All rates remain constant throughout training.

The numerical least-squares fit is copied from the raw baseline with its original relative cutoff $10^{-14}$. A new coordinate-dependent numerical truncation could produce a different apparent floor despite identical exact spans, so it is not substituted into these comparisons.

## 4. Results

All 128 new runs completed. Neighboring took 33.0 seconds and square-root scaling 31.5 seconds for training, evaluations, and checkpoints. These timings exclude code preparation and figure rendering.

At the predefined reference lambda .25, final-step relative L2 errors are:

| Target | Raw GD | Neighbor GD | Scaled GD | Raw Adam | Neighbor Adam | Scaled Adam |
|---|---:|---:|---:|---:|---:|---:|
| Sine | 2.294e-3 | 1.156e-3 | 9.217e-3 | 2.398e-4 | 2.302e-5 | 1.145e-3 |
| Quadratic | 4.628e-3 | 4.981e-3 | 4.997e-3 | 8.317e-4 | 3.885e-6 | 6.020e-4 |
| Mixed sine | 4.999e-2 | 2.866e-3 | 9.155e-2 | 5.059e-4 | 1.072e-5 | 5.701e-4 |
| Runge | 9.334e-5 | 1.763e-3 | 8.062e-4 | 2.803e-5 | 1.298e-7 | 1.278e-4 |

These are final snapshots, not converged-error claims. The complete trajectories and best-recorded figures expose the constant-rate Adam excursions. Neighboring gives lower final Adam error on all four targets at this lambda; GD improves on two and worsens on two. Square-root scaling alone does not produce a uniform improvement.

### Figures

- **Neighboring early:** rows 1/4 compare step2k performance versus lambda; rows 2/3 show the neighboring GD/Adam trajectories with vertical 2k markers.
- **Neighboring middle:** same layout with a 10k slice.
- **Neighboring final:** same layout with a 20k slice.
- **Neighboring best:** rows 1/4 use independently selected best recorded errors within 20k for each condition. Dots identify the neighboring trajectories' selected steps.
- **Scaling early:** the corresponding 2k slice for square-root allowance coordinates.
- **Scaling middle:** the corresponding 10k slice.
- **Scaling final:** the corresponding 20k slice.
- **Scaling best:** independently selected best recorded errors within the common 20k budget, with selected points marked below.

In row 4, orange means GD and blue means Adam; solid circles identify the ablation and dashed crosses the raw baseline. Axes match across function columns and snapshots within an arm. The trajectory/comparison error range is chosen from the arm and baseline together; it differs between the two arms because their attained errors differ. The first row retains the full range needed to show the numerical reference.

### Code & data

- [Ablation runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/ablations.py), [shared training](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/run.py), [plotting](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/plot.py), [map tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/test_ablations.py).
- Neighboring figures: [early](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/neighbor_unscaled/figures/early.png), [middle](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/neighbor_unscaled/figures/middle.png), [final](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/neighbor_unscaled/figures/final.png), [best](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/neighbor_unscaled/figures/best.png).
- Scaling figures: [early](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/sqrt_allowance/figures/early.png), [middle](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/sqrt_allowance/figures/middle.png), [final](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/sqrt_allowance/figures/final.png), [best](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/sqrt_allowance/figures/best.png).
- Resumable data: [neighboring](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/neighbor_unscaled/data), [scaling](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/ablations/sqrt_allowance/data). Each contains metadata, reference/map arrays, trajectory/optimizer state, and a physical-prediction equivalence check. [Continuation instructions](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/README.md).

## 5. Additional details

Ten implementation tests passed, covering the original optimizer/normalization checks plus map identities, the gradient chain rule, reference preservation, full map rank, allowance values, and exact native-coordinate checkpoint resumption. Both maps have rank306. At the final states, the largest difference between mapped and decoded-physical predictions, relative to the target norm, is $1.56\times10^{-14}$ for neighboring and $6.82\times10^{-15}$ for scaling, well below these trained errors.

The saved-data audit verified identical grids, lambdas, snapshot steps, and reference errors; finite completed trajectories; correct best-snapshot indices; and unchanged baseline data hashes. Saved coefficient snapshots are physical readouts. The continuation field and Adam moments are native coordinates, with the map stored alongside them.

## 6. Conclusions

These are two separate readout-coordinate comparisons with fixed geometry. They establish neither a combined recipe nor the theorem mechanism. Their results can be inspected against the original baseline without another training run.

## 7. Open questions

Combined neighboring/scaling, epsilon compensation, schedule changes, and initialization changes remain separate unrun ablations. Select the next intervention after reviewing these two comparisons.
