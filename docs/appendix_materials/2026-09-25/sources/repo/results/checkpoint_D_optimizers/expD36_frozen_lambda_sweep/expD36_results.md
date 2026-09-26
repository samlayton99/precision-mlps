# Frozen-lambda readout sweep — complete first phase

Status: 64 runs completed through 20,000 updates; four requested figures saved. No geometry training or theorem tests. Scientific interpretation remains for discussion with Sam.

**21 September extension:** lambda1 (gamma128) was added independently for GD/Adam on all four functions, making 72 runs over nine lambdas. The four figures and continuation data include it. The initial eight-lambda record and table below describe the original run; those saved cases were preserved exactly. Twelve implementation tests passed, including the append/merge checks.

## 1. TL;DR

- Four targets, eight frozen lambda values, ordinary GD and Adam, zero readout. All runs reached 20,000 steps.
- Four 3×4 figures show the same trajectories with top-row slices at 2,000, 10,000, and 20,000 steps, plus best recorded performance. The corresponding steps are marked below.
- Training, evaluation, and checkpoint saves took 31.5 seconds on the local CPU using batched feature products. Data and Adam state are saved for continuation.

## 2. Question

How does frozen hidden scale affect readout optimization at a fixed step budget? This first phase establishes a raw-coordinate baseline before the PR's conditioning maps are introduced. The numerical least-squares reference measures available fits under one stated rank policy; reaching that curve is not the sole success criterion.

## 3. Experiment design

The network is

$$f(x)=c_0+\sum_{j=1}^{W}c_j\tanh(\gamma(x-z_j)).$$

Here $c_0$ is the trained output bias, $c_j$ are the trained readout coefficients, $z_j$ are fixed centers, and the positive slope $\gamma$ is identical across neurons in each run. All readouts begin at zero. Geometry is frozen throughout.

The grid convention is $h=2/N$ with $N=256$, and $\lambda=\gamma h$. There are 257 centers on $[-1,1]$ plus 24 halo centers on each side, giving $W=305$ neurons and 306 trained readout parameters. Training uses 1,021 uniform midpoints; evaluation uses 4,093 uniform midpoints on the same interval. The coprime counts avoid repeated alignment; the two odd midpoint grids share only the central point.

The four targets, in column order, are

$$\sqrt{2}\sin(2\pi x),\qquad \sqrt{5}x^2,\qquad \frac{\sin(2\pi x)+0.1\sin(20\pi x)}{\sqrt{0.505}},\qquad \frac{1}{1+25x^2}.$$

The first three use the revised note/PR's definitions. Runge uses the existing repository definition. No additional target normalization is applied during training.

The smallest slope is the PR's Xavier RMS scale, $\gamma_X=(5/3)\sqrt{2/(W+1)}\approx0.134742$. Eight initially geometric lambda points run from $\lambda_X\approx0.00105267$ to $.5$; the nearest interior point to $.25$ is replaced by $.25$. The actual lambda grid is

$$0.00105267,\ 0.00253907,\ 0.00612432,\ 0.0147720,\ 0.0356306,\ 0.0859419,\ 0.25,\ 0.5.$$

The training objective is $L=\frac{1}{2m}\sum_{i=1}^{m}(f(x_i)-y_i)^2$, where $m=1021$. Let $A$ be the physical feature matrix, including the constant column, and $B=A/\sqrt m$. GD uses the constant per-geometry rate $\eta=1/\sigma_{\max}(B)^2$. Adam uses constant rate $.001$, betas $(.9,.999)$, epsilon $10^{-8}$, and no weight decay. Both use raw readout coordinates. No line search, momentum in GD, refit, or geometry update enters training.

The runner computes fresh direct products $B^\top(Bc-y/\sqrt m)$ in FP64. It does not train on rounded normal equations. The four targets and two optimizers are columns of batched products with separate readouts and separate Adam moment entries. The batching does not couple their updates.

All plotted quantities are evaluation relative L2 error, $\|f-y\|_2/\|y\|_2$, not MSE. Evaluations are recorded at every step through 100 and every 20 steps thereafter: 1,096 snapshots including step zero. Physical readouts are additionally saved every 200 steps. The final checkpoint includes the full Adam state and step count, so the constant-rate run can continue without resetting its history.

The separate least-squares reference uses SVD of the same physical training features with relative singular-value cutoff $10^{-14}$. Its plotted error uses the independent evaluation grid. This is a numerical reference, not a proof of a true approximation floor or a lower bound on evaluation error.

## 4. Results

All runs completed with finite recorded errors. Every step-zero error equals one. The table reports the smallest **final-step** error among the eight lambda values for each optimizer; it does not mix in earlier snapshots.

| Target | GD at 20k, best lambda | Adam at 20k, best lambda |
|---|---:|---:|
| Sine | 1.81482e-3 | 2.39819e-4 |
| Quadratic | 3.39209e-3 | 5.41749e-4 |
| Mixed sine | 5.46506e-3 | 5.05900e-4 |
| Runge | 6.69919e-5 | 2.80341e-5 |

### Figures

- **Early:** top row at step 2,000. Middle and bottom rows show all 20k GD/Adam trajectories, with the slice indicated by a vertical dashed line.
- **Middle:** identical layout, top row at step 10,000 and corresponding vertical markers.
- **Final:** top row at step 20,000 and corresponding vertical markers.
- **Best:** each top-row point uses the minimum recorded evaluation error from that specific optimizer/lambda/function run. Dots in the lower rows mark those minima, which may occur at different steps. No averaging, smoothing, or cross-lambda substitution is applied to these points.

The top row has common log error limits across columns and all four figures. Both trajectory rows share another common set of log error limits chosen from all trajectories; their horizontal axes are linear gradient steps. Lambda colors are shared across functions and optimizers. Least squares is dotted in the top row only.

### Code & data

- [Configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/config.yaml), [runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/run.py), [plotting](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/plot.py), [validation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/test_run.py), [continuation instructions](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD36_frozen_lambda_sweep/README.md).
- [Early figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/figures/early.png), [middle figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/figures/middle.png), [final figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/figures/final.png), [best figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/figures/best.png).
- [Trajectory and continuation state](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/data/trajectory.npz), [geometry and reference fits](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/data/reference.npz), [metadata](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/data/metadata.json).

## 5. Additional details

Four tests passed: batched GD/Adam against standard PyTorch updates; zero-start and gradient-normalization checks; bitwise-identical interrupted/resumed training; and reference evaluation against the actual feature matrix. Batched and single-case matrix products have slightly different FP64 reductions, so the independent PyTorch comparison allows that rounding, including Adam amplification of near-symmetry-zero gradients. Continuation with the same batched implementation is exact in the test.

The saved-data audit checked all 64 final states, all 1,096 metric snapshots, the three requested slice steps, finite errors, and consistency of the best indices with the minimum of the stored trajectories. Figure layout and markers were inspected.

## 6. Conclusions

This supplies the requested 20k-step raw-coordinate baseline and numerical reference curves. It does not establish a theorem mechanism or compare the PR's conditioning maps.

## 7. Open questions

Interpret the raw-coordinate lambda dependence before selecting the next ablation. Constant-rate Adam and conditioned readout coordinates remain distinct experimental choices; neither was changed within this run.
