# expD29 — Sweep the approximation-gradient weight — draft-pending-Sam

## TL;DR

- The six-weight sweep changes the conclusion from the initial mu=1000 test: **mu=100000 produces substantial refitted approximation improvements on all four Xavier targets**. Those gains persist on an independent evaluation grid and at all three tested readout cutoffs.
- Better refitted geometry does not ensure a better current readout fit. At the largest weight, actual loss improves on sine and Gaussian but worsens on mixed sine and Runge; several trajectories have large nonmonotone jumps.
- Scale motion also depends on the target. Xavier mixed sine reaches mean gamma 1.76 and Gaussian reaches 0.757; Runge and sine stay much smaller. The network does not acquire uniform QI gamma 16.
- All 72 trajectories complete 500 updates. Large solved coefficients, numerical gradient sensitivity, a single seed, and the dense-SVD cost remain limitations of this diagnostic.

## Question / hypothesis

If ordinary GD mostly follows the readout-gap geometry gradient, does amplifying the approximation gradient produce useful geometry improvement, and how does that response vary with its weight mu at the unchanged base learning rate?

## Experiment design

Use the same normalized model and decomposition as expD28:

\[
 A(\theta)=\frac{1}{\sqrt n}[\tanh(x_i a_k+b_k),\mathbf1],\quad
 L=\tfrac12\|Av-y/\sqrt n\|^2,
 \quad F_\tau=\tfrac12\|(I-U_rU_r^T)y/\sqrt n\|^2,
 \quad G_\tau=L-F_\tau.
\]

The retained left singular vectors satisfy sigma_i > 1e-13 sigma_1. As in expD28, DF_tau includes the derivative of the moving retained subspace, not just the untruncated envelope formula evaluated at truncated coefficients.

Compare two runs from identical states:

\[
 \theta_{t+1}=\theta_t-\eta\,[w\,DF_\tau+DG_\tau],
 \qquad v_{t+1}=v_t-\eta\,\nabla_vL,
 \qquad w\in\{1,1000\}.
\]

Since DG_tau=DL-DF_tau, implementation adds **999 DF_tau** to the ordinary geometry gradient. This is gradient descent on H=L+999F_tau when w=1000. F_tau is independent of the current readout, so its added term leaves the readout gradient rule unchanged. All blocks update simultaneously from the current state. Solved coefficients are used to calculate F_tau and DF_tau; they are **never installed as the trained readout**.

The four targets are sine, mixed sine, Runge, and Gaussian-envelope mixed sine, with identical definitions to expD28. All train on the same 1,024 midpoint samples in [-1,1], including Gaussian. Use 500 updates, eta=0.002, fp64, seed 0, N=128 intervals and 177 neurons including 24 halo slots per side. The three initializations are ordinary Xavier; center-preserving scaled Xavier with mean initial lambda=0.25 and the same random readout; and uniform QI gamma=16 with zero readout. Geometry coordinates are every raw slope and bias. Gamma is |a|.

The initial comparison used weights 1 and 1000. The subsequent sweep uses **mu = 1, 10, 100, 1000, 10000, 100000**, with geometry direction mu DF_tau + DG_tau and objective H=L+(mu-1)F_tau. All other settings remain identical. The 24 initial trajectories are reused and 48 new trajectories are added, giving 72 unique cases. The extension has separate figures and compact data; earlier two-curve figures remain available.

The weighted run computes the numerical profile every update. Ordinary GD computes it only for evaluation at saved states. Plots of F_tau use identical saved steps in both runs to avoid comparing different diagnostic sampling densities; the full weighted trajectory remains saved. Ordinary loss and gamma are recorded every step. Independent numerical checks occur at steps 0, 1, 10, 50, 100, 250, and 500. They report sensitivity but do not gate, clip, or retune the requested update.

The three initial comparison figures have function columns and rows for actual loss L, numerical approximation loss F_tau, and change in mean gamma. Loss axes are logarithmic, gamma change is linear, and each panel fits its own vertical limits. Blue solid is ordinary GD; orange dashed is the weighted update. Initial mean gamma is stated in each figure. Rank-changing panels are annotated, and QI approximation-floor fluctuations are identified as numerically unresolved.

**Code & data**

- [Runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD29_weighted_profile/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD29_weighted_profile/config.yaml), [implementation tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD29_weighted_profile.py).
- [Data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/data): 24 compressed cases, approximately 5.8 MB, with saved parameters, per-step measurements, numerical audits, and timing. Summary and terminal-cutoff JSON files record endpoint measurements and independent-grid refit errors.
- [Xavier figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/xavier.png).
- [Scaled-Xavier figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/scaled_xavier.png).
- [QI figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/figures/qi_zero.png).
- [Sweep runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD29_weighted_profile/mu_sweep.py) and [mu values](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD29_weighted_profile/mu_sweep.yaml).
- [Sweep data and endpoint audits](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/data).
- [Xavier mu sweep](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/xavier.png).
- [Scaled-Xavier mu sweep](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/scaled_xavier.png).
- [QI mu sweep](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD29_weighted_profile/mu_sweep/figures/qi_zero.png).

## Results

### Initial comparison: mu=1 versus mu=1000

Xavier starts at mean gamma 0.088044. Its final numerical approximation losses are:

| Target | F_tau, ordinary GD | F_tau, weighted update | Final mean gamma, weighted |
|---|---:|---:|---:|
| Sine | 6.7424e-6 | 6.7011e-6 | 0.086521 |
| Mixed sine | 0.065950 | 0.065889 | 0.084113 |
| Runge | 0.0016982 | 0.00074423 | 0.088293 |
| Gaussian envelope | 0.028096 | 0.028087 | 0.088634 |

The Runge improvement corresponds to a 56% decrease in the numerical approximation loss relative to ordinary GD. An independent 8,192-point evaluation with solved readouts confirms a smaller residual: relative L2 is 0.1473 for ordinary GD geometry and 0.0975 for weighted geometry. Actual training loss only changes from 0.042752 to 0.042663, about a 0.21% improvement. The distinction between better refitted geometry and an unfinished actual readout remains visible.

Mixed sine moves its mean gamma farther downward and has about 1.6% worse actual loss at the end. Its numerical approximation loss repeatedly jumps as the retained rank alternates between 9 and 10. Runge alternates between ranks 10 and 11. These jumps are not smooth changes in an exact fixed-rank profile.

For scaled Xavier, the final approximation loss is about 0.94% smaller on mixed sine and 1.69% smaller on Gaussian, compared with ordinary GD. The other changes are very small. All four actual-loss ratios differ from one by less than 7e-6. Thus the visual separation in the tightly zoomed approximation panels should not be mistaken for a large change in overall fitting. Mean gamma stays essentially at 16.

QI's actual-loss and gamma trajectories are indistinguishable at the relevant plotting precision. Its profiled losses are already around 1e-30 to 1e-28; their fluctuations do not establish approximation gains or losses.

### Expanded sweep: mu=1 through 100,000

The final Xavier numerical approximation losses F_tau are:

| Target | mu=1 | 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|---:|---:|---:|
| Sine | 6.7424e-06 | 6.742e-06 | 6.7381e-06 | 6.7011e-06 | 3.8123e-06 | 1.0658e-08 |
| Mixed sine | 0.06595 | 0.065944 | 0.065911 | 0.065889 | 0.06575 | 0.014892 |
| Runge | 0.0016982 | 0.0016979 | 0.0016947 | 0.00074423 | 0.00023783 | 1.278e-05 |
| Gaussian envelope | 0.028096 | 0.028096 | 0.028094 | 0.028087 | 0.027312 | 0.0045177 |

Small weights barely change the initial comparison. At 10,000, sine and Runge improve more noticeably. At 100,000, all four targets have substantially lower numerical approximation losses than ordinary GD. This improvement is not monotone at every training step, and individual rank crossings still create jumps.

Independent-grid refitting and the actual GD loss separate the two outcomes:

| Xavier target | Refit relative L2, mu=1 | Refit relative L2, mu=100,000 | Actual loss change | Final mean gamma, mu=100,000 |
|---|---:|---:|---:|---:|
| Sine | 0.00519396 | 0.000206721 | -1.02% | 0.0909356 |
| Mixed sine | 0.44832 | 0.21307 | +15.57% | 1.75871 |
| Runge | 0.147289 | 0.0127785 | +14.24% | 0.139493 |
| Gaussian envelope | 0.491559 | 0.197114 | -27.18% | 0.75741 |

The actual-loss change compares mu=100000 with ordinary GD at the same final step; negative means a lower actual loss. Refitted errors use the same 8,192-point evaluation grid after solving on the training grid. All Xavier targets begin at mean gamma 0.088044. At the largest weight, mixed sine has maximum gamma 34.0 while its mean is only 1.76: the geometry changes unevenly across neurons, rather than becoming uniform QI geometry.

In scaled Xavier, mu=100000 reduces F_tau relative to ordinary GD by about 44% on mixed sine and 61% on Gaussian. The actual losses change by less than 0.08%. Sine and Runge approximation losses change little. QI remains at its numerical approximation floor, with no meaningful change in actual fitting.

### Figures

- **Xavier:** top panels show almost overlapping actual-loss curves; middle panels show small sine/Gaussian changes and the rank-dependent Runge and mixed-sine jumps; bottom panels show that amplifying DF_tau changes scale motion without finding gamma of order 16.
- **Scaled Xavier:** the middle row is deliberately zoomed to the very small numerical approximation losses. Mixed sine and Gaussian separate somewhat while actual loss and mean scale remain almost unchanged. Both F_tau curves use the same evaluation steps.
- **QI:** the actual fitting curves overlap, while profiled-loss fluctuations remain at numerical precision. The bottom row displays changes relative to initial mean gamma 16, making the small ordinary motion visible without printing long strings of nearly identical 16.0000 labels.

- **Xavier mu sweep:** same function columns and three quantities, with six viridis curves ordered by mu. Ordinary GD is dashed. Look for the strong refitted gains at the largest weight and the different response of actual loss. The gamma-change row uses a signed log scale to expose both small and large movements.
- **Scaled-Xavier mu sweep:** larger weights reduce the already-small mixed-sine and Gaussian approximation losses, while actual GD fitting barely changes. Vertical limits are local to each panel; the visual size of a separation is not its absolute size.
- **QI mu sweep:** all actual fitting and mean-gamma curves overlap to plotting precision; changes in F_tau remain numerical-floor fluctuations.

## Additional details

### Cutoff and coefficient sensitivity

The terminal Runge result illustrates the limitation of the numerical reference:

| Readout cutoff | Ordinary-GD refit relative L2 | Weighted-geometry refit relative L2 |
|---|---:|---:|
| 1e-12 sigma_1 | 0.14735 | 0.14626 |
| 1e-13 sigma_1 | 0.14729 | 0.09751 |
| 1e-14 sigma_1 | 0.14729 | 0.09751 |

At the primary cutoff, the solved coefficient norm grows from 8.49e9 for the ordinary-GD geometry to 2.36e11 for the weighted geometry. A more permissive readout uses the newly retained direction and produces a better function on the independent grid; the gain is therefore not merely a training-point reporting effect. However, its magnitude depends strongly on retaining that nearly singular direction. This is not yet a robust path toward the QI approximation regime with modest coefficients.

The alternative-SVD/tanh checks also find unresolved DF_tau estimates at some weighted Xavier states. The 1,000× intervention amplifies that uncertainty along with the gradient. This experiment therefore cannot establish that increasing the exact approximation gradient would universally fail, nor that numerical spikes are a property of the note's exact loss.

### Numerical audit of the larger weights

At mu=100000, the refitted evaluation error is lower than the ordinary-GD control on all four Xavier targets at each tested cutoff: 1e-12, 1e-13, and 1e-14 times the largest singular value. This is stronger evidence than the initial mu=1000 Runge result, whose advantage largely disappeared at 1e-12. The magnitudes still depend on the cutoff, especially for sine and mixed sine.

For Runge, mu=100000 gives refit relative L2 0.01278 at both the primary and stricter cutoffs, compared with about 0.1473 for the control. Its solved coefficient norm is 3.35e8, lower than the control's 8.49e9. This larger-weight improvement does not require the coefficient-norm increase seen at mu=1000. Nevertheless, the terminal solved coefficient norms across the four mu=100000 Xavier runs remain large, from approximately 3e8 to 1.4e10.

The independent numerical-gradient checks remain relevant: five of seven checked states are unresolved for the mu=100000 Xavier sine trajectory, and one of seven for each of the other three Xavier targets. The improved terminal functions are observable, but the trajectory cannot be treated as a numerically exact realization of unrestricted VarPro gradient descent.

### Verification and cost

The implementation checks confirm that the modified geometry gradient agrees with independent autograd of the weighted scalar loss through a moving singular subspace; weight 1 reproduces ordinary GD; and the first readout update matches ordinary GD for both weights while geometry updates differ. All twelve full baseline trajectories match the existing expD28 ordinary losses bitwise through step 500, with exact agreement of saved parameters at shared steps. Every paired run starts from identical parameters.

The existing expD28 derivative tests supply additional independent checks of the numerical profile and its matrix derivative. No training loss comparisons, line search, clipping, learning-rate changes, or solve replacement enter this experiment. All 24 cases finish the requested 500 steps; aggregate measured training and diagnostic time is about 83 seconds on this machine.

The sweep adds 48 new runs while reusing the original 24; all 72 reach the requested final step. The new runs take about 281 seconds in aggregate and add about 12 MB of compressed data. Six focused tests pass after extending the independent scalar-gradient check to multiple weights and checking that a stopped trajectory preserves its last valid unscheduled state. Matched initial parameters and terminal-state consistency are verified across all sweep cases.

Each weighted step performs a dense feature-matrix SVD. This is an explicitly requested small-problem mechanism test and does not meet the project's production optimizer cost or architecture-independence requirements.

## Conclusions

Amplifying the numerical approximation gradient can produce substantial geometry improvement from Xavier, beyond what the mu=1000 pilot showed. The sweep also exposes a tradeoff: stronger improvement after readout refitting can coexist with worse current GD fitting and irregular training trajectories.

## Open questions

- Does the useful part of this response persist with an independently validated approximation objective that avoids hard-cutoff discontinuities and large-coefficient sensitivity?
- Can a stable weighting rule preserve these approximation gains while fitting the current readout effectively, with modest coefficients and consistency across seeds? This 500-step sweep does not establish such a rule.
