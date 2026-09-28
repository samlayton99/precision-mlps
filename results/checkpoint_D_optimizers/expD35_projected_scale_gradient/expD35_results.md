# expD35 — Current-readout gamma-gradient projection split

Status: complete; descriptive results recorded, mechanism interpretation pending Sam's review.

## TL;DR

- Requested diagnostic: four targets, ordinary joint GD, Xavier and uniform gamma-16/QI initializations, 10,000 steps each.
- One 4×4 figure: gradient norms and cosine similarity for Xavier in the top two rows; the same quantities for gamma 16 in the bottom two rows.
- The current readout is used throughout. These are the note's projection terms, not the VarPro F/G gradient split.

## Question / hypothesis

How large are the readout-shared and out-of-span parts of the current gamma gradient, and do their directions align or oppose one another during ordinary training?

## Experiment design

The four columns are sine, mixed sine, Runge, and Gaussian envelope, using the same functions as the preceding matched experiments. All use 1,024 midpoint samples on [-1,1], float64, seed 0, and half-mean-square loss. Full-batch GD trains raw slopes, biases, and readout coefficients together at constant learning rate 0.002 for 10,000 updates. Xavier retains its random readout; the uniform gamma-16 initialization uses zero readout. N=128 follows the previous grid convention: 129 interior centers and 24 halo centers per side, totaling 177 neurons. Norms include all 177 trainable slopes.

Sam selected the actual GD slope component, not the center-preserving Fourier tangent. Locally at nonzero slope, gamma_j=|a_j| and the diagnostic Jacobian is

$$J_{\gamma,ij}=\frac{\operatorname{sign}(a_j)c_jx_i\operatorname{sech}^2(a_jx_i+b_j)}{\sqrt n}.$$

Bias b is held fixed in this derivative; it still trains in the actual trajectory. Thus these quantities exclude the bias-gradient components, without freezing biases or centers during training.

With normalized A=[tanh(ax+b),1]/sqrt(n), r=(f-y)/sqrt(n), and a retained SVD basis U, the measured vectors are

$$g_{\rm shared}=(U^TJ_\gamma)^T(U^Tr),\qquad Z_\gamma=J_\gamma-U(U^TJ_\gamma),\qquad g_{\rm independent}=Z_\gamma^T[r-U(U^Tr)].$$

The shared expression evaluates $C_\gamma^T\nabla_vL$ without explicitly multiplying an ill-conditioned pseudoinverse into the residual. The total is independently calculated as $g_{\rm total}=J_\gamma^Tr$ and compared with the training autodiff gradient after the sign conversion. The plotted cosine is $g_{\rm shared}^Tg_{\rm independent}/(\|g_{\rm shared}\|\|g_{\rm independent}\|)$.

The numerical projector retains singular values above 1e-13 times the largest. It is a retained-space approximation to the exact column-space projector. Measurements are offline at 200 unique snapshots: every step through 20, then logarithmically distributed through 10,000 (the requested snapshot budget of 201 loses one duplicate endpoint). Training losses and mean gamma are stored at every step. No coefficient solve or projection affects training.

Each snapshot compares two SVD algorithms and NumPy versus PyTorch tanh evaluation. A cosine is marked resolved only when ranks agree and both vector norms exceed ten times their estimated discrepancy; the independent term also has a scale-aware roundoff guard. Gray dotted cosines show unresolved numerical estimates, not reliable evidence of alignment or cancellation. At zero readout both terms vanish, making the step-zero QI cosine undefined.

**Code & data**

- [Implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD35_projected_scale_gradient/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD35_projected_scale_gradient/config.yaml), [tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD35_projected_scale_gradient.py).
- [Saved trajectories and diagnostics](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD35_projected_scale_gradient/data).
- [Requested 4×4 figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD35_projected_scale_gradient/figures/projected_gamma_gradient.png).

## Results

All eight trajectories completed 10,000 steps. Three focused implementation tests pass: literal projection-formula agreement, signed-slope autodiff agreement including zero readout, and nonmutation of the ordinary-GD trajectory by diagnostics. Across all snapshots the maximum Euclidean reconstruction discrepancy is 7.52e-16; maximum discrepancy between the directly calculated total and autodiff is 2.53e-16.

The total and shared-term norm curves visually overlap. The independent-term norms are much smaller in these measurements. Numerical direction resolution varies substantially, so the cosine rows retain explicit uncertainty markings rather than interpreting every finite number as a physical direction.

| Initialization | Sine resolved cosines | Mixed sine | Runge | Gaussian envelope |
|---|---:|---:|---:|---:|
| Xavier | 105/200 | 37/200 | 200/200 | 0/200 |
| Gamma 16, zero readout | 0/200 | 29/200 | 0/200 | 19/200 |

These are counts passing the numerical checks described above, not statistical confidence estimates. Agreement of the two large terms to roundoff does not independently certify every digit of the much smaller term.

### Figures

- **Projected gamma-gradient decomposition:** columns are the four functions. Rows 1/3 show Euclidean norms of the shared term (orange), independent term (green), and total gradient (black dashed) for Xavier/gamma 16, with common logarithmic limits. Rows 2/4 show the cosine between the two terms on [-1,1]. Steps are linear through 10 and logarithmic thereafter. The total may overlap the shared term; zero norms are omitted rather than replaced with a floor. No loss or bias-gradient norm is shown.

## Additional details

This is a diagnostic of the current-readout Section 2 identity. It does not measure singular-value-squared convergence, differentiate the profiled loss, or change the optimizer. In particular, a tiny current-readout independent term is not the same statement as a tiny gradient evaluated with solved readout coefficients.

## Conclusions

The requested current-readout slope decomposition is measured and plotted for both initializations. Mechanism claims beyond these measurements are deferred to user review.

## Open questions

Interpretation of the measured alignment and imbalance is deferred until the requested figure is reviewed.
