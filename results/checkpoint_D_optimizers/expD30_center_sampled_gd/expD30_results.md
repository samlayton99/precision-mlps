# expD30 — Gamma 128 with training samples at the centers

Status: data-obvious; completed requested comparison, awaiting Sam's interpretation.

## TL;DR

- The center-sampled problem is interpolatable with modest readout coefficients. Initial least-squares training losses are approximately $10^{-30}$–$10^{-29}$.
- After 200,000 actual GD updates, the best tested rate gives training relative errors from $1.2\times10^{-6}$ to $6.1\times10^{-4}$ with fixed geometry. Joint training chiefly helps mixed sine, reaching $3.2\times10^{-4}$.
- The larger constant rate beats both the previous rate and warmup/cosine decay on every target and geometry policy. The fixed readout remains poorly conditioned; feasibility does not imply fast high-precision GD.
- Independent-grid errors remain around $10^{-3}$–$5\times10^{-3}$. Fitting these samples is a different accuracy goal from fitting the continuous target.

## Question / hypothesis

If gamma is deliberately large and data are sampled only at the exact initial centers, can ordinary GD fit the training problem accurately, how many updates does it take, and do geometry training or learning-rate schedules help?

## Experiment design

Sam requested this extreme setting and confirmed comparing fixed geometry against jointly trained geometry. Use the same four finite-domain targets: sine $\sin(2\pi x)$, mixed sine $m(x)=\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$, Runge $(1+25x^2)^{-1}$, and Gaussian envelope $\exp[-x^2/(2(0.4)^2)]m(x)$.

There are 128 intervals on $[-1,1]$, spacing $h=1/64$, and 129 in-domain centers. Those 129 centers, including both endpoints, are the training samples. There are 24 halo neurons on each side, giving 177 neurons plus output bias. Halo centers are not additional training samples. Initially $a_j=128$, $b_j=-128z_j$, so $\gamma=128$ and $\lambda=2$. All readout coefficients, including bias, start at zero. Training samples stay fixed when learned centers move.

$$
f(x)=\sum_j c_j\tanh(a_jx+b_j)+d,\qquad
L=\frac{1}{2n}\sum_{i=1}^n(f(x_i)-g(x_i))^2,\qquad
E=\sqrt{\frac{\sum_i(f(x_i)-g(x_i))^2}{\sum_i g(x_i)^2}}.
$$

Fixed-geometry GD updates only $(c,d)$. Joint GD updates raw $(a,b,c,d)$ at the same rate, without center constraints. Independent targets are batched for computation, with their own networks and loss normalization. They do not share trained parameters or receive an extra factor of four in their gradients. PyTorch fp64 training uses an analytic backward pass checked independently against autograd.

The three prescribed choices, each run for 200,000 actual updates, are:

1. Constant $\eta=0.002$.
2. Constant $\eta=1.9/\sigma_1(A_0)^2=0.01971069$, where $A_0=[\tanh(128(x_i-z_j)),1]/\sqrt{129}$. The fixed-geometry stability limit is $2/\sigma_1^2=0.02074810$. This limit is not asserted as a joint-network stability theorem.
3. A 1,000-update linear warmup from 0.002 to 0.01971069, followed by cosine decay ending at 1% of that peak.

This gives 24 target/policy/schedule trajectories. There is no momentum, clipping, stochastic sampling, coefficient solve in training, or loss-based acceptance test. A separate SVD sets the diagnostic rate and supplies the fixed-geometry prediction. Fresh model residuals are recorded every step; sparse independent evaluations use 8,192 uniform midpoint samples. Terminal refits remain offline. Mean and maximum absolute gamma changes, center displacement, and coefficient norms are recorded.

**Code & data**

- [Implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD30_center_sampled_gd/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD30_center_sampled_gd/config.yaml), [verification tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD30_center_sampled_gd.py), [design and practicality checklist](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/expD30_plan.md).
- [Training comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/training_convergence.png), [matrix spectrum, weak directions, convergence prediction, and schedules](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/spectrum_and_schedules.png), [between-center evaluation](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/figures/between_centers.png).
- [Data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/data), [all 24 endpoint and threshold summaries](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD30_center_sampled_gd/data/summary.csv): six compressed trajectory files, one spectral diagnostic, and one summary CSV; 31.6 MiB total. The driver supports `--resume` with configuration matching and `--plot-only` without training.

## Results

**The problem is feasible but remains slow at high accuracy.** The initial matrix has all 129 row singular values numerically resolved, with largest singular value 9.818 and smallest 0.0009244, giving condition number 10,621. Initial minimum-norm readout norms range from 0.124 to 1.044, so this feasibility result does not depend on the huge coefficients seen in earlier small-gamma solves.

The larger constant rate is best at the executed endpoint for every target and policy. After 200,000 steps:

| Target | Fixed GD loss $L$ | Joint GD loss $L$ | Fixed train $E$ | Joint train $E$ | Joint independent-grid $E$ |
|---|---:|---:|---:|---:|---:|
| Sine | 6.383e-09 | 5.258e-09 | 1.604e-04 | 1.456e-04 | 1.719e-03 |
| Mixed sine | 1.200e-07 | 3.386e-08 | 6.072e-04 | 3.225e-04 | 5.340e-03 |
| Runge | 1.174e-13 | 1.174e-13 | 1.229e-06 | 1.229e-06 | 9.061e-04 |
| Gaussian envelope | 1.789e-10 | 1.781e-10 | 3.938e-05 | 3.929e-05 | 4.709e-03 |

These are half-MSE losses and relative errors, respectively; a loss of $10^{-13}$ is not a relative error of $10^{-13}$. All runs remain finite and all remain above the interpolation floor. The continued descent in the traces is not a measured nonzero limiting loss.

At the larger rate, the first recorded step reaching each relative-error threshold is:

| Target | Fixed: $E\leq10^{-2}$ | Joint: $E\leq10^{-2}$ | Fixed: $E\leq10^{-3}$ | Joint: $E\leq10^{-3}$ |
|---|---:|---:|---:|---:|
| Sine | 766 | 766 | 17,102 | 16,879 |
| Mixed sine | 5,135 | 4,984 | 102,834 | 58,228 |
| Runge | 224 | 224 | 700 | 700 |
| Gaussian envelope | 2,729 | 2,727 | 5,083 | 5,081 |

**Joint training helps chiefly on mixed sine.** Its endpoint relative training error falls by about 47% compared with fixed geometry, and it reaches $10^{-3}$ in roughly 58,000 rather than 103,000 updates. This does not improve its independent-grid error: the final joint error there is slightly worse than the fixed counterpart. In the larger-rate joint runs, mean absolute gamma displacement ranges from $8.4\times10^{-7}$ to $8.9\times10^{-4}$; the largest individual displacement is 0.0289 for mixed sine. Mean center displacement is at most $1.6\times10^{-5}$. Both slopes and centers can contribute to the training benefit.

**Cosine decay does not improve this comparison.** Its final relative errors are larger than those of the larger constant rate for every target in both modes. For small fixed-matrix eigenvalues, per-step contraction is approximately $1-\eta_k\sigma_j^2$, so reducing the accumulated learning rate slows those directions. This explains the fixed-geometry behavior; it does not rule out every other schedule, especially schedules allowing temporarily expansive steps.

**The fixed-geometry long horizon is quantitatively predictable.** Decompose the normalized target as $\alpha_j=u_j^Ty$, where $A_0=U\Sigma V^T$. With zero readout and full row rank,

$$
L_k=\frac12\sum_j\alpha_j^2(1-\eta\sigma_j^2)^{2k}.
$$

Using the larger constant rate gives these exact-arithmetic predictions, not executed training:

| Target | Steps to $E\leq10^{-4}$ | Steps to $E\leq10^{-8}$ | Steps to $E\leq10^{-12}$ |
|---|---:|---:|---:|
| Sine | 376,059 | 109,491,505 | 246,090,616 |
| Mixed sine | 2,216,111 | 129,204,470 | 265,803,580 |
| Runge | 1,663 | 107,922,136 | 654,726,602 |
| Gaussian envelope | 57,734 | 88,633,627 | 225,232,737 |

The smallest singular direction alternates sign at all 128 adjacent sample pairs. The second smallest alternates at 127 pairs. Odd targets barely load onto the weakest even direction, so predictions retain each target's actual mode weights instead of quoting only the worst condition number. The three odd targets nevertheless load onto the second-weakest direction. This is a matrix-conditioning diagnosis, not a measurement of a whole-line Fourier spectrum.

### Figures

- **Training convergence:** four target rows and two columns, fixed geometry left and joint training right. Horizontal axis is actual GD update number, linear through 10 and logarithmic afterward; vertical axis is training half-MSE with common limits $10^{-16}$ to 1. Colors identify the three schedules. Each panel reports final relative errors and the much smaller initial LS loss, which lies below the displayed range.
- **Spectrum and schedules:** a 2×2 diagnostic. Top left is the initial singular-value spectrum; top right shows its two weakest sample-space directions. Bottom left predicts target-specific fixed-GD relative errors out to two billion steps, marking the executed 200,000-step horizon. Bottom right shows the three rates and the initial fixed-geometry stability limit.
- **Between centers:** columns are targets. Top row shows final spatial residuals for the larger constant rate and the initial LS interpolant. Bottom row compares training-center relative error with independent midpoint-grid error over actual steps; line style separates joint and fixed geometry. Small sample losses coexist with persistent errors between centers.

## Additional details

Four focused tests pass: exact sampling/initial interpolation, raw joint updates versus independent autograd with correct target batching, zero-readout first-step/frozen-readout update consistency, and schedule/spectral recurrence checks. Full-run fixed-geometry predictions agree with actual relative errors within $2.86\times10^{-14}$ for the old rate and $9.66\times10^{-15}$ for the larger rate. Frozen slopes and biases are checked exactly, and every recorded training loss is finite. Endpoints are actual fresh residuals rather than recursively carried residual estimates. Training plus sparse evaluation inside the six batched arms took approximately 192 seconds; this excludes figure rendering and external test time.

The prediction approaches zero in exact arithmetic; it does not promise that hundreds of millions of freshly evaluated fp64 GD updates will reach the plotted precision. The reference solves and step predictions are small-problem diagnostics, not a scalable optimizer proposal. Because this experiment changes both gamma and sampling relative to the previous dense-grid runs, it does not isolate their individual effects. It also does not compare gamma 128 against other gamma levels on these same samples.

## Conclusions

Overblown gamma and center-only data remove the observed training approximation floor while leaving an ordinary-GD conditioning problem. The best tested constant rate accelerates fitting, but conventional decay does not overcome the weak directions, and joint training only partly changes the outcome.

## Open questions

- Whether a different approved GD schedule can accelerate these weak directions without sacrificing stable numerical progress.
- How much the conditioning depends specifically on exact center alignment, as opposed to nearby sampling locations. No shifted-sample control was run.
