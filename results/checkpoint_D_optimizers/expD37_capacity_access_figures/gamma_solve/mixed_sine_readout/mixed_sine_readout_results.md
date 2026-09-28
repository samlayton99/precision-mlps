# Mixed sine: actual-kernel GD and 20,000-step Adam

Status: complete; computed results, September 23, 2026.

## TL;DR

The target and geometry match the collaborator note's screenshot. GD is calculated from the actual finite tanh features through one billion steps. Adam executes 20,000 full-batch updates, with the four gamma cases batched independently. Plot values are percentages, so the threshold is 1 on the vertical axis.

## Question

How long do ordinary readout GD and Adam take to reach 1% error on the mixed sine at four frozen gamma values?

## Experiment design

The target is $f(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)$ on $[-1,1]$. Use 8,193 uniform endpoint-inclusive samples, center spacing $h=2/512$, 513 interior centers and 23 halo centers per side: 559 tanh features plus bias. The common frozen slopes are $8,12,16,64$. Both optimizers start from zero physical readout coefficients, with no whitening or coefficient scaling. All calculations use FP64.

The objective is $L(v)=\|\Phi v-y\|^2/(2m)$, with $\Phi=[\mathbf1,\tanh(\gamma(x_i-c_j))]$. The plotted error is $100\|\Phi v-y\|/\|y\|$ on the training samples. With $J=\Phi/\sqrt m=U\Sigma V^T$, actual kernel eigenvalues are $\sigma_i^2$. The GD step is recomputed as $\eta=0.5/\sigma_1^2$. GD error uses $E(n)^2=\sum_i p_i(1-\eta\sigma_i^2)^{2n}+p_0$, including the orthogonal target remainder. The original note used saved approximately curvature-normalized steps; our direct recomputation need not reproduce every last integer of its archived counts.

Adam uses learning rate $10^{-3}$, betas $(0.9,0.999)$, epsilon $10^{-8}$, no weight decay and no schedule. Its gradient is evaluated from the original features each step, and ordinary PyTorch Adam updates the raw readouts. All four runs are batched together; every step is recorded. The final optimizer state is saved for continuation. First crossing does not imply that subsequent Adam errors stay below the threshold.

**Code and data.** `experiments/expD37_capacity_access_figures/gamma_solve/mixed_sine_readout/run.py` reuses the verified spectral and Adam routines from `sqrt_readout/run.py`. Configuration, input arrays, trajectories, summaries, verification and resumable Adam state are in `data/` beside this writeup. The earlier square-root experiment is unchanged.

## Results

| Gamma | GD first step ≤1% | Adam first step ≤1% | GD error at 20,000 | Adam error at 20,000 |
|---:|---:|---:|---:|---:|
| 8 | 15,798,313 | Not within 20,000 | 21.8281% | 6.6864% |
| 12 | 186,057 | 4,035 | 14.9494% | 0.4752% |
| 16 | 61,792 | 1,372 | 5.4615% | 1.7147% |
| 64 | 16,013 | 332 | 0.8668% | 0.0630% |

### Figures

- [GD versus Adam](gd_vs_adam.png): both methods share one set of axes; solid lines are actual-kernel GD through $10^9$ steps, dashed lines are executed Adam through 20,000. Colors identify gamma. The horizontal threshold is 1%; diamonds mark GD's first crossings. Curves below 0.1% leave the display. No smoothing or best-so-far replacement.
- [Long-range GD, screenshot-style scale](gd_long_range.png): GD displayed from 1,000 to one billion steps; diamonds and vertical guides mark the calculated first 1% crossings. These are predictions, not executed GD measurements or the note's Fourier construction.

## Verification and limits

A short direct-GD trajectory is compared to the spectral residual formula. The batched analytic Adam gradient is compared to ordinary autograd, and final errors are independently recomputed from saved coefficients using the original sample matrix. Discrepancies are recorded in `data/verification.json`. The GD curves describe exact-arithmetic iteration evaluated numerically in FP64; billions of physical updates may behave differently due to accumulated rounding. The error is training error, not held-out error. No Fourier or center-integral kernel approximation is used.

## Conclusions

The plotted curves give the requested direct comparison on the specified mixed sine. Broader claims about other targets or training geometry are outside this run.

## Open questions

No additional ablations were run.
