# Square-root target: spectral GD and 20,000-step Adam

Status: complete; computed results, September 23, 2026.

## TL;DR

- The GD curves use the actual finite tanh feature matrix, with eigenvalues obtained from its rectangular SVD; there is no center-integral or periodic approximation.
- Adam runs the same frozen readout problems for 20,000 full-batch steps, with all six gamma cases batched in parallel.
- Both main panels show training relative L2, the same axes, and a 1% line. The extra GD figure extends the analytic trajectory to one billion steps.

## Question

How does changing frozen gamma affect plain GD and Adam on a target with a square-root endpoint singularity?

## Experiment design

The target is $f(x)=\sqrt{(x+1)/2}$ on $[-1,1]$. This keeps the geometry and gamma interpretation consistent with the preceding mixed-sine diagnostics. There are 263 equally spaced training samples, including both endpoints. Centers have spacing $h=2/128$ with 129 interior centers and 12 halo centers on each side: 153 tanh features, plus a bias. Gamma values are $1,4,8,16,32,64$, corresponding to $\gamma h=0.015625,0.0625,0.125,0.25,0.5,1$. Geometry is fixed, readout coefficients and bias start at zero, and the readout coordinates are unscaled.

The objective is $L(v)=\frac1{2m}\|\Phi v-y\|^2$, with $\Phi=[\mathbf1,\tanh(\gamma(x_i-c_j))]$. All computation is FP64. The metric is $E(n)=\|\Phi v_n-y\|/\|y\|$ on the training samples, not a held-out-grid error.

For GD, form $J=\Phi/\sqrt m$ and compute $J=U\Sigma V^\top$. Thus the actual kernel $K=JJ^\top$ has positive eigenvalues $\sigma_i^2$. With $\eta=0.5/\sigma_1^2$, evaluate $E(n)^2=\sum_i p_i(1-\eta\sigma_i^2)^{2n}+p_0$, where $p_i=|u_i^\top y|^2/\|y\|^2$ and $p_0$ is the squared target fraction orthogonal to the thin $U$. Powers use `log1p` for stability. The recorded GD curves are spectral evaluations, not executed training loops. First-hit counts are obtained by integer bisection.

Adam uses the identical feature and target arrays, learning rate $10^{-3}$, betas $(0.9,0.999)$, epsilon $10^{-8}$, no weight decay, and no schedule. A single batched parameter tensor holds six independent readouts; the analytic gradient $J^\top(Jv-y/\sqrt m)$ is passed to ordinary PyTorch Adam. Every step's error is saved, with no smoothing or best-so-far filtering. Final parameters and optimizer state are retained for continuation.

An idle H200 was reachable through the laptop, but automatic approval review rejected transferring the research payload to that external server without more explicit authorization. No payload was uploaded. The batched CPU run completed in about one second, so no GPU training was needed.

**Code and data.** Source: `experiments/expD37_capacity_access_figures/gamma_solve/sqrt_readout/run.py`. All inputs, configuration, trajectories, summaries, checks, and Adam continuation state are in this folder's `data/`. Figures are linked below. No other experiment outputs were modified.

## Results

| Gamma | GD first step at or below 1% | Adam first step at or below 1% | GD error at step 20,000 | Adam error at step 20,000 |
|---:|---:|---:|---:|---:|
| 1 | 40,644,819 | Not within 20,000 | 1.7848% | 1.3448% |
| 4 | 7,702 | 1,985 | 0.8067% | 0.5269% |
| 8 | 2,302 | 885 | 0.5469% | 0.3538% |
| 16 | 1,040 | 531 | 0.4181% | 0.2432% |
| 32 | 701 | 409 | 0.3311% | 0.1590% |
| 64 | 592 | 366 | 0.2673% | 0.0889% |

### Figures

- [GD versus Adam through 20,000 steps](gd_vs_adam_20k.png): left, actual-kernel spectral GD; right, executed Adam. Colors identify gamma. Both axes ranges are shared; the step axis is linear up to 10 and logarithmic afterward, and the error axis is logarithmic. The dashed horizontal line is 1%.
- [Extended spectral GD curves](gd_spectral_long_range.png): the same actual-kernel calculation through one billion steps. The vertical dotted line marks the Adam run's 20,000-step budget. No extra training was executed for this figure.

## Verification and limits

A 200-step direct GD check matches the spectral residual vector to about $1.1\times10^{-15}$ relative to target norm. Thirty batched Adam steps using the analytic gradient match ordinary autograd exactly in the CPU check. Recomputing final Adam errors from saved coefficients and the original unnormalized feature matrix agrees to floating-point accuracy; exact discrepancy is saved in `data/verification.json`. The problem archive's SHA-256 is recorded with the Adam output and checked locally.

The GD calculation describes exact-arithmetic iteration on the supplied finite matrix, evaluated numerically in FP64; very long physical training runs may accumulate roundoff differently. Adam uses its standard fixed learning rate, whereas GD uses curvature normalization, so this is not a matched-learning-rate optimizer comparison. The target is not smooth at the left endpoint. The experiment tests one width, one sample grid, one target, and a fixed raw readout parameterization.

## Conclusions

For this target and configuration, increasing gamma reduces the computed GD time to 1% across the tested values. Adam also reaches 1% earlier at the larger tested gammas; gamma 1 remains above the threshold at step 20,000. These results concern frozen-readout optimization and do not measure joint geometry learning.

## Open questions

No extra sweep was performed. Sensitivity to Adam's learning rate and evaluation on a held-out grid remain separate questions.
