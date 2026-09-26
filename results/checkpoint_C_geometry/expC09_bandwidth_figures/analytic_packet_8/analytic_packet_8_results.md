# Milder oscillatory Gaussian — measured

## TL;DR

- Both constants are reduced from 80 to 8: $f(x)=e^{-8(x-0.2)^2}\sin(8\pi x)$.
- All 41 widths were remeasured; the other curves and panels retain their previous data.

## Question

How does the less oscillatory, broader analytic target converge under the existing settings?

## Experiment design

The same FP64 fixed-grid tanh least-squares construction uses $\lambda=0.25$, widths 64 through 2048, and 24 halo nodes per side. With $W=N+49$, spacing is $h=2/N$ and slope is $\gamma=\lambda/h$. Training and evaluation use $\max(4801,4W+1)$ and $\max(8001,8W+1)$ points respectively, with SVD cutoff $2^{-52}$. The metric is $\|\hat f-f\|_2/\|f\|_2$ on the evaluation grid.

**Code & data**

- Measurement and plotting: `experiments/expC09_bandwidth_figures/analytic_packet.py` and `combined.py`.
- Config, coefficients, and measurements: `data/` beside this report.
- Figure: `../combined/figures/all_targets_chirp_bandwidth.png`.
- Checks: `tests/test_analytic_packet.py`.

## Results

Relative error drops from about 0.156 at width 64 to below $10^{-12}$ at width 112, then settles around $10^{-13}$ to $10^{-14}$.

### Figures

- **Combined PNG:** the gray Oscillatory Gaussian curve in panel (a) uses the new target; panels (b) and (c) retain the chirp sweeps and approved layout.

## Additional details

Both regression checks pass: agreement with the existing convergence solver on a shared target, and the analytic whole-line Gaussian energy integral. Four doubled, shifted evaluation grids were checked; their largest relative error change is 1.88%. Halo and denser-fitting controls were not rerun for this target.

## Conclusions

Reducing both constants produces an analytic target that reaches small error at substantially lower widths.

## Open questions

No further changes are proposed.
