# Smooth absolute value — measured

## TL;DR

- Replaces Gaussian envelope in panel (a) with $f(x)=\log(\cosh(10x))/10$.
- The measured error falls below $10^{-12}$ by the sampled width 144, reaching a numerical floor around $10^{-14}$ to $10^{-13}$.

## Question

How does a compactly expressed, real-analytic target with a rounded corner converge under the approved settings?

## Experiment design

The same FP64 tanh least-squares construction uses $\lambda=0.25$ and 24 halo nodes per side. Total width is $W=N+49$, spacing is $h=2/N$, and slope is $\gamma=\lambda/h$. Independent SVD solves use cutoff $2^{-52}$ on $\max(4801,4W+1)$ training points; relative sampled $L^2$ error is $\|\hat f-f\|_2/\|f\|_2$ on $\max(8001,8W+1)$ evaluation points. The 41-width source sweep spans 64 through 2048; the approved figure displays only 64 through 1024. Other curves and both chirp panels retain their saved measurements.

**Code & data**

- Runner: `experiments/expC09_bandwidth_figures/smooth_absolute.py`; renderer: `combined.py` in the same directory.
- Measurements, coefficients, source hash, and configuration: `data/` beside this report.
- PNG: `../combined/figures/all_targets_chirp_bandwidth.png`.
- Regression checks: `tests/test_smooth_absolute.py`.

## Results

At width 64 the relative error is approximately $1.9\times10^{-4}$, falling to approximately $1.3\times10^{-12}$ at width 128. The curve reaches the numerical floor by about width 160.

### Figures

- **Combined three-panel PNG:** the brown Smooth absolute value curve replaces Gaussian envelope in panel (a), preserving the other curves, colors, and formatting.

## Additional details

Two checks pass: the target matches a 60-decimal-digit evaluation of its formula, and the solver agrees with the existing convergence implementation on a shared sine target. Four doubled, shifted evaluation grids were checked; their largest relative error change is 11.88%. Halo-size and denser-fitting controls were not rerun for this target.

## Conclusions

The function produces a visible convergence regime without the delayed resolution of the discarded high-frequency Gaussian.

## Open questions

No further changes are proposed.
