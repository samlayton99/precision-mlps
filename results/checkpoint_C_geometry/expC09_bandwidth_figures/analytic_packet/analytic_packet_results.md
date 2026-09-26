# Analytic replacement for the compact bump — measured

Superseded by the milder variant with both constants reduced from 80 to 8. This report describes the preserved 80-variant measurements.

## TL;DR

- The combined figure replaces the compact bump with $f(x)=e^{-80(x-0.2)^2}\sin(80\pi x)$, labeled “Oscillatory Gaussian.” This product is entire.
- Its measured error stays near one through width 192 and reaches approximately $10^{-12}$ at the largest widths.

## Question

Can an analytic target provide a difficult eighth curve in the existing convergence comparison?

## Experiment design

The target has a narrow Gaussian envelope and carrier angular frequency $80\pi$. All 41 even widths from the existing sweep, spanning 64 to 2048, use $\lambda=0.25$, FP64, and 24 halo nodes per side. Total width includes halo neurons: $W=N+49$, with $h=2/N$, centers $c_j=-1+jh$, and tanh slope $\gamma=\lambda/h$.

The network is $b+\sum_j a_j\tanh(\gamma(x-c_j))$. Independent SVD readout solves use cutoff $2^{-52}$ and $\max(4801,4W+1)$ training points. Relative sampled $L^2$ error is $\|\hat f-f\|_2/\|f\|_2$, evaluated at $\max(8001,8W+1)$ points in chunks of 1024. These numerical conventions match the existing convergence comparison. Other curves and panels retain their saved observations.

**Code & data**

- Measure: `experiments/expC09_bandwidth_figures/analytic_packet.py`; render: `experiments/expC09_bandwidth_figures/combined.py`.
- Saved observations, coefficients, and hashed configuration: `data/` beside this writeup.
- Archived figure: `figures/all_targets_chirp_bandwidth_80.png`.
- Checks: `tests/test_analytic_packet.py`.

## Results

At widths 256, 512, and 2048, the relative errors are approximately $4.2\times10^{-4}$, $4.7\times10^{-11}$, and $1.0\times10^{-12}$ respectively. The target requires more width than the other analytic examples to achieve a comparable error.

### Figures

- **Combined three-panel PNG:** the gray curve in panel (a) is the new analytic target; the other seven curves and both chirp panels retain the previously approved data and layout.

## Additional details

Two regression checks pass: matching the existing convergence solver exactly on a shared sine target, and matching the new target's sampled squared norm to its analytic Gaussian integral. A doubled, shifted evaluation grid at widths 256, 512, 1024, and 2048 changes relative errors by at most 1.03%. Halo size and fitting-density sensitivity were not independently rechecked for this replacement; the plotted error at large widths is an observed numerical result, not a certified optimum.

## Conclusions

This analytic target supplies a delayed convergence curve while preserving the comparison's numerical settings.

## Open questions

Whether larger halos or denser fitting grids materially change its large-width error remains untested.
