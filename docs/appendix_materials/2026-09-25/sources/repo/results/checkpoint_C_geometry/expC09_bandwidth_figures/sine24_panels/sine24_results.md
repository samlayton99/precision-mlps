# Bandwidth and precision for sin(24 pi x) — measured

## TL;DR

- A separate two-panel PNG shows $\sin(24\pi x)$ with the existing width, precision, halo, and bandwidth settings.
- Width 96 remains poorly resolved; wider networks produce pronounced bandwidth minima. The width-128 prediction is substantially left of the observed minimum.

## Question

What do panels (b) and (c) look like for the higher-frequency sine instead of the chirp?

## Experiment design

Fit $f(x)=\sin(24\pi x)$ on $[-1,1]$ with $b+\sum_j a_j\tanh(\gamma(x-c_j))$, using $h=2/N$, $c_j=-1+jh$, $\gamma=\lambda/h$, 24 halo centers per side, and $W=N+49$. Panel (b) uses FP64 at widths 96, 128, 192, and 256. Panel (c) fixes width 256 and varies significand bits $p=24,32,40,53$. Each curve has 85 logarithmically spaced bandwidths from 0.05 to 1.5, supplemented by its valid refined prediction.

The shared numerical routine uses 4,801 training points, SVD cutoff $2^{1-p}$, and 8,001 evaluation points. Errors are sampled relative $L^2$ norms. Lower precisions round features, labels, coefficients, products, and accumulated sums; SVD internals remain FP64.

The refined selector uses the exact single-mode angular frequency $24\pi$. At width 96, $h\omega=48\pi/47>\pi$, outside the selector's valid range; this curve has no prediction diamond. Predictions use the same effective tolerance $2^{1-p}$ as before.

**Code & data**

- Runner: `experiments/expC09_bandwidth_figures/sine24_panels.py`; numerical routine: `run.py` in the same directory.
- Saved configuration, predictions, measurements, and control checks: `data/` beside this writeup.
- PNG: `figures/sine24_bandwidth_precision.png`.
- Shared regression checks: `tests/test_bandwidth_figures.py`.

## Results

Measured FP64 minima improve from about 0.53 at width 96 to $3.5\times10^{-9}$ at width 128 and approximately $10^{-12}$ at widths 192 and 256. At width 128, the predicted bandwidth is about 0.107, while the sampled minimum occurs near 0.190. At width 256, the precision curves separate clearly, with lower precision favoring larger bandwidths.

### Figures

- **Two-panel sine PNG:** bandwidth versus relative error, with width varied on the left and precision varied on the right. Diamonds mark refined predictions. The existing combined chirp figure remains separate.

## Additional details

The width-128 prediction solves an alias-only budget: at $\lambda=0.10745$ the alias score is $2^{-52}$, but measured error is $3.87\times10^{-3}$ and readout coefficient norm is $1.53\times10^{11}$. Near the sampled optimum $\lambda=0.19023$, the alias score is $1.44\times10^{-9}$, measured error is $3.46\times10^{-9}$, and coefficient norm is $1.46\times10^6$. The desired normalized kernel response rises from $4.24\times10^{-11}$ to $4.50\times10^{-6}$ between these bandwidths. This shows why an alias-only budget is insufficient to predict the finite-precision error minimum. A separate diagnostic PNG (`figures/sine24_prediction_diagnosis.png`) compares measured error with alias score and plots coefficient norms over the same sweep.

The single sine supplies an exact frequency, so the discrepancy here does not arise from choosing a representative frequency for a mixture. At the width-128 prediction, doubling halos changes error from 0.00387 to 0.00312, and doubling both fitting and evaluation density changes it to 0.00365; neither control resolves the mismatch. The plotted finite-halo least-squares model also differs from the boundary-corrected construction underlying the alias bound. Finite-width approximation, truncation, and numerical recovery must be considered in addition to aliasing.

All 601 configured cells are present. All 18 shared numerical regression tests pass, and explicit checks confirm exact target agreement with the earlier convergence comparison and valid prediction status for every marked curve. Seven representative configurations were checked with doubled evaluation density, doubled fitting/evaluation density, and twice as many halo centers at fixed interior spacing. Evaluation-only differences are at most 4.64%; refitting changes individual errors by up to a factor of 2.76. These checks support the broad curves, not precise comparisons of numerical minima.

## Conclusions

The high-frequency sine produces distinct width and precision curves; the refined prediction is not uniformly close to the measured optimum, particularly at width 128.

## Open questions

### Follow-up: relaxing the alias budget at width 128

Four direct measurements retain FP64 arithmetic while changing only the selector and its tolerance:

| Rule | Alias budget | Selected $\lambda$ | Measured relative error |
|---|---:|---:|---:|
| Basic | $2^{-52}$ | 0.24408 | $1.25\times10^{-7}$ |
| Basic | $10^{-8}$ | 0.44429 | $1.58\times10^{-4}$ |
| Refined | $2^{-52}$ | 0.10745 | $3.87\times10^{-3}$ |
| Refined | $10^{-8}$ | 0.21025 | $1.10\times10^{-8}$ |

The best point in the original sampled sweep is $\lambda=0.19023$, with error $3.46\times10^{-9}$. Relaxing the refined budget to $10^{-8}$ yields an error close to that budget, within a factor of 3.2 of the sampled minimum. Relaxing the basic budget moves too far right for this high-frequency target. The comparison is saved in `data/budget_comparison_W128.json` and plotted against the measured curve in `figures/budget_comparison_W128.png`. These observations do not establish a general width-dependent choice of tolerance.

No change to the approved combined figure has been made.
