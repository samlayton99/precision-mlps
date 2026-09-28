# Four target bandwidth figures

Status: measured; revised 2026-09-21.

## TL;DR

- Each left panel measures 97 total widths: $64,66,\ldots,256$, at $\lambda=0.25$ and $p=53$.
- Each middle panel contains only $W=96,128,192,256$. The precision panel remains at $W=256$ with $p=24,32,40,53$.
- Regular sine is now $\sin(4\pi x)$, with its observations and refined predictions recomputed. Current figure exports are PNG only.

## Question

How do width, bandwidth, and precision affect these four targets, and where do the basic and refined bandwidth rules choose to operate?

## Experiment design

All targets are fitted and scored on $[-1,1]$:

| Target | Function |
|---|---|
| Runge | $1/(1+25x^2)$ |
| Regular sine | $\sin(4\pi x)$ |
| Mixed sine | $m(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x)$ |
| Gaussian envelope | $e^{-x^2/(2\cdot0.4^2)}m(x)$ |

The network is $q(x)=b+\sum_{j=-H}^{N+H}a_j\tanh(\gamma(x-c_j))$, with 24 halo nodes per side, $c_j=-1+jh$, $h=2/N$, and $\gamma=\lambda/h$. Total hidden width includes halo nodes and excludes the output bias: $W=N+49$. Widths for geometric refinement and bandwidth comparison are configured separately.

Readouts are solved on 4,801 equally spaced points with SVD cutoff $2^{1-p}$. Relative sampled $L^2$ error is measured on 8,001 evaluation points. Bandwidth sweeps use 85 logarithmically spaced values over $[0.05,1.5]$, supplemented with direct solves at the predictions and control bandwidths. No smoothing is applied. The dashed geometric guides fit observations with errors between $10^{-12}$ and $0.5$; they are empirical fits, not theoretical bounds.

For $p<53$, feature values, targets, coefficients, products, and sequential accumulated sums are rounded to $p$ significand bits. Feature arguments and SVD internals remain FP64. These are the earlier bandwidth experiment's precision-model conventions, not native low-precision solves.

The September 14 selector uses effective tolerance $2^{1-p}$. Refined frequency summaries are the Fourier-amplitude-weighted mean absolute angular frequencies: $5$, $4\pi$, $34\pi/7$, and approximately $15.364$ for Runge, regular sine, mixed sine, and the envelope. Prediction markers have directly measured error coordinates. The shaded strip spans refined choices across the four middle-panel widths; it is not an uncertainty interval.

**Code & data**

- Runner, target definitions, configuration, and validation: `experiments/expC09_bandwidth_figures/{run.py,targets.py,config.yaml,validate.py}`.
- Regression checks: `tests/test_bandwidth_figures.py`.
- Selector: `docs/lambda_theorem_compatibility/choosing_optimal_lambda/choose_lambda.py`.
- Measurements, prediction/source hashes, summaries, validation, and reuse provenance: `data/` beside this writeup.
- Main PNGs: `runge/figures/runge_bandwidth.png`, `mixed_sine/figures/mixed_sine_bandwidth.png`, `sine/figures/sine_bandwidth.png`, and `gaussian_envelope/figures/gaussian_envelope_bandwidth.png`.
- Each target also has a `figures/halo_check.png`.
- Earlier outputs and the previous experiment code/configuration are preserved under `archive/five_widths_sine2pi/`. The initial Runge-only work remains under `archive/initial_runge/`.

Reproduce from the repository root:

```sh
.venv/bin/python experiments/expC09_bandwidth_figures/run.py
.venv/bin/python experiments/expC09_bandwidth_figures/validate.py
.venv/bin/python -m pytest -q tests/test_bandwidth_figures.py
```

Use `--plot-only` to redraw saved observations.

## Results

The dense refinement resolves the descent and small numerical fluctuations that the five-width plot missed. At $W=64$, the revised sine has relative error about $7.6\times10^{-9}$; by $W=76$ it is below $10^{-12}$. The other targets retain their decline from larger initial errors before reaching numerical floors of roughly $10^{-14}$ to $10^{-13}$.

### Figures

- **Runge:** dense width refinement, four bandwidth curves, and four precision curves.
- **Mixed sine:** the same panels show the effect of resolving the high-frequency component.
- **Regular sine:** all panels now use $\sin(4\pi x)$, including its new refined frequency scale and precision measurements.
- **Gaussian envelope:** the mixture is localized with $\sigma=0.4$; errors concern the finite interval, not the earlier whole-line training objective.
- **Halo controls:** four panels per target preserve the interior spacing while extending each halo from 24 to 48 and 96 nodes. Total widths increase in these controls.

## Validation and limits

All 3,972 configured measurements are present. This revision reuses 2,703 unchanged observations and adds 1,269 measurements. Before reuse, the measurement function was checked unchanged, retained target values were checked identical, and sampling counts were checked against the new configuration. Every regular-sine observation was recomputed.

All 18 regression tests pass, including the separate dense-refinement and middle-panel width selections, exclusion of $W=64$ from bandwidth sweeps, and the revised sine values/frequency summary. Current output folders contain PNGs only. All four main figures were visually inspected.

The 952 paired halo controls retain sensitivity in some cells: the largest FP64 improvement above $10^{-12}$ is about a factor 15, and the largest improvement at a rule choice across precisions is about a factor 13. Thus 24 halo nodes per side is not certified sufficient everywhere. These controls include both boundary and numerical-conditioning effects.

Seventy-two sampling controls double evaluation density and then fitting density. Evaluation-only changes are at most 3.8%; refitting can move precision floors by factors of a few. The figures support broad trends, not precise rankings of numerical minima or native low-precision performance.

## Conclusions

The denser width sweeps show the transition from approximation error to numerical recovery floors. Removing $W=64$ from the bandwidth panel preserves the requested four-width comparison, while the higher-frequency regular sine now has a visible initial refinement regime.

## Open questions

Halo sensitivity and native intermediate-precision arithmetic remain outside the claims of these plots.
