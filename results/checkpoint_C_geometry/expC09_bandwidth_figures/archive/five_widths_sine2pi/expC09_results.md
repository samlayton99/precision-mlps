# Four target bandwidth figures

Status: measured, 2026-09-21. These replace the initial Runge-only figure and its width choices.

## TL;DR

- Four separate three-panel figures use exactly $W=64,96,128,192,256$ for both width panels. The precision panel fixes $W=256$ and uses $p=24,32,40,53$.
- All widths count the halo neurons. With 24 halo nodes per side, $W=N+49$, so the interior interval counts are $N=15,47,79,143,207$.
- The sine target is already close to the FP64 recovery floor throughout the requested refinement range. The other three targets show a strong decline followed by a floor.

## Question

How do the requested widths and precisions affect approximation of Runge, mixed sine, regular sine, and a Gaussian-enveloped sine mixture, and where do the recent bandwidth rules choose to operate?

## Experiment design

Every target is fitted and scored on $[-1,1]$. The definitions match the existing four-target residual-spectrum comparison:

| Target | Function |
|---|---|
| Runge | $1/(1+25x^2)$ |
| Regular sine | $\sin(2\pi x)$ |
| Mixed sine | $m(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x)$ |
| Gaussian envelope | $e^{-x^2/(2\cdot0.4^2)}m(x)$ |

The network is $q(x)=b+\sum_{j=-H}^{N+H}a_j\tanh(\gamma(x-c_j))$, with $H=24$, $c_j=-1+jh$, $h=2/N$, and $\gamma=\lambda/h$. Width includes all hidden neurons and excludes the output bias. Readouts are solved on 4,801 equally spaced points with SVD cutoff $2^{1-p}$. Relative sampled $L^2$ error is measured on 8,001 points. The bandwidth grid has 85 logarithmically spaced values over $[0.05,1.5]$, supplemented with direct solves at the predictions and control bandwidths.

Panel (a) fixes $\lambda=0.25$ and $p=53$. Panel (b) sweeps bandwidth at $p=53$. Panel (c) fixes $W=256$ and varies precision. Below 53 bits, feature values, targets, coefficients, products, and sequential accumulated sums are rounded to $p$ binary significand bits. SVD internals and feature arguments remain FP64, as in the earlier bandwidth experiment. These are precision-model results, not native low-precision solves. No smoothing is applied.

Predictions use the September 14 revised basic/refined selector with effective tolerance $2^{1-p}$. The refined frequency summaries are the Fourier-amplitude-weighted mean absolute angular frequencies: $5$, $2\pi$, $34\pi/7$, and approximately $15.364$ for Runge, sine, mixed sine, and the envelope respectively. For the envelope, the exact summary is $\sum_j a_jk_j\pi/\sum_j a_j\operatorname{erf}(0.4k_j\pi/\sqrt2)$. Its implementation is checked against independent spectral quadrature. Prediction markers have directly measured error coordinates; the blue strip spans the refined choices across widths, not an uncertainty interval.

**Code & data**

- Runner and configuration: `experiments/expC09_bandwidth_figures/run.py`, `config.yaml`.
- Target definitions and frequency summaries: `experiments/expC09_bandwidth_figures/targets.py`.
- Numerical controls: `experiments/expC09_bandwidth_figures/validate.py`.
- Regression checks: `tests/test_bandwidth_figures.py`.
- Selector: `docs/lambda_theorem_compatibility/choosing_optimal_lambda/choose_lambda.py`.
- Raw observations, predictions, configuration, source hashes, summaries, and controls: `data/` beside this writeup.
- Figures: `runge/figures/runge_bandwidth`, `mixed_sine/figures/mixed_sine_bandwidth`, `sine/figures/sine_bandwidth`, and `gaussian_envelope/figures/gaussian_envelope_bandwidth`, each in PNG, PDF, and SVG. Each folder also contains `halo_check.png`.
- Superseded initial Runge work is preserved under `archive/initial_runge/`.

Reproduce from the repository root:

```sh
.venv/bin/python experiments/expC09_bandwidth_figures/run.py
.venv/bin/python experiments/expC09_bandwidth_figures/validate.py
.venv/bin/python -m pytest -q tests/test_bandwidth_figures.py
```

Use `--plot-only` on the runner to redraw saved observations.

## Results

At fixed $\lambda=0.25$, Runge declines from about $2\times10^{-3}$ at $W=64$ to $7\times10^{-12}$ at $W=128$, then reaches the FP64 floor. Mixed sine and the envelope start near $0.2$ and reach approximately $3\times10^{-12}$ by $W=128$. Their larger widths approach $10^{-13}$. Regular sine stays between roughly $10^{-14}$ and $10^{-13}$ over the requested widths; there are no resolved pre-floor observations from which to infer a geometric rate.

### Figures

- **Runge:** the three requested panels show refinement, the five total-width bandwidth curves, and four precision curves.
- **Mixed sine:** the same panels expose the poorly resolved $W=64$ case and its improvement with width.
- **Regular sine:** the left panel shows the already-small numerical error; it deliberately omits a geometric fit. The bandwidth and precision panels remain informative.
- **Gaussian envelope:** the same sine mixture is spatially localized by the Gaussian; all errors concern $[-1,1]$, not the whole-line objective used in the earlier training experiment.
- **Halo checks:** five panels per target preserve each interior grid while increasing halo counts to 48 and 96 per side. These controls necessarily increase total width.

## Validation and limits

All 4,116 measurements are present. Seventeen regression tests pass, covering total-width accounting at every requested width, exact preservation of interior centers when extending the halo, precision rounding, scalar prediction thresholds, agreement with the earlier solver, all four target definitions, the envelope's spectral summary, and the configured widths/precisions.

The 1,088 halo comparisons expose residual sensitivity; 24 nodes per side must not be described as a proved sufficient halo everywhere. The largest FP64 improvement above $10^{-12}$ is from $1.55\times10^{-12}$ to $3.30\times10^{-14}$ for sine at $W=128$, $\lambda=0.05$. Runge at $W=64$ near $\lambda=0.215$ changes by almost a factor ten, and mixed sine at $W=96$, $\lambda\approx0.168$ changes by about a factor twelve. The controls retain both finite-domain and numerical-conditioning effects; they do not isolate a pure boundary-error term. Predictions are practical choices, not guarantees of an error minimum or halo independence.

Seventy-six sampling controls double evaluation density, then fitting density. Evaluation-only changes are at most 5.3%, with the largest relative differences near the FP64 floor. Refitting on the denser grid moves some precision floors by factors of a few. The figures support their large-scale trends, not precise rankings of numerical minima. The geometric dashed guides fit the three pre-floor widths for Runge, mixed sine, and the envelope; they are empirical fits, not theoretical error bounds.

## Conclusions

At the requested total widths, Runge, mixed sine, and the envelope show a strong refinement trend before numerical recovery limits accuracy. Regular sine is already near that limit. The four precision curves remain distinct, with $p=48$ excluded as requested.

## Open questions

Halo sensitivity at selected small widths and small bandwidths remains visible. Native intermediate-precision arithmetic is outside this figure's scope.
