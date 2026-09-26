# Chirp, Runge 100, and smooth compact bump — measured

## TL;DR

- Three separate PNGs use the same width, bandwidth, halo, and precision settings as the previous four-target suite.
- The bump's precision curves nearly coincide at width 256 across a broad bandwidth range; chirp and Runge 100 retain distinct precision-dependent curves.
- The chirp's refined marker uses mean local frequency as a heuristic, not a Fourier-moment certificate.

## Question

How do the three-panel bandwidth comparisons change for the chirp, sharper Runge function, and compact bump from the eight-target convergence plot?

## Experiment design

The targets on $[-1,1]$ are $\sin(8\pi(x+1)^2)$, $1/(1+100x^2)$, and $\exp[-1/(1-4x^2)]$ inside $|x|<1/2$, zero outside. They exactly match the earlier convergence comparison.

We fit $b+\sum_j a_j\tanh(\gamma(x-c_j))$ with $c_j=-1+jh$, $h=2/N$, and $\gamma=\lambda/h$. The main runs use 24 halo centers per side, so total hidden width is $W=N+49$, excluding the output bias. Readouts use independent SVD least-squares solves on 4,801 training points; relative sampled $L^2$ error is $\|\hat f-f\|_2/\|f\|_2$ on 8,001 evaluation points.

Left panels use all even widths from 64 through 256 at $\lambda=0.25$, FP64. Middle panels sweep $\lambda\in[0.05,1.5]$ at widths 96, 128, 192, and 256. Right panels fix width 256 and compare $p=24,32,40,53$. The sweeps include 85 logarithmically spaced bandwidths plus prediction and halo-control points. Dashed geometric trends are empirical log-linear fits over errors between $10^{-12}$ and $0.5$, not theoretical rates; in particular the compact bump is not analytic.

For $p<53$, features, labels, readouts, products, and accumulated sums are rounded to the specified significand length. Feature arguments and SVD internals remain FP64. These are simulated rounding experiments, not native low-precision solves.

Basic and refined prediction markers use the existing selector with tolerance $2^{1-p}$. The Runge frequency summary is 10. The bump's Fourier-amplitude-weighted mean frequency is approximately 5.63186, computed by zero-padded Fourier quadrature through angular frequency 1000; doubling padding from length 64 to 128 changes the estimate by less than $2\times10^{-5}$ relatively. The chirp uses the spatial mean of its absolute phase derivative, $16\pi$, explicitly labeled as a heuristic. The shaded strip spans refined predictions across widths and is not an uncertainty interval.

**Code & data**

- Runner and configuration: `experiments/expC09_bandwidth_figures/run.py`, `additional_config.yaml`; extra target definitions: `additional_targets.py`.
- Run with `.venv/bin/python experiments/expC09_bandwidth_figures/run.py --config experiments/expC09_bandwidth_figures/additional_config.yaml --output results/checkpoint_C_geometry/expC09_bandwidth_figures/additional_targets`; append `--plot-only` to redraw.
- Validate with `.venv/bin/python experiments/expC09_bandwidth_figures/validate.py --output results/checkpoint_C_geometry/expC09_bandwidth_figures/additional_targets`.
- Measurements and manifests: `data/measurements.jsonl`, `data/predictions.json`, `data/summary.json`, `data/validation.json` beside this report.
- PNGs: `chirp/figures/chirp_bandwidth.png`, `runge100/figures/runge100_bandwidth.png`, `smooth_bump/figures/smooth_bump_bandwidth.png`; each directory also contains `halo_check.png`.
- Regression checks: `tests/test_bandwidth_additional_targets.py`, plus the existing bandwidth and convergence tests.

## Results

At width 256 and bandwidth 0.25, the chirp and Runge 100 reach relative errors around $10^{-12}$ and $10^{-13}$ respectively. The compact bump remains substantially less accurate over this width range. Its broad overlap of precision curves is consistent with approximation error dominating the rounding effects in this configuration.

### Figures

- **Chirp:** width refinement, bandwidth curves, and precision curves show the difficult low-width regime and a pronounced bandwidth optimum at larger widths. The mean-frequency prediction does not capture the full range of local frequencies.
- **Runge 100:** the same three panels show width refinement down toward FP64 precision and a broad separation of precision curves.
- **Smooth compact bump:** refinement is slower; the middle-panel plateaus fall with width, while most precision curves overlap at width 256.
- **Halo controls:** four panels per target compare 24, 48, and 96 nodes per side at fixed interior spacing; total width increases for these controls.

## Additional details

All 2,979 configured measurements are present. All 24 tests pass, including exact agreement of the added target values with the eight-target comparison, stable bump frequency quadrature, width counting, precision rounding, and regression against the earlier bandwidth implementation.

There are 714 paired halo comparisons and 54 doubled-grid controls. Doubling evaluation density changes relative errors by at most about 5%; doubling fitting and evaluation density can move individual errors by factors of a few. Larger halos improve some FP64 errors above $10^{-12}$ by up to 4.42 times and errors at prediction choices by up to 2.56 times across precisions. Thus the main halo choice is retained for comparability, not certified sufficient in every cell. All three main PNG layouts were visually inspected; no PDFs were exported.

## Conclusions

These matched settings produce clearly different bandwidth and precision profiles for the three functions. The compact bump's curves remain well above FP64 accuracy at these widths.

## Open questions

How do the bump's precision curves separate at the larger widths needed to resolve it? How should a prediction represent the chirp's range of local frequencies rather than a single mean?
