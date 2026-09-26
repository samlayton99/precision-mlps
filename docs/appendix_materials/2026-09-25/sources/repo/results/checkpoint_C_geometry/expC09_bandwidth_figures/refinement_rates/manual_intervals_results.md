# User-selected semilog intervals — measured fits

Fit $\log_{10}E=a+mW$ by ordinary least squares using every original measured point in each requested interval. Bounds snap to the nearest existing width, with ties resolved downward. No automatic trimming or floor exclusion is applied. Supplemental endpoint probes made before the instruction to use nearest neighbors are excluded from these fits.

| Target | Width interval used | Slope (decades/neuron) | $R^2$ | Points |
|---|---|---:|---:|---:|
| sine4 | 50–70 | -0.538176 | 0.99477 | 11 |
| sine24 | 100–140 | -0.175421 | 0.98114 | 21 |
| weak_high_frequency | 124–170 | -0.137597 | 0.99046 | 24 |
| chirp | 100–160 | -0.137326 | 0.98201 | 31 |
| runge25 | 50–150 | -0.130713 | 0.98900 | 51 |
| runge100 | 50–224 | -0.071125 | 0.99056 | 88 |
| smooth_absolute | 50–140 | -0.122739 | 0.99257 | 46 |
| analytic_packet | 64–90 | -0.335912 | 0.99176 | 14 |

The shaded bands show the actual fitted widths, and dashed lines show the fitted exponential trends. Wider selected ranges contain visible curvature and intermediate dips, reflected in the displayed fit quality. These are descriptive local fits.

- PNG: `figures/refinement_rates_semilog_manual.png`.
- Fit metadata and table: `data/manual_fits.json`, `data/manual_rates.csv`.
- Requested intervals: `experiments/expC09_bandwidth_figures/manual_rate_intervals.json`.
- Reproduce with `experiments/expC09_bandwidth_figures/refinement_rates.py --manual-intervals experiments/expC09_bandwidth_figures/manual_rate_intervals.json`.
