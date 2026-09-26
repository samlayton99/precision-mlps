# Local refinement rates — measured, descriptive fits

## TL;DR

- Two 2×4 PNGs show all eight current targets, one using log–log coordinates and one using log(error) versus width. Every fit uses a contiguous interval highlighted by a vertical band.
- All sixteen selected fits have at least eight points, span at least 2.5 decades, and have $R^2\geq0.995$. These are selection requirements, not independent validation.
- The two views select intervals separately. Same-point comparisons and endpoint sensitivity are saved alongside the slopes.

## Question

Which well-supported, steep middle segments of the refinement curves are approximately linear, and what local slopes do they have?

## Experiment design

The targets and numerical conventions match the approved panel (a): fixed $\lambda=0.25$, FP64, 24 halo centers per side, and total hidden width $W=N+49$. The original independent SVD solver, precision cutoff, fitting grid, and chunked evaluation are retained. Existing observations are reused where available. Dense measurements cover every even width from 50 through 384, supplemented by the original coarser points through 1024: 180 widths and 1,440 target-width errors. Width 50 is the smallest admissible even width with these halos; smaller diagnostic widths are needed to inspect the rapid sine decay. The approved combined plot is unchanged.

For each target, estimate the numerical floor from the median log error at widths at least 512. Candidate fits must stay at least one decade above that estimate. Enumerate contiguous windows with at least six points and 2.5 fitted decades of decay. Require $R^2\geq0.995$, RMS residual at most 0.15 decades, and maximum residual at most 0.35 decades. Retain windows whose fitted slope magnitude is at least 85% of the steepest qualified window, then select the one spanning the largest fitted error drop. This balances a steep slope against sufficient support; no interior observations are discarded.

The log–log fit is $\log_{10}E=a+m\log_{10}W$, so $E\approx10^a W^m$ locally. The semilog fit is $\log_{10}E=a+mW$, so $E\approx10^a10^{mW}$ locally. A semilog slope is measured in decades per additional neuron. The shaded limits are the selected fit interval, not claims of uniquely identified physical transition points.

**Code & data**

- Runner and selector: `experiments/expC09_bandwidth_figures/refinement_rates.py`; use `--plot-only` to refit and redraw saved observations.
- Measurements and provenance: `data/measurements.jsonl` and `data/config.json`.
- Selection settings and fitted results: `data/fit_method.json`, `data/fits.json`, and `data/rates.csv`.
- PNGs: `figures/refinement_rates_loglog.png` and `figures/refinement_rates_semilog.png`.
- Synthetic selector checks: `tests/test_refinement_rates.py`.

## Results

| Target | Log–log slope | Width interval | Semilog slope | Width interval |
|---|---:|---|---:|---|
| sine4 | -73.9 | 50–70 | -0.526 | 54–68 |
| sine24 | -65.5 | 96–110 | -0.277 | 96–110 |
| weak_high_frequency | -46.5 | 126–174 | -0.155 | 126–156 |
| chirp | -32.6 | 128–174 | -0.101 | 132–160 |
| runge25 | -33.7 | 84–104 | -0.151 | 72–108 |
| runge100 | -25.4 | 118–160 | -0.0773 | 80–168 |
| smooth_absolute | -30.9 | 88–116 | -0.125 | 64–120 |
| analytic_packet | -69.8 | 72–88 | -0.392 | 72–86 |

### Figures

- **Log–log fits:** each subplot plots measured error against total width on logarithmic axes; the red dashed fit covers the blue shaded interval. Each panel reports slope, $R^2$, selected widths, and point count.
- **Semilog fits:** the same targets use a linear width axis and logarithmic error axis, with independently selected fitting intervals. Panels zoom to show the decay and entry into the numerical floor.

## Additional details

All four synthetic checks pass: recovery of a known exponential middle segment, recovery of a known power-law middle segment, rejection of a flat curve, and refusal to silently exclude a large interior outlier. Endpoint sensitivity shifts each endpoint independently by one observed width sample; the largest relative slope change is 14.1%. These sensitivity ranges are descriptive, not statistical confidence intervals.

For the $\sin(4\pi x)$ log–log fit, the selected region begins at the smallest admissible width, so its upper cutoff is not fully observed. Other intervals can end before the numerical floor because curvature or a visible bend makes a single line inappropriate. In particular the Runge curves have intermediate dips and shoulders; these are not treated as a universal numerical floor.

Raw $R^2$ values from different selected intervals do not establish which convergence law is preferable. The CSV supplies the alternative-coordinate $R^2$ on exactly the same points. On the broad semilog-selected intervals, Runge 100 and smooth absolute value fit substantially better in semilog coordinates; on the log–log-selected intervals, the weak mixture and chirp fit better in log–log coordinates. Those are local comparisons, not asymptotic rate claims. No new halo or fitting-density study was performed for these added widths.

## Conclusions

Both representations admit well-supported local linear segments, with substantially different useful interval lengths. The plots expose the selected support and numerical-floor exclusion rather than implying one constant rate over the entire curve.

## Open questions

Sam can inspect whether these selected intervals match the intended middle portions. Global asymptotic rates and uniquely defined change points are not established by these local fits.
