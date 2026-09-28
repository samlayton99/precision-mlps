# Chirp precision law at total width 512 — data-obvious

## TL;DR

- Two three-panel PNGs show relative error versus significand precision, using either the predicted bandwidth or the best sampled bandwidth.
- All 46 integer precisions from 8 through 53 bits were measured. The rule's error is within a factor of 2.38 of the best sampled error throughout this sweep.

## Question

How does chirp approximation error change with arithmetic precision at fixed total width, using a bandwidth prediction versus an empirical bandwidth choice?

## Experiment design

The target is $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$. Total width is $W=512$, including 24 halo centers on each side. Thus $N=463$, $h=2/N$, and the centers are $c_j=-1+jh$ for $j=-24,\ldots,N+24$. The model is a bias plus a linear combination of $\tanh((\lambda/h)(x-c_j))$ features. Readouts are fitted on 4,801 uniform points; relative sampled $L^2$ error is evaluated on 8,001 uniform points.

At every integer precision $p=8,\ldots,53$, we retain the existing experiment's rounding convention: round features, labels, readout coefficients, products and sequential accumulated sums to $p$ significand bits. Feature arguments and SVD internals remain FP64; the singular-value cutoff is $2^{1-p}$. The $p=53$ path uses the original FP64 matrix-vector evaluation. These are simulated significand-precision experiments with unrestricted exponent range, not native low-precision SVD solves.

The rule version uses the existing refined aliasing selector with mean local angular frequency $\bar\omega=16\pi$ and $e_{\mathrm{tol}}=2^{1-p}$. This remains a heuristic frequency summary for the chirp. At low precisions the threshold lies above the old upper search limit of 1.5; the selector search is extended to 3.0 when needed, and every returned threshold is checked against the requested alias score.

The best-bandwidth version takes the minimum measured error over a common 103-point bandwidth grid, plus that precision's basic and refined predictions: 105 points per precision. The common grid retains the original 85 logarithmic points on $[0.05,1.5]$ and adds 18 logarithmic points up to 3.0. Best means the best sampled value, not a certified continuous minimum. It is selected using the same evaluation errors displayed in the figure, so it is a retrospective comparison.

**Code & data:** runner `experiments/expC09_bandwidth_figures/precision_law.py`; renderer `experiments/expC09_bandwidth_figures/combined.py` (defaults to the predicted-bandwidth variant; use `--precision-law both` for both). Configuration, source hashes, predictions, measurements, validation and CSV/JSON summaries are in `results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W512/data/`. Figures are `results/checkpoint_C_geometry/expC09_bandwidth_figures/combined/figures/all_targets_chirp_precision_law_rule.png` and `all_targets_chirp_precision_law_best.png`; each has a separate provenance manifest in the combined directory. The canonical `all_targets_chirp_bandwidth.png` now displays the rule version. The former precision-level figure is preserved in `combined/archive/precision_levels/`.

## Results

| Significand bits | Error using predicted $\lambda$ | Error using best sampled $\lambda$ |
|---:|---:|---:|
| 8 | $6.13\times10^{-2}$ | $4.35\times10^{-2}$ |
| 24 | $6.23\times10^{-7}$ | $4.38\times10^{-7}$ |
| 40 | $8.98\times10^{-12}$ | $6.97\times10^{-12}$ |
| 53 | $1.82\times10^{-13}$ | $1.21\times10^{-13}$ |

Both curves fall roughly linearly on the log-error versus precision axes before approaching an error floor near $10^{-13}$. The rule-to-best error ratio ranges from 1 to 2.38. No sampled minimum occurs at a search boundary.

### Figures

- **Rule version:** panel (a) retains the eight-target width scaling; panel (b) retains six viridis chirp bandwidth curves, with the prediction diamond moved into the bottom-right legend; panel (c), titled Precision law, plots measured error at predicted bandwidth against significand bits at $W=512$.
- **Best-bandwidth version:** panels (a) and (b) are identical; panel (c) uses the minimum sampled error at each precision. Both versions use matching axes and colors for direct comparison.

## Additional details

The current predicted-bandwidth figure includes the reference $\varepsilon=C\,2^{-p}$, equivalently $p=\log_2(1/\varepsilon)+\log_2 C$. The slope is fixed at one halving of error per added bit; only the log-space intercept is fitted by least squares over $p=16,\ldots,40$, giving $C=11.10196$. This is a reference scaling with a fitted offset, not a fitted convergence exponent. Panel (c) now has its legend in the upper right and omits the width annotation; the measured width remains 512.

There are 4,830 observations, including 87 reused FP64 observations and 4,743 new observations. Validation checks complete and unique coverage, finite positive errors, geometry, fitting/evaluation counts, selector thresholds, and best-error ordering. All 21 existing bandwidth and additional-target tests passed. Both PNGs were visually inspected. The existing halo and sampling settings were retained; no additional halo-sufficiency study was performed.

## Conclusions

For this target, width and precision simulation, the predicted-bandwidth curve closely tracks the best sampled curve across the measured precision range.

## Open questions

How the precision curves change under native low-precision solves or different halo counts is not tested here.
