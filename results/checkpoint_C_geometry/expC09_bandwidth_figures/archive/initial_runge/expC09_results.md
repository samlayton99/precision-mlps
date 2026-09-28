# Runge: refinement, bandwidth, and precision

Superseded by the four-target figures two directories above. This is the preserved initial report; its original paths and reproduction commands describe the earlier version.

Status: measured; data-obvious observations, 2026-09-21.

## TL;DR

- The three-panel figure replaces the schematic with fresh tanh least-squares measurements on $f(x)=1/(1+25x^2)$ over $[-1,1]$.
- The bandwidth sweep uses actual hidden widths $W=128,256,512,1024$, including 32 halo neurons on each side. The precision panel fixes $W=256$ and uses $p=24,32,40,53$; $p=48$ is omitted.
- The fixed-bandwidth refinement reaches an FP64 recovery floor of a few $10^{-14}$. Lower precisions use the earlier experiment's rounding model with FP64 SVD internals.

## Question

How do measured Runge errors change with total network width, bandwidth, and precision, and where do the recent basic and refined bandwidth rules place their choices?

## Experiment design

Use $q(x)=b+\sum_{j=-H}^{N+H}a_j\tanh(\gamma(x-c_j))$, where $c_j=-1+jh$, $h=2/N$, $H=32$, and $\gamma=\lambda/h$. The total hidden width is $W=N+1+2H=N+65$; the output bias is excluded. Consequently the four bandwidth-sweep widths correspond to $N=63,191,447,959$. Interior interval counts are derived from the requested total widths.

Fit the readout on 4,801 equally spaced points with an SVD cutoff of $2^{1-p}$. Score relative sampled $L^2$ error, $\|q-f\|_2/\|f\|_2$, on 8,001 equally spaced evaluation points. These are dense function measurements, not a statistical test set or a certified continuous norm. The bandwidth grid has 85 logarithmically spaced values from $0.05$ to $1.5$, supplemented with direct measurements at both predicted choices and the halo-control bandwidths. Refinement uses $\lambda=0.25$ and widths 80 through 256 in increments of eight.

For $p<53$, round feature values, target values, solved coefficients, products, and sequential accumulated sums to $p$ binary significand bits, using nearest-even rounding. Feature arguments and SVD internals remain FP64; exponent limits are not emulated. At $p=53$, use native FP64 features, solve, and evaluation. This reproduces the precision model in the previous anchor experiment; it is not native low-precision validation.

The basic and refined choices use the September 14 revised selector, effective tolerance $2^{1-p}$, and the refined frequency scale $\bar\omega=5$. This scale is the amplitude-weighted mean absolute angular frequency of Runge's whole-line extension. Predictions are saved before solving. Every marker's vertical coordinate is a fresh measured error at the predicted bandwidth. The shaded strip spans the refined FP64 predictions across widths; it is not a confidence interval or a guaranteed low-error band.

**Code & data**

- Configuration and runner: `experiments/expC09_runge_bandwidth_figure/config.yaml`, `run.py`.
- Selector: `docs/lambda_theorem_compatibility/choosing_optimal_lambda/choose_lambda.py`.
- Controls: `experiments/expC09_runge_bandwidth_figure/validate.py`.
- Tests: `tests/test_runge_bandwidth_figure.py`.
- Local observations, predictions, configuration, selector hash, and validation: `data/measurements.jsonl`, `data/predictions.json`, `data/summary.json`, `data/validation.json` beside this writeup. Pilot data are preserved separately under `data/`.
- Main outputs: `figures/runge_bandwidth.png`, `.pdf`, `.svg` beside this writeup.
- Halo control: `figures/halo_check.png` beside this writeup.

Reproduce from the repository root:

```sh
.venv/bin/python experiments/expC09_runge_bandwidth_figure/run.py
.venv/bin/python experiments/expC09_runge_bandwidth_figure/validate.py
.venv/bin/python -m pytest -q tests/test_runge_bandwidth_figure.py
```

## Results

At fixed $\lambda=0.25$, the measured error first falls below $10^{-13}$ at $W=168$ and then fluctuates around a few $10^{-14}$. The dashed geometric trend is fitted to the six pre-floor observations with errors between $10^{-10}$ and $10^{-2}$; it is an empirical guide, not the theorem's error term.

The basic FP64 rule gives $\lambda=0.2441$, while the refined predictions range from $0.2601$ to $0.2855$ across the displayed widths. Both choices lie near the observed transition into aliasing error. The $W=128$ curve is less accurate and has a local dip; neither rule is a guarantee of the exact minimum.

At $W=256$, the refined choices for $p=24,32,40,53$ are approximately $0.646,0.466,0.366,0.272$. Their measured relative errors are approximately $4\times10^{-7},2\times10^{-9},3\times10^{-12},2\times10^{-14}$.

### Figures

- **Main three-panel figure:** (a) width refinement with the empirical geometric guide; (b) measured bandwidth curves at four total widths, with basic/refined choices; (c) the precision-model sweep at fixed total width. All curves are unsmoothed measurements; small floor fluctuations remain visible.
- **Halo control:** four panels hold each original interior spacing fixed while increasing the halo to 64 and 128 nodes per side. The total width increases in these controls, so they isolate halo extension from interior resolution.

## Validation and limits

All 919 configured measurements are present. The 238 paired halo comparisons show at most a factor $1.79$ improvement for FP64 errors above $10^{-12}$, and at most a factor $2.18$ at the rule choices across the displayed precisions. This supports the chosen halo for this figure's scale and range, without claiming exact independence from the halo.

Seventeen sampling controls double evaluation density and then double fitting density. Changing the evaluation grid moves errors by at most 7%, with the largest relative changes at the FP64 floor. Changing the fitting grid changes some rounded-precision floors and poorly conditioned small-bandwidth solves by factors of a few. The largest-width, largest-bandwidth endpoint agrees to better than $10^{-4}$ relatively after doubling both grids. These controls support the broad curves, not percent-level claims about numerical minima.

Thirteen regression checks pass: total-width accounting, preservation of the interior grid under halo extension, rounding against binary32 and halfway cases, scalar-budget thresholds, invalid-width rejection, and numerical agreement with the previous bandwidth experiment at three precisions. The figure is a finite-grid least-squares study, not a new validation of the explicit cardinal-QI coefficient construction or of a continuous error bound.

## Conclusions

The measured Runge figure shows geometric refinement ending at an FP64 recovery floor and precision-dependent bandwidth choices near the right side of the low-error region. Halo extension does not change these broad features over the displayed range.

## Open questions

Native arithmetic at intermediate precisions remains outside this figure's scope.
