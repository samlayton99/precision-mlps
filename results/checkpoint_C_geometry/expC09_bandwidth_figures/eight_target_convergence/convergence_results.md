# Eight-function width convergence

Status: measured, 2026-09-21.

## TL;DR

- One PNG compares the eight selected functions over 41 total widths from 64 to 2048.
- Fits use the preceding figures' fixed-grid tanh least-squares method: FP64, $\lambda=0.25$, and 24 halo neurons per side, included in width.
- The weak high-frequency mixture has a visible intermediate error shoulder; the compact smooth bump converges substantially more slowly than the analytic targets.

## Question

How does the width needed for accurate approximation differ across oscillatory, localized, sharp, and nonanalytic smooth functions?

## Experiment design

Each function is fitted and scored on $[-1,1]$:

| Curve | Function |
|---|---|
| Low-frequency sine | $\sin(4\pi x)$ |
| High-frequency sine | $\sin(24\pi x)$ |
| Weak HF mixture | $\sin(2\pi x)+10^{-3}\sin(40\pi x)$ |
| Chirp | $\sin(8\pi(x+1)^2)$ |
| Runge (25) | $1/(1+25x^2)$ |
| Runge (100) | $1/(1+100x^2)$ |
| Gaussian envelope | $e^{-x^2/(2\cdot0.4^2)}[\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x)]$ |
| Smooth compact bump | $e^{-1/(1-(x/0.5)^2)}$ for $|x|<0.5$, and zero elsewhere |

Widths have eight samples per doubling, beginning with $64,72,80,\ldots,120,128$ and ending at 2048. Every width is even. The network has centers $c_j=-1+2j/N$ for $j=-24,\ldots,N+24$, so $W=N+49$, spacing $h=2/N$, and slope $\gamma=0.25/h$. The output bias does not count toward $W$.

Each target gets an independent FP64 `gelsd` least-squares solve with relative cutoff $2^{-52}$ and its own vector evaluation, matching the earlier figures. The fitting grid has $\max(4801,4W+1)$ equally spaced points; evaluation has $\max(8001,8W+1)$ points. The plotted metric is relative sampled $L^2$ error. Evaluation is divided into chunks to limit memory. Curves show raw measurements without smoothing or fitted convergence envelopes.

**Code & data**

- Runner and configuration: `experiments/expC09_bandwidth_figures/convergence.py`, `convergence_config.yaml`.
- Tests: `tests/test_bandwidth_convergence.py`.
- Main PNG: `figures/eight_target_convergence.png` beside this writeup.
- Control PNG: `figures/convergence_validation.png` beside this writeup.
- Observations, source fingerprint, exact widths, readout coefficients, and numerical checks: `data/` beside this writeup.

Reproduce from the repository root:

```sh
.venv/bin/python experiments/expC09_bandwidth_figures/convergence.py --validate
.venv/bin/python -m pytest -q tests/test_bandwidth_convergence.py
```

Use `--plot-only` to redraw existing observations. Exports are PNG only.

## Results

The simple sine begins with small error already at $W=64$. The higher-frequency sine and chirp remain poorly represented at the first few widths before falling sharply. The weak high-frequency mixture initially leaves error near the amplitude of its small component and improves once enough width is available. The narrower Runge peak needs more width than the original Runge function. The compact bump has the longest declining curve.

On the sampled width grid, the compact bump first drops below relative error $10^{-12}$ at $W=896$, compared with $W=144$ for Runge (25) and $W=224$ for Runge (100). These are first crossings, not guarantees that every subsequent point is smaller.

### Figures

- **Eight-target convergence:** logarithmic width and error axes, one curve per function, all widths counting halo neurons.
- **Numerical controls:** eight panels compare the main curve against a doubled, shifted evaluation grid; doubled fitting/evaluation density at three widths; and doubled halo counts at fixed interior spacing. Halo controls increase total width, while the horizontal labels retain the original width for comparison.

## Validation and limits

Regression checks cover the requested width range, all eight target formulas and compact-support boundaries, and agreement with independent single-target solves. Saved coefficients permit direct reevaluation. Numerical controls use original widths 64, 128, 256, 512, 1024, and 2048; fitting-density controls use 64, 256, and 2048. Their full values are retained with the observations.

All three regression tests pass, and all 328 target/width fits are present. The largest relative change under denser shifted evaluation is 15.6%, at a Runge error of approximately $2\times10^{-14}$. Doubling fitting density changes a checked error by at most a factor 3.05; doubling the halo improves one by a factor 9.87. Numerical floors therefore should be read by order of magnitude. The final main and control PNGs were visually inspected.

An initial shared-right-hand-side solve was replaced by independent solves after a regression check found visible differences on poorly resolved targets with very large coefficients. Its observations and source are preserved under `archive/shared_rhs_pilot/`. This is an FP64 cancellation issue in difficult coarse-width fits; floor-scale differences should not be interpreted as exact-arithmetic convergence rates.

The construction method is fixed-grid feature placement followed by least-squares readout recovery. The title deliberately identifies that method. These measurements are not a new run of the explicit cardinal-QI coefficient construction, a continuous-norm certificate, or proof of sufficient halo size for every target and width.

## Conclusions

The selected functions produce distinct width-convergence curves, including a small-component shoulder and a much slower smooth-bump regime, under one fixed bandwidth and precision convention.

## Open questions

The plot does not establish optimal bandwidths or native lower-precision behavior. Numerical and halo controls constrain how precisely its floors can be interpreted.
