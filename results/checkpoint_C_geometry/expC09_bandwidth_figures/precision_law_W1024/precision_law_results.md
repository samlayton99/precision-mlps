# Predicted-bandwidth precision law at width 1024 — data-obvious

This records the earlier partial-precision simulation. The current figure uses the corrected full-forward experiment documented in `../precision_law_W1024_strict/precision_law_results.md`; the measurements below are preserved for comparison.

## TL;DR

- Panel (c) now uses measured chirp errors at total width 1024 for every integer precision from 8 to 53 bits.
- Its reference is labeled $O(\log(1/\varepsilon))$, expressing required significand bits versus inverse error. Only the reference's intercept is fitted.

## Question

How does the predicted-bandwidth precision curve change at width 1024?

## Experiment design

The target is $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$. Total width $W=1024$ includes 24 halo centers per side, leaving $N=975$ intervals with $h=2/N$. Features are $\tanh((\lambda/h)(x-c_j))$, with $c_j=-1+jh$ for $j=-24,\ldots,N+24$, plus a bias. The existing readout implementation uses 4,801 fitting points and 8,001 evaluation points; the error is the sampled relative $L^2$ norm.

For each $p=8,\ldots,53$, the refined selector uses $\bar\omega=16\pi$ and $e_{\mathrm{tol}}=2^{1-p}$. Its search extends to 3.0 where necessary, and every returned threshold is verified. Features, labels, readouts, products and accumulated sums follow the prior significand-rounding simulation; arguments and SVD internals remain FP64. Only predicted-bandwidth points are measured here; no best-bandwidth sweep is claimed at this width.

The reference is $\varepsilon=C2^{-p}$, equivalently $p=\log_2(1/\varepsilon)+\log_2 C$. Fixing slope $-1$ in log-base-2 error and fitting its intercept over $p=16,\ldots,40$ gives $C=12.54995$. The big-O legend denotes this reference scaling, not a proof of an asymptotic law beyond the observed precision floor.

**Code & data:** runner `experiments/expC09_bandwidth_figures/precision_rule.py`; renderer `experiments/expC09_bandwidth_figures/combined.py` (default width 1024 and predicted bandwidth). Configuration, source hashes, predictions, measurements, summary and validation are in `results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W1024/data/`. The figure is `results/checkpoint_C_geometry/expC09_bandwidth_figures/combined/figures/all_targets_chirp_precision_law_rule_W1024.png`. The previous width-512 figures and provenance are preserved in `combined/archive/precision_law_W512/` under the same experiment results directory.

## Results

The measured error falls from about $6.34\times10^{-2}$ at 8 bits to $8.23\times10^{-14}$ at 53 bits, with a near-linear descending region in log error and a floor at high precision.

### Figures

- **Three-panel PNG:** the width-scaling and bandwidth-selection panels retain their data. The Precision law panel shows the new width-1024 curve and the offset-fitted reference; its top-right legend omits width as requested.

## Additional details

All 46 precision points are present and finite. One FP64 measurement is reused; 45 points are newly measured. Checks verify prediction thresholds, unique coverage, geometry and sampling counts. The PNG was visually inspected. Halo and sampling-density sufficiency were not re-tested.

## Conclusions

The width-1024 curve follows the reference trend through its descending region before leveling near $10^{-13}$.

## Open questions

The dependence on halo count and native low-precision solver arithmetic remains untested.
