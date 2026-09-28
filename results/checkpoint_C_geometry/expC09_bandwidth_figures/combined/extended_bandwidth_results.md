# Combined figure: bandwidth sweeps at widths 512 and 1024 — data-obvious

## TL;DR

- Added measured chirp curves at total widths 512 and 1024 to panel (b).
- Both added curves reach relative errors near $10^{-13}$; the existing width-refinement and precision panels are retained.

## Question

How do the chirp bandwidth curves extend to larger total widths under the existing experimental settings?

## Experiment design

The target is $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$. Each added curve uses $W\in\{512,1024\}$, 24 halo centers per side, $N=W-49$, $h=2/N$, centers $c_j=-1+jh$ for $j=-24,\ldots,N+24$, and features $\tanh((\lambda/h)(x-c_j))$. A bias-augmented SVD least-squares readout is fitted on 4,801 uniform points in FP64 with singular-value cutoff $2^{-52}$. Relative sampled $L^2$ error is measured on 8,001 uniform points.

Each curve has the original 85 logarithmically spaced bandwidths from 0.05 to 1.5 plus the basic and refined prediction locations, totaling 87 measurements. Only the refined prediction is displayed, labeled Predicted $\lambda$, using the existing mean-frequency heuristic $\bar\omega=16\pi$ and tolerance $2^{-52}$. All prior observations and the established figure formatting are reused.

**Code & data:** runner `experiments/expC09_bandwidth_figures/extend_bandwidth.py`; renderer `experiments/expC09_bandwidth_figures/combined.py`; new measurements, source hashes, predictions and validation in `results/checkpoint_C_geometry/expC09_bandwidth_figures/combined/data/extra_bandwidth/`. The original precision-level figure and provenance are preserved in `combined/archive/precision_levels/` under the same experiment results directory; the current renderer replaces panel (c) with the precision law documented separately. The extension runner resumes saved measurements.

## Results

| Total width | Best sampled $\lambda$ | Minimum relative $L^2$ error | Predicted $\lambda$ | Error at prediction |
|---:|---:|---:|---:|---:|
| 512 | 0.232912 | $1.21\times10^{-13}$ | 0.254932 | $1.82\times10^{-13}$ |
| 1024 | 0.265508 | $8.23\times10^{-14}$ | 0.265508 | $8.23\times10^{-14}$ |

### Figures

- **Combined three-panel PNG:** (a) eight targets versus width, (b) six chirp bandwidth curves at widths 96, 128, 192, 256, 512 and 1024, and (c) four chirp precision levels at width 256. The six bandwidth curves use viridis, ordered from purple at the smallest width to yellow-green at the largest. Diamonds mark predictions.

## Additional details

All 174 added measurements are present, unique and finite. Checks verify geometry, bandwidth scaling, precision, sampling counts, sweep coverage and prediction locations. The existing bandwidth and additional-target test modules passed all 21 tests. The rendered PNG was visually inspected. No additional halo-sufficiency or sampling-density study was performed for these new widths.

## Conclusions

Under the retained settings, both added widths achieve sampled relative errors near $10^{-13}$ in the bandwidth sweep.

## Open questions

Whether increasing the halo or sampling density materially changes these new curves remains untested.
