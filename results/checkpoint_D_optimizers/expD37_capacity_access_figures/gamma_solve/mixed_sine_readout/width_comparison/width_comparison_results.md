# Mixed-sine width comparison

Status: complete; September 23, 2026.

## TL;DR

Repeat the saved mixed-sine experiment at N=1024 and N=256. GD is computed spectrally through one billion steps; Adam executes 20,000 updates. Every panel uses the same limits: steps 1–10^9 and relative L2 error 0.1%–110%.

## Question

Compare GD and Adam across the two requested widths with identical plot scales.

## Experiment design

The target is $\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)$ on $[-1,1]$. The same 8,193 endpoint-inclusive samples and gammas 8,12,16,64 are used at both widths. Centers have spacing $h=2/N$, N+1 interior centers, and $\lceil\sqrt N\rceil$ halo centers on each side. Thus N=256 uses 289 tanh features and N=1024 uses 1,089, each with an additional bias. The width change follows the existing halo-count rule, so the physical halo extent also changes.

Geometry is frozen and raw readout parameters start at zero. The objective is half mean squared error. GD uses the SVD of the actual sample-normalized tanh feature matrix and step $0.5/\lambda_{max}(K)$, evaluating the eigenvalue power formula without long GD training. Adam uses learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, no schedule and no weight decay. Four independent gammas run in a batched tensor. Everything uses FP64. Error is training relative L2, displayed in percent. Adam curves end at 20,000 even though every horizontal axis extends to one billion. Curves below 0.1% are outside the displayed range. No smoothing or best-so-far transformation is applied.

**Code and data.** `mixed_sine_readout/run.py` generates the width-specific problems and reuses the preceding spectral and Adam implementations. `mixed_sine_readout/plot_widths.py` generates these figures. Configuration, inputs, trajectories, optimizer state, and checks are in `data/N256/` and `data/N1024/`.

## Results

| N | Gamma | GD first step ≤1% | Adam first step ≤1% | Adam error at 20,000 |
|---:|---:|---:|---:|---:|
| 1024 | 8 | 13,666,769 | >20,000 | 6.3944% |
| 1024 | 12 | 178,269 | 2,801 | 0.3300% |
| 1024 | 16 | 59,927 | 977 | 0.0620% |
| 1024 | 64 | 15,176 | 219 | 0.0630% |
| 256 | 8 | 17,242,987 | >20,000 | 7.5721% |
| 256 | 12 | 198,898 | 7,153 | 0.5103% |
| 256 | 16 | 65,907 | 2,257 | 0.8282% |
| 256 | 64 | 17,151 | 602 | 0.8319% |

### Figures

- [Width by optimizer](gd_vs_adam_widths.png): top row N=1024, bottom row N=256; GD left, Adam right. Four gamma lines per panel. All axes have the same limits.
- [Width and optimizer by gamma](gd_vs_adam_widths_by_gamma.png): columns are gamma 8,12,16,64; rows are N=1024 GD, N=1024 Adam, N=256 GD, N=256 Adam. All axes have the same limits.

## Verification and limits

The previously verified GD and Adam implementations are reused. Saved trajectories are checked for finite values and unit initial relative error. Final Adam errors are independently recomputed from saved coefficients and the original sample matrix; discrepancies are in each width's `plot_checks.json`. First crossings do not imply Adam remains below the threshold afterward. The spectral calculation describes exact-arithmetic GD evaluated in FP64; no claim is made about accumulation of rounding over billions of executed updates.

## Conclusions

The figures provide the requested comparison at common scales. No new optimizer tuning was performed.

## Open questions

None added for this plot request.
