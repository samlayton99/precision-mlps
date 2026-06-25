# expC01 -- The lambda tradeoff (U-shaped error vs bandwidth)

**Status: conclusions approved by Sam.**

## TL;DR

- The error-vs-$\lambda$ curve is U-shaped for both QI and lstsq -- ill-conditioning on the left, aliasing/cancellation on the right, a viable minimum between.
- QI's optimum is a narrow band (fp64 $\approx0.30$, mpmath $\approx0.25$); lstsq's bottom is wide and flat. The robust basin study is expC03.

## Question

Does the error trace the U-shape in $\lambda=\gamma h$ that theory predicts, and where is the optimum for each method and precision?

## Experiment design

For each $(N,\lambda)$ build a QI construction and an lstsq readout on the *same* geometry: centers $-1+nh$ with $h=2/N$, a single shared $\gamma=\lambda/h$, $K_c=160$, halo $=\texttt{default\_halo}(N,\lambda)$. The lstsq readout is the direct $\Phi$ solve (not normal equations) on $\max(512,4N)$ points with a fitted bias. Both are scored on 2048 eval points by $L_\infty=\max_x|\hat f-f|$ and rel $L_2=\|\hat f-f\|/\|f\|$. Four sweep variants cover the walls and the bottom: a coarse fp64 sweep ($N\in\{16,\dots,256\}$, $\lambda\in[0.01,1.0]$), an mpmath-QI sweep ($N\in\{128,256,512\}$, $\lambda\in[0.10,0.40]$), and two fine zooms over $\lambda\in[0.22,0.28]$ at step $0.005$ (one with QI in fp64, one in mpmath). Targets: sine, runge, exp, sine_mixture.

**Code & data.** `experiments/expC01_lambda_tradeoff/` (`run.py`, `plot_consolidated.py`); run with `--mpmath`, `--fine`, `--fine-fp64`, then `plot_consolidated.py`. Data: `data.json`, `data_mpmath.json`, `data_fine*.json`, merged `all_data.json`. Figures: `consolidated_linf.png` (headline) plus per-precision/zoom `lambda_tradeoff_*`.

## Results

- **The U-shape holds for both methods** -- the lstsq readout on the same geometry shows the same U as QI, so the tradeoff is not a QI-formula artifact.
- **Optima differ in location and width:** QI-fp64 bottoms near $0.30$ (~$5\times10^{-12}$ at $N=128$ on sine); QI-mpmath near $0.25$ at machine $\varepsilon$; lstsq reaches ~$10^{-13}$ with a wide flat bottom (no single pinned optimum). runge/sine_mixture descend into the well only once $N$ is large enough.

### Figures

- **`consolidated_linf.png`** (headline) -- 3x4 grid: rows = precision combo (fp64, mpmath-QI, both-mpmath), cols = targets; x = $\lambda$, y = $L_\infty$ (log), color = width, dashed = QI, solid = lstsq. Read down a column to watch the floor drop with precision; within a panel, the dashed QI trough sits near $0.30$ (fp64) or $0.25$ (mpmath) and the solid lstsq trough is wider/flatter.
- **`lambda_tradeoff_*`** -- per-precision and fine-zoom views (same 2x2-per-target layout), magnifying the bottom of the U.

## Conclusions

The bandwidth tradeoff is real and method-independent: both QI and lstsq are U-shaped. QI needs a narrow viable band; lstsq has a wide flat bottom. (Approved by Sam; the quantitative basin is expC03.)

## Open questions

None here -- pursued in expC03.
