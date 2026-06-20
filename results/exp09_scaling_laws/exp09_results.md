# Exp09 -- scaling laws for the fixed-geometry least-squares readout (1D)

**Status: draft -- pending Sam's review and sign-off on conclusions.**

## Question

The fixed uniform QI geometry (uniform centers + halo, $\gamma = \lambda/h$) with a plain least-squares readout reaches the fp64 floor in 1D (exp03/0B/exp11/exp12). This experiment traces three scaling laws of that method, across 6 targets and 3 activations ($\tanh$, gelu, relu), all fp64: (1) how the clean best-over-$\lambda$ error falls as the grid size $N$ (hence width $W$) grows; (2) how the clean error behaves as the number of training points $n_\text{data}$ crosses the under- $\to$ over-determined threshold $W+1$ at fixed geometry; and (3) how, under fixed $y$-noise, adding data buys accuracy along the statistical $\sigma\,n^{-1/2}$ law.

## What the code runs

Code: `experiments/exp09_scaling_laws/run.py`. All fp64, all numpy. Geometry per $N$: `uniform_geometry(N)` builds QI centers + halo (`default_halo`, `HALO_LAMBDA = 0.25`), fixing the width $W$, the span, and $h = 2/N$; $\gamma = \lambda/h$ is set separately. The readout is a bias-augmented $[\Phi, \mathbf 1]$ solved by truncated-SVD least squares with `RCOND = 1e-13` (relative to $s_\max$). Feature entry $\Phi_{ik} = \text{act}(\gamma_k (x_i - c_k))$. Metrics on a prime eval grid (`DATA_N_EVAL`/`WIDTH_N_EVAL` $= 8009/8009$, misaligned with the train grid): eval $L_\infty = \max_i |\hat f(x_i) - f(x_i)|$ and relative $L_2 = \lVert \text{resid} \rVert / \lVert y \rVert$ on the eval grid.

- Targets (6): `sine`, `sine_8pi`, `runge`, `sine_mixture`, `exp`, `abs_cubed`.
- Activations (3): `tanh`, `gelu` ($\tfrac12 z(1+\operatorname{erf}(z/\sqrt2))$), `relu`.
- Figure 1 (size scaling): $N$ on a 32-point logspace $16 \to 1024$ (the realized integer grid is $16, 18, \ldots, 1024$, giving $W$ from $157$ at $N{=}16$ to $1843$ at $N{=}1024$). Train grid $N_\text{train} = 4001$ (prime, overdetermined for all tested $W$). For each $(N, \text{act}, \text{target})$ the error is the best over the bandwidth grid $\lambda \in \{0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00\}$. A $5\times10^{-14}$ reference line is drawn on every panel.
- Figure 2 (data scaling, clean): fixed geometry $N = 128$ ($W = 269$), $\lambda = 0.25$; $n_\text{data}$ on a 30-point logspace $16 \to 512$ that starts underdetermined ($n < W$) and ends overdetermined. Clean targets, single seed.
- Figure 3 (data scaling, noisy): same fixed geometry, with additive Gaussian $y$-noise std $\text{NOISE\_STD} = 10^{-4}$, mean over 3 seeds $\{0,1,2\}$; $n_\text{data}$ on a 42-point logspace $16 \to 500{,}000$ extended far into the statistical regime. A reference slope $-1/2$ and the $n = W+1$ threshold line are drawn.

Layout of every figure: rows = activations, left column = relative $L_2$, right column = $L_\infty$, one colored line per target. Metric note: eval $L_\infty$ on a finite grid is a lower bound on the true sup error -- the standard convention across these 1D experiments.

## Results

Data: `results/exp09_scaling_laws/data.json` (keys `config`, `width`, `data_clean`, `data_noise`). Figures in the same directory.

### Figure 1 -- `error_vs_width.png` (size scaling, clean)

Best eval error over the $\lambda$ grid at the largest grid size $N = 1024$ ($W = 1843$):

| target | $\tanh$ rel $L_2$ | $\tanh$ $L_\infty$ | gelu rel $L_2$ | gelu $L_\infty$ | relu rel $L_2$ | relu $L_\infty$ |
|---|---|---|---|---|---|---|
| sine | 1.6e-14 | 1.4e-13 | 1.1e-13 | 8.4e-13 | 5.6e-6 | 1.2e-5 |
| sine_8pi | 1.4e-14 | 2.3e-13 | 4.2e-13 | 1.3e-11 | 9.0e-5 | 2.0e-4 |
| runge | 2.4e-14 | 5.3e-13 | 7.1e-15 | 1.5e-14 | 4.4e-6 | 1.5e-5 |
| sine_mixture | 1.2e-14 | 7.1e-14 | 5.8e-13 | 1.5e-11 | 6.4e-5 | 1.9e-4 |
| exp | 3.5e-14 | 2.4e-12 | 6.6e-15 | 1.2e-13 | 1.4e-7 | 8.3e-7 |
| abs_cubed | 3.2e-11 | 3.4e-10 | 2.0e-11 | 2.3e-10 | 1.3e-6 | 1.8e-6 |

The lowest values reached anywhere on the grid: $\tanh$ min rel $L_2 = 3.3\times10^{-15}$ (min $L_\infty = 9.0\times10^{-15}$), gelu min rel $L_2 = 1.9\times10^{-15}$ (min $L_\infty = 1.2\times10^{-14}$), relu min rel $L_2 = 1.4\times10^{-7}$ (min $L_\infty = 8.3\times10^{-7}$). For `tanh`/`sine` the descent is: $N{=}16$ rel $L_2 = 1.4\times10^{-13}$, $N{=}179$ $9.4\times10^{-15}$, $N{=}400$ $6.3\times10^{-15}$, $N{=}1024$ $1.6\times10^{-14}$ -- i.e. it reaches the floor early and rides it. relu `sine` descends as a power law: $N{=}16$ $2.5\times10^{-2}$, $N{=}80$ $9.2\times10^{-4}$, $N{=}400$ $3.7\times10^{-5}$, $N{=}1024$ $5.6\times10^{-6}$.

How to read it: 3 rows (activations) $\times$ 2 columns (rel $L_2$, $L_\infty$), $x = N$ (log), $y = $ best-over-$\lambda$ error (log), one line per target, black line at the $5\times10^{-14}$ floor. In the top two rows ($\tanh$, gelu) most targets descend and then sit on/just above the black floor line; `abs_cubed` (brown) descends but stays $\sim10^{-11}$, well above the floor at the largest $N$. The bottom row (relu) shows straight descending log-log lines for every target with no floor reached in range -- a power-law decay, not convergence to machine precision.

### Figure 2 -- `error_vs_data_clean.png` (data scaling, clean, fixed $N=128$, $W=269$)

How to read it: same $3\times2$ panel layout; $x = n_\text{data}$ (log), gray dotted line at $n = W+1 = 270$. For `tanh`/`sine` the clean rel $L_2$ falls steeply with $n$ from $1.7\times10^{-2}$ at $n{=}16$ through $2.1\times10^{-8}$ at $n{=}138$, reaching $5.0\times10^{-15}$ by $n{=}454$ -- i.e. error collapses as $n$ approaches and passes the $W+1$ threshold, then floors. Minimum clean rel $L_2$ reached: $\tanh$ $5.0\times10^{-15}$, gelu $2.6\times10^{-14}$, relu $9.2\times10^{-6}$. Read the panels for the sharp drop to the floor near/just past the dotted $W+1$ line ($\tanh$, gelu top rows), and for the relu row plateauing above $\sim10^{-5}$ regardless of how much data is added (relu is resolution-limited here, not data-limited).

### Figure 3 -- `error_vs_data_noise.png` (data scaling under $y$-noise $\sigma = 10^{-4}$, mean over 3 seeds)

How to read it: same $3\times2$ layout; $x = n_\text{data}$ (log, out to $5\times10^5$), $y = $ error (log), reference slope $-1/2$, threshold line at $n=W+1=270$. Each $\tanh$/gelu line shows a spike right at the $W+1$ transition (the augmented matrix is near-square and ill-conditioned there: `tanh`/`sine` rel $L_2 = 1.9\times10^{-2}$ at $n{=}200$, dropping to $4.9\times10^{-4}$ by $n{=}257$ and $1.1\times10^{-4}$ by $n{=}331$), then a long straight descent. In the overdetermined regime ($n > 2W$) the fitted log-log slopes cluster tightly around $-1/2$ for every $\tanh$/gelu target: rel $L_2$ slopes range $-0.487$ to $-0.522$ (e.g. $\tanh$/`sine` $-0.491$, gelu/`exp` $-0.493$), and $L_\infty$ slopes $-0.43$ to $-0.51$. Across the full overdetermined range `tanh`/`sine` falls from $1.1\times10^{-4}$ ($n{=}331$) to $2.5\times10^{-6}$ ($n{=}500{,}000$). The relu row is essentially flat (rel $L_2$ slopes $\approx 0$ for most targets, $-0.19$ for `exp`): relu `sine` sits at $\sim3.8\times10^{-4}$ at $n{=}300$ and $3.6\times10^{-4}$ at $n{=}500{,}000$ -- its bias/resolution error exceeds the $10^{-4}$ noise contribution, so adding data does not help.

## Conclusions

Plainly visible in the data (pending Sam's sign-off):

- **Clean size scaling: $\tanh$ and gelu descend with width and floor near $\sim5\times10^{-14}$; relu does not reach the floor.** On the smooth targets the best-over-$\lambda$ error drops to the $5\times10^{-14}$ reference and rides it ($\tanh$ min rel $L_2 = 3.3\times10^{-15}$, gelu $1.9\times10^{-15}$), whereas relu rides a descending power law topping out at $\sim10^{-6}$ rel $L_2$ at $N{=}1024$. `abs_cubed` is the one $\tanh$/gelu target that stays well above the floor ($\sim3\times10^{-11}$ rel $L_2$ at the largest $N$).
- **$\tanh$ (and gelu) reach a much lower floor than relu** -- by roughly 8-9 orders of magnitude in the clean size and data figures.
- **Clean data scaling: error collapses as $n_\text{data}$ crosses $W+1$ and then floors.** At fixed $N{=}128$ ($W{=}269$) the clean $\tanh$ error falls steeply through the $W+1 = 270$ threshold and reaches $\sim5\times10^{-15}$ rel $L_2$ once overdetermined; relu plateaus above $\sim10^{-5}$ regardless of $n$.
- **Noisy data scaling follows the $\sigma\,n^{-1/2}$ law for $\tanh$/gelu.** In the overdetermined regime the noisy lines fall as straight log-log lines with slope $\approx -0.5$ (fitted rel $L_2$ slopes $-0.487$ to $-0.522$ across all $\tanh$/gelu targets), with no plateau out to $n = 5\times10^5$. There is a conditioning spike exactly at the $n = W+1$ transition.
- **relu does not show the noise law at this geometry** because its approximation (bias) error exceeds the $10^{-4}$ noise level; its noisy lines are flat in $n$ (slopes $\approx 0$).

Flag (not independent evidence): the size-scaling and clean-data-scaling figures both report best-over-$\lambda$ error -- they share the same $\lambda$-selection step, so "floors near $\sim5\times10^{-14}$" in Figure 1 and "collapses past $W+1$" in Figure 2 are not fully independent observations of the same method's accuracy. The noisy-figure slope ($\approx -0.5$) is independent of the $\lambda$ selection (fixed $\lambda = 0.25$).
