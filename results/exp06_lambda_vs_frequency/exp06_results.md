# Exp06 -- Optimal $\lambda$ vs target frequency (QI vs lstsq)

**Question:** does the optimal bandwidth $\lambda = \gamma h$ depend on the target frequency, and does it depend the same way for the QI construction and the lstsq readout?

Code: `experiments/exp06_lambda_vs_frequency/` (`run.py`, `config.yaml`, `sweep_utils.py`; tests in `tests/test_exp06.py`). Run: `python3 experiments/exp06_lambda_vs_frequency/run.py`.

## Experiments run (exactly as set up in the code)

fp64 throughout; eval grid $N_{\text{eval}} = 4096$; $K_c = 160$. Frequency ladder $\sin(k\pi x)$ for $k \in \{1, 2, 4, 8, 16\}$; widths $N \in \{128, 256\}$; fine $\lambda$ grid $\{0.02, 0.03, \dots, 0.40\}$ (step $0.01$). For each $(N, k, \lambda)$ we compute the eval $L_\infty$ of (a) the full QI construction and (b) the lstsq readout on the same geometry, then take the $\lambda$ that minimizes each (`best_lambda`).

## Results

`data.json` holds one row per $(N, k)$ with the full $\lambda$ grid, both error curves, and the argmin $\lambda$/error for each method.

**Optimal $\lambda$ (argmin of eval $L_\infty$):**

| $N$ | method | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}8$ | $k{=}16$ |
|---|---|---|---|---|---|---|
| 128 | QI | 0.30 | 0.32 | 0.30 | 0.29 | 0.28 |
| 128 | lstsq | 0.09 | 0.17 | 0.29 | 0.27 | 0.25 |
| 256 | QI | 0.30 | 0.30 | 0.32 | 0.30 | 0.30 |
| 256 | lstsq | 0.04 | 0.09 | 0.16 | 0.11 | 0.21 |

QI's optimum is pinned at $\lambda \approx 0.30$ for every frequency and both widths (best error $2.5\times10^{-12}$ to $5\times10^{-11}$). lstsq's optimum rises with frequency and is lower at the larger width.

**The same optima as absolute bandwidth $\gamma = \lambda N/2$:**

| $N$ | method | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}8$ | $k{=}16$ |
|---|---|---|---|---|---|---|
| 128 | lstsq $\gamma$ | 5.8 | 10.9 | 18.6 | 17.3 | 16.0 |
| 256 | lstsq $\gamma$ | 5.1 | 11.5 | 20.5 | 14.1 | 26.9 |
| both | QI $\gamma$ | $0.15N$ | $0.15N$ | $0.15N$ | $0.15N$ | $0.15N$ |

For $k \le 4$ the lstsq optimal $\gamma$ is nearly width-independent and grows with $k$ (≈ 5, 11, 19); at $k = 8, 16$ it is noisier (those frequencies are near the grid/cancellation limits). QI fixes $\lambda$, so its optimal $\gamma = 0.15N$ grows with width regardless of target.

### Figures

**`optimal_lambda_vs_frequency.png`.** Optimal $\lambda$ vs frequency; lstsq solid, QI dashed/open; blue $N{=}128$, red $N{=}256$. *How to read:* flat dashed lines $\Rightarrow$ QI optimum is frequency-independent; rising solid lines $\Rightarrow$ lstsq optimum tracks frequency; the solid red sitting below solid blue $\Rightarrow$ lstsq optimum falls with width.

**`error_vs_lambda_curves.png`.** Eval $L_\infty$ vs $\lambda$, one panel per width, one color per frequency; lstsq solid, QI dashed. *How to read:* QI U-curves all bottom near $\lambda = 0.30$; lstsq U-curves have a wide flat bottom whose left (low-$\lambda$) wall rises with frequency, shifting the minimum rightward. The high-$\lambda$ blowup (fp64 cancellation) is common to both.

## Conclusions (pending Sam's review -- NOT yet approved)

*Proposed, for discussion -- do not treat as final.*

- QI's optimal $\lambda$ is $\approx 0.30$, **independent of target frequency and width** (consistent with the cardinal coefficients $c_j$ being target-independent). The error *magnitude* grows mildly with frequency; the optimum *location* does not move.
- lstsq's optimal $\lambda$ is **not fixed**: it rises with frequency and falls with width. Recast as absolute bandwidth, lstsq's optimal $\gamma$ is set mainly by the target frequency and is roughly width-independent (clean for $k \le 4$) -- i.e. lstsq picks the bandwidth needed to resolve the target, whereas QI always uses $\gamma = O(N)$. This connects to the $\gamma$-scaling theme: QI mandates $\gamma = O(N)$; lstsq only needs $\gamma \sim$ (target frequency).
- Both methods share the high-$\lambda$ wall (fp64 cancellation); lstsq has a much wider flat bottom than QI's sharp optimum.
