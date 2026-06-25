# expC02 -- Optimal bandwidth vs target frequency

**Status: conclusions approved by Sam (with the correction below).**

## TL;DR

- QI's optimal $\lambda\approx0.25 - 0.30$ is constant across frequency and width -- only the error magnitude grows with frequency.
- lstsq has a much wider, flatter bottom (the useful contrast).
- **Correction (Sam):** the earlier reading that lstsq's optimum drifts with frequency/width is *not real* -- it is jitter in that flat bottom (expC03 confirms the optimum is essentially constant).

## Question

Does the optimal bandwidth depend on target frequency, and does it depend the same way for QI and lstsq?

## Experiment design

Frequency ladder $\sin(k\pi x)$, $k\in\{1,2,4,8,16\}$, widths $\{128,256\}$, fp64, $K_c=160$, eval on 4096 points. Sweep a fine $\lambda$ grid ($0.02\to0.40$, step $0.01$) and, for each $(N,k,\lambda)$, record eval $L_\infty$ for the full QI construction and the lstsq readout on the same geometry; report each method's argmin $\lambda$. The optimum is also recast as absolute bandwidth $\gamma=\lambda N/2$, to separate "dimensionless $\lambda$" from "resolution $\gamma$".

**Code & data.** `experiments/expC02_lambda_vs_frequency/` (`run.py`, `sweep_utils.py`; tests in `tests/test_expC02_lambda_vs_frequency.py`). Data: `data.json`. Figures: `optimal_lambda_vs_frequency.png`, `error_vs_lambda_curves.png`.

## Results

QI's optimum stays pinned at $\lambda\approx0.25 - 0.30$ for every frequency and width (error grows mildly with $k$, location doesn't move). lstsq's curve is shallow over a broad band, so its reported argmin scatters -- but per expC03 that scatter is flat-bottom jitter, not a real trend. The correct reading: lstsq tolerates a wide $\lambda$ range; QI needs its narrow band.

### Figures

- **`error_vs_lambda_curves.png`** -- eval $L_\infty$ vs $\lambda$, one panel per width, one color per frequency, QI dashed, lstsq solid. QI U-curves all bottom near $0.30$; lstsq curves have a wide flat bottom (the headline). The high-$\lambda$ blowup is common to both.
- **`optimal_lambda_vs_frequency.png`** -- argmin $\lambda$ vs frequency. The flat QI dashed lines are the real signal; the lstsq markers move around, but that movement is the flat-bottom jitter corrected by expC03.

## Conclusions

QI's optimal $\lambda$ is constant (~$0.30$) across frequency and width; lstsq just has a wider flat bottom. The apparent drift of lstsq's optimum is numerical jitter, not a real effect. (Approved by Sam.)

## Open questions

None -- the robust bandwidth law is settled in expC03.
