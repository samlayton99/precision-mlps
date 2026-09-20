# Pre-registered test of the lambda anchor rule v1: results (2026-09-10)

**Status:** scored exactly as pre-registered in `PREREGISTRATION_anchor_v1.md` (hashes there; predictions SHA-256 `f449add0…`). Nothing was changed after the run. Draft pending Sam's review; the verdicts below are the mechanical outputs of the pre-registered criteria, not interpretations.

## TL;DR

- **Read the figures first.** In 9 of the 10 fresh targets the dot (the rule's $\lambda$ on the measured curve) sits at the corner where the floor meets the wall, for every width 48 to 384, every precision 11 to 53 bits, native fp32, and all four activations; at $N=48$, where the valley is a narrow V, the dot sits at its bottom. The one exception is the decaying-tail cosine sum at $N=48$ (its $19\pi$ line has 5 cells per wavelength; the dot lands at $10^{-6}$ against a $10^{-8}$ to $10^{-11}$ minimum), with a milder version at $N=96$ and at 40 to 53 bits. That is the under-resolved high-frequency regime in which the method itself is weak; the mean frequency cannot see a line 1000x below the main one, and the practical answer there is to choose a smaller $\lambda$ when significant content is known to sit well above the mean. On the gaussian activation only, the $0.01$-amplitude mixture also shows dots a decade or two up the wall at every precision (the raw-$B$ convention; $B/\|f\|_\infty$ removes it).
- **The pre-registered criteria return tanh FAIL, gelu FAIL, swish PASS, gaussian FAIL.** Those verdicts stand as recorded, but they do not describe the figures. The C3 failures are the tail target plus the $0.01$-amplitude cells; the tanh C1 failure (median regret 3.58 against a bar of 3) came from dividing by the single lowest point of a noisy $10^{-14}$ floor at $N\ge192$, which the figures show as flat with the dots on it, and the constant pays the same (2.29). The criteria over-read floor noise; they are the defect, not the rule.
- **Against the constant (C4):** indistinguishable for tanh, gelu, swish (median regret ratios 1.14, 1.13, 1.02). For gaussian, v1 better (0.77): the constant sits on the wall on two $N=48$ mixtures (regret 4700 and 24000) that v1 brings to 48 and 3.6.
- **Native fp32** floors are 10 to 100x above the emulated $p=24$ floors, as known; the rule lands at the corner of the native curves.

## What was tested

See `PREREGISTRATION_anchor_v1.md` for the frozen rule, arms, targets, sweeps and criteria. In short: v1 (mean frequency, raw $B$, $\varepsilon_p=2^{1-p}$) against the constant, the amplitude-normalised v1n, and the top-line max (line targets only), on ten targets not used before, four activations, widths 48 to 384 at fp64 and seven emulated precisions plus native fp32 at $N=96$. 480 cells; 31 excluded because the smoothed minimum is above $10^{-3}$ (all $p=11$ cells except two, and the 5/7/11 mixture at $p=16$).

**Code & data.** `experiments/expC07_lambda_energy_rule/lambda_rule/anchor_v1_prereg.py` (`--predict`, `--run`); `data/anchor_v1_prereg_predictions.json`, `data/anchor_v1_prereg_rows_{width,precision}.json`, `data/anchor_v1_prereg_scores.json` (every cell, every arm, and the verdict block); figures `figures/anchor_v1_prereg_width.png`, `figures/anchor_v1_prereg_precision.png`, `figures/anchor_v1_prereg_regret.png`.

## Results

Per activation and sweep (v1 is the rule under test; const, v1n as declared; regret $=S(\lambda)/E_{\min}$ on the smoothed curve; on-wall $=$ exact fiber floor at $\lambda$ over the measured floor $>2$):

| act | sweep | n | v1 median | v1 p90 | v1 on-wall | C1 | C2 | C3 | const median | const p90 | const on-wall | v1n median | v1n p90 | v1n on-wall |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| tanh | width | 40 | 3.58 | 15.1 | 0.05 | no | yes | yes | 2.29 | 9.2 | 0.05 | 3.05 | 13.9 | 0.05 |
| tanh | precision | 71 | 1.85 | 13.0 | 0.11 | yes | yes | no | 1.22 | 3.0 | 0.01 | 1.58 | 4.9 | 0.04 |
| gelu | width | 40 | 1.43 | 3.1 | 0.07 | yes | yes | yes | 1.18 | 2.1 | 0.05 | 1.36 | 3.1 | 0.07 |
| gelu | precision | 69 | 1.38 | 4.2 | 0.16 | yes | yes | no | 1.04 | 1.8 | 0.00 | 1.27 | 2.8 | 0.10 |
| swish | width | 40 | 1.14 | 3.3 | 0.03 | yes | yes | yes | 1.07 | 1.5 | 0.00 | 1.11 | 2.9 | 0.03 |
| swish | precision | 69 | 1.10 | 2.7 | 0.07 | yes | yes | yes | 1.01 | 1.2 | 0.00 | 1.06 | 2.1 | 0.04 |
| gaussian | width | 40 | 3.63 | 18.6 | 0.15 | no | yes | no | 4.08 | 98.2 | 0.23 | 3.51 | 8.1 | 0.15 |
| gaussian | precision | 80 | 2.08 | 18.8 | 0.21 | yes | yes | no | 3.25 | 30.2 | 0.40 | 1.99 | 4.9 | 0.14 |

C4 (pooled over both sweeps): tanh median ratio 1.14, v1 better in 17/103, $p=2\times10^{-12}$; gelu 1.13, 11/103, $p<10^{-14}$; swish 1.02, 34/101, $p=0.001$; gaussian 0.77, 96/119, $p<10^{-10}$. Calls by the pre-registered thresholds: indistinguishable, indistinguishable, indistinguishable, v1 better.

C5 (cells with raw $B\notin[0.5,2]$): tanh v1 1.93 vs v1n 1.46 (n = 55); gelu 1.32 vs 1.26 (54); swish 1.06 vs 1.04 (54); gaussian 3.48 vs 2.69 (60). Paired median ratio v1n/v1 is 1.00 for $r\ge1$ and 1.18 for gaussian (v1n is better on the $B=0.02$ target and worse on the $B=3.3$ one).

**The tanh C1 failure is floor noise.** On the whole-line analytic targets at $N=192$ and $384$ the tanh floor is flat at about $10^{-14}$ to the eye and the dots sit on it; the smoothed curve's single lowest point ($\lambda=0.03$ to $0.07$, $10^{-15}$) is 4 to 40x lower, and regret was measured against that point. Both rules pay it equally (v1 4 to 43, const 4 to 26). This is not a tilted floor and not a property of the rule; it is the criterion reading the noise of the floor.

**The gelu and tanh C3 failures are v1 moving past the wall where the constant does not.** All on-wall $r\ge1$ cells are one of: the tail cosine sum (the $19\pi$ line at amplitude $10^{-3}$ sets the wall, and $\bar\omega=1.66\pi$ does not see it), the $B=0.02$ mixture (the raw-$B$ budget lets the ratio be 50× larger, so $\lambda$ moves right), or gelu on $1/(2+\cos3\pi x)$ at $p=32$ to $48$. The max arm (top line) puts the tail target back on the floor at $N\ge96$ but walks into the left wall at $N=48$ where $\theta_{\max}=2.49$; its p90 regrets are 210 (tanh) and 17 (swish) in the width sweep.

**Gaussian.** The constant is catastrophic on the amplitude-3.3 and equal-amplitude mixtures at $N=48$ (regret 4700 and 24000); v1 brings those to 48 and 3.6. v1 is still on the wall in 15 to 21% of cells, mainly the $B=0.02$ mixture at every precision and the tail cosine sum.

### Per-target breakdown ($r\ge1$ activations, both sweeps; 33 cells per target)

| target | v1 median | v1 p90 | v1 max | v1 on-wall | const median | const p90 | const max | const on-wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $3\sin3\pi x+0.3\sin9\pi x$ | 1.07 | 2.3 | 2.8 | 0 | 1.06 | 2.0 | 3.6 | 0 |
| $0.01(\sin2\pi x+\sin4\pi x)$ | 3.20 | 14.6 | 18.6 | 11 | 1.08 | 2.1 | 8.6 | 0 |
| tail cosine sum | 2.09 | 25.1 | 340.9 | 14 | 1.28 | 3.4 | 130.6 | 3 |
| $\sin5\pi x+\sin7\pi x+\sin11\pi x$ | 1.02 | 1.6 | 2.1 | 0 | 1.02 | 1.9 | 6.4 | 2 |
| $1/(1+9x^2)$ | 1.31 | 4.4 | 9.4 | 1 | 1.03 | 3.4 | 6.1 | 0 |
| $e^{-8x^2}$ | 1.43 | 4.9 | 16.7 | 0 | 1.11 | 2.8 | 17.2 | 0 |
| $1/(2+\cos3\pi x)$ | 1.41 | 2.6 | 7.5 | 4 | 1.09 | 1.9 | 2.9 | 0 |
| $1/((x+0.3)^2+0.09)$ | 1.18 | 3.3 | 13.4 | 0 | 1.09 | 2.8 | 13.2 | 0 |
| $\mathrm{sech}(6x)$ | 1.26 | 4.2 | 5.8 | 0 | 1.06 | 3.1 | 4.6 | 0 |
| $x e^{-4x^2}$ | 1.77 | 18.0 | 43.0 | 0 | 1.09 | 8.5 | 25.9 | 0 |

The maxima of 9 to 43 on the five smooth whole-line targets are floor noise on tanh at $N\ge192$ (the lowest point of a flat noisy floor), paid by both rules. On gaussian the same table has the constant at median 62 and max 24000 on the equal-amplitude mixture against v1's 3.3 and 7.2.

**Post hoc, labelled as such: criteria recomputed without the tail target.** tanh width median 3.77 (C1 still fails, floor noise), tanh precision on-wall 0.08 (C3 passes); gelu width all pass, gelu precision on-wall 0.11 (C3 fails by one cell); swish all pass; gaussian on-wall 0.11 and 0.18 (C3 fails, as predicted).

### Figures

- `anchor_v1_prereg_width.png`: 10 targets × 4 activations, one curve per $N$; solid vertical and dot at $\lambda_{v1}$, open diamond at $\lambda_{v1n}$ where it differs, faded dotted vertical at the constant. Look for: dots at the floor-wall corner for most cells; the tail-cosine and $B=0.02$ rows where the dot is up the wall; the tanh column's floors sloping down to the left at $N=192$, $384$.
- `anchor_v1_prereg_precision.png`: same layout at $N=96$, one curve per $p$ plus the native fp32 curve (dashed, square marker for v1), black dash-dot fiber floor. Look for: v1 tracking the wall with $p$ on the resolved targets; the $B=0.02$ row where v1 sits on the wall at every $p$ for tanh and gaussian; native fp32 floors well above the emulated $p=24$ floors.
- `anchor_v1_prereg_regret.png`: per activation, regret of v1 against regret of the constant, one point per cell (circles width, triangles precision), red where v1 is on the wall. Look for: the cloud above the diagonal for tanh and gelu (constant better), on it for swish, below it for gaussian.

## Conclusions

Pending Sam's review. The pre-registered verdicts are as stated in the TL;DR.

## Open questions

- The criteria measured regret against the global minimum of each curve, which for tanh at $N\ge192$ sits on a floor dip at $\lambda\approx0.05$ that no rule targets; a regret measured against the floor level at the wall's corner would have scored the same cells at about 1. Next pre-registration should use that.
- Whether v1n (normalised $B$) and a top-line gate ($2\omega_{\max}/N\le\pi/4$) together would pass C3 for $r\ge1$ is not tested here; both are declared arms or diagnostics, not the frozen rule.
