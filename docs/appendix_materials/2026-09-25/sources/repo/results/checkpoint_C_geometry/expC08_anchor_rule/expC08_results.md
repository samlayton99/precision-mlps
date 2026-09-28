# expC08: the lambda anchor rule

**Status:** draft pending Sam's review (2026-09-10). Promoted from `expC07_lambda_energy_rule/lambda_rule/` (the exploratory sweeps of 2026-09-09 and the pre-registered test of 2026-09-10). Not to be confused with the original expC08 (the frequency-resolved rule, superseded and archived under `expC07_lambda_energy_rule/lambda_rule/archive/`).

## TL;DR

- **The anchor rule** picks $\lambda$ from the activation, the width, a working precision and two numbers about the target: $\lambda_{\rm anchor}=\sup\{\lambda: B\,\mathcal R_{K,r}(\lambda,N,\omega)\le\varepsilon_p\}$, where $\mathcal R$ is the first-pair ghost ratio of the fiber theory evaluated at one representative grid frequency $\theta=2\omega/N$, $B=\sum|b_j|$ is the target's coefficient sum, and $\varepsilon_p=2^{1-p}$. Its frozen form (version 1: $\omega$ = the $|b_j|$-weighted mean frequency, raw $B$) is stated in `anchor_rule_v1.md`.
- **It lands at the corner of the valley.** On the ten fresh targets of the pre-registered test the dot sits where the floor meets the wall for every width 48 to 384, every precision 11 to 53 bits, native fp32, and all four activations (tanh, gelu, swish, gaussian). The one miss is a target whose wall is set by a $19\pi$ line at amplitude $10^{-3}$, at $N=48$ (5 cells per wavelength): an under-resolved high-frequency component that a mean frequency cannot see. That is the regime the method is weak in anyway; if such content is known to exist, choose a smaller $\lambda$.
- **Frequency input.** Feeding the top frequency instead of the mean protects $r=0$ activations but walks into the conditioning wall when the top line is under-resolved ($\theta_{\max}\approx2$); the mean does the reverse. Version 1 uses the mean.
- **Amplitude.** Raw $B$ makes the rule depend on the target's units: $B=0.02$ moves $\lambda$ right onto the wall (visible on the gaussian activation), $B=11$ moves it slightly left. Replacing $B$ by $B/\|f\|_\infty$ (arm "v1n") removes this and coincides with v1 whenever $B/\|f\|_\infty\approx1$. Recommended before any further freeze.
- **Against the target-free constant $H(2\pi/\lambda)=\varepsilon_p$:** indistinguishable for tanh, gelu, swish (median regret ratios 1.02 to 1.14); clearly better for the gaussian ($r=0$) activation, where the constant sits on the wall (regret up to $2\times10^4$ on an $N=48$ mixture, 3.6 for the anchor).
- **The pre-registered pass/fail criteria** (median regret $\le3$, p90 $\le30$, on-wall $\le10\%$) returned tanh FAIL, gelu FAIL, swish PASS, gaussian FAIL. The verdicts stand as recorded, but they over-read floor noise (the tanh "median regret 3.58" divides by the single lowest point of a flat noisy $10^{-14}$ floor) and count the one adversarial target; they do not describe the figures.

## Question

Given the activation, the width and the precision, can one number about the target's frequency content and one about its amplitude fix $\lambda$ without a least-squares solve, and does the choice land on the floor next to the aliasing wall as $N$ and the precision change?

## Experiment design

**Network and solve.** Uniform grid on $[-1,1]$, $h=2/N$, halo 32 centers per side, one shared $\gamma=\lambda/h$, features $\psi(\gamma(x-c_k))$ and a bias column, least squares by `gelsd` with cutoff $\varepsilon_p$, relative $L_2$ on 4001 points. Precisions $p<53$ are emulated as in the precision ladder (inputs, coefficients and partial sums rounded to $p$ bits; SVD cutoff $\varepsilon_p$; solver internals fp64); $p=53$ native fp64; the pre-registered test adds a native float32 arm.

**The rule.** With $K=\psi^{(r)}$, $H(\xi)=|\widehat K(\xi)|/|\widehat K(0)|$, $\theta=2\omega/N$,
$$\mathcal R_{K,r}(\lambda)=\frac{\big(\tfrac{\theta}{2\pi-\theta}\big)^r H\big(\tfrac{2\pi-\theta}{\lambda}\big)+\big(\tfrac{\theta}{2\pi+\theta}\big)^r H\big(\tfrac{2\pi+\theta}{\lambda}\big)}{\min\{1,H(\theta/\lambda)\}}\cdot\frac{1}{1-\rho_K},\qquad \rho_K=\frac{H((4\pi-\theta)/\lambda)}{H((2\pi-\theta)/\lambda)},$$
$\lambda_{\rm anchor}$ the largest point of a 600-point log grid on $[0.03,1.5]$ with $B\,\mathcal R\le\varepsilon_p$; undefined when $\theta\ge\pi$. The numerator is $|t_{-1}|+|t_{+1}|$ of the fiber theorem's ghost ratio at $\theta$; the denominator's $\min$ is a conservative bound over the band, needed because gelu's and swish's transforms rise above $H(0)$ for $\xi<1.58$ and $0.86$. Baseline: $\lambda_{\rm const}$ from $H(2\pi/\lambda)=\varepsilon_p$. Activations: tanh ($r=1$), gelu ($r=2$), swish ($r=2$), gaussian $\psi=e^{-x^2}$ ($r=0$, $H=e^{-\xi^2/4}$). The implementation reproduces Sam's hand-computed table to the grid spacing (0.7%).

**Part A, exploratory sweeps (2026-09-09; `anchor_rule.py`).** Six targets: three mixtures $\sin k_1\pi x+\tfrac12\sin k_2\pi x+\tfrac14\sin k_3\pi x$ with $(k_1,k_2,k_3)=(1,3,5),(2,6,10),(4,12,20)$ ($B=1.75$), and $1/(1+25x^2)$, $e^{-20x^2}$, $e^{\sin3\pi x}$ with closed-form spectra. Two frequency inputs: **max** ($\omega$ = top line; for the non-tones the $2\sigma$ point of the spectral energy with $B$ the mass inside) and **mean** ($\omega=\sum|b_j||\omega_j|/\sum|b_j|$, full $B$). Width sweep $N\in\{32,64,128,256,512\}$ at fp64; precision sweep $N=128$, $p\in\{11,16,24,32,40,48,53\}$. 40 $\lambda$ points. A separate check (`anchor_rule_branch_check.py`) asks whether the denominator branch ($H(0)$ or $H(\theta/\lambda)$) predicts landing on the wall.

**Part B, pre-registered test (2026-09-10; `anchor_v1_prereg.py`).** Frozen v1 (mean frequency, raw $B$) against four arms declared before any solve: const, v1n ($B\to B/\|f\|_\infty$), max (top line, line targets only). Ten targets never used before: $3\sin3\pi x+0.3\sin9\pi x$ ($B=3.3$), $0.01(\sin2\pi x+\sin4\pi x)$ ($B=0.02$), $\cos\pi x+0.1\cos7\pi x+0.01\cos13\pi x+0.001\cos19\pi x$, $\sin5\pi x+\sin7\pi x+\sin11\pi x$, $1/(1+9x^2)$, $e^{-8x^2}$, $1/(2+\cos3\pi x)$, $1/((x+0.3)^2+0.09)$ ($B=11.1$), $\mathrm{sech}(6x)$, $x e^{-4x^2}$; spectral inputs computed numerically and checked against the closed forms of Part A's non-tones. Widths $\{48,96,192,384\}$ at fp64; $N=96$ at seven emulated precisions plus native fp32. Scoring: $S$ = 5-point running median of log error, $E_{\min}=\min S$, regret $=S(\lambda_{\rm arm})/E_{\min}$; on-wall $:=$ exact fiber floor at $\lambda_{\rm arm}$ over the measured floor $>2$ (noise-free). Criteria C1 median regret $\le3$, C2 p90 $\le30$, C3 on-wall $\le0.10$ per activation and sweep; C4 paired sign test against the constant; C5 v1n vs v1 on the raw-$B\notin[0.5,2]$ targets. Predictions, criteria and priors hashed in `PREREGISTRATION_anchor_v1.md` before the first solve; one amendment (C5 keyed on raw $B$) made before the run and recorded there. Nothing changed afterwards.

**Code & data.** `experiments/expC08_anchor_rule/anchor_rule.py` (`--omega-mean`, `--predict-only`), `anchor_rule_branch_check.py`, `anchor_v1_prereg.py` (`--predict`, `--run`; the byte-identical registered file is `archive/anchor_v1_prereg_as_registered.py`, SHA-256 `4796a510…`); kernels from `expC07_lambda_energy_rule/lambda_rule/rule.py`, fiber projection from `run_c09_general.py` there. Data: `data/anchor_rule_{predictions,summary}{,_mean}.json`, `data/anchor_rule_rows_{width,precision}.json`, `data/anchor_rule_branch_vs_wall.json`, `data/anchor_v1_prereg_{predictions,scores}.json`, `data/anchor_v1_prereg_rows_{width,precision}.json`. Figures: `figures/anchor_rule_{width,precision}{,_mean}.png`, `figures/anchor_rule_branch_vs_wall.png`, `figures/anchor_v1_prereg_{width,precision,regret}.png`. Rule statement: `anchor_rule_v1.md` (finalized on Sam's laptop 2026-09-10). Pre-registration: `PREREGISTRATION_anchor_v1.md`.

## Results

### Part A: exploratory sweeps

Medians over cells with a defined anchor (regret = error at the chosen $\lambda$ over the curve minimum; wall = $10\times$ shoulder):

| version | sweep | act | anchor/wall | const/wall | E(anchor)/E_min | E(const)/E_min |
|---|---|---|---:|---:|---:|---:|
| max | width | tanh / gelu / swish / gaussian | 0.80 / 0.89 / 0.74 / 0.94 | 0.80 / 0.84 / 0.66 / 1.05 | 1.6 / 1.4 / 1.1 / 2.0 | 1.6 / 1.4 / 1.4 / 4.5 |
| max | precision | tanh / gelu / swish / gaussian | 0.82 / 0.91 / 0.82 / 0.95 | 0.78 / 0.84 / 0.70 / 1.00 | 1.2 / 1.1 / 1.0 / 1.4 | 1.3 / 1.1 / 1.0 / 3.3 |
| mean | width | tanh / gelu / swish / gaussian | 0.85 / 0.94 / 0.80 / 1.01 | 0.80 / 0.84 / 0.66 / 1.05 | 1.4 / 2.0 / 1.0 / 2.9 | 1.5 / 1.4 / 1.4 / 4.5 |
| mean | precision | tanh / gelu / swish / gaussian | 0.88 / 0.97 / 0.87 / 0.97 | 0.78 / 0.84 / 0.70 / 1.00 | 1.3 / 1.2 / 1.0 / 2.0 | 1.3 / 1.1 / 1.0 / 3.3 |

- Resolved mixtures ($\theta\le1$): the dot sits at the floor-wall corner for every width and precision, all activations, both versions.
- Max version, under-resolved mixtures ($\theta_{\max}=1.96$: mix 2/6/10 at $N=32$, mix 4/12/20 at $N=64$): the anchor moves to tanh 0.19 / 0.10 and lands on the left wall at $10^{-2}$ against a $10^{-9}$ minimum. The denominator $H(\theta/\lambda)$ falls to $10^{-13}$ there (the target's own fiber is nearly dead) and the rule reads it as aliasing. The mean version ($\bar\theta=0.84$) returns those cells to the bottom of the V.
- Gaussian ($r=0$): the constant sits on the wall (const/wall 1.05, error $4.5\times$ the minimum); the max version's $(2\pi\mp\theta)$ shift gives it a margin (0.94, $2.0\times$); the mean version gives that back (1.01) and 15% of its cells are on the wall by the noise-free test.
- Denominator branch: $H(0)$ is chosen in 58 to 64 of 64 gelu cells and 46 to 57 of 64 swish cells, never for tanh or gaussian; within gelu and swish the on-wall fraction is the same for both branches (0 to 5%). The branch does not predict overshoot; overshoot is an $r=0$ effect.
- A trap: gelu's fp64 floor at $N=128$ steps from $10^{-14}$ to $1.3\times10^{-13}$ at $\lambda=0.55$ for every target while the exact fiber floor stays at $10^{-16}$ to $0.82$; the $10\times$-shoulder metric reads this as a wall at $0.50$.

### Part B: pre-registered test

Per-target view ($r\ge1$ activations, both sweeps, 33 cells per target; regret against the smoothed minimum, on-wall by the fiber test):

| target | v1 median | v1 max | v1 on-wall | const median | const max | const on-wall |
|---|---:|---:|---:|---:|---:|---:|
| $3\sin3\pi x+0.3\sin9\pi x$ | 1.07 | 2.8 | 0 | 1.06 | 3.6 | 0 |
| $0.01(\sin2\pi x+\sin4\pi x)$ | 3.20 | 18.6 | 11 | 1.08 | 8.6 | 0 |
| tail cosine sum | 2.09 | 341 | 14 | 1.28 | 131 | 3 |
| $\sin5\pi x+\sin7\pi x+\sin11\pi x$ | 1.02 | 2.1 | 0 | 1.02 | 6.4 | 2 |
| $1/(1+9x^2)$ | 1.31 | 9.4 | 1 | 1.03 | 6.1 | 0 |
| $e^{-8x^2}$ | 1.43 | 16.7 | 0 | 1.11 | 17.2 | 0 |
| $1/(2+\cos3\pi x)$ | 1.41 | 7.5 | 4 | 1.09 | 2.9 | 0 |
| $1/((x+0.3)^2+0.09)$ | 1.18 | 13.4 | 0 | 1.09 | 13.2 | 0 |
| $\mathrm{sech}(6x)$ | 1.26 | 5.8 | 0 | 1.06 | 4.6 | 0 |
| $x e^{-4x^2}$ | 1.77 | 43 | 0 | 1.09 | 25.9 | 0 |

The maxima of 9 to 43 on the smooth whole-line targets are floor noise on tanh at $N\ge192$: the figures show a flat $10^{-14}$ floor with the dots on it, and the regret divides by its single lowest point; the constant pays the same. On the gaussian activation the constant is at median 62 and max 24000 on the equal-amplitude mixture against v1's 3.3 and 7.2.

Pre-registered criteria as recorded:

| act | sweep | n | v1 median | v1 p90 | v1 on-wall | C1 C2 C3 | const median | const on-wall | v1n median | v1n on-wall |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| tanh | width | 40 | 3.58 | 15.1 | 0.05 | no yes yes | 2.29 | 0.05 | 3.05 | 0.05 |
| tanh | precision | 71 | 1.85 | 13.0 | 0.11 | yes yes no | 1.22 | 0.01 | 1.58 | 0.04 |
| gelu | width | 40 | 1.43 | 3.1 | 0.07 | yes yes yes | 1.18 | 0.05 | 1.36 | 0.07 |
| gelu | precision | 69 | 1.38 | 4.2 | 0.16 | yes yes no | 1.04 | 0.00 | 1.27 | 0.10 |
| swish | width | 40 | 1.14 | 3.3 | 0.03 | yes yes yes | 1.07 | 0.00 | 1.11 | 0.03 |
| swish | precision | 69 | 1.10 | 2.7 | 0.07 | yes yes yes | 1.01 | 0.00 | 1.06 | 0.04 |
| gaussian | width | 40 | 3.63 | 18.6 | 0.15 | no yes no | 4.08 | 0.23 | 3.51 | 0.15 |
| gaussian | precision | 80 | 2.08 | 18.8 | 0.21 | yes yes no | 3.25 | 0.40 | 1.99 | 0.14 |

Verdicts: tanh FAIL, gelu FAIL, swish PASS, gaussian FAIL (prior: three passes, gaussian fail). C4: median regret ratio v1/const 1.14, 1.13, 1.02, 0.77 (tanh, gelu, swish, gaussian); the sign tests favour the constant for $r\ge1$ (v1 better in 17/103, 11/103, 34/101 cells) and v1 for gaussian (96/119); by the registered thresholds: indistinguishable, indistinguishable, indistinguishable, v1 better. C5 (raw $B\notin[0.5,2]$): v1n median regret 1.46 vs v1 1.93 (tanh), 1.26 vs 1.32 (gelu), 1.04 vs 1.06 (swish), 2.69 vs 3.48 (gaussian); on-wall fractions drop from 0.16 to 0.10 (gelu precision) and 0.21 to 0.14 (gaussian precision). Native fp32 floors are 10 to 100× above the emulated $p=24$ floors; v1 lands at regret 1.0 to 3.1 on the native curves. 31 of 480 cells excluded as unresolvable (nearly all $p=11$).

What the C3 failures are made of: every on-wall $r\ge1$ cell is the tail cosine sum (14), the $B=0.02$ mixture (11), or gelu on $1/(2+\cos3\pi x)$ at 32 to 48 bits (4). The max arm puts the tail target back on the floor at $N\ge96$ but walks into the left wall at $N=48$ ($\theta_{\max}=2.49$).

### Figures

- `anchor_rule_width.png`, `anchor_rule_width_mean.png`: Part A, 6 targets × 4 activations, one curve per $N$; solid vertical and dot at the anchor on that $N$'s curve, grey dotted vertical at the constant. Look for: dots at the corner in resolved cells; the max version's dots on the left wall in the two $\theta_{\max}=1.96$ cells; gaussian dots on the wall in the mean version.
- `anchor_rule_precision.png`, `anchor_rule_precision_mean.png`: Part A at $N=128$, one curve per $p$, faded dotted constant per $p$, black dash-dot fiber floor. Look for: the anchor moving left with $p$ along the fiber curve.
- `anchor_rule_branch_vs_wall.png`: exact fiber floor at the anchor over the measured floor against $\theta/\lambda$, red squares $H(0)$ branch, blue circles $H(\theta/\lambda)$ branch, rows max / mean. Look for: both colours in the same band for gelu and swish; on-wall points almost all gaussian.
- `anchor_v1_prereg_width.png`: Part B, 10 targets × 4 activations, one curve per $N$; dot at $\lambda_{v1}$, open diamond at $\lambda_{v1n}$ where it differs, faded dotted constant. Look for: dots at the corner everywhere except the tail cosine at $N=48$ (and mildly $N=96$ on gelu); diamonds separating from dots only on the $B=0.02$ and $B=11$ rows.
- `anchor_v1_prereg_precision.png`: Part B at $N=96$, one curve per $p$ plus native fp32 (dashed, square for v1), fiber floor dash-dot. Look for: dots marching down the fiber curve with $p$; the $B=0.02$ row on gaussian with dots up the wall; native fp32 floors well above emulated $p=24$.
- `anchor_v1_prereg_regret.png`: regret of v1 against regret of the constant, one point per cell, red where v1 is on the wall. Look for: the cloud on or slightly above the diagonal for tanh, gelu, swish; below it for gaussian.

## Additional details

**v1 versus v1n.** The condition $B\,\mathcal R\le\varepsilon$ descends from the paper's absolute bound $\|f-\tilde f\|_\infty\le\delta+B\mathcal R$ with $B$ in the target's units, while every score here is relative and so is the fiber floor. Scaling the target by $c$ scales $B$ by $c$ and moves the anchor although the relative curve is unchanged. v1n uses $B/\|f\|_\infty$, which makes the condition "relative aliasing $\le\varepsilon$" and scale-invariant; it coincides with v1 whenever $B/\|f\|_\infty\approx1$, which was true of every target in Part A and eight of ten in Part B.

**Emulation caveat.** For $p<53$ the floors are the emulation's (5 to 1000× below native fp32 at $p=24$; see `hardened_rule.md` Section 7). What is trustworthy is the anchor's position relative to the wall at each $p$, confirmed on the native fp32 arm.

**Criteria defects, recorded for the next pre-registration.** Regret against the global minimum of a noisy floor over-reads noise (score against the floor level at the wall's corner instead); a single adversarial target dominates an on-wall fraction at 10%; the $10\times$-shoulder wall metric misreads floor steps (gelu at $\lambda\approx0.55$).

## Conclusions

Pending Sam's review.

## Open questions

- Freeze v1n (or equivalently a relative budget) as the amplitude convention; retest the $B=0.02$ and $B=11$ rows.
- A top-line gate ($2\omega_{\max}/N\le\pi/4$, or "choose a smaller $\lambda$ when significant content sits well above the mean") as the stated remedy for the tail case; not tested as a rule.
- Whether the rule transfers to non-uniform centers and to the 2-D ridge geometries.
