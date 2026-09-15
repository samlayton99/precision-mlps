# The output-ratio anchor rule: width and precision sweeps (2026-09-09)

**Status:** draft, pending Sam's review. Data-obvious statements only; no conclusions signed off.

## TL;DR

- The anchor rule $\lambda_{\rm anchor}=\sup\{\lambda: B\,\mathcal R_{K,r}(\lambda,N,\omega)\le\varepsilon\}$ was tested on 6 targets $\times$ 4 activations, across widths $N=32$ to $512$ at fp64 and across precisions $p=11$ to $53$ bits at $N=128$, with the exact per-cell predictions written before any solve. Two versions of the frequency input: $\omega$ = top frequency / $2\sigma$ of the spectral energy (**max**), and $\omega$ = $|b_j|$-weighted mean $|\omega_j|$ (**mean**).
- For resolved targets on tanh, gelu and swish the anchor lands on the floor at the corner of the wall for every width and every precision, within noise of where the constant rule $A_K(\lambda)=\varepsilon$ lands. Median anchor/wall 0.80 to 0.97 depending on activation and version.
- The **max** version walks into the conditioning (left) wall when the top line is under-resolved ($\theta_M\approx2$): error $10^{-2}$ at the anchor against a $10^{-9}$ minimum. The **mean** version fixes those cells but removes most of the margin on $r=2$ and all of it on $r=0$.
- On the gaussian activation ($r=0$, no attenuation factor) the mean version puts 15% of cells on the aliasing wall, up to $10^3\times$ above the floor. This is the only systematic overshoot; it is an $r=0$ effect, not a denominator-branch effect (Section "Branch check").

## Question

Does choosing $\lambda$ from the first-pair output ratio, with the target entering through $B=\sum|b_j|$ and one frequency $\omega$, land on the floor next to the wall as $N$ and the working precision change, and does it improve on the target-free constant?

## Experiment design

**Network and solve.** Uniform grid on $[-1,1]$, $h=2/N$, halo 32 centers per side, one shared $\gamma=\lambda/h$, features $\psi(\gamma(x-c_k))$ plus a bias column, least squares by `gelsd` with cutoff $\varepsilon_p$, relative $L_2$ on 4001 points. Precisions $p<53$ are emulated as in `precision_ladder.py` (inputs, coefficients and partial sums rounded to $p$ bits; SVD cutoff $2^{-p}$; solver internals not emulated), $p=53$ is native fp64.

**The rule.** With $K=\psi^{(r)}$, $\theta=2\omega/N$,
$$\mathcal R_{K,r}(\lambda)=\frac{\big(\tfrac{\theta}{2\pi-\theta}\big)^r\big|\widehat K\big(\tfrac{2\pi-\theta}{\lambda}\big)\big|+\big(\tfrac{\theta}{2\pi+\theta}\big)^r\big|\widehat K\big(\tfrac{2\pi+\theta}{\lambda}\big)\big|}{\min\{|\widehat K(0)|,\,|\widehat K(\theta/\lambda)|\}}\cdot\frac{1}{1-\rho_K},\qquad \rho_K=\frac{|\widehat K((4\pi-\theta)/\lambda)|}{|\widehat K((2\pi-\theta)/\lambda)|},$$
and $\lambda_{\rm anchor}$ is the largest $\lambda$ on a 600-point log grid with $B\,\mathcal R\le\varepsilon_p$, $\varepsilon_p=2^{-(p-1)}$. The old rule is $\lambda_{\rm const}$: $|\widehat K(2\pi/\lambda)|/|\widehat K(0)|=\varepsilon_p$. The implementation reproduces the values in Sam's table (mixed wave at $N=64,256$; sine at $N=64$; three activations) to the grid spacing (0.7%). Undefined when $\theta\ge\pi$.

**Activations.** tanh ($r=1$), gelu ($r=2$), swish ($r=2$), gaussian $\psi(x)=e^{-x^2}$ ($r=0$, $\widehat K(\xi)/\widehat K(0)=e^{-\xi^2/4}$).

**Targets and rule inputs.** Three mixtures $\sin k_1\pi x+\tfrac12\sin k_2\pi x+\tfrac14\sin k_3\pi x$ with $(k_1,k_2,k_3)=(1,3,5),(2,6,10),(4,12,20)$: $B=1.75$ exactly. Three non-tones with closed-form spectra: $1/(1+25x^2)$ ($\hat f=\tfrac{\pi}{5}e^{-|\omega|/5}$), $e^{-20x^2}$ ($\hat f\propto e^{-\omega^2/80}$), $e^{\sin3\pi x}$ ($|c_n|=I_n(1)$ at $3\pi n$).

| target | max version: $\omega$, $B$ | mean version: $\omega$, $B$ |
|---|---|---|
| mix 1/3/5 | $5\pi$, 1.75 | $2.14\pi$, 1.75 |
| mix 2/6/10 | $10\pi$, 1.75 | $4.29\pi$, 1.75 |
| mix 4/12/20 | $20\pi$, 1.75 | $8.57\pi$, 1.75 |
| $1/(1+25x^2)$ | $2.25\pi$ ($2\sigma$), 0.757 (mass inside; 0.24 outside) | $1.59\pi$, 1 |
| $e^{-20x^2}$ | $2.85\pi$ ($2\sigma$), 0.843 (0.16 outside) | $1.61\pi$, 1 |
| $e^{\sin3\pi x}$ | $3.54\pi$ ($2\sigma$), 2.40 (0.32 outside) | $2.02\pi$, $e$ |

Max version: $\sigma_\omega$ is the standard deviation of $|\hat f|^2$ and $B$ the $L^1$ mass of $\hat f/2\pi$ inside $|\omega|\le\omega_M$. Mean version: $\omega=\int|\omega||\hat f|/\int|\hat f|$ (same weights as $B$), $B$ the full mass.

**Sweeps.** Width: $N\in\{32,64,128,256,512\}$, fp64. Precision: $N=128$, $p\in\{11,16,24,32,40,48,53\}$. 40 $\lambda$ points in $[0.03,1.5]$; 4800 + 6720 solves, shared by both rule versions.

**Wall test.** Two measures per cell: the $10\times$-shoulder wall of the smoothed curve (noisy near the floor), and a noise-free one: the exact fiber floor evaluated at $\lambda_{\rm anchor}$ divided by the measured floor level (median of the smoothed curve within $5\times$ its minimum). A ratio above 1 means exact arithmetic cannot reach the measured floor at the anchor, i.e. the anchor is on the aliasing wall.

**Code & data.** `experiments/expC07_lambda_energy_rule/lambda_rule/anchor_rule.py` (`--omega-mean` for the mean version; `--predict-only`), `anchor_rule_branch_check.py`. `data/anchor_rule_predictions{,_mean}.json` (written before the solves), `data/anchor_rule_rows_{width,precision}.json`, `data/anchor_rule_summary{,_mean}.json`, `data/anchor_rule_branch_vs_wall.json`. Figures: `figures/anchor_rule_width{,_mean}.png`, `figures/anchor_rule_precision{,_mean}.png`, `figures/anchor_rule_branch_vs_wall.png`.

## Results

Medians over the cells with a defined anchor (E = relative $L_2$; $E_{\min}$ = minimum of the smoothed curve; wall = $10\times$ shoulder):

| version | sweep | act | anchor/wall | const/wall | $E$(anchor)/$E_{\min}$ | $E$(const)/$E_{\min}$ |
|---|---|---|---|---|---|---|
| max | width | tanh | 0.80 | 0.80 | 1.6 | 1.6 |
| max | width | gelu | 0.89 | 0.84 | 1.4 | 1.4 |
| max | width | swish | 0.74 | 0.66 | 1.1 | 1.4 |
| max | width | gaussian | 0.94 | 1.05 | 2.0 | 4.5 |
| max | precision | tanh | 0.82 | 0.78 | 1.2 | 1.3 |
| max | precision | gelu | 0.91 | 0.84 | 1.1 | 1.1 |
| max | precision | swish | 0.82 | 0.70 | 1.0 | 1.0 |
| max | precision | gaussian | 0.95 | 1.00 | 1.4 | 3.3 |
| mean | width | tanh | 0.85 | 0.80 | 1.4 | 1.5 |
| mean | width | gelu | 0.94 | 0.84 | 2.0 | 1.4 |
| mean | width | swish | 0.80 | 0.66 | 1.0 | 1.4 |
| mean | width | gaussian | 1.01 | 1.05 | 2.9 | 4.5 |
| mean | precision | tanh | 0.88 | 0.78 | 1.3 | 1.3 |
| mean | precision | gelu | 0.97 | 0.84 | 1.2 | 1.1 |
| mean | precision | swish | 0.87 | 0.70 | 1.0 | 1.0 |
| mean | precision | gaussian | 0.97 | 1.00 | 2.0 | 3.3 |

**Resolved mixtures ($\theta\le1$), all activations, both versions:** the dot sits at the corner where the floor meets the wall at every width and every precision; the anchor tracks the wall as $p$ changes.

**Under-resolved mixtures (max version):** mix 2/6/10 at $N=32$ and mix 4/12/20 at $N=64$ have $\theta_M=1.96$; the anchor moves to tanh 0.19 / 0.10 and lands on the left wall at $10^{-2}$ while the minimum is $10^{-9}$ near $\lambda\approx0.25$. Same failure mode as expC08's frequency-resolved rule: the denominator $\widehat K(\theta_M/\lambda)$ becomes $10^{-13}$ (the target's own fiber is nearly dead) and the rule reads that as aliasing. The mean version ($\bar\theta=0.84$ in those cells) puts the anchor back at the bottom of the V.

**Non-tones:** with either frequency input $\theta$ is small at every width, so the anchor is within a few percent of the constant and lands where it does.

**Gaussian activation ($r=0$):** the constant sits on the wall (const/wall 1.05); the max version's $(2\pi\mp\theta)$ shift moves the anchor to 0.94 of the wall with error $2.0\times$ the minimum instead of $4.5\times$. The mean version gives back that margin (anchor/wall 1.01) and 15% of its cells are on the aliasing wall by the noise-free test, up to $10^3\times$ above the floor (mix 2/6/10 at $N=32$, mix 4/12/20 at $N=64$ and $128$, $e^{-20x^2}$ at $N=32$; all with $\bar\theta/\lambda\gtrsim0.5$).

### Figures

- `anchor_rule_width.png` / `anchor_rule_width_mean.png`: 6 targets (rows) $\times$ 4 activations (columns), rel $L_2$ vs $\lambda$, one curve per $N$ (viridis). Solid vertical in the $N$ colour at $\lambda_{\rm anchor}$, dot at its intersection with that $N$'s curve, faint horizontal through the dot; grey dotted vertical at the $N$-independent constant. Look for: dots at the floor-wall corner in resolved cells; the max version's dots on the left wall in the two $\theta_M=1.96$ cells; gaussian dots on the wall in the mean version.
- `anchor_rule_precision.png` / `anchor_rule_precision_mean.png`: same layout at $N=128$, one curve per $p$; solid vertical + dot for the anchor, faded dotted vertical in the same colour for the constant at that $\varepsilon_p$, black dash-dot fiber floor (exact per-line sum for mixtures, expC09 projection for the non-tones). Look for: the anchor moving left with $p$ along the fiber curve; $p<53$ floors are the emulation's.
- `anchor_rule_branch_vs_wall.png`: $y$ = exact fiber floor at the anchor over the measured floor, $x=\theta/\lambda_{\rm anchor}$, red squares where the denominator was $\widehat K(0)$, blue circles where it was $\widehat K(\theta/\lambda)$; rows = max / mean. Look for: red and blue in the same band for gelu and swish; the on-wall points ($y>1$) almost all gaussian.

## Additional details

**Denominator branch.** tanh and gaussian transforms are monotone, so $\widehat K(\theta/\lambda)$ is always chosen. Gelu's transform exceeds $\widehat K(0)$ for $\xi<1.58$ and swish's for $\xi<0.86$, so $\widehat K(0)$ is chosen in 58/64 (max) and 64/64 (mean) gelu cells and 46/64 and 57/64 swish cells. Within gelu and swish the on-wall fraction is the same for both branches (0 to 5%); the branch does not predict overshoot. In resolved cells the two branches differ by a few percent.

**A floor step, not a wall.** For gelu at $N=128$, fp64, the measured floor steps from $10^{-14}$ to $1.3\times10^{-13}$ at $\lambda=0.55$ and is flat to the real wall at $0.85$, for Runge and $e^{-20x^2}$ alike, while the exact fiber floor stays at $1.6\times10^{-16}$ through $0.82$. The $10\times$-shoulder metric reads this as a wall at $0.50$ and flags the anchor at $0.78$ as an overshoot; the fiber test says it is on the floor's upper step. Cells flagged only by the shoulder metric should be read with this in mind.

**Emulation caveat.** For $p<53$ the floors are those of the emulation, which understates real reduced-precision floors (see `hardened_rule.md`, Section 7). What is trustworthy in the precision figures is the anchor's position relative to the wall at each $p$, not the floor level.

## Conclusions

Pending Sam's review.

## Open questions

- Which frequency input to standardise on: the max protects $r=0$ and fails under-resolved cells; the mean does the reverse. A margin factor on $B$ or a cap $\theta\le\pi/4$ would separate the two failure modes; not tested.
- The gelu floor step at $\lambda\approx0.55$ ($N=128$, fp64) is unexplained and affects any shoulder-based wall metric.
