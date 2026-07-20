# expA07 -- The inner-band norm rule: how big should the readout start?

**Status: draft -- rule proposed by Sam, verified below; conclusions pending sign-off.**

## TL;DR

- The inner-band readout norm follows a **zero-free-parameter law** once the width resolves the target: $\|v_{inner}\|_2 = \|f'\|_{L_2[-1,1]}/\sqrt{2N}$ for tanh (slope $-1/2$; prefactor lands within 0.2--3% on all 6 targets) and $(2/N)^{3/2}\|f''\|/\lambda$ for gelu (slope $-3/2$; holds for 5/6, breaks on the rough target).
- **Two regimes**: an under-resolution transient where the norm sits far above the law, then a clean lock onto the $-1/2$ line once $N$ resolves the target.
- **The practical initializer**: a Gaussian vector normalized to **vector norm** $\|v\|_2 = 1/\sqrt N$ (interior nodes; halo 0) captures most of the benefit -- it beats standard init everywhere, keeps improving with width (standard init does not), and is within $\sim 2\times$ of the full derivative-shaped init at $N=512$.
- Standard PyTorch readout init has $\Theta(1)$ vector norm independent of width -- exactly the wrong scaling, which is why random init plateaus as $N$ grows (expD06).

## Question

What magnitude should the readout be initialized at for regression -- and is there a single rule across targets?

## Experiment design

**Norm sweep.** SVD min-norm readout $[\Phi,\mathbf 1][v;b]\approx y$ on the frozen QI geometry ($\lambda^*=0.25$, halo default, $n_{train}=\max(2003, 2W{+}3)$ equispaced), recording $\|v_{inner}\|_2$ over the centers in $[-1,1]$ only (halo excluded). Grid: $N = 32,64,\dots,1024$ (linspace, 32 points), 6-target family, activations tanh and gelu. Candidate laws (power, power-to-floor, exponential, power-log) fitted to $\log_{10}\|v_{inner}\|$; separately, the **zero-parameter predictions** from the expA06 derivative law:

- tanh ($v_k \approx \tfrac h2 f'(c_k)$): $\|v_{inner}\| \approx \sqrt{\tfrac h4 \int f'^2} = \|f'\|_{L_2}/\sqrt{2N}$.
- gelu ($v_k \approx \tfrac{h^2}{\lambda} f''(c_k)$): $\|v_{inner}\| \approx (2/N)^{3/2}\,\|f''\|_{L_2}/\lambda$.

Tail slopes are fitted on $N\ge 384$ (past the under-resolution transient).

**Shape-vs-magnitude control (run under expD06).** Two Gaussian-direction variants added to the expD06 protocol (train everything, full-batch Adam, warmup 100, 1000 steps): `seed6` = Gaussian at the calibrated magnitude $\|p/a\|$, `seed7` = Gaussian at bare $1/\sqrt N$; both interior-only with halo 0. Compared against the derivative-shaped inits and standard random init.

**Code & data.** `experiments/expA07_inner_norm_rule/run.py`; data `results/checkpoint_A_numerics/expA07_inner_norm_rule/{expA07_rows.json, expA07_fits.json}`; controls in `experiments/expD06_derivative_readout_init/run.py` (`--seeds seed6_gauss_calibmag,seed7_gauss_sqrtN`), data `results/checkpoint_D_optimizers/expD06_derivative_readout_init/expD06_rows.json`.

## Results

**tanh: the law holds for every target.** Tail slopes $-0.50$ to $-0.66$ (prediction $-1/2$); the known prefactor lands at data/prediction ratios 1.000--1.030 at $N=1024$. The same rule fits all six targets -- the target enters only through the single number $\|f'\|_{L_2}$.

**gelu: slope $-3/2$ confirmed for 5/6** (tail slopes $-1.37$ to $-1.50$, prefactor ratios 0.95--1.001); `abs_cubed` breaks entirely (ratio $\sim 10^4$) -- its $f''=6|x|$ is too rough for the ramp-kernel law, the same family as the runge-gelu anomaly in expA06. Smoothness requirement: the *kernel-relevant derivative order* must be smooth; tanh (first order) is forgiving, gelu (second) is not. *(Update, smoothness stress test below: the abs\_cubed break was a bandwidth artifact -- at gelu's aliasing-rule $\lambda=0.707$ (expC07) instead of 0.25, gelu locks onto the law on abs\_cubed with ratio 0.996.)*

**Two regimes.** Below the resolution width the norm rides the geometric under-resolution transient orders of magnitude above the law (sine/exp are on the line from $N=32$; sine\_8pi joins near $N\approx200$; mixture/abs\_cubed near $N\approx300$--$400$). The best whole-range empirical fit (power-log, $R^2$ 0.93--1.00) is just transient+asymptote blended; the physical law is the $-1/2$ power.

**Shape vs magnitude** (median final eval rel $L_2$ over 6 targets, 1000 Adam steps, from expD06):

| init | shape | magnitude | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---:|---:|---:|
| calibrated | $f'$ | $\|p/a\|$ | 1.2e-2 | 3.9e-3 | 1.9e-3 |
| noisy | $f'$+12.5% noise | $\|p/a\|$ | 1.4e-2 | 3.3e-3 | 2.5e-3 |
| gauss-calibmag | random | $\|p/a\|$ | 2.5e-1 | 5.5e-3 | 3.5e-3 |
| **gauss-$1/\sqrt N$** | random | $1/\sqrt N$ | 9.3e-3 | 4.7e-3 | 4.2e-3 |
| standard random | random | $\Theta(1)$ | 1.2e-2 | 1.6e-2 | 1.3e-2 |

The magnitude scaling is the dominant lever: bare $1/\sqrt N$ with a random direction recovers most of the gap to the shaped init and, unlike standard init, keeps improving with width. The derivative shape adds a further $\sim2\times$ at large $N$ and much faster early convergence. The $\|f'\|$ prefactor without the shape helps at large $N$ but hurts in the transient regime (gauss-calibmag at $N{=}128$).

### Figures

- **`figures/expA07_inner_norm_{tanh,gelu}.png`** -- 2x3 log-log grids, one panel per target: blue dots = measured $\|v_{inner}\|_2$ vs $N$, red = best empirical fit (title shows the $N\ge384$ tail slope), green dashed = the zero-parameter derivative-law prediction. Look for the data collapsing onto the green line once past the transient; on the gelu figure note `abs_cubed` never joins its line.

## Smoothness stress test (expA07/smoothness) -- does the law need $f^{(r)}$ to exist?

**Status: data reported; conclusions pending Sam.**

**Question.** The norm law's input is $\|f^{(r)}\|_{L_2}$ with $r$ the kernel order. Does its predictive power vanish when the target is rougher than the kernel order -- and does it generalize across all three kernel orders at the expC07 aliasing-rule bandwidths?

**Design.** The generalized zero-parameter law, $\|v_{inner}\|_2 = C_\psi\,\|f^{(r)}\|_{L_2}\,(2/N)^{r-1/2}$, derived per order: $r{=}0$ (bumps; Riemann/delta argument) $v_k\approx\tfrac{\lambda}{M}f(c_k)$, $C=\lambda/M$ with $M=\int K$ (sech$^2$: 2, $e^{-x^2}$: $\sqrt\pi$), slope $+1/2$; $r{=}1$ (steps; telescoping) $v_k\approx\tfrac{h}{\Delta}f'(c_k)$, $C=1/\Delta$ with $\Delta=\psi(\infty)-\psi(-\infty)$ (tanh 2, sigmoid 1), slope $-1/2$; $r{=}2$ (ramps; slope-change matching) $v_k\approx\tfrac{h^2}{s\lambda}f''(c_k)$, $C=1/(s\lambda)$ with ramp slope $s{=}1$ (gelu, swish), slope $-3/2$. Each activation runs at its expC07 aliasing-rule $\lambda$ (sech$^2$ 0.25, gaussian 0.530, tanh 0.25, sigmoid 0.50, gelu 0.707, swish 0.455). Same protocol as the main sweep ($[\Phi,\mathbf 1]$ SVD lstsq, inner-band norm, $N=32..1024$ step 32, tail slopes on $N\ge384$), plus eval rel $L_2$ per cell. Targets: two sets of three spanning smoothness classes -- set A: $|x|$ ($C^0$), $x|x|$ ($C^1$), $|x|^3$ ($C^2$); set B: the same classes built on $\sin(2\pi x)$ ($|s|$, $s|s|$, $|s|^3$). For $C^0$ targets $f''$ contains Dirac deltas, so the $r{=}2$ law is undefined ($\|f''\|_{L_2}=\infty$) -- the cells where prediction should be lost outright.

**Code & data.** `experiments/expA07_inner_norm_rule/smoothness/run.py`; data `results/checkpoint_A_numerics/expA07_inner_norm_rule/smoothness/expA07s_rows.json`; figures `smoothness/figures/expA07s_norm_law_set{A,B}.png`.

**Results.** Read from the compensated figures (measured/predicted ratio vs $N$; on the law = flat at 1). Lock status by (kernel order $\times$ smoothness class), ratio at $N{=}1024$ (set A / set B):

| | $C^0$ | $C^1$ | $C^2$ |
|---|---|---|---|
| $r{=}0$ sech$^2$ | far above ($10^8$) | far above, noisy ($10^4$) | touches 1 near $N\approx400$, then departs upward, noisy (6.6) |
| $r{=}0$ gaussian | far above ($2\cdot10^4$/$3\cdot10^5$) | approaching (1.3 / 190) | **locked** from $N\approx300$ (scatter 0.75--1.3 / 10) |
| $r{=}1$ tanh, sigmoid | off-law, ratio $\sim2\cdot10^4$ declining slowly | above, converging (39 / 690, slope $-2$) | **locked** from $N\approx300$ (1.000 / 4.6) |
| $r{=}2$ gelu | law undefined | above, noisy (420 / $1.4\cdot10^3$) | **locked** from $N\approx250$ (0.996 / 5.5) |
| $r{=}2$ swish | law undefined | above ($10^4$) | approaching (11 / 270) |

- **When the target is smooth enough, all three kernel orders lock onto their zero-parameter laws** (gaussian, tanh/sigmoid, gelu on $|x|^3$: flat-at-1 from $N\approx250$--$400$; spot checks at $N{=}1024$: tanh law $\|3x|x|\|_{L_2}/\sqrt{2048}=0.0419$ vs measured $0.0421$; gelu law $(2/N)^{3/2}\cdot\sqrt{24}/0.707=5.98\times10^{-4}$ vs measured $5.96\times10^{-4}$).
- **Below the required smoothness, the data is still lawful -- just not the predicted law.** The norm follows clean, *steeper* power laws sitting above the prediction (tanh on $x|x|$: slope $-2$ vs law $-1/2$; on $|x|$: slope $-1$), i.e. the transient never ends within reach: the lock width grows rapidly with the roughness of $f^{(r)}$. Where $f^{(r)}$ has delta content the law has no input at all and norms are large and erratic. Set B (multi-kink) repeats set A with larger prefactors and later locks.
- **Eval error is activation-independent; norm is not.** At fixed (target, $N$) all six activations sit at the same rel $L_2$ (e.g. $|x|^3$ at $N{=}1024$: all $\approx2.4\times10^{-11}$) while their inner norms span 4+ decades (gelu $6\times10^{-4}$, tanh $4\times10^{-2}$, gaussian 2.7, sech$^2$ 10). The error is set by target smoothness and width alone; the norm law is about *which coefficients* buy that error.
- **The gelu abs\_cubed anomaly of the main sweep is resolved**: at $\lambda=0.707$ gelu locks (0.996) where at $\lambda=0.25$ it broke by $10^4$ -- the aliasing-rule bandwidth, not the target, was the binding constraint there.
**Theory: the master formula (added after the sweep; verified against it).** On the uniform grid the dense-sample lstsq operator is Toeplitz, so it diagonalizes over grid frequencies $\theta=\omega h\in[-\pi,\pi]$; each fiber is a scalar least-squares over the aliases $\theta+2\pi m$, and where the principal band dominates the min-norm solution is pure deconvolution $V(\theta)=F(\theta)/\hat g(\theta)$ with $\hat g(\theta)=\lambda^{r-1}\hat K(\theta/\lambda)/(i\theta)^r$. Parseval then gives

$$\|v\|_2^2=\frac{h^{2r-1}}{2\pi\lambda^{2r-2}}\int_{|\omega|\le\pi/h}\frac{|\widehat{f^{(r)}}(\omega)|^2}{|\hat K(\omega h/\lambda)|^2}\,d\omega,$$

one integral with two regimes:

- **Law regime.** Spectrum of $f^{(r)}$ concentrated where $\hat K\approx\hat K(0)=M$: $\|v\|=\frac{\lambda^{1-r}}{M}\|f^{(r)}\|_{L_2}h^{r-1/2}$ -- the zero-parameter law, with one prefactor $\lambda^{1-r}/M$ unifying all three orders ($M$ = kernel mass = step rise $\Delta$ at $r{=}1$, ramp slope $s$ at $r{=}2$).
- **Edge (Nyquist-deconvolution) regime.** A jump of strength $J$ in $f^{(s)}$ gives $|\widehat{f^{(r)}}|\sim J\omega^{-q}$, $q=s+1-r$; the exponentially growing $1/|\hat K|$ makes the integral endpoint-dominated at $\omega=\pi/h$: $\|v\|\approx J(h/\pi)^q\,\lambda^{1-r}\pi^r h^{r-1}\,\frac{\sqrt{\lambda/4\pi a}}{|\hat K(\pi/\lambda)|}$. The constant $1/|\hat K(\pi/\lambda)|$ is the *Nyquist amplification* -- $\approx(\varepsilon^*)^{-1/2}$ for pole-type kernels ($2.4\times10^6$ for sech$^2$/tanh), only $\sim10^3$ for Gaussian-type.
- **Verified quantitatively at $r{=}1$**: predicted off-law slopes $-q$: $|x|$ ($q{=}1$) $\to-1$ vs measured $-0.98$; $x|x|$ ($q{=}2$) $\to-2$ vs measured $-2.00$. Predicted magnitudes at $N{=}1024$: 340 vs 610 ($|x|$), 0.4 vs 1.4 ($x|x|$) -- endpoint-Laplace accuracy. Predicted law/edge crossover $N^*$ (the elbow): $|x|^3$: 210 vs observed $\approx300$; $x|x|$: $5\times10^3$ (beyond the sweep, hence "converging, not locked"); $|x|$: $10^{11}$ (never).
- **The $r{=}0$/noisy-cell anomalies are solver artifacts, resolved.** SVD decomposition at $N{=}512$: the kernel's spectrum only reaches down to $\sigma/\sigma_{max}\sim\hat K(\pi/\lambda)/M$ ($\approx2\times10^{-8}$ sech$^2$, $\approx10^{-4}$ gaussian); within that physical band the solution norm matches the theory (sech$^2$/$|x|$: $\approx190$), while the observed gelsd norms live entirely in fp64-noise modes *below* the floor that `gelsd`'s internal threshold fails to cut. A clean truncated-SVD solve reproduces the theory at identical eval error: sech$^2$/$|x|$ at $N{=}1024$: $\|v\|=96$ (vs gelsd $3\times10^8$), rel $L_2=1.34\times10^{-5}$ both; sech$^2$/$|x|^3$: 1.14 $\approx$ the law (vs gelsd 10, drifting). sech$^2$ is the exposed case because its pole-type spectrum grades exponentially across 8 decades before the noise floor.
- **Figures.**
  - `expA07s_norm_law_set{A,B}.png` -- raw view: 3x3 grids, row = kernel order (both activations per panel, blue/red dots; tail slope in legend), column = smoothness class; dashed lines = the zero-parameter law per activation; the $r{=}2\times C^0$ panel is annotated "law undefined". The elbow where a curve bends onto its dashed line is the lock.
  - `expA07s_ratio_set{A,B}.png` -- compensated view (the readable one): same layout, y = measured/predicted; the law holds where a curve goes flat at the dashed ratio-1 line. Look for: flat-at-1 tails in the $C^2$ column for gaussian, tanh/sigmoid, gelu; the still-descending curves everywhere else; sech$^2$ touching 1 then bouncing away ($C^2$ column, top row).

## Additional details

**Norm bookkeeping.** The rule is stated for the vector $L_2$ norm; per element it means $\mathrm{std}(v_k) \approx \|f'\|/(\sqrt2\,N)$, i.e. $O(1/N)$ per weight. Standard `nn.Linear` init draws $U(\pm1/\sqrt W)$ per element, giving vector norm $\approx 1/\sqrt3 = \Theta(1)$ -- constant in width, which is the wrong regime and explains the random-init plateau in expD06.

**Practical recipe (no oracle).** Estimate $\|f'\|_{L_2}$ from training data by finite differences; initialize interior readout nodes with shape = local data slopes (or a Gaussian if lazy) at vector norm $\|\hat f'\|/\sqrt{2N}$ (or bare $1/\sqrt N$), halo = 0. expD06's noise tolerance (12.5% perturbation indistinguishable from exact) covers finite-difference estimation error.

## Conclusions

*Pending Sam review.* The inner-band readout norm obeys a single target-independent law, $\|v_{inner}\|_2 = \|f'\|_{L_2}/\sqrt{2N}$ for tanh, valid once the width resolves the target; and initializing a Gaussian readout at vector norm $1/\sqrt N$ (halo 0) captures most of the trainability benefit of the full derivative-shaped initialization, because the width-scaling of the magnitude -- not the direction -- is the dominant lever that standard initialization gets wrong.

## Open questions

- Where exactly is the regime boundary $N_{res}(f)$, and can it be predicted from the target's spectrum (it should track the analyticity/frequency content)?
- Does the $1/\sqrt N$-normalized Gaussian init transfer to the high-dimensional ridge-bundle setting (expF04's real-data tasks)?
- ~~gelu on rough targets: what law replaces $f''$?~~ Answered by the master formula: the Nyquist-deconvolution edge term, slope $-q$ with $q=s+1-r$.
- ~~The sech$^2$ norm upturn~~ -- resolved: `gelsd` threshold artifact (fp64-noise modes below the kernel's spectral floor); a clean truncated-SVD solve restores the theory values at identical error.
- Re-run the sweep with the truncated-SVD solver and overlay the full two-term master-formula prediction (law + edge) per cell -- the definitive zero-parameter version of the figures. Also: do the $r{=}2$ noisy cells (gelu/swish on $C^1$) collapse onto the edge term once the solver artifact is removed?
- The master formula assumes free extension at $\pm1$ (halo absorbs boundary); set B's boundary kinks ($|s|$ at $x=\pm1$) were not counted in $J$ -- quantify their contribution.
- Long-horizon: does any of these inits change the 20k-step floor (expD05 scale), or only the approach speed?
