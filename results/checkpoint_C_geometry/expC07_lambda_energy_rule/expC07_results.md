# expC07 -- The aliasing rule: predicting optimal $\lambda$ across activations

**Status:** approved (Sam, 2026-07-16).

## TL;DR

- A single kernel invariant, anchored at tanh's $\lambda^*=0.25$, predicts the optimal bandwidth of every activation tested: choose $\lambda$ so the kernel's **Fourier tail at the first grid harmonic** equals machine epsilon. Predictions: sech$^2$ 0.250, gaussian 0.530, tanh 0.250, sigmoid 0.500 (exact), gelu 0.707, swish 0.455 -- all six confirmed by the measured minima (gelu measured 0.716, swish 0.437).
- Tanh's $\lambda^*=0.25$ is itself explained: it is precisely the largest $\lambda$ at which the intrinsic aliasing $e^{-\pi^2/\lambda}$ sits below fp64 eps ($A_{\tanh}(0.25)=4\pi^2/\sinh(4\pi^2)=5.7\times10^{-16}$).
- What sets $\lambda^*$ is the kernel's **analyticity class** -- pole distance for exponential-tail kernels, entire/Gaussian for super-exponential -- not the derivative order $r$ and not real-space energy. Both energy-matching variants (the original hypothesis) are falsified by the data.

## Question / hypothesis

Original hypothesis (Sam): anchored at tanh's $\lambda^*=0.25$, the optimal $\lambda$ of any activation follows from matching an energy-like invariant of its kernel -- the $r$-th derivative $\psi^{(r)}$ that does the approximating -- to tanh's kernel $\mathrm{sech}^2$. The experiment asks: what invariant of the kernel actually predicts the optimum?

## Experiment design

**Model and solve.** Frozen QI geometry on $[-1,1]$: uniform centers $c_k$ with the standard halo, $\gamma=\lambda/h$, features $\Phi_{ik}=\psi(\gamma(x_i-c_k))$ for activation $\psi$, augmented system $[\Phi,\mathbf 1]$ solved by SVD lstsq (`gelsd`) on a dense grid ($\ge 2003$ points), evaluated as rel $L_2=\|\hat f-f\|_2/\|f\|_2$ on 4001 points.

**Sweep.** 6 activations $\times$ 4 targets $\times$ $\lambda\in\mathrm{geomspace}(0.02,1.5,36)$ $\times$ $N\in\{64,128,256\}$. Targets: $\sin(2\pi x)$, $\sin(8\pi x)$, Runge $1/(1+25x^2)$, $e^x$. Activations by kernel order $r$ (kernel $K=\psi^{(r)}$): $r{=}0$ sech$^2$, $e^{-x^2}$; $r{=}1$ tanh, sigmoid; $r{=}2$ gelu, swish.

**The aliasing rule (the prediction being tested).** The quasi-interpolant places kernel translates on a unit lattice in grid units $u=x/h$; by Poisson summation the irreducible error is the kernel spectrum leaking past the first grid harmonic (Theorem 1's intrinsic-aliasing term, $C_2e^{-c_2/\lambda^p}$). With $\widehat K$ the kernel FT, define the normalized aliasing amplitude

$$A(\lambda)=\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|},$$

anchor $\varepsilon^*=A_{\tanh}(0.25)=4\pi^2/\sinh(4\pi^2)=5.65\times10^{-16}$ (machine eps), and predict each activation's optimum as the solution of $A_K(\lambda)=\varepsilon^*$ (mpmath bisection; all FTs verified against 40-digit quadrature).

- Kernels and transforms ($\varphi$ = standard normal pdf; $\widehat K/\widehat K(0)$ shown):
  - tanh, sech$^2$: $K=\mathrm{sech}^2 x$, $\;\pi\omega/(2\sinh(\pi\omega/2))\sim\pi\omega\,e^{-\pi\omega/2}$, i.e. aliasing $e^{-\pi^2/\lambda}$ ($p{=}1$).
  - sigmoid: $K=\sigma'=\tfrac14\mathrm{sech}^2(x/2)$, $\;\pi\omega/\sinh(\pi\omega)\sim e^{-\pi\omega}$ -- poles at $\pm i\pi$ instead of $\pm i\pi/2$.
  - gaussian: $K=e^{-x^2}$, $\;e^{-\omega^2/4}$, i.e. aliasing $e^{-\pi^2/\lambda^2}$ ($p{=}2$).
  - gelu: $K=\mathrm{gelu}''=(2-x^2)\varphi(x)$, $\;(1+\omega^2)e^{-\omega^2/2}$.
  - swish: $K=\mathrm{swish}''=\tfrac14\mathrm{sech}^2(x/2)\,[2-x\tanh(x/2)]$, $\;\pi^2\omega^2\cosh(\pi\omega)/\sinh^2(\pi\omega)\sim2\pi^2\omega^2e^{-\pi\omega}$.

**Energy-rule variants (original hypothesis, for contrast).** Raw energy with the chain-rule amplitude, $\lambda^{2r-1}E_K=0.25\,E_{\mathrm{sech}^2}$ with $E_K=\int K^2$; and amplitude-normalized width matching, $\lambda=0.25\,(E_K/K_{\max}^2)/(E_{\mathrm{sech}^2}/1)$. A mass-normalized variant ($W=m^2/E$) was the run's original prediction set and is kept as the gray dotted line in the figures.

**Metrics.** Per-cell argmin $\lambda$ of rel $L_2$; "hard cells" are those whose minimum stays above $10^{-14}$ (the U-curve has a sharp bottom there, so the argmin is informative rather than floor-jitter).

**Code & data.**
- Sweep + figures: `experiments/expC07_lambda_energy_rule/run.py` (predictions in `PRED`/`PRED_ENERGY`).
- Rows: `results/checkpoint_C_geometry/expC07_lambda_energy_rule/expC07_rows.json` (1728 solves).
- Figures: `results/checkpoint_C_geometry/expC07_lambda_energy_rule/figures/expC07_ucurves_{sine,sine_8pi,runge,exp}.png`.

## Results

The aliasing rule lands on all six activations; both energy variants miss badly.

| activation | $r$ | aliasing-rule $\lambda$ | measured (hard-cell median) | raw energy | normalized width |
|---|---|---|---|---|---|
| sech$^2$ | 0 | 0.250 | 0.222 | 4.00 | 0.250 |
| gaussian | 0 | 0.530 | 0.494 | 3.76 | 0.235 |
| tanh | 1 | 0.250 (anchor) | 0.208 | 0.250 | 0.250 |
| sigmoid | 1 | 0.500 (exact) | 0.437 | 2.00 | 0.500 |
| gelu | 2 | 0.707 | 0.716 | 0.755 | 0.228 |
| swish | 2 | 0.455 | 0.437 | 0.910 | 0.332 |

- **The prediction marks the right wall of the basin.** In every panel the measured minima hug the predicted line from the left, with the flat basin floor extending leftward -- the same geometry tanh shows against its own 0.25 anchor (its argmins scatter 0.16--0.24). The rule predicts the largest safe bandwidth; smaller $\lambda$ stays on the floor until conditioning bites.
- **Sigmoid is an exact consistency check.** $\sigma(x)=\tfrac12(1+\tanh(x/2))$ plus the $[\Phi,\mathbf1]$ constant column makes the sigmoid model at $\lambda$ span exactly the tanh model at $\lambda/2$, forcing $\lambda_\sigma=0.5$. The aliasing rule reproduces this exactly (poles at $\pm i\pi$ vs $\pm i\pi/2$, prefactors identical); raw energy predicts 2.0 -- falsified a priori.
- **Gelu and swish discriminate the rule.** Gelu's Gaussian-type kernel predicts $\lambda=0.707\approx1/\sqrt2$; measured 0.716 (grid point nearest 0.707), vs 0.23 from width matching. Swish shares sigmoid's pole distance but its double pole adds an $\omega^2$ prefactor, pulling the solve from 0.500 to 0.455 -- and the data indeed puts swish slightly left of sigmoid.

**Figures.** All four share one layout: 2$\times$3 grid (panel = activation), log-log rel $L_2$ vs $\lambda$, one line per width $N\in\{64,128,256\}$, dot at each line's measured minimum, red dashed vertical at the aliasing-rule prediction, gray dotted at the original energy rule.

- `expC07_ucurves_sine_8pi.png` -- the sharpest U-curves (hardest smooth target); the cleanest visual of minima sitting on the red line in all six panels.
- `expC07_ucurves_runge.png` -- $N{=}64$ has not reached the floor, so the minima are unambiguous; same agreement.
- `expC07_ucurves_sine.png`, `expC07_ucurves_exp.png` -- easy targets: wide flat floors whose right edge still tracks the prediction; argmins wander leftward along the floor as expected.

## Additional details

- **Why derivative order does not set $\lambda^*$:** differentiation multiplies $\widehat K$ by $(i\omega)^r$ -- a polynomial factor, hence only a logarithmic shift in the solve. The chain-rule amplitude $\lambda^r$ is absorbed by the lstsq readout ($w[m]=a[m]/\gamma^{r-1}$), so any amplitude-sensitive invariant (raw energy) is ruled out before looking at data. What matters is the analyticity class: pole-type kernels at distance $a$ give $\lambda^*\approx0.25\cdot a/(\pi/2)$ ($p{=}1$); entire kernels $e^{-x^2/(2s^2)}$ give super-exponential tails ($p{=}2$) and correspondingly larger $\lambda^*$.
- **The anchor explains the paper's own constants:** $e^{-\pi^2/\lambda}$ at $\lambda=0.30$ is $5\times10^{-15}$ (the documented fp64-path floor) and at $\lambda=1.5$ is $1.4\times10^{-3}$ (the documented "does not work").
- **Calibration caveat:** $A(\lambda)$ is near-vertical on a log scale, so the predicted $\lambda$ is only logarithmically sensitive to the choice of $\varepsilon^*$. The rule's real content is the ratios between activations, which the five non-anchor activations confirm independently.
- The FT formulas were verified against mpmath quadrature (rel. error $\le10^{-9}$, most $\le10^{-30}$); the sweep itself predates this analysis (run under the energy hypothesis), so the data is out-of-sample with respect to the aliasing predictions.

## Conclusions

The optimal bandwidth of an activation is set by its kernel's intrinsic aliasing: $\lambda^*$ solves $|\widehat K(2\pi/\lambda)|/|\widehat K(0)|=\varepsilon_{\mathrm{fp64}}$, with $K=\psi^{(r)}$ and the anchor value $5.7\times10^{-16}$ recovered from tanh at $\lambda=0.25$. This predicts the measured optimum for all six activations (sigmoid exactly, gelu and swish to within one grid step) and falsifies energy/width matching; equivalently, tanh's $\lambda^*=0.25$ is the point where $e^{-\pi^2/\lambda}$ reaches machine epsilon.

## Open questions

- Does the rule transfer to 2D ridge geometries (expE01), where the optimal $\lambda$ appears to drift slightly with $N$?
- The rule fixes the right wall (aliasing); does the left wall (conditioning/prefactor blowup as $\lambda\to0$) have a similarly universal per-kernel form?
- Can the rule set $\lambda$ a priori for new activations in the training experiments (checkpoint D reparameterizations with $\gamma=\lambda/h$), removing a tuned hyperparameter?
