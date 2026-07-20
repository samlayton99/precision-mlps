# expA06 -- Readout structure: the solved coefficients are sampled derivatives

**Status: draft -- core finding spotted by Sam in the interactive explorer, verified numerically below; conclusions pending Sam sign-off.**

## TL;DR

- In the working regime ($\lambda \gtrsim 0.25$) the min-norm lstsq readout on the QI geometry is **the sampled derivative of the target**: for tanh, $v_k \approx \tfrac{h}{2} f'(c_k)$ -- correlation $\geq 0.986$ on all four test targets, fitted scale $\to h/2$ exactly as $\lambda$ grows.
- For gelu the same mechanism gives the **second** derivative: $v_k \approx \tfrac{h^2}{\lambda} f''(c_k)$ (corr $\approx 0.99$ on oscillatory targets, parameter-free scale within 6%).
- **Alternating coefficients are the small-$\lambda$ pathology, not the solution structure**: below $\lambda \approx 0.1$ the derivative correlation collapses to $\sim 0$ and the sign-alternation rate jumps to $\sim 0.8$.
- Norm-vs-width follows a **power law decaying to a floor**, $\|v\| = A W^b + C$ -- the best universal form across all 35 activation$\times$target combos (mean $R^2$ 0.91 in log space).

## Question

What structure does the SVD-solved readout $v$ have on the frozen QI geometry -- and can that structure be written down in closed form as a readout initialization?

## Experiment design

Two pieces: a live Dash explorer and a precomputed scaling-law sweep.

**Explorer.** Single frozen QI geometry (equispaced centers $c_k$ on $[-1,1]$ plus halo, $\gamma = \lambda/h$, $h = 2/N$); features $\Phi_{ik} = \sigma(\gamma(x_i - c_k))$ for $\sigma \in$ {tanh, gelu, relu, swish, sigmoid}. Readout solved as $[\Phi, \mathbf 1][v; b] \approx y$ by SVD with an rcond truncation slider (drop $\sigma_i < \text{rcond}\cdot\sigma_{\max}$) and a separate ridge slider (filter factors $\sigma_i/(\sigma_i^2+\alpha)$). Six live panels: fit, symlog residual, coefficients $v_k$ vs center $c_k$ (with a $(-1)^k$ demodulation overlay toggle), coefficient histogram, coefficient FFT (DC$\leftrightarrow$Nyquist), and the singular-value spectrum colored by the sign-alternation rate of each right singular vector. Dials: $N$, $\lambda$, activation, target (presets + free text), $n_{train}$, $n_{test}$, sampling, noise, rcond, ridge, halo.

**Verification protocol for the derivative law.** For interior centers ($|c_k| \leq 0.9$, excluding the halo band): Pearson correlation of $v_k$ against $f'(c_k)$ (and $f''$), plus the best scalar $\alpha$ in $v \approx \alpha f^{(r)}(c_k)$, compared against the closed-form predictions $\alpha = h/2$ (tanh) and $\alpha = h^2/\lambda$ (gelu). Targets: sine, sine\_8pi, runge, exp; $N = 128$; $\lambda \in \{0.05, 0.1, 0.25, 0.5, 1.0\}$; pure min-norm solve (no rcond/ridge).

**Scaling-law sweep.** $\|v\|_2$ over $N \in [16, 2048]$ (15 log-spaced points, $n_{train} = \max(2003, 2W+3)$, always overdetermined), all 5 activations $\times$ 7 targets. Six candidate forms fitted to $\log_{10}\|v\|$ per curve (power, exponential, stretched, power$\cdot$log, power-to-floor, exponential-to-floor); judged by $R^2$ in log space, aggregated by mean and worst-combo across all 35 curves.

**Code & data.** `experiments/expA06_readout_structure/{app.py, run.py}` (explorer, port 8051), `experiments/expA06_readout_structure/scaling_law/run.py`. Data: `results/checkpoint_A_numerics/expA06_readout_structure/scaling_law/{scaling_rows.json, fit_summary.json}`. Figures: `scaling_law/figures/{scaling_curves, scaling_form_comparison}.png`.

## Results

**tanh: $v_k \approx \tfrac{h}{2} f'(c_k)$.** Correlation of interior $v_k$ with $f'(c_k)$, and the fitted scale vs the prediction $h/2 = 0.00781$ ($N=128$):

| target | $\lambda=0.05$ | $\lambda=0.10$ | $\lambda=0.25$ | $\lambda=0.50$ | $\lambda=1.0$ | scale at $\lambda=1$ |
|---|---:|---:|---:|---:|---:|---:|
| sine | 0.82 | 0.80 | 1.0000 | 1.0000 | 1.0000 | 0.00784 |
| sine\_8pi | 0.86 | 0.91 | 1.0000 | 1.0000 | 1.0000 | 0.00832 |
| runge | -0.00 | 0.00 | 0.9865 | 0.9993 | 1.0000 | 0.00787 |
| exp | 0.06 | 0.09 | 1.0000 | 1.0000 | 1.0000 | 0.00781 |

At $\lambda = 0.25$ the fitted scale sits $\sim 7\%$ above $h/2$ (finite-bandwidth correction) and converges onto it as $\lambda$ grows. Below $\lambda \approx 0.1$ the law collapses and the alternation rate of the interior coefficients jumps from $\lesssim 0.1$ to $0.6$--$0.8$: the alternating regime is the breakdown mode, not the structure.

**gelu: $v_k \approx \tfrac{h^2}{\lambda} f''(c_k)$** (prediction $9.77\times10^{-4}$ at $N=128$, $\lambda=0.25$): corr(v, f'') = 0.988 / 0.986 for sine / sine\_8pi with fitted scales $9.1$/$9.7\times10^{-4}$, while corr(v, f') $\approx 0$ -- clean separation of the two laws. Caveats: exp cannot distinguish them ($f' = f'' = e^x$, corr 0.29 for both), and runge shows the predicted projection scale but corr $\approx 0$ -- something beyond the $f''$ law dominates there (open).

**Halo.** Halo coefficients follow neither law -- irregular boundary-correction structure in both activations. At $\lambda=0.25$, $N=128$, sine: mean $|v|$ halo/interior $= 16.5$ (tanh) vs $0.4$ (gelu); the tanh ratio is large because its interior derivative signal is tiny ($\sim h/2 \cdot f'$), not because its halo is larger in absolute terms.

**Singular structure.** The right singular vectors of $[\Phi,\mathbf 1]$ sort by alternation: mean sign-alternation rate 0.74 for the bottom-20 singular directions vs 0.04 for the top-20 (tanh, $N=128$, $\lambda=0.25$). rcond $=10^{-8}$ keeps 112/270 modes and cuts $\|v\|$ from 8.7 to 0.46 at the cost of rel $L_2$ $3.7\times10^{-14} \to 3\times10^{-9}$ -- the norm-vs-precision tradeoff lives entirely in the alternating tail.

**Scaling law.** Best universal form for $\|v\|$ vs $W$: **power-to-floor** $\|v\| = A W^b + C$ (mean $R^2$ 0.910, median 0.947 over 35 combos), ahead of exponential-to-floor (0.839), power$\cdot$log (0.806), pure power (0.641), pure exponential (0.422). Both floor-forms beat their pure counterparts decisively: the norm decays polynomially and saturates at a target/activation-dependent constant. Worst combos ($R^2 \approx 0.36$) are the rough/high-frequency targets in the small-$N$ pre-asymptotic regime, which no single 3-parameter form captures.

### Figures

- **`scaling_law/figures/scaling_curves.png`** -- $\|v\|$ vs $W$, one panel per activation, log-log; dots = data (one color per target), lines = the fitted power-to-floor form. Look for the polynomial descent flattening onto per-curve floors.
- **`scaling_law/figures/scaling_form_comparison.png`** -- bar chart of mean and worst-combo $R^2$ per candidate form; the two floor-forms lead, power-to-floor best.
- **The explorer itself** (port 8051) is the primary instrument for the derivative finding: panel 3 with the demodulation toggle, and panel 6's alternation coloring.

## Additional details

**Mechanism (tanh).** tanh is a smoothed step with jump 2, so a superposition of steps builds $f$ from its increments: $2 v_k = f(c_k + h/2) - f(c_k - h/2) \approx h f'(c_k)$. Continuum check: differentiating $f \approx \sum_k v_k \tanh(\gamma(x - c_k))$ with $v_k = \alpha f'(c_k)$ gives $f' \approx \alpha f' \gamma \sum_k \mathrm{sech}^2(\gamma(x - c_k)) \approx 2\alpha f'/h$, forcing $\alpha = h/2$ with no free parameters. Equivalently: $\tfrac{d}{dx}\tanh(\gamma(x-c)) = \gamma\,\mathrm{sech}^2$ is a delta family, so the readout is the Stieltjes measure of $f$.

**Mechanism (gelu).** gelu is a smoothed ramp (asymptotically linear); a ramp's second derivative is a delta family with unit mass, so the coefficients carry local curvature: the same continuum argument gives $\alpha = h/\gamma = h^2/\lambda$.

**Why alternation at small $\lambda$.** Flat kernels make adjacent columns of $\Phi$ near-identical; representing anything then requires finite-difference (alternating) combinations of near-duplicate columns -- the same reason the QI cardinal coefficients alternate. At working $\lambda$ the target's projection onto those small-$\sigma$ directions is negligible and the smooth derivative solution wins the min-norm competition.

## Conclusions

*Pending Sam review; the derivative law was proposed by Sam from the explorer and is verified above.* In the working regime the min-norm readout on the frozen QI geometry is, to first order, a closed-form object: the target's derivative sampled at the centers with a known scale ($\tfrac{h}{2} f'(c_k)$ for tanh). This gives an immediate candidate readout initialization that requires no solve -- and when $f'$ is unknown, local finite-difference slopes of the training data estimate it.

## Open questions

- **Does $v^{init}_k = \tfrac{h}{2} f'(c_k)$ close the Adam gap?** Direct checkpoint-D test: initialize the readout in closed form (plus a halo rule) and see how far Adam gets from there vs standard init.
- **The runge-gelu anomaly**: right projection scale but zero correlation -- what dominates the gelu solution for peaked targets?
- **Halo initialization rule**: the interior law is closed-form; the halo boundary correction is not. Characterize it (it is the dominant $\|v\|$ mass for tanh).
- **relu/swish laws**: relu is an exact ramp (expect $f''$); swish interpolates -- does the explorer show a mixed law?
- **The $\lambda$-dependent scale correction** ($\sim 1.07 \times h/2$ at $\lambda = 0.25$): derive it, so the init is exact at the working $\lambda$.
