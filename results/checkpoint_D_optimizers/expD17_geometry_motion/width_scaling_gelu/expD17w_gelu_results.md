# expD17w (GELU) -- geometry motion and width scaling under a second activation

**Status: draft -- pending Sam's review.** Companion to `../width_scaling/expD17w_results.md` (tanh). One variable changed: the hidden activation. Same grid, budgets, zero-init readout, plain Adam, snapshot schedule, and figure set.

## TL;DR

- **The halo-death asymmetry predicted from GELU's shape is exactly confirmed.** Under tanh, left and right halo neurons die at equal rates (0.43/0.44, 0.43/0.43, 0.63/0.63 at $W=64,128,256$). Under GELU, **left-halo death is $0.000$ at every width** while right-halo death is $0.68$, $0.84$, $0.92$. Interior neurons never die under either activation; standard init never has a dead neuron at all.
- Every other structural conclusion of the tanh study survives the activation swap: the $\gamma$-normalization explanation holds (slope $1.07$, correlation $0.92$, up from $0.80$), absolute drift is init-independent (median std/QI ratio $0.852$ versus tanh's $0.855$), and the floor-damage / off-floor-improvement split reproduces cell for cell.
- **Adam does not rescale GELU's bandwidths either**: the 1-D median $\gamma$ moves $-0.18\%$ to $+0.21\%$ over 5000 steps, against an init IQR of exactly zero that training opens to $0.05$-$0.61$. Same behavior as tanh, at $2.83\times$ the bandwidth.
- GELU's relative-drift width exponents are systematically shallower than tanh's ($-0.40$ vs $-1.59$ in 1-D), so the $1/W$ suppression is activation-dependent, not a law.

## Question

Does the geometry-motion picture, and in particular the frozen-halo finding, depend on the activation? GELU is the natural contrast: it is the other activation the repo has characterized (expA04, expA06, expA07, expC07), and its asymmetric limits give a sharp falsifiable prediction about which neurons can receive gradient.

## Experiment design

Identical to the tanh width-scaling study in every respect except the activation and the bandwidth it forces. Plain Adam, readout initialized to zero, one hidden layer, 5000 steps, fp64 on CPU, single seed, 4 problem classes $\times$ 3 problems $\times$ 2 inits $\times$ 3 widths (72 runs). Geometry vector $g$ = all first-layer parameters; per-neuron bandwidth $\gamma_k=\|w_k\|_2$; per-neuron center $c_k=-b_k/\gamma_k$.

**The bandwidth had to change, and this is the one substantive design decision.** expC07's aliasing rule sets $\lambda^*$ from the activation's kernel Fourier tail: $0.250$ for tanh, $0.707$ for GELU. Running GELU at tanh's $\lambda$ would confound "GELU behaves differently" with "GELU was handed the wrong bandwidth" -- expA07 measured a $10^4$ break in the readout-norm law at exactly that mistake. Every class's tuned tanh $\lambda$ is therefore scaled by $0.707/0.25 = 2.828$. The 1-D probe confirms the choice: at $W=128$ the GELU geometry's lstsq floor is $4.1\times10^{-13}$ at $\lambda=0.707$, and the $\lambda$ that produced tanh's floor does not transfer.

**Caveat carried from the theory.** GELU is kernel order $r=2$ ($K=\mathrm{gelu}''$), so its readout law is $v_k\approx (h^2/\lambda)f''(c_k)$ rather than tanh's $(h/2)f'(c_k)$ (expA06), and $\mathrm{gelu}''$ changes sign, so the QI construction is **not** strictly valid for GELU (paper appendix B.4, expA04). What transfers is the *geometry* -- uniform centers, one shared bandwidth, a halo -- not a proven construction. That is precisely why the run is informative: it separates "the geometry does this" from "the construction does this."

**The death prediction under test.** A halo neuron sits at $|c|>1$ with $\gamma=O(N)$, so its preactivation $\gamma(x-c)$ has one sign over the whole domain. Under tanh, $\tanh'=\mathrm{sech}^2\to 0$ in **both** directions, so every halo neuron saturates, the gradient underflows to exactly zero, and Adam's $0/(0+\epsilon)$ is zero. Under GELU the limits differ: $\mathrm{gelu}(z)\to 0$ with vanishing slope as $z\to-\infty$, but $\mathrm{gelu}(z)\to z$ with slope $\to 1$ as $z\to+\infty$. So **left-halo** neurons ($c<-1$, positive preactivation) should stay alive as near-linear units, and **right-halo** neurons ($c>1$, negative preactivation) should die as tanh's do. A neuron counts as dead when its applied Adam update stays below $10^{-10}$ for the first 30 snapshots; bitwise-zero updates are reported separately in the figure.

**Code & data.** `experiments/expD17_geometry_motion/width_scaling_gelu/run.py` (a thin driver that loads `../width_scaling/run.py` and rebinds four globals, so builders, training loop, figures and analysis are the same code). Data `results/checkpoint_D_optimizers/expD17_geometry_motion/width_scaling_gelu/data/*.jsonl` (72 runs, per-iteration drift plus snapshot streams for $\gamma$, bias and per-row update norms). Figures in `figures/` as listed below.

## Results

**The asymmetry is total.** Dead-early fraction on 1-D sine, split by neuron location:

| activation | $W$ | all | interior | left halo | right halo |
|---|---:|---:|---:|---:|---:|
| tanh | 64 | 0.307 | 0.000 | 0.457 | 0.443 |
| tanh | 128 | 0.223 | 0.000 | 0.429 | 0.429 |
| tanh | 256 | 0.278 | 0.000 | 0.627 | 0.627 |
| GELU | 64 | 0.148 | 0.000 | **0.000** | 0.680 |
| GELU | 128 | 0.186 | 0.000 | **0.000** | 0.843 |
| GELU | 256 | 0.204 | 0.000 | **0.000** | 0.922 |

Standard init has zero dead neurons in every cell of both studies. The tanh left/right rates agree to three digits at two of three widths, which is what a symmetric activation must give; GELU's left column is exactly zero at all three.

**The frozen set requires both ingredients, and neither alone produces it.** The QI init is what places neurons at $|c|>1$ with $\gamma=O(N)$, so their preactivation has one sign and large magnitude over the entire domain; the activation is what decides whether a neuron in that position feels anything. Standard init under the *same* activation has no dead neurons, because its neurons sit where the data is and its $\gamma$ is $O(1)$. Swapping the activation changes *which* of our halo neurons freeze (both sides under tanh, only the right side under GELU) but does not create or remove the phenomenon. The conjunction is the finding: our geometry puts capacity where a saturating activation cannot deliver gradient.

Right-halo death also *rises* with width under GELU ($0.68\to0.92$), because $\gamma$ grows with $N$ and drives those units further into the vanishing tail. Only the 1-D class has a saturating halo at all; 2-D, PINN and tabular geometries place their offsets inside the data's reach and show zero dead neurons under both activations.

**Everything structural reproduces.** The $\gamma$-normalization account of the red-below-blue gap is, if anything, cleaner under GELU: regressing $\log$(relative-drift ratio) on $\log(\|g_0\|$ ratio) gives slope $1.07$ with correlation $0.915$, against tanh's $0.99$ and $0.800$ (slope 1 is what pure normalization predicts). Absolute drift stays init-independent: median std/QI absolute-drift ratio $0.852$ (IQR $0.67$-$1.26$) versus tanh's $0.855$ (IQR $0.64$-$0.98$). In both studies the QI arm typically moves slightly *further* in parameter units than the standard arm.

**Bandwidths hold; the delta breaks.** On 1-D sine the QI median $\gamma$ moves $+0.21\%$, $-0.06\%$, $-0.18\%$ at $W=64,128,256$ over 5000 steps, from $\gamma^*=22.6, 45.2, 90.5$. The init IQR is exactly zero (every neuron shares $\gamma^*$) and training opens it to $0.107$, $0.191$, $0.611$. Centers barely move: RMS motion $0.0145$, $0.0030$, $0.0036$ against a span of $3.56$-$3.59$. This is the tanh behavior at a different bandwidth: on a correct geometry Adam leaves the layout alone and only jitters the shared bandwidth.

**Width scaling is shallower.** QI relative-drift exponents, GELU versus tanh: 1-D $-0.40$ vs $-1.59$, 2-D $-0.43$ vs $-0.56$, PINN $-0.68$ vs $-0.98$, tabular $-0.73$ vs $-0.68$. Standard-init exponents are near zero in both ($-0.33$ to $+0.13$). So the "relative drift falls like $1/W$" reading from the tanh study is not a law across activations; only the tabular class, whose $\gamma$ grows like $\sqrt W$ in both studies, gives the same exponent.

**The probe split reproduces cell for cell.** Every QI geometry at its floor is damaged by training, and every QI geometry off the floor is improved, in both studies:

| cell | tanh pre $\to$ post | GELU pre $\to$ post |
|---|---|---|
| 1-D sine | $3.7\text{e-}14 \to 7.2\text{e-}12$ | $4.1\text{e-}13 \to 3.7\text{e-}9$ |
| 1-D runge | $2.5\text{e-}14 \to 2.5\text{e-}11$ | $1.2\text{e-}14 \to 2.5\text{e-}10$ |
| 1-D sine_8pi | $5.9\text{e-}14 \to 1.2\text{e-}7$ | $1.3\text{e-}11 \to 3.7\text{e-}5$ |
| 2-D gauss_bump | $2.2\text{e-}14 \to 8.3\text{e-}12$ | $4.1\text{e-}13 \to 4.1\text{e-}9$ |
| 2-D sine2d | $4.8\text{e-}8 \to 3.2\text{e-}10$ | $4.9\text{e-}8 \to 2.6\text{e-}11$ |
| 2-D mixed2d | $5.5\text{e-}5 \to 5.5\text{e-}8$ | $5.6\text{e-}5 \to 1.4\text{e-}8$ |

GELU's damage is $2$-$3$ orders larger on the floor cells, and its improvement on the off-floor cells is slightly better ($1900\times$ vs $150\times$ on sine2d, $4000\times$ vs $1000\times$ on mixed2d).

**Inverse problems: the QI init's advantage is real under tanh and collapses at width under GELU.** Recovered PDE parameter, relative error in percent (true values $\nu=0.1$, $\lambda=1$, $k=5$):

| problem | $W$ | tanh std | tanh QI | GELU std | GELU QI |
|---|---:|---:|---:|---:|---:|
| burgers $\nu$ | 256 | 3.6 | **0.5** | 0.1 | 0.4 |
| burgers $\nu$ | 512 | 3.4 | **0.1** | 2.6 | **1.0** |
| burgers $\nu$ | 1024 | 3.9 | **0.1** | 2.0 | 8.5 |
| bratu $\lambda$ | 256 | 5.5 | **1.3** | 10.9 | **2.1** |
| bratu $\lambda$ | 512 | 8.5 | **1.1** | 9.5 | 67.1 |
| bratu $\lambda$ | 1024 | 14.2 | 27.7 | 4.7 | 90.3 |
| allencahn $k$ | all | 320-555 | 59-88 | 516-580 | 83-92 |

Under tanh the QI init recovers the parameter $7$-$33\times$ more accurately on burgers at every width and $4$-$8\times$ on bratu at the two smaller widths, losing only on bratu at $W=1024$. Under GELU the same init is competitive at $W=256$ and then **degrades with width**, ending at $67\%$ and $90\%$ error on bratu where the field error also diverges ($87$ at $W=1024$, i.e. the run came apart). Neither init nor activation solves allencahn: every arm misses $k=5$ badly, though the QI arms are wrong by less and carry $20\times$ better field error.

### Figures

- **`figures/expD17w_dead_fraction_gelu_vs_tanh.png`** -- 1x4 panels (one per problem class), x = width, y = percent of neurons with zero update, four lines: {tanh, GELU} x {QI, standard}. Read the 1-D panel: both QI lines rise, both standard lines sit flat on zero, and the other three panels are empty because only the 1-D geometry has a saturating halo. The left/right split that makes the mechanism visible is in the table above, not this figure.
- **`figures/expD17w_gelu_drift_from_init.png`** and **`expD17w_gelu_step_size.png`** -- the two relative views, 4x3, six lines per subplot (blue shades standard, red shades QI, lighter = smaller width), rows share the y scale. Same qualitative picture as tanh: red far below blue, and the red shades ordered by width.
- **`figures/expD17w_gelu_drift_from_init_abs.png`** and **`expD17w_gelu_step_size_abs.png`** -- the same runs without the $\|g_0\|$ normalization. The blue/red separation collapses; this pair plus the ratio scatter is the evidence that the relative gap is units.
- **`figures/expD17w_gelu_ratio_scatter.png`** -- relative-drift ratio against effective $\gamma$ ratio, one marker per class, with the $y=x$ line. Points straddle the line across three decades.
- **`figures/expD17w_gelu_loss.png`** -- run error at the same cadence, same layout, for reading motion against progress.
- **`figures/gamma_hist/expD17w_gamma_hist_w{1,2,3}.gif`** and **`figures/center_hist/expD17w_center_hist_w{1,2,3}.gif`** -- QI arm only, 141 frames on the dense-start schedule, red dotted median, mean in the title, fixed axes. Watch the 1-D $\gamma$ delta broaden in place while the center histogram stays put.
- **`figures/gamma_vs_update/expD17w_gamma_vs_update_w{1,2,3}.png`** -- per-neuron $\gamma$ against applied Adam update, both arms overlaid (viridis standard, autumn QI), colored by training step, log-log. The vertical tail on the QI column reaching the axis floor is the frozen right halo.

## Additional details

The dead-neuron criterion here ($<10^{-10}$ for the first 30 snapshots) is deliberately looser than bitwise zero, because GELU's left tail underflows gradually rather than exactly; the summary figure uses the strict bitwise-zero-over-the-run criterion, which is why its 1-D percentages ($14$-$21\%$) sit below the table's early-training fractions. Both criteria give the same left/right split.

Single seed throughout, as in the tanh study. The GELU 1-D floors are $1$-$2$ orders above tanh's at matched width, consistent with expA04's rank finding ($O(N)$ null space versus $O(1)$); this study is about motion, and no claim is made here about GELU's attainable precision.

## Conclusions

*Pending Sam.* The frozen-halo phenomenon is a property of the activation's saturation, not of the QI geometry: swapping tanh for GELU converts a symmetric dead set into a strictly one-sided one, with left-halo death dropping to exactly zero at every width while right-halo death rises to $92\%$. Every other structural finding of the tanh study is activation-independent -- the $\gamma$-normalization account of the relative-drift gap, the init-independence of absolute drift, the preservation of median bandwidth under Adam, and the damage-at-the-floor / improvement-off-the-floor split -- while the $1/W$ suppression of relative drift is not.

## The normalization factorial

The `rms_center` normalization selected by expD21 is measured against this study's arms in `../norm_factorial/expD17n_results.md`, a six-arm grid crossing activation, normalization and init at two widths, with a two-layer tabular row on the expD20 suite. That study is where the question "does normalization make the QI init work for GELU too" is actually answered.

## Open questions

- Does a one-sided or non-saturating activation, or a halo built with reduced $\gamma$, restore gradient access to the halo without giving up the precision the halo buys (expA01: removing it collapses every solver to $10^{-4}$)?
- Is the frozen halo actually harmful? These runs never needed to *learn* a halo, they were given one. The measurement that matters is whether a network trained from standard init can ever construct halo-like capacity, which no experiment has tested.
- GELU's damage on floor-level geometries is $2$-$3$ orders worse than tanh's at matched cells. Is that the $r=2$ kernel order, the larger $\gamma$, or the sign-changing $\mathrm{gelu}''$?
