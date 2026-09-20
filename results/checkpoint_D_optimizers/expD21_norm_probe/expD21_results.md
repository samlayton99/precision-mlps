# expD21 -- which feature normalization should the factorial carry?

**Status: draft -- pending Sam's review.** A selection probe run before the large factorial, not the factorial itself. Follows `../expD19_gelu_init/expD19_results.md`, which found that repairing the GELU QI-init's feature scale fixes the inverse-PINN divergence but could not say *which* normalization to adopt.

## TL;DR

- **No variant is consistently good, and the honest answer is that the data does not support a single winner.** Across 18 cells the five variants tie on mean rank ($2.67$ to $3.11$ out of 5), and every normalization is worse than doing nothing on at least 8 of 18 cells.
- **The winner differs by activation, and this is the headline.** On GELU, centered normalization wins (median $1.36\times$ better than baseline, best mean rank $2.56$). On tanh, the same transform is the second *worst* option (median $0.79\times$, one cell $32\times$ worse) and plain baseline ties for best.
- **Centering, not scaling, is the active ingredient -- and its sign flips with the activation.** Holding the code path fixed and toggling one flag: on GELU centering moves the median ratio from $0.88$ (scaling only, worse than baseline) to $1.36$; on tanh it moves it from $1.00$ to $0.79$.
- **BatchNorm without affine is the least consistent variant tested** (ratio spread $329\times$, worst single regret $80\times$, 10 of 18 cells worse than baseline). Its expD19 win survives only on the inverse problems.
- **The consistency ranking is the inverse of the benefit ranking.** The two safest variants (LayerNorm spread $4.2\times$, scaling-only spread $5.8\times$) are the two that do the least; the two most useful on GELU (centered RMS, BatchNorm) carry the two largest tail risks.
- **Recommendation: carry `rms_center` and `layernorm_noaffine` into the factorial, as an activation-conditional pair, and drop BatchNorm.** Reasons in §Conclusions.

## Question

expD19 compared six arms but could not decide between them, for two reasons. Its BatchNorm and LayerNorm arms carried $2W$ extra learnable parameters, so they were not compared at equal capacity. And BatchNorm won the inverse-PINN row while barely reducing the feature-column-norm spread, which the pure-scaling diagnosis cannot explain -- the obvious missing ingredient being *centering*, which BatchNorm does and the static reparameterization does not. This probe fixes both and asks: which normalization is most **consistent**, not merely best somewhere?

## Experiment design

Five variants, **identical trainable parameter counts**, halo **held fixed at the standard rule** in every one so nothing is conflated with the halo question expD19 raised.

| variant | transform applied to the hidden features |
|---|---|
| `baseline` | none |
| `rms_nocenter` | $h \mapsto h / \mathrm{rms}(h_{\rm init})$, a fixed buffer (expD19's `static_colnorm`) |
| `rms_center` | $h \mapsto (h - \mathrm{mean}(h_{\rm init})) / \mathrm{std}(h_{\rm init})$, fixed buffers |
| `batchnorm_noaffine` | `nn.BatchNorm1d(W, affine=False)` |
| `layernorm_noaffine` | `nn.LayerNorm(W, elementwise_affine=False)` |

The architecture is `x -> Linear(d,W) -> act -> [(h - shift) * scale] -> [norm] -> Linear(W,1)`; normalization always sits **after** the geometry, on the hidden activations, never on the input. With the readout zeroed, variants 2, 3, 5 and primed-4 are **pure reparameterizations at init**: every one represents the identical function (zero), so the comparison isolates gradient geometry and nothing else. A gate test pins this, along with bit-identical inner layers across variants.

- **The pair structure is what makes the mechanism readable.** `rms_center` is exactly a BatchNorm frozen at its init statistics, so (`rms_nocenter`, `rms_center`) isolates centering from scaling with one flag in one code path, and (`rms_center`, `batchnorm_noaffine`) isolates frozen-at-init statistics from running statistics. The second pair is *not* a clean single-variable contrast, and the writeup does not treat it as one: BatchNorm also applies an $\varepsilon$ floor, dividing by $\sqrt{\sigma^2+\varepsilon}$ with $\varepsilon=10^{-5}$, so columns with $\sigma^2 < \varepsilon$ are scaled but never normalized. At $N=64$, 23 of 115 GELU columns sit below that floor, which is the mechanical reason BatchNorm leaves a $10^{12}$ spread where the static transforms reach exactly $1.0$.
- **Grid.** Both activations at their own aliasing-rule bandwidth ($\lambda^*=0.25$ tanh, $0.707$ GELU, expC07); QI init only; three classes $\times$ three problems -- 1-D interpolation {sine, runge, sine_8pi} at $N=128$, 2-D interpolation {gauss_bump, sine2d, mixed2d} at $W\approx576$, 2-D inverse PINN {burgers, bratu, allencahn} at $W=512$; **3 seeds**; 2000 full-batch Adam steps on expD17's schedule; fp64 on CPU. 270 runs.
- **Seeds are the data realization, and they had to be constructed.** expD19's runs are fully deterministic -- the QI init is a formula, the readout starts at zero, the loss is full-batch, the sample sets were fixed -- so repeating a cell reproduces it bitwise and a robustness claim over seeds would be vacuous. Here the 1-D training grid is jittered within its own spacing (expB01: $x$-jitter is harmless to the floor), the 2-D disk sample is redrawn, and the PINN interior data points are redrawn. Measured seed spread in final error is $1.008\times$ median, $1.13\times$ max, so seed noise is small against the variant differences below; **robustness here means robustness to the data, not to initialization, of which there is none by design.**
- **Judgement is consistency.** Reported per variant: mean rank, worst-case rank, rank variance, median/min/max of the ratio (baseline error / variant error), the spread of that ratio, the count of cells where the variant loses to baseline (regret), and the worst single regret. A variant that is $5\times$ better somewhere and $10\times$ worse elsewhere is worse than one uniformly $1.5\times$ better.

**Code & data.** `experiments/expD21_norm_probe/{run.py, analysis.py}`; gate tests `tests/test_expD21_norm_probe.py` (8 passed: identical parameter counts per activation, zero-parameter norm layers, bit-identical geometry, identical represented function at init, the centering/scaling contract, the BatchNorm $\varepsilon$-floor mechanism, and a live seed axis). Data `results/checkpoint_D_optimizers/expD21_norm_probe/data/*.jsonl` (270 runs, 90 jobs). Figures `figures/expD21_{ranks,curves}.png`.

## Results

**The parameter gate passes.** Every variant trains the same number of parameters within an activation (1-D: 808 tanh, 694 GELU; 2-D: 2281; PINN: 2074). Counts differ *between* activations only because the standard halo rule follows $\lambda^*$ (halo 70 at tanh's 0.25, 51 at GELU's 0.707); every variant comparison is within-activation, so that is not a confound. expD19's arms differed by $2W$; these do not.

**Consistency, all 18 cells.** Ratio $=$ baseline error / variant error, so $>1$ is better:

| variant | mean rank | worst rank | med ratio | ratio spread | regret | worst regret |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 3.06 | 5 | 1.000 | 1.0 | 0/18 | -- |
| `rms_nocenter` | 3.06 | 5 | 1.002 | 5.8 | 9/18 | $1.96\times$ |
| `rms_center` | **2.67** | 5 | **1.149** | 149.8 | 8/18 | $32.4\times$ |
| `batchnorm_noaffine` | 3.11 | 5 | 0.932 | **329.1** | 10/18 | $\mathbf{79.9\times}$ |
| `layernorm_noaffine` | 3.11 | 5 | 1.017 | **4.2** | 8/18 | $2.26\times$ |

Five variants spread across $2.67$-$3.11$ mean rank on a 1-to-5 scale is a tie. Every normalization loses to doing nothing on roughly half the cells. The rankings are stable across the last two thirds of training, so this is not an artefact of the shortened budget.

**Split by activation, the tie resolves into a disagreement:**

| | tanh: med ratio (regret, worst) | GELU: med ratio (regret, worst) |
|---|---|---|
| `baseline` | 1.000 (--) mean rank **2.78** | 1.000 (--) mean rank 3.33 (worst) |
| `rms_nocenter` | **1.005** (4/9, $1.63\times$) | 0.879 (5/9, $1.96\times$) |
| `rms_center` | 0.786 (5/9, $32.4\times$) mean rank **2.78** | **1.361** (3/9, $4.35\times$) mean rank **2.56** |
| `batchnorm_noaffine` | 0.779 (6/9, $79.9\times$) mean rank 3.44 (worst) | 1.339 (4/9, $11.9\times$) |
| `layernorm_noaffine` | 0.856 (5/9, $1.52\times$) | 1.162 (3/9, $2.26\times$) |

On GELU every normalization except scaling-only beats baseline at the median, and baseline is the *worst* arm. On tanh no normalization beats baseline at the median, and the best-performing GELU transforms are the worst tanh ones. **The answer to "does the winner differ between tanh and GELU" is yes, unambiguously.**

**Centering versus scaling, the mechanistic question.** Toggling one flag in one code path: GELU $0.879 \to 1.361$ (centering helps, and is the whole benefit -- scaling alone is worse than baseline); tanh $1.005 \to 0.786$ (centering hurts). Scaling alone on tanh is a near-exact no-op, as it must be: tanh's baseline column-norm spread is already $1.0$, so there is nothing to rescale. That null result is a useful check on the whole diagnosis -- the transform only does something where there is something to fix.

**Why centering is dangerous on tanh, verified causally.** A saturated tanh halo column is a near-constant $\pm1$, so its standard deviation is roundoff-level ($1.1\times10^{-16}$ at $N=128$). Centering and dividing by that std amplifies the residual fluctuation -- largely roundoff -- to unit variance, a factor of $8.8\times10^{15}$. A separate sweep clamping the std at a floor relative to the largest column std confirms the mechanism and its limits, on tanh/runge (baseline $3.64\times10^{-4}$):

| std floor (relative) | max amplification | final error | vs baseline |
|---|---:|---:|---:|
| none | $8.8\times10^{15}$ | $1.18\times10^{-2}$ | $0.031\times$ |
| $10^{-12}$ | $1.0\times10^{12}$ | $2.07\times10^{-3}$ | $0.176\times$ |
| $10^{-8}$ | $1.0\times10^{8}$ | $1.97\times10^{-3}$ | $0.185\times$ |
| $10^{-2}$ | $1.0\times10^{2}$ | $1.45\times10^{-3}$ | $0.251\times$ |

Capping recovers most of the catastrophe ($32\times \to 4\times$) but **never restores parity**: centering costs tanh something beyond the amplification artefact. The same sweep on GELU/runge is flat at $\approx1.97\times$ better than baseline across 300 orders of amplification, so the GELU benefit is genuine and not an artefact of extreme scaling. This also explains BatchNorm's behaviour from the other side: its $\varepsilon$ floor caps amplification at $1/\sqrt\varepsilon=316$, which is protective, at the cost of never normalizing the columns that most need it.

**Secondary measurements.** Column-norm spread at init, GELU 1-D: baseline $9.5\times10^{16}$, both static transforms exactly $1.0$, LayerNorm $6.6$, BatchNorm $3.5\times10^{12}$ (the $\varepsilon$ floor). On tanh 1-D the baseline spread is already $1.0$ and **BatchNorm manufactures a spread of $2.8\times10^{13}$ where none existed** -- a direct argument against it on an activation with no scale pathology. Dead-neuron fractions are unchanged by normalization on GELU ($18.6\% \to 17.3$-$18.6\%$), confirming expD19's finding that the frozen-halo and column-scale pathologies need different fixes; on tanh, `rms_center` alone drives the dead fraction to $0.0\%$ (from $23\%$), which is the same amplification acting usefully. Geometry damage on the floor-quality 1-D GELU cells worsens under every normalization ($1.5\times10^4$ baseline to $10^{7}$-$10^{9}$), reproducing expD19's preservation-versus-refinement tension at equal parameter count.

**Inverse-problem absolute accuracy** (median over seeds, $W=512$; reported per request, not plotted; plain Adam, so machine precision is not expected):

| activation | problem | best variant | recovered | absolute error | correct decimals | baseline decimals |
|---|---|---|---|---:|---:|---:|
| tanh | burgers $\nu=0.1$ | `rms_center` | 0.099980706 | $1.93\times10^{-5}$ | **3.71** | 2.57 |
| GELU | burgers $\nu=0.1$ | `layernorm` | 0.10014199 | $1.42\times10^{-4}$ | 2.85 | 2.72 |
| tanh | bratu $\lambda=1$ | `rms_center` / BN | 0.824 | $1.76\times10^{-1}$ | 0.75 | 0.12 |
| GELU | bratu $\lambda=1$ | BN | 0.873 | $1.27\times10^{-1}$ | 0.90 | 0.25 |
| both | allencahn $k=5$ | -- | 0.47-0.84 | $\approx4.4$ | 0.04-0.08 | 0.04-0.07 |

So: **two to four correct decimals on burgers, under one on bratu, and total failure on allencahn in every arm.** The centered transforms improve bratu by a factor of four to six on absolute error under both activations, which is expD19's inverse-PINN result reproduced at equal parameter count and with seeds. Allen-Cahn is untouched by anything tested, under either activation -- consistent with expD17 and expD19, and still undiagnosed.

### Figures

- **`figures/expD21_ranks.png`** -- the consistency picture. Rows are the five variants, columns the 18 (activation, class, problem) cells with tanh left of the black divider and GELU right; colour is the ratio to baseline on a log diverging scale, green better. A variant that deserved to be adopted would show as a uniformly green row; none does. Read the tanh half against the GELU half for the headline: the `rms_center` and BatchNorm rows change sign across the divider.
- **`figures/expD21_curves.png`** -- eval relative $L_2$ against iteration, median over the three seeds, one line per variant, six rows (activation $\times$ class) by three problems, log $y$. Use it to confirm the rankings are not an artefact of stopping at 2000 steps: the orderings are set well before the end and the curves are flat by then.

## Additional details

The PINN class keeps expD19's frozen-BatchNorm treatment (prime the running statistics on the full point set, then pin to eval mode), because the PINN loss makes three separate forward passes and train-mode batch statistics would normalize each block differently, making the residual inconsistent with the boundary fit. Frozen BatchNorm is a fixed data-derived affine map, so it tests the normalization rather than a known BatchNorm/PINN incompatibility.

The static transforms clamp the divisor only at $10^{-300}$, which protects exact zeros but not the roundoff-scale standard deviations that cause the tanh failure. A relative floor is the obvious repair and is quantified above; it was not adopted as a variant here because the directive fixed the five arms.

Single width per class and a 2000-step budget; the ranking-stability check shows the orderings are settled by two thirds of the run, but nothing here speaks to longer training or other widths.

## Conclusions

*Pending Sam.* The probe does not identify a normalization that is consistently better than doing nothing: across 18 cells the five variants tie on mean rank and each normalization loses to baseline on roughly half the cells. What it does establish is sharper than a winner. **The choice is activation-conditional**: centered normalization is worth a median $1.36\times$ on GELU, where baseline is the worst arm, and is actively harmful on tanh, where baseline ties for best. **Centering rather than scaling is the active ingredient**, verified by a one-flag toggle in a single code path, and its sign flips between the two activations for a mechanical reason -- tanh's saturated halo columns are near-constant, so centering amplifies their roundoff to unit variance, while GELU's left-halo columns are ramps that centering leaves informative. **BatchNorm should be dropped**: at equal parameter count it is the least consistent variant tested (spread $329\times$, worst regret $80\times$), its $\varepsilon$ floor prevents it from normalizing the columns that most need it, and on tanh it manufactures a column-norm spread where none existed.

Recommendation for the factorial: carry **`rms_center`** (the GELU-side candidate, parameter-free, gate-compatible, and the strongest on the inverse problems) and **`layernorm_noaffine`** (the lowest-tail-risk variant, spread $4.2\times$, mildly positive on GELU and mildly negative on tanh), and treat "no normalization" as the tanh default rather than an arm to be beaten. Adding a relative std floor to `rms_center` before the factorial is cheap and would remove its single worst failure mode.

## Open questions

- Does a relative std floor make `rms_center` safe on tanh without giving up its GELU benefit? The sweep above says it recovers $32\times \to 4\times$ but not to parity, so a floor alone is not sufficient and the residual cost of centering on tanh is unexplained.
- The GELU benefit of centering is measured on 9 cells at one width and 2000 steps. Whether it survives longer training and larger widths is untested, and expD17's width scaling suggests the pathology's magnitude grows with $\gamma=O(N)$.
- Allen-Cahn's parameter is missed by every variant under both activations, as in expD17 and expD19. Three experiments have now failed on it without diagnosis.
- Normalization worsens geometry preservation on floor-quality geometries under every variant, reproducing expD19. The variant that conditions only the badly-scaled columns and leaves correctly-placed ones alone remains unbuilt and is the obvious next design.
