# expD17 / norm factorial -- how much geometry movement each activation allows, and what a frozen normalization changes

**Status: draft -- pending Sam's review.** Third study in the expD17 family, after `../width_scaling` (tanh) and `../width_scaling_gelu` (GELU). Same protocol; the two arms become six, crossing activation, normalization and init. 192 runs, one seed.

## TL;DR

- **`rms_center` is a brake on geometry motion, and it brakes the standard init hardest.** GELU standard-init relative drift falls $8.8\times$ (1-D), $6.9\times$ (2-D), $3.1\times$ (PINN), $2.2\times$ (tabular) when the normalization is switched on. The two inits' motion is compressed toward each other rather than the QI arm being freed.
- **It rescues the GELU + QI failure on inverse problems, decisively.** bratu_$\lambda$ at $W=512$ went from a diverged run (recovered $\hat\lambda=0.329$, field error $3.66\times10^{-1}$) to the **best arm in the study** ($\hat\lambda=0.9983$, field $3.51\times10^{-3}$): a $104\times$ field-error improvement and $0.17\to2.78$ correct decimals.
- **And it is not a general win.** On the same grid it *hurts* 2-D interpolation ($0.23$-$1.0\times$) and tabular ($0.62$-$0.94\times$), and it hurts the GELU standard arm nearly everywhere. Mean rank over 32 cells: `tanh/no-norm/QI` **2.25**, then `gelu/rms_center/QI` 3.41 and `gelu/no-norm/QI` 3.44 (a tie), with `gelu/rms_center/std` last at 4.22.
- **The best arm in the study is still tanh + QI + no normalization**, first in 9 of 32 cells. Nothing here displaces it.
- **`rms_center` fixes the column-scale pathology completely and the dead-neuron pathology not at all**: the GELU + QI column-RMS spread goes $5.8\times10^{16}\to1.00$ in 1-D, while the dead fraction only moves $15.9\%\to14.7\%$. Two distinct diseases, one cure each, and this cure treats only the first.
- **On real tabular data at depth 2 the QI init does not help** (six parity tasks, std/QI ratio $0.62$-$1.05$) with **one striking exception: `naval`**, where QI is $2.6$-$3.2\times$ better and the best arm reaches $1.08\times10^{-2}$ -- still $10\times$ short of a degree-3 polynomial's $1.07\times10^{-3}$ on the same split.

## Question

Two questions, and the factorial exists to separate them.

1. **How much parameter movement does our init allow**, for tanh and for GELU, against a standard-init baseline -- and does adding a frozen normalization change that?
2. **How do the arms comparatively perform**, and does normalization change the ranking?

The prior studies answered the first question two arms at a time, and the answers disagreed: under tanh the QI init preserves a correct geometry and wins inverse problems, while under GELU the same init produces a feature matrix whose column norms span sixteen orders and destabilizes training at width. expD21 then selected one normalization out of five. This study puts all of it in one grid where activation, normalization and init vary independently.

## Experiment design

Identical to `../width_scaling` in every respect except the arm set and the tabular class: plain Adam and no solves, readout **zero-initialized in every arm**, 5000 full-batch steps with a cosine schedule and 100-step warmup, fp64 on CPU, **one seed**, drift tracked every iteration, per-neuron snapshots on a schedule that is dense at the start and capped at 50 steps.

### The six arms

| # | activation | normalization | init |
|---|---|---|---|
| 1 | tanh | none | standard |
| 2 | tanh | none | QI |
| 3 | GELU | none | standard |
| 4 | GELU | none | QI |
| 5 | GELU | `rms_center` | standard |
| 6 | GELU | `rms_center` | QI |

**tanh $\times$ `rms_center` is deliberately absent.** expD21 measured it at $0.79\times$ baseline with a $32\times$ worst case, and identified the mechanism: a saturated tanh halo column has RMS exactly $1$ but standard deviation $\sim10^{-14}$, so centering divides by roundoff and amplifies it to unit variance. Re-measuring that would spend a sixth of the grid on a known artifact.

**`rms_center`** is $h \mapsto (h - \operatorname{mean}(h_{\rm init}))/\operatorname{std}(h_{\rm init})$ with both statistics frozen at initialization, i.e. exactly a BatchNorm pinned to its initial statistics. It carries **no learnable parameters**, so arms differ only in the reparameterization (a gate test asserts equal parameter counts; expD19's BatchNorm and LayerNorm arms carried $2W$ extra parameters, which confounded every comparison there). Because the readout is zeroed at init, the transform changes nothing about the represented function and nothing about what the geometry can express -- only the gradient geometry. A second gate test asserts this directly: the pre-train lstsq probe agrees to $10^{-12}$ with and without it.

The normalization is applied to the output of the **QI-initialized layer** in every class, which is the layer carrying the column-scale pathology.

### Bandwidths, halo, widths

Bandwidths follow expC07's aliasing rule per activation: $\lambda^*=0.25$ for tanh and $0.707$ for GELU, with every class's tuned tanh $\lambda$ scaled by the same ratio $2.828$ -- identical to `../width_scaling_gelu`. The **halo rule is held fixed** in every arm; the halo question is not asked here. One consequence to read carefully: the 1-D halo rule $R=\max(\lceil 35/(2\lambda)\rceil, \lceil 0.4N\rceil)$ depends on $\lambda$, so the GELU arms have a *smaller* network than the tanh arms at the same $N$ ($W=115$ vs $205$ at $N=64$). Parameter counts are identical across norm and init within an activation, and differ across activations for this reason alone.

Two widths per class, the largest of the `width_scaling` triple dropped:

| class | problems | widths |
|---|---|---|
| 1-D interp | sine, runge, sine_8pi | $N\in\{64,128\}$ |
| 2-D interp | gauss_bump, sine2d, mixed2d | requested ridges $\{288,576\}$ |
| 2-D inverse PINN | burgers_$\nu$, bratu_$\lambda$, allencahn_$k$ | requested ridges $\{256,512\}$ |
| tabular | pol, kin8nm, parkinsons, bike_sharing, airfoil, sarcos, naval | $W\in\{128,256\}$ per layer |

### The tabular class is different, deliberately

**Two hidden layers.** expD20 measured one hidden layer as width-saturated on real tabular regression: a sixteen-fold width increase changes test error by under 5% on all seventeen tasks tested, so the headroom there is depth-headroom. A single-layer tabular row would have measured nothing.

**Only the first layer is the geometry.** The QI-family init (`scaled_psqrt`, expF04's best variant) is applied to layer 1 only; layer 2 keeps standard init; and every drift, $\gamma$, center and dead-neuron metric reads layer 1 and nothing else. The consequence for comparability is stated plainly: the tabular row is **not** comparable to the single-layer rows on width scaling.

**A corrected metric.** Tabular uses expD20's suite and expD20's preprocessing, which standardizes the target on the train split, so the reported error is variance-normalized and a mean-predictor scores $1.0$. The expF04 cache used previously stores min-max normalized targets, and the old relative $L_2$ therefore gave every model free credit for predicting the offset (a mean-predictor scored $0.52$ on bike_sharing). The other three classes keep the existing relative $L_2$ unchanged, since their targets are centered and it is honest there.

**Training size is capped at 6000 rows**, applied identically to every arm, because a full-batch fp64 two-layer net is quadratic in width and the class is 84 runs. The cap does not move the reference point: a degree-3 polynomial ridge on `naval` reaches $1.069\times10^{-3}$ under this exact pipeline against expD20's $1.04\times10^{-3}$ on the full split.

### Metrics

Geometry vector $g_i$ = first-layer weight and bias concatenated (readout excluded; the PDE scalar excluded and logged separately). Per-neuron bandwidth $\gamma_k := \|w_k\|_2$ and center $c_k := -b_k/\gamma_k$, as in `../width_scaling`.

Per run: relative drift $\|g_i-g_0\|/\|g_0\|$ and absolute drift $\|g_i-g_0\|$; the two lstsq probes of PROGRAM_FRAMING §4.3 (pre-train on the init geometry, post-train on the trained geometry with the readout discarded), giving a **geometry score** (post/pre) and a **readout score** (final/post); dead-neuron fraction split by preactivation sign class; column-RMS spread of the QI layer at init and at the end; parameter count; and for the PINN class the recovered PDE parameter with absolute error and correct decimals.

**Code & data.** `experiments/expD17_geometry_motion/norm_factorial/{run.py, report.py}`; gate tests `tests/test_expD17n_norm_factorial.py` (7 passed). Data `results/checkpoint_D_optimizers/expD17_geometry_motion/norm_factorial/data/*.jsonl` (192 runs, one file per class/width/problem) plus `analysis.json`. Figures in `figures/`.

## Results

### Q1. How much movement is allowed

Relative drift at the end of training, geometric mean over each class's problems and both widths:

| arm | 1-D interp | 2-D interp | inverse PINN | tabular |
|---|---:|---:|---:|---:|
| tanh / no-norm / std | 1.50 | 8.44 | 4.26 | 1.39 |
| tanh / no-norm / QI | 1.46e-2 | 0.321 | 0.385 | 0.467 |
| GELU / no-norm / std | 8.52 | 5.91 | 2.66 | 1.51 |
| GELU / no-norm / QI | 1.15e-2 | 0.128 | 0.117 | 0.197 |
| GELU / `rms_center` / std | 0.973 | 0.860 | 0.858 | 0.704 |
| GELU / `rms_center` / QI | 2.19e-2 | 7.67e-2 | 6.04e-2 | 0.155 |

**The QI init allows one to three orders less relative movement than the standard init**, in every class and under both activations. QI/standard ratios: $0.010$/$0.038$/$0.090$/$0.336$ (tanh), $0.001$/$0.022$/$0.044$/$0.130$ (GELU no-norm), $0.022$/$0.089$/$0.070$/$0.220$ (GELU + `rms_center`). The ordering is the same everywhere and is the clearest single pattern in the study.

**`rms_center` acts as a brake, and it brakes the standard arm hardest.** Switching it on cuts GELU standard-init relative drift by $8.8\times$ (1-D), $6.9\times$ (2-D), $3.1\times$ (PINN) and $2.2\times$ (tabular); absolute drift falls similarly ($16.5\to1.88$, $14.4\to2.09$, $6.44\to2.08$, $12.2\to5.67$). On the QI arms it reduces motion in 2-D, PINN and tabular ($0.128\to0.077$, $0.117\to0.060$, $0.197\to0.155$) and *raises* it in 1-D ($1.15\to2.19\times10^{-2}$). Net effect: the two inits' motion is compressed toward each other, by slowing the standard arm rather than by freeing the QI arm.

This is worth stating against the expD19 reading, which suggested conditioning fixes "free up" motion. At equal parameter count and with a frozen transform, the opposite happens.

### Q2. How they comparatively perform

Final eval error, geometric mean over problems and widths:

| arm | 1-D interp | 2-D interp | inverse PINN | tabular |
|---|---:|---:|---:|---:|
| tanh / no-norm / std | 2.65e-1 | 1.31e-2 | 1.66e-1 | 1.70e-1 |
| **tanh / no-norm / QI** | **9.75e-4** | 4.21e-3 | 1.83e-2 | 1.49e-1 |
| GELU / no-norm / std | 9.21e-2 | 2.22e-2 | 1.73e-1 | 1.73e-1 |
| GELU / no-norm / QI | 5.59e-2 | **3.53e-3** | 5.22e-2 | 1.66e-1 |
| GELU / `rms_center` / std | 3.52e-1 | 9.29e-2 | 2.62e-1 | **1.40e-1** |
| GELU / `rms_center` / QI | 2.28e-2 | 8.02e-3 | **1.35e-2** | 1.79e-1 |

Rank over the 32 cells (1 = best of six):

| arm | mean rank | worst | # best |
|---|---:|---:|---:|
| **tanh / no-norm / QI** | **2.25** | 4 | 9 |
| GELU / `rms_center` / QI | 3.41 | 6 | 6 |
| GELU / no-norm / QI | 3.44 | 6 | 4 |
| GELU / no-norm / std | 3.69 | 6 | 5 |
| tanh / no-norm / std | 4.00 | 6 | 3 |
| GELU / `rms_center` / std | 4.22 | 6 | 5 |

**Normalization does not change the overall ranking**: `tanh/no-norm/QI` is first by a clear margin and the two GELU QI arms are tied in the middle. What normalization changes is *where* GELU + QI works. Head-to-head (`gelu_none`/`gelu_rmsc`, $>1$ means `rms_center` better), on the QI init: 1-D $1.1$-$4.1\times$ better, PINN $1.5$-$104\times$ better on burgers and bratu, but 2-D interp $0.23$-$1.0\times$ (worse) and tabular $0.66$-$0.92\times$ (worse, `naval` excepted at $1.27$-$1.77\times$). On the standard init `rms_center` is worse nearly everywhere ($0.08$-$1.2\times$), with `naval` the one large exception at $5.2$-$5.3\times$.

So the honest answer to "does `rms_center` rescue GELU + QI" is: **yes on inverse problems and 1-D interpolation, no on 2-D interpolation and tabular.** It is a regime-dependent fix, not a general one.

### Q3. The probes

Geometry score (post/pre; $<1$ = training improved the geometry) and readout score (final/post; $>1$ = Adam left accuracy on the table), geometric means:

| arm | geometry 1-D | geometry 2-D | geometry PINN | geometry tab | readout tab |
|---|---:|---:|---:|---:|---:|
| tanh / no-norm / std | 0.28 | 0.000 | 0.27 | 0.48 | 1.62 |
| tanh / no-norm / QI | 1.1e4 | 0.104 | 35.6 | 0.43 | 1.28 |
| GELU / no-norm / std | 0.000 | 0.001 | 0.16 | 0.48 | 1.66 |
| GELU / no-norm / QI | 1.7e5 | 0.034 | 17.4 | 0.47 | 1.18 |
| GELU / `rms_center` / std | 0.063 | 0.295 | 0.96 | 0.51 | 1.23 |
| GELU / `rms_center` / QI | 4.4e7 | 0.039 | 17.6 | 0.48 | 1.05 |

The damage-at-the-floor / improvement-off-the-floor split of the earlier studies reproduces exactly, and `rms_center` makes the 1-D floor damage **worse** ($1.7\times10^5 \to 4.4\times10^7$), consistent with expD19 and expD21. Where the geometry starts far from its floor (2-D, tabular) every arm improves it.

The readout score on the interpolation classes is enormous for every arm ($10^2$ to $10^6$): plain Adam leaves the available readout accuracy unclaimed, which is the founding result of the program and is not disturbed here. On tabular it is small ($1.05$-$1.66$), because a second hidden layer plus a noise-floored target leaves little for a terminal solve to recover.

### Q4. Inverse problems, absolute accuracy

Plain Adam, so machine precision is not expected. Correct decimals $=-\log_{10}(|\hat p - p|/|p|)$:

| problem | $W$ | best arm | $\hat p$ | abs err | decimals | field err |
|---|---:|---|---:|---:|---:|---:|
| burgers $\nu$ ($0.1$) | 256 | GELU/no-norm/std | 0.099888 | 1.12e-4 | 2.95 | 1.62e-2 |
| burgers $\nu$ | 512 | tanh/no-norm/QI | 0.099898 | 1.02e-4 | 2.99 | 9.31e-3 |
| bratu $\lambda$ ($1.0$) | 256 | **GELU/`rms_center`/QI** | 0.996572 | 3.43e-3 | 2.47 | 5.28e-3 |
| bratu $\lambda$ | 512 | **GELU/`rms_center`/QI** | 0.998330 | 1.67e-3 | 2.78 | 3.51e-3 |
| allencahn $k$ ($5.0$) | both | none | 0.46-6.04 | $\ge$6.7e-1 | $\le$0.87 | $\ge$7.0e-2 |

**Two to three correct decimals** is the ceiling under plain Adam, matching expD19's range. The bratu $W=512$ cell is the study's sharpest single result: `gelu/no-norm/QI` diverges there ($\hat\lambda=0.329$, $0.17$ decimals, field $3.66\times10^{-1}$) and `gelu/rms_center/QI` is the best arm anywhere on that problem ($\hat\lambda=0.9983$, $2.78$ decimals, field $3.51\times10^{-3}$).

**Allen-Cahn fails in every arm for the fourth consecutive experiment.** Every QI arm recovers $\hat k\in[0.46,2.04]$ against $k^*=5$. The one arm that gets the parameter roughly right, `gelu/rms_center/std` ($\hat k = 5.67$ and $6.04$, $0.68$-$0.87$ decimals), has the *worst* field error in the class ($1.95$), so it is a degenerate fit rather than a solution. This is now a standing unexplained failure and should be diagnosed rather than re-measured.

### Q5. Tabular

Variance-normalized test error; $1.0$ is a mean-predictor. Best arm per row starred:

| task | $W$ | tanh std | tanh QI | GELU std | GELU QI | GELU rc std | GELU rc QI |
|---|---:|---:|---:|---:|---:|---:|---:|
| pol | 256 | 0.0990 | 0.0988 | **0.0862** | 0.1308 | 0.1024 | 0.1524 |
| kin8nm | 256 | 0.3351 | 0.3300 | **0.2982** | 0.3420 | 0.3324 | 0.3865 |
| parkinsons | 256 | **0.3077** | 0.3808 | 0.4604 | 0.5306 | 0.4511 | 0.5948 |
| bike_sharing | 256 | 0.2832 | 0.2989 | **0.2447** | 0.3139 | 0.2567 | 0.4759 |
| airfoil | 256 | 0.1876 | 0.1766 | 0.1884 | 0.2139 | **0.1742** | 0.2107 |
| sarcos | 256 | **0.1356** | 0.1427 | 0.1365 | 0.1541 | 0.1377 | 0.1647 |
| naval | 256 | 0.0574 | 0.0242 | 0.0572 | 0.0158 | **0.0108** | 0.0124 |

**On the six parity tasks the QI init does not help.** Standard/QI ratios are $0.62$-$1.05$: under tanh it is a near-tie (five of six within 10%), under GELU the QI arm loses by $8$-$25\%$, and `rms_center` widens the loss. This is the cleanest measurement yet of PROGRAM_FRAMING §7.4 on real data at depth 2, and it is negative.

**`naval` is the exception, and a large one.** The QI init is $2.6\times$ (tanh) and $3.2\times$ (GELU) better than standard, and `rms_center` gives the study's best number on it, $1.08\times10^{-2}$. But the reference on the same split is a degree-3 polynomial at $1.07\times10^{-3}$: **every arm is still $10\times$ worse than a linear-in-parameters smooth basis.** The gap expD20 identified survives contact with the QI init and the normalization.

**The `naval` plateau is an approximation limit, not label quantization.** The target takes 51 distinct values on an exact $10^{-3}$ grid over $[0.95,1.0]$, which is a grid step of $0.0682$ in standardized units. A model at rel $L_2 = 1.04\times10^{-3}$ has an RMS residual of $0.0011$, i.e. $1.6\%$ of one grid step; the neural nets at $1.35\times10^{-2}$ sit at $20\%$ of a step. Both are far inside the quantization, so nothing here is floored by the labels, and the task can carry a precision claim.

### Q6-Q7. The two pathologies are separate

The column-RMS spread of the QI layer at init (max/min over live columns):

| arm | 1-D | 2-D | PINN | tabular |
|---|---:|---:|---:|---:|
| GELU / no-norm / QI | 5.8e16 | 3.8e7 | 3.0e5 | 1.3e2 |
| GELU / `rms_center` / QI | 1.00 | 1.00 | 1.00 | 1.00 |

`rms_center` removes the column-scale pathology exactly, by construction. It does **not** remove the dead neurons: in 1-D the GELU + QI dead fraction moves only $15.9\%\to14.7\%$, with the sign asymmetry intact (dead among always-negative preactivations $72.5\%\to67.2\%$; dead among always-positive $0.0\%$ in both). The frozen-halo pathology and the column-scale pathology are independent, and this normalization treats only the second.

### Figures

- **`figures/expD17n_drift_from_init.png`** -- 4x3, rows = problem class, columns = problems (tabular row shows pol / kin8nm / naval), $y=\|g_i-g_0\|/\|g_0\|$ on a log scale, rows sharing the y scale. Colour family = activation + normalization (blue tanh, red GELU, green GELU + `rms_center`), lightness = init (light standard, dark QI), linestyle = width (dashed small). Read the light-versus-dark separation for the init effect and the green-versus-red light lines for the brake `rms_center` applies to the standard arm.
- **`figures/expD17n_step_size.png`** -- same layout, per-iteration motion $\|g_i-g_{i-1}\|/\|g_0\|$. All arms decay together with the cosine schedule; the arm ordering is the same as the drift figure.
- **`figures/expD17n_drift_from_init_abs.png`**, **`expD17n_step_size_abs.png`** -- the same two views without the $\|g_0\|$ normalization. The init separation collapses relative to the normalized view, reproducing the earlier studies' conclusion that most of the relative gap is units.
- **`figures/expD17n_loss.png`** -- run error, same layout and colours. The 1-D panel is where `tanh/no-norm/QI` (dark blue) separates from everything by two orders; the bratu panel shows the diverged red `gelu/no-norm/QI` line sitting at $3.7\times10^{-1}$ while dark green descends to $3.5\times10^{-3}$.
- **`figures/gamma_hist/expD17n_gamma_hist_<arm>_w{1,2}.gif`** and **`figures/center_hist/...`** -- per-neuron $\gamma$ and center histograms over training, QI-init arms only, one gif per arm and width, red dotted median, mean in the title, axes fixed across frames, frames dense at the start. Watch the 1-D $\gamma$ delta broaden in place while the center histogram stays put.
- **`figures/gamma_vs_update/expD17n_gamma_vs_update_w{1,2}.png`** -- per-neuron $\gamma$ against applied Adam update, all six arms overlaid by colour, opacity ramping with training step, log-log. The `rms_center` arms collapse onto a single $\gamma$ column, which is the same fact the Q6 table reports.

## Additional details

**The tanh no-norm arms reproduce `../width_scaling` bit for bit**, which is the validity check on the refactor: bratu_$\lambda$ at $W=512$, standard init, recovers $\hat\lambda=0.9146485891571732$ in both studies. The activation switch and the normalization buffers leave the original code path numerically untouched (the buffers are $0$ and $1$, and $(x-0)\cdot 1 = x$ exactly in IEEE arithmetic).

**Caveats.** One seed; our QI runs are deterministic by construction (formula init, zeroed readout, full batch, fixed sample grids), so there is no initialization randomness to average over and a seed axis would vary only the data realization. tanh $\times$ `rms_center` is absent by design, so no statement here covers it. The halo is fixed, so nothing here speaks to halo sizing. The tabular class is two layers, capped at 6000 training rows, and its width axis is not comparable to the other three rows.

## Conclusions

*Pending Sam.* Under plain Adam the QI init allows one to three orders less relative geometry movement than a standard init, in every problem class and under both activations, and a frozen `rms_center` normalization brakes movement further -- most strongly on the standard init, which it slows by $2$ to $9\times$, compressing the two inits toward each other rather than freeing the QI arm.

On performance the normalization does not change the overall ranking: tanh with the QI init and no normalization is the best arm in the study, first in 9 of 32 cells. What `rms_center` changes is where GELU + QI is usable. It converts the GELU + QI inverse-PINN failure into the study's best inverse-problem result ($104\times$ better field error on bratu at $W=512$, $2.78$ correct decimals), helps 1-D interpolation, and hurts 2-D interpolation and tabular. It removes the column-scale pathology exactly and the dead-neuron pathology not at all.

On real tabular data at depth 2 the QI init does not help on six parity tasks, and does help by $2.6$-$3.2\times$ on `naval`, where the target is a near-noiseless simulator output -- though every arm there remains $10\times$ short of a degree-3 polynomial.

## Open questions

- Why does `rms_center` help inverse problems and 1-D interpolation while hurting 2-D interpolation and tabular? The column-spread fix is identical in all four; the divergence must come from something else the transform does.
- Allen-Cahn has now defeated every arm in four consecutive experiments. It needs a diagnosis (is the manufactured setup identifiable at all from this data?) rather than another measurement.
- The dead-neuron pathology survives every normalization tested so far. It is a property of where our init puts neurons relative to the data, so the candidate fixes are geometric (halo sizing, softened halo bandwidth), not normalization-based.
- `naval` is the one real dataset where a smooth linear-in-parameters basis beats every network by an order of magnitude, and the QI init closes part but not all of that gap. Whether a readout solve on the QI geometry closes the rest is the obvious next measurement, and it is cheap.
