# expD19 -- does a scale-aware GELU init fix what expD17 broke?

**Status: draft -- pending Sam's review.** Follows `../expD17_geometry_motion/width_scaling_gelu/expD17w_gelu_results.md`, which measured the pathologies this experiment tries to repair.

## TL;DR

- **The PINN divergence is fixed, decisively, and every arm fixes it.** On bratu, where expD17's GELU-plus-QI run came apart, the recovered PDE parameter goes from $67.1\%$ error (baseline) to $0.2$-$1.5\%$, and the field error from $3.7\times10^{-1}$ to $3.3\times10^{-3}$: $432\times$ better on the parameter and $110\times$ on the field. This is the pathology the scale fix was aimed at, and it lands.
- **On plain interpolation the same fixes buy a factor of one to three and nothing more.** Every arm ends four to five orders above its geometry's own lstsq floor. The scale repair does not turn GELU training into precision training.
- **The split between those two results is the whole finding**, and it matches expD17's damage-at-the-floor / improvement-off-the-floor law: conditioning matters when the geometry still has to be *learned* (PINN, off-floor), and is nearly irrelevant when the geometry is already at its floor and only has to be *preserved* (1-D interpolation).
- **Two pathologies, two disjoint fixes.** Shrinking the halo removes the frozen neurons completely ($18$-$20\%\to0.0\%$) and barely moves the column-norm spread; column normalization flattens the spread exactly ($10^{17}\to1.0$) and leaves the frozen fraction untouched. Neither addresses the other's problem.
- **Answering the norm-layer question: BatchNorm is the best arm where it matters and the worst where it does not.** It wins every PINN cell (bratu $0.2\%$) and is the weakest arm on 1-D and 2-D interpolation (2-D gauss_bump $1.8\times10^{-2}$ against baseline's $7.8\times10^{-4}$, $23\times$ worse). LayerNorm is the most consistent, never worst anywhere. The parameter-free static reparameterization achieves the same spread fix at zero parameter cost.
- **The halo rule $R\approx0.4N$ is oversized for both activations**: tanh reaches $4.4\times10^{-15}$ at halo 8 versus $3.4\times10^{-14}$ at halo 59, with $41\%$ fewer neurons.

## Question

expD17 found three GELU-plus-QI-init pathologies, all absent under tanh: 68-92% of right-halo neurons frozen, 2-3 orders more floor damage, and inverse-PINN divergence at width. The diagnosis was that the init is scale-blind: with $\gamma=O(N)$ and $\mathrm{gelu}(z)\to z$, feature-column norms grow linearly in width on the left halo and underflow to zero on the right, so $\Phi^\top\Phi$'s diagonal spans ten orders. Does repairing the scale repair the training?

## Experiment design

Six arms, all with the QI geometry, plain Adam, readout zero-initialized, one hidden layer, 5000 full-batch steps on expD17's cosine schedule, fp64 on CPU, single seed. Only the parameterization differs.

| arm | halo | feature scaling | extra terms | params ($N{=}256$, 1-D) |
|---|---|---|---|---:|
| `baseline` | $R=\texttt{default\_halo}\approx0.4N$ | none | bias | 1384 |
| `halo8` | 8 per side (2-D/PINN: radon collar $2.5\to1.3$) | none | bias | 820 |
| `static_colnorm` | full | $\phi_k\mapsto\phi_k/\|\phi_k\|_{\rm rms}$, computed once at init | bias | 1384 |
| `recommended` | 8 per side | same | bias **and a linear term** | 821 |
| `batchnorm` | full | `BatchNorm1d`, affine | bias | 2306 |
| `layernorm` | full | `LayerNorm`, affine | bias | 2306 |

- **`static_colnorm` is a pure reparameterization.** The column scales live in a fixed buffer: no trainable weights, no running statistics, no batch dependence. With the readout zeroed the represented function at init is unchanged; only the gradient geometry differs. This is the variant compatible with `docs/REQUIREMENTS.md`'s practicality gate (optimizer state stays $O(m)$, Adam's class), which is why it is separated from the norm layers.
- **The linear term is theory, not a hyperparameter.** GELU has kernel order $r=2$ ($K=\mathrm{gelu}''$), so the paper's integration polynomial $p_{r-1}$ is affine; tanh at $r=1$ needs only the single bias our init has always supplied. `recommended` is the first arm to give GELU the term its own construction calls for.
- **Bandwidth.** $\lambda^*=0.707$ for GELU throughout (expC07's aliasing rule), $0.25$ for tanh, per-class $\lambda$ scaled by the same ratio, exactly as expD17 did.
- **Metrics.** Final eval relative $L_2$; the two lstsq probes of PROGRAM_FRAMING §4.3 (truncated SVD at rcond $10^{-13}$ on the init geometry and on the trained geometry, with the $x$ columns included for arms that carry a linear term, so each arm is scored against the best readout *its* parameterization can express); RMS feature-column norms by preactivation sign class; the dead-neuron fraction (applied Adam update below $10^{-10}$ across the first 30 snapshots, expD17's criterion) split by whether a neuron's preactivation is always negative, always positive, or sign-spanning over the data; and the recovered PDE parameter for the inverse cells.
- **BatchNorm on the PINN class is frozen after priming.** The PINN loss makes three separate forward passes (collocation, boundary, data); in train mode each block would be normalized by its own batch statistics, so the PDE residual and the boundary fit would see three different networks. Priming the running statistics on the full point set and then freezing makes BN a fixed data-derived affine map, testing the scale hypothesis rather than a known BN/PINN incompatibility. Single-block full-batch losses (1-D, 2-D) need no such handling: with `momentum=None` and one repeated batch, train-mode BN and the primed statistics coincide.
- **tanh cross-check.** `baseline` and `recommended` on the 1-D class at both widths.

**Scope cuts, stated plainly.** The inverse-PINN row runs at $W=512$ only; $W=1024$ was dropped for runtime. The divergence already appears at 512 in expD17 (bratu: 67.1% parameter error, field error 0.366) and this experiment reproduces that number exactly, so the test lands, but the fixes are untested at $W=1024$, where expD17 recorded 90.3% and field error 87. What is untested is whether the fixes hold at the larger width, not whether the pathology exists there. The 2-D tanh `recommended` cross-check was also cut, so the tanh comparison rests on the six 1-D cells. Single seed throughout, as in expD17; the 1-D and 2-D arm differences below are mostly within a factor of three and should be read as trends, whereas the PINN differences span two orders.

**Code & data.** `experiments/expD19_gelu_init/{run.py, analysis.py, static_probe.py}`; gate test `tests/test_expD19_gelu_init.py` (8 tests pinning the static diagnosis). Data `results/checkpoint_D_optimizers/expD19_gelu_init/data/*.jsonl` (87 runs). Figures `figures/expD19_{arms,colnorm,pinn_recovery}.png`.

## Results

### The static diagnosis, reproduced

1-D sine, $N=128$, RMS column norms and the truncated-SVD floor:

| variant | $W$ | interior | left halo | right halo | max/min | lstsq floor |
|---|---:|---:|---:|---:|---:|---:|
| GELU baseline | 248 | 20.9 | 71.5 | $4\times10^{-4}$ | $4\times10^{303}$ | $2.25\times10^{-13}$ |
| GELU + colnorm + linear | 249 | 1.0 | 1.0 | 0.19 | -- | $2.89\times10^{-14}$ |
| GELU halo 8/8 + colnorm + linear | 147 | 1.0 | 1.0 | 1.0 | 1.0 | $2.73\times10^{-14}$ |
| tanh baseline | 248 | 0.97 | 1.00 | 1.00 | 1.03 | $3.37\times10^{-14}$ |
| tanh halo 8/8 | 146 | 0.97 | 1.00 | 1.00 | 1.03 | $4.42\times10^{-15}$ |

The geometry was never the problem: GELU's lstsq floor is $2.3\times10^{-13}$ despite column norms spanning 300 orders, because truncated SVD is scale-robust. Normalization improves the floor about $8\times$, and the small halo matches the full halo at 40% fewer neurons -- for tanh it beats it, by $8\times$.

Two halo results worth separating. Dropping either side alone costs 3+ orders under GELU (halo 8/0: $5.3\times10^{-9}$; halo 0/8: $3.9\times10^{-5}$; halo 8/8: $4.1\times10^{-12}$ raw), so the naive reading of expD17's asymmetry -- "the right halo is dead, delete it" -- is wrong: the frozen neurons still carry boundary correction the solve needs, they simply cannot learn it. And no halo at all fails for both activations, which is what the small halo trades against.

### The inverse-PINN divergence: fixed by every arm

Recovered PDE parameter, relative error, $W=512$ (true values $\nu=0.1$, $\lambda=1$, $k=5$):

| problem | baseline | halo8 | static_colnorm | recommended | batchnorm | layernorm |
|---|---:|---:|---:|---:|---:|---:|
| burgers $\nu$ | 1.0% | 0.0% | 0.5% | 0.4% | **0.0%** | 1.1% |
| bratu $\lambda$ | **67.1%** | 9.0% | 0.8% | 1.5% | **0.2%** | 0.9% |
| allencahn $k$ | 90.9% | 86.7% | 88.6% | 88.5% | **85.8%** | 87.6% |

Field errors on bratu: baseline $3.66\times10^{-1}$, halo8 $3.80\times10^{-2}$, layernorm $9.16\times10^{-3}$, static_colnorm $9.90\times10^{-3}$, recommended $1.43\times10^{-2}$, **batchnorm $3.33\times10^{-3}$**. The baseline number reproduces expD17's $W=512$ divergence exactly, so this is a like-for-like repair, not a different setup.

Ordering the fixes by how much they flatten the column-norm spread predicts their ranking on bratu: baseline ($6.9\times10^{6}$, 67.1%), halo8 ($9.9\times10^{1}$, 9.0%), then the three arms that reach $\lesssim5$ (0.2-1.5%). No arm has any dead neurons in this class -- the radon collar keeps every ridge within reach of the data -- so the entire effect here is conditioning, cleanly separated from the frozen-neuron pathology.

Allen-Cahn resists every arm on the parameter (85.8-90.9%), as it did in expD17 under both activations, though the field error still improves $4\times$ (baseline $2.91\times10^{-1}$, best $7.47\times10^{-2}$). Whatever defeats that problem is not the feature scale.

### Interpolation: the same fixes buy almost nothing

Final eval relative $L_2$, GELU:

| cell | baseline | halo8 | static_colnorm | recommended | batchnorm | layernorm |
|---|---:|---:|---:|---:|---:|---:|
| 1-D sine $W{=}256$ | 2.5e-2 | 1.9e-2 | 2.8e-2 | 2.7e-2 | 3.6e-2 | **1.2e-2** |
| 1-D runge $W{=}256$ | 2.0e-2 | 1.1e-2 | 7.9e-3 | **5.4e-3** | 1.2e-2 | 5.5e-3 |
| 1-D sine_8pi $W{=}256$ | 6.7e-1 | 5.5e-1 | 5.2e-1 | 4.7e-1 | **3.3e-1** | 4.7e-1 |
| 2-D gauss_bump | 7.8e-4 | 5.5e-4 | **5.3e-4** | 7.4e-4 | 1.8e-2 | 6.2e-4 |
| 2-D sine2d | 2.0e-2 | 2.2e-2 | **9.2e-3** | 2.4e-2 | 6.8e-2 | 1.9e-2 |
| 2-D mixed2d | 5.2e-3 | 4.3e-3 | 6.4e-3 | 5.2e-3 | 5.0e-2 | **4.3e-3** |

Median over the six 1-D cells of (baseline error / arm error): layernorm $2.37\times$, recommended $1.45$, halo8 $1.37$, static_colnorm $1.25$, batchnorm $1.20$. Every arm ends between $5\times10^{-3}$ and $6\times10^{-1}$ against geometries whose lstsq floor is $10^{-13}$. **BatchNorm is the weakest arm on 2-D interpolation by a wide margin** ($23\times$ worse than baseline on gauss_bump), the mirror image of its PINN result.

### Geometry preservation: the fixes make it worse where the geometry is already right

Post-train probe divided by pre-train probe, so $>1$ means training damaged the geometry:

| cell | baseline | halo8 | static_colnorm | recommended | batchnorm | layernorm |
|---|---:|---:|---:|---:|---:|---:|
| 1-D sine $W{=}256$ | $4.6\times10^{2}$ | $\mathbf{2.9\times10^{1}}$ | $2.3\times10^{9}$ | $7.9\times10^{8}$ | $3.8\times10^{7}$ | $1.9\times10^{8}$ |
| 1-D runge $W{=}256$ | $2.3\times10^{2}$ | $\mathbf{1.6\times10^{1}}$ | $6.1\times10^{5}$ | $3.9\times10^{5}$ | $6.6\times10^{6}$ | $1.5\times10^{4}$ |
| 2-D sine2d | $\mathbf{5.2\times10^{-4}}$ | $1.9\times10^{-1}$ | $2.4\times10^{-4}$ | $1.2\times10^{-1}$ | $2.3\times10^{-4}$ | $8.0\times10^{-1}$ |

In the 1-D cells, where the init already sits at its floor, `baseline` and `halo8` preserve the geometry best and every normalized arm is $10^3$-$10^4\times$ worse. The mechanism is the one expD17 identified from the other side: dividing a large-norm column by its norm amplifies the gradient reaching that neuron's inner weights, so the geometry moves more, and on a geometry already at its floor motion is damage. The ill-conditioning was acting as an accidental brake.

In the 2-D cells, where the init starts off-floor, ratios below 1 mean training *improved* the geometry, and the normalized arms improve it most. The same mechanism, opposite sign, exactly as PROGRAM_FRAMING's preservation-versus-refinement tension predicts.

### Dead neurons and column norms: disjoint fixes

Dead fraction and column-norm spread at init, GELU:

| cell | baseline | halo8 | static_colnorm | recommended | batchnorm | layernorm |
|---|---:|---:|---:|---:|---:|---:|
| 1-D sine $W{=}256$, dead | 20.4% | **0.0%** | 19.7% | **0.0%** | 20.2% | 20.4% |
| 1-D sine $W{=}256$, spread | $2.3\times10^{17}$ | $9.0\times10^{10}$ | **1.0** | **1.0** | $4.3\times10^{12}$ | 6.6 |
| 2-D gauss_bump, dead | 10.5% | **0.0%** | 2.6% | **0.0%** | 5.3% | 10.5% |
| PINN burgers, dead | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Only the arms that move capacity inside the data revive the frozen neurons, and they do it completely; only the normalizing arms flatten the spread. This is mechanically necessary rather than surprising: a right-halo neuron is frozen because its preactivation sits so deep in GELU's vanishing tail that the gradient underflows, and multiplying its numerically-zero output by any finite constant leaves the gradient zero. **BatchNorm barely helps the spread** ($4.3\times10^{12}$) because a dead column has zero variance, so BN's output for it is $0/\sqrt{\epsilon}$; normalizing per neuron is the right axis in principle and fails here on exactly the columns that are identically zero.

### tanh cross-check

`recommended` versus `baseline` under tanh, six 1-D cells: $4.1\times$, $0.68\times$, $1.06\times$, $15.0\times$, $0.41\times$, $4.3\times$ (ratios $>1$ favour `recommended`). Better on four of six, notably $15\times$ on sine at $W=256$, worse on runge at both widths. With one seed this is a trend, not a result; what it does establish is that the fix is not harmful to the activation that already worked, and that the smaller halo costs nothing at $41\%$ fewer neurons.

### Figures

- **`figures/expD19_arms.png`** -- rows = (class, width), cols = problem; one bar per GELU arm, final eval rel $L_2$ on a log axis, with the tanh baseline and tanh recommended as horizontal reference lines. Read bar heights within a panel for the arm ordering, and note that the PINN rows are the only ones where the ordering spans orders rather than factors.
- **`figures/expD19_colnorm.png`** -- one panel per class, x = arm, y = RMS feature-column norm at init, three markers per arm for the interior, positive-preactivation (left halo) and negative-preactivation (right halo) regions. An arm has fixed the scale pathology when its three markers coincide: `static_colnorm` and `recommended` collapse to a point, `layernorm` nearly does, `baseline` and `batchnorm` span the full panel.
- **`figures/expD19_pinn_recovery.png`** -- relative error in the recovered PDE parameter against iteration, one line per arm, dotted guide at 10%. The divergence test: baseline's bratu line rides above the guide for the whole run while every other arm descends below 2%.

## Additional details

Probes for arms with a linear term include the $x$ columns, so no arm is scored against a readout it cannot represent. BatchNorm features are taken in eval mode with primed running statistics, so both halves of a probe use one deterministic feature map.

`halo8` and `recommended` use 41% fewer neurons than the other arms ($W=273$ versus 461 at $N=256$), so their gains are a lower bound at equal cost. The norm layers add $2W$ trainable parameters (2306 versus 1384); the static reparameterization adds none; the linear term adds $d_{\rm in}$.

## Conclusions

*Pending Sam.* The scale diagnosis is correct and the repair matters exactly where the geometry still has to be learned. On the inverse-PINN problem that expD17 saw diverge, every arm fixes it and the best is $432\times$ better on the recovered parameter than the baseline, with the ranking following how completely each arm flattens the feature-column spread. On plain interpolation, where the init is already at its floor, the same repairs are worth a factor of one to three and leave every arm four to five orders above the lstsq floor, and they make geometry preservation worse by up to four orders because the conditioning they remove was suppressing the motion that damages a correct geometry. Answering the norm-layer question directly: BatchNorm is the strongest arm on the PINN cells and the weakest on interpolation (it cannot normalize a zero-variance dead column); LayerNorm is the most consistent; the parameter-free static reparameterization achieves the exact spread fix at no parameter cost and is the variant compatible with the practicality gate. Separately, the halo rule $R\approx0.4N$ is oversized for both activations, and both halo sides remain necessary under GELU despite the one-sided death.

## Open questions

- The two pathologies want opposite things on a floor-quality geometry (fix the conditioning and you also unfreeze the damage). Is there a variant that conditions only the neurons that are badly scaled and leaves correctly-placed ones alone -- normalizing only columns whose norm exceeds the interior scale?
- LayerNorm gives the best 1-D run error while leaving a factor-7 spread, so its benefit is probably not the scale fix. Per-sample mean subtraction is a different mechanism and was not isolated here.
- Does the PINN repair hold at $W=1024$, where expD17's baseline was worst? The scope cut leaves that open, and it is the single cheapest follow-up.
- Allen-Cahn's parameter is missed by 86-91% in every arm and both activations. That failure is not conditioning and has never been diagnosed.
- Interpolation still ends four to five orders above the floor in every arm. The next test is not another init variant but the readout solve on these trained geometries, which is what PROGRAM_FRAMING's division of labor says should be the actual answer for GELU as much as for tanh.
