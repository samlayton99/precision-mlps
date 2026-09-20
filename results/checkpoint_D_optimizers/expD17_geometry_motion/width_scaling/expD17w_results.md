# expD17 / width scaling -- geometry motion vs width, and the $\gamma$-normalization question

**Status: draft -- pending Sam's review. Extends `../expD17_results.md`: same protocol, now 3 widths per problem class, plus per-neuron $\gamma$/center/update instrumentation.**

## TL;DR

- **The red-below-blue gap in relative drift is a normalization effect, and the data says so.** Absolute motion is init-independent: the median std/QI absolute-drift ratio over 36 cells is **0.86** (IQR 0.64-0.98), i.e. the two arms move about equally far in parameter units, while the *relative* ratio spans 2 to $6\times10^3$. $\log$(relative-drift ratio) tracks $\log(\|g_0\|$ ratio) with correlation **0.80** and slope **0.99**; against the $\gamma$ ratio, correlation 0.80 and slope 0.93. Sam's explanation holds.
- **QI relative drift falls with width; standard does not.** Fitted exponents in $W$: 1-D $-1.36$, 2-D $-1.01$, PINN $-0.90$, tabular $-0.65$ (QI) versus $-0.19$, $+0.17$, $+0.06$, $-0.39$ (standard). Absolute drift is roughly width-flat in both arms ($-0.34$ to $+0.29$). The QI geometry does not move less because gradients die; $\|g_0\|$ grows as $\gamma=O(N)$ while the motion does not.
- **Adam barely touches the 1-D $\gamma$s and never rescales them**: at $W=256$ the median $\gamma$ moves $0.00$-$0.12\%$ off $\lambda^*N/2=32$ in 5000 steps. It does not collapse toward $O(1)$. What it does is *break the delta*: the init has zero IQR (one shared $\gamma$), training opens a spread of $0.05$-$3.1$ absolute.
- **The 1-D center grid is preserved almost exactly** (RMS center motion $0.0034$-$0.116$ against a grid span of $3.6$-$6.4$), while 2-D/PINN centers move substantially (RMS $0.16$-$3.3$). Where the geometry is already right, Adam leaves the *layout* alone and perturbs the shared bandwidth.
- **Update magnitude is set by the learning-rate schedule, not by $\gamma$.** Within a snapshot the QI arm's $\gamma$ spans a factor $1.6$ (1-D: $1.01$, effectively a delta) while $\|\Delta w_k\|$ spans $9.5\times$; across arms the median update at mid-run differs by only $1.2\times$ despite $\gamma$ ratios of $2$-$580$. Per-row updates decay $10^{-4}\to10^{-8}$ over the run, tracking the cosine schedule.

## Question

As width grows, $\gamma=\lambda^*N/2$ grows with it. Does the expD17 geometry-motion picture change with width, and is the QI-vs-standard drift gap explained by the $\gamma$ (equivalently $\|g_0\|$) difference between the inits rather than by a real difference in how far training moves the geometry?

## Experiment design

Identical protocol to the parent experiment (plain Adam, no solves; one hidden tanh layer; **readout zero-initialized in every arm**; 5000 full-batch steps, cosine LR with 100-step warmup; fp64; CPU; single seed), repeated at three widths per class, both arms:

| class | widths | QI arm |
|---|---|---|
| 1-D interp (sine, runge, sine_8pi) | $N\in\{64,128,256\}$ ($W=141,269,525$) | construction geometry, uniform grid + halo, $\gamma=\lambda^*/h$, $\lambda^*=0.25$, $b=-\gamma c$ |
| 2-D interp (gauss_bump, sine2d, mixed2d) | requested ridges $\{288,576,1152\}$ | expE01 `radon_tensor` at each cell's best $\lambda$ (0.26/0.08/0.12), $\gamma=\lambda/h_{\rm ref}$, $h_{\rm ref}=2.8/\sqrt N$ |
| 2-D inverse PINN (burgers_$\nu$, bratu_$\lambda$, allencahn_$k$) | requested ridges $\{256,512,1024\}$ | `radon_tensor` at $\lambda=0.15$; one learnable log-parameterized PDE scalar started $10\times$ low, trained jointly |
| tabular (superconductivity, sarcos, parkinsons) | $W\in\{128,256,512\}$ | expF04 `scaled_psqrt` ridge-bundle init, `centers_per_dir` $=\sqrt W$, random centers from data projections |

Standard arm is Glorot (PyTorch default for tabular) on the identical architecture.

**Definitions.** Geometry vector $g_i$ = first-layer weight and bias concatenated (readout excluded; the PDE scalar excluded). Per-neuron bandwidth $\gamma_k := \|w_k\|_2$, the $L_2$ norm of neuron $k$'s first-layer weight row: exact for every class here, since each init writes $w_k=\gamma_k u_k$ with $\|u_k\|_2=1$ (in 1-D the direction is the scalar 1, so $\gamma_k=|w_k|$), and after training the same canonical decomposition defines $\gamma_k$. Per-neuron center $c_k := -b_k/\gamma_k$, in input units: the 1-D center, the ridge's signed distance from the origin in 2-D/PINN, the offset in the ridge's own projection coordinate for tabular. Raw biases are deliberately not plotted, because $b=-\gamma c$ conflates bandwidth drift with center motion. For Sam's hypothesis test the standard arm additionally gets an **effective $\gamma$** by his prescription: factor each row $w_k=s_k u_k$ with $\|u_k\|_1=1$, so $s_k=\|w_k\|_1$, and take $\bar s$ over neurons (the QI arm gets the same statistic).

**Tracked every iteration:** relative drift $\|g_i-g_0\|/\|g_0\|$, relative step $\|g_i-g_{i-1}\|/\|g_0\|$, and both **without** the $\|g_0\|$ normalization. A $/\|g_{i-1}\|$ variant was considered and skipped: a moving denominator conflates step decay with norm growth.

**Snapshot schedule.** Every step for the first 25, then the interval grows geometrically ($\times1.2$) and is capped at 50 steps for the rest of the run (141 snapshots). The same schedule drives the gif frames, so at constant display framerate early training plays in slow motion. At each snapshot, both arms record the per-neuron $\gamma$ vector, the per-neuron bias, and the per-row $L_2$ norm of the Adam update actually applied that step.

**Probes** (PROGRAM_FRAMING §7.1): truncated-SVD readout solve on the init geometry (pre) and on the final geometry (post), scored on the eval grid / test set; PINN probes fit the data-fit block only.

**Code & data.** `experiments/expD17_geometry_motion/width_scaling/run.py` (reuses the parent's builders by explicit-path import). Data `results/checkpoint_D_optimizers/expD17_geometry_motion/width_scaling/data/{class}_w{W}.jsonl` (72 runs) plus `analysis.json`. Figures in `width_scaling/figures/`: `expD17w_drift_from_init.png`, `expD17w_step_size.png`, their `_abs` variants, `expD17w_loss.png`, `expD17w_ratio_scatter.png`, `gamma_hist/expD17w_gamma_hist_w{1,2,3}.gif`, `center_hist/expD17w_center_hist_w{1,2,3}.gif`, `gamma_vs_update/expD17w_gamma_vs_update_w{1,2,3}.png`.

## Results

### The $\gamma$-normalization hypothesis: it holds

Over all 36 (class, problem, width) cells, regressing in logs:

| relation | correlation | slope |
|---|---:|---:|
| rel-drift ratio (std/QI) vs $\gamma$ ratio (QI/std) | 0.795 | 0.93 |
| rel-drift ratio vs $\|g_0\|$ ratio (QI/std) | 0.800 | 0.99 |

A slope of 1 is what "the gap is pure normalization" predicts, and $\|g_0\|$ delivers it. The complementary check is stronger: the **absolute** drift ratio has median **0.855** (IQR 0.64-0.98; 0.84 excluding the two cells where the standard arm blows up, runge and gauss_bump). Both arms move about the same distance in parameter units; the QI arm merely starts from a vector 2 to 500 times longer.

### Width scaling

Median over the three problems in each class, fitted exponent of drift versus $W$ (log-log):

| class | QI rel | std rel | QI abs | std abs |
|---|---:|---:|---:|---:|
| 1-D interp | $-1.36$ | $-0.19$ | $-0.34$ | $-0.17$ |
| 2-D interp | $-1.01$ | $+0.17$ | $-0.00$ | $+0.17$ |
| inverse PINN | $-0.90$ | $+0.06$ | $+0.09$ | $+0.07$ |
| tabular | $-0.65$ | $-0.39$ | $+0.29$ | $+0.12$ |

QI relative drift falls close to $1/W$ in the three geometric classes, exactly as $\|g_0\|\propto\gamma\propto N$ with width-flat absolute motion predicts. The tabular exponent is shallower ($-0.65$) because its ridge-bundle $\gamma$ grows like $\sqrt W$ (centers per direction $=\sqrt W$), not like $W$. Nothing collapses at the largest widths: absolute motion is flat to mildly increasing, and the per-step update magnitudes stay in the same band across widths.

**Probes across width.** The parent experiment's split persists and sharpens: every 1-D QI geometry is damaged (sine $3.2\times10^{-14}\to1.1\times10^{-12}$ at $W=256$; sine_8pi $4.3\times10^{-14}\to1.1\times10^{-8}$), and the damage *shrinks* with width (sine_8pi loses 7 orders at $W=64$, 6 at $W=128$, 5 at $W=256$) -- consistent with relative motion falling as $1/W$. Off-floor QI geometries still improve at every width (mixed2d $8.7\times10^{-7}\to3.5\times10^{-10}$ at $W=1152$), though the improvement shrinks as the init's own floor drops (sine2d at $W=1152$ starts at $3.6\times10^{-11}$ and gains only $2\times$). Tabular is unchanged: both arms land on the same noise floor.

### What Adam does to the $\gamma$s (gif deliverable 4)

From QI init, Adam does **not** rescale the bandwidths and does not pull them toward $O(1)$. In 1-D the median $\gamma$ moves by $0.00$-$0.12\%$ over 5000 steps at every width (at $W=256$: $32\to32.0$, and $32\to32.04$ for runge). What changes is the *shape* of the distribution: the QI init is a delta (all neurons share one $\gamma$, IQR exactly 0) and training opens a spread -- absolute IQR $0.048$ (sine, $W=256$) to $3.06$ (sine_8pi, $W=64$), i.e. a few percent of $\gamma$ except on the hardest target at the smallest width. The gif reads as a spike that stays put and grows shoulders, with the median line essentially frozen.

2-D and PINN behave differently and the difference is informative: there the median $\gamma$ moves a lot (sine2d $+188\%$ at $W=288$, burgers $+74\%$ at $W=256$), and the movement *shrinks with width* ($+188\%\to+80\%\to+27\%$ for sine2d across $W$). Those are the cells whose init $\lambda$ is furthest from ideal, and Adam is correcting the bandwidth toward a better value -- the same off-floor refinement seen in the probes.

### What Adam does to the centers (gif deliverable 6)

In 1-D the grid is preserved: RMS center displacement is $0.0034$-$0.116$ in input units against an init span of $3.6$-$6.4$ (grid plus halo), i.e. well under 2% of the layout, and it falls with width. The final histogram is still the flat, evenly-filled grid the construction started from. In 2-D/PINN the centers genuinely rearrange (RMS $0.16$-$3.3$, largest at the smallest widths), and the tabular offsets concentrate toward the data (mean $|c|$ ends at $0.26$-$0.47$).

Read together with the $\gamma$ story: **on a correct geometry Adam leaves the layout alone and only jitters the shared bandwidth; on an incorrect one it moves both.** That is preservation where preservation is right, and refinement where refinement is available.

### $\gamma$ versus update magnitude (deliverable 5)

Update magnitude is governed by the learning-rate schedule, not by the bandwidth. Per-row updates decay from $\sim10^{-4}$ early to $1.8\times10^{-8}$ at the end of the cosine schedule, four decades, in lockstep across neurons. Across arms at mid-run the median update differs by only $1.2\times$ (IQR 0.13-2.19) even though $\gamma$ differs by 2 to 580 times -- which is why absolute drift comes out init-independent.

The strict "flat in $\gamma$" version of the sign-step prediction is *not* what the data shows, and the honest statement is weaker: within a snapshot there is a modest positive rank correlation between $\gamma_k$ and $\|\Delta w_k\|$ (median Spearman $+0.55$ standard, $+0.40$ QI), so larger-bandwidth neurons do take somewhat larger steps. But the dynamic range is small and mostly independent of $\gamma$'s: in the QI arm $\gamma$ spans $1.57\times$ (p90/p10) while updates span $9.5\times$, and in 1-D $\gamma$ spans $1.013\times$ -- a delta -- while updates span seven decades. Update magnitude is essentially decoupled from $\gamma$; what sets it is the schedule and the local gradient statistics.

### Figures

- **`expD17w_drift_from_init.png`** -- 4$\times$3, rows = class (each labeled with its three widths), x = iteration, log y = $\|g_i-g_0\|/\|g_0\|$, blues = standard init, reds = QI, lighter = smaller width, rows share the y scale. Look for: red bands 1-3 decades below blue everywhere; within the red band, darker (wider) lines sit lower, the $1/W$ law; blue bands show no such ordering.
- **`expD17w_drift_from_init_abs.png`** -- same without the $\|g_0\|$ normalization. This is the headline comparison: the blue/red separation largely collapses, the two colors interleave in most panels, and what remains is a per-problem spread, not a per-init one.
- **`expD17w_step_size.png`** and **`expD17w_step_size_abs.png`** -- per-iteration motion, same layout. Look for: no dead plateau after the single zero-readout step; motion decaying with the cosine schedule; the same collapse between arms in absolute units.
- **`expD17w_loss.png`** -- eval rel $L_2$ per cell (field error for the PINN row), same styling. Read alongside the drift figures to see which motion bought error reduction.
- **`expD17w_ratio_scatter.png`** -- one point per (class, problem, width): x = $\gamma_{\rm QI}/\gamma_{\rm std}$ (effective, mean $L_1$ row norm), y = std/QI relative-drift ratio, log-log, marker per class, $y=x$ dashed. Points scatter about the diagonal, which is the $\gamma$ hypothesis; the 1-D runge cells sit above it (the standard arm genuinely moves further there, not just relatively).
- **`gamma_hist/expD17w_gamma_hist_w{1,2,3}.gif`** -- QI arm only, one gif per width index, 4$\times$3 histograms of $\gamma_k$ with a red dotted median line and mean $|\gamma|$ in each title; axes fixed across frames at robust $[0.1\%,99.9\%]$ quantiles with the clip fraction annotated. Watch the 1-D row: the spike stays at $\lambda^*N/2$ and grows shoulders. Watch the 2-D/PINN rows: the whole distribution translates.
- **`center_hist/expD17w_center_hist_w{1,2,3}.gif`** -- same construction for $c_k=-b_k/\gamma_k$ in input units. The 1-D row is the grid holding its shape; the 2-D/PINN rows deform.
- **`gamma_vs_update/expD17w_gamma_vs_update_w{1,2,3}.png`** -- 4$\times$3, log-log scatter of ($\gamma_k$, $\|\Delta w_k\|_2$) over all (neuron, snapshot) pairs, viridis = standard arm, autumn = QI arm, color = training step, two colorbars outside the axes, axes capped at robust quantiles with clip fraction annotated. Uniform random subsample, at most 3000 points per arm per panel. Look for: horizontal banding (update roughly independent of $\gamma$) and the vertical color gradient (updates shrinking with the schedule); the QI arm is a narrow vertical column because its $\gamma$s barely spread.

## Additional details

- Single seed per cell throughout, as in the parent. The 1-D runge cells are the largest outliers in every ratio (the standard arm's drift reaches $\sim10$ relative, $23\times$ the QI arm's in absolute terms), so the medians above are the honest summary and the means are not.
- The absolute-drift ratio's IQR reaching 0.64 (rather than sitting at 1.00) means the QI arm typically moves slightly *further* in absolute terms than the standard arm, not less.
- Effective $\gamma$ by the $L_1$ prescription and $\|g_0\|$ are nearly collinear here (they differ by the bias block and the $L_1$-vs-$L_2$ convention), which is why both regressions give the same correlation; $\|g_0\|$ is the better predictor by slope.

## Conclusions

*Pending Sam's review.* The relative-drift gap between the inits is a normalization artifact of $\gamma=O(N)$: absolute geometry motion is init-independent (median ratio 0.86) and roughly width-flat, so QI relative drift falls like $1/W$ while the standard arm's stays flat. Adam does not rescale a correct geometry -- in 1-D the median bandwidth moves under $0.12\%$ and the center grid holds to within 2% of its span at every width -- but it does open a bandwidth spread where the init had none, and that spread is what damages floor-quality geometries. Where the init is off-floor (2-D, PINN), the same mechanism moves bandwidths and centers substantially and improves the geometry. Per-row update magnitude is set by the learning-rate schedule and is only weakly rank-correlated with $\gamma$, which is the mechanism behind the init-independence of absolute motion.

## The GELU companion

The same grid was rerun with GELU at its own aliasing-rule bandwidth ($\lambda^*=0.707$, expC07): `../width_scaling_gelu/expD17w_gelu_results.md`. Two results worth carrying back here. First, **the frozen halo is a property of the activation's saturation, not of the geometry**: tanh kills left and right halo neurons at equal rates, while GELU kills $68$-$92\%$ of the right halo and **exactly none** of the left, which is what its asymmetric limits ($\mathrm{gelu}'\to1$ as $z\to+\infty$, $\to0$ as $z\to-\infty$) predict. Second, every structural conclusion of this study survives the activation swap -- the $\gamma$-normalization account, the init-independence of absolute drift, the preserved median bandwidth, the damage-at-the-floor / improvement-off-the-floor split -- **except** the $1/W$ suppression of relative drift, whose exponent is activation-dependent ($-0.40$ GELU vs $-1.59$ tanh in 1-D).

## The normalization factorial

Both studies above are superseded on the arm axis by `../norm_factorial/expD17n_results.md`, which crosses activation, normalization and init in one grid (six arms, two widths, plus a two-layer tabular row on the expD20 suite under a variance-normalized metric). This study's tanh no-norm arms are reproduced there bit for bit, so the two are directly comparable.

## Open questions

- Multi-seed, particularly the 1-D runge outlier and the tabular cells.
- The damage-shrinks-with-width trend (sine_8pi: 7, 6, 5 orders lost at $W=64,128,256$) suggests a width at which plain Adam stops meaningfully degrading a floor geometry. Extrapolating the $1/W$ law, is there a crossover width past which no protection mechanism is needed?
- Does the bandwidth spread (not the center motion) account for all the floor damage? A run that freezes $\gamma$ and trains only centers, versus the reverse, would separate them and is cheap.
