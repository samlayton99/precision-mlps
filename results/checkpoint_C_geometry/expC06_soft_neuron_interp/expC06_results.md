# expC06 -- Soft-neuron weight interpolation (histogram + threshold test)

**Status: conclusions approved by Sam.**

Two views of one question: what do the **soft (small / low-bandwidth) inner-weight neurons** do under expC05's Xavier $\to$ uniform weight interpolation?
- **Part 1 (`run_histogram.py`):** visualize the $|w|$ distribution deforming.
- **Part 2 (`run_threshold.py`):** test whether protecting the soft tail prevents expC05's error "hump".

## TL;DR

- **Histogram:** no reshaping -- the $|w|$ spread just squeezes symmetrically toward the uniform value (support $[s,2-s]$). So at the hump's onset the *only* change is the loss of the smallest and largest weights.
- **Protecting the soft neurons flattens the hump; protecting the same number of *random* neurons does not** -- the causal result. The soft neurons (a spread of which span a low-degree polynomial basis) do the cheap smooth-approximation work.
- **The hump tracks convexity:** prominent for the convex targets ($e^x, x^2, -\cos$), marginal for non-convex $\sin(2\pi x)$ -- as expected if the effect is the soft neurons' polynomial basis.
- **Uniform centers:** $\sim\!10$ protected soft neurons (raw $\tau{=}1.5$) cut the hump $\sim\!250\times$ ($4\times10^{-10}\to2\times10^{-12}$, $N{=}256$); soft *fractions* keep it flattest. Placebo stays several$\times$ above protect at every $N$ ($\sim\!4$–$13\times$).
- **Init centers:** orders worse, no $s{=}1$ recovery -- error climbs to its *worst* at perfect uniformity. Here a *fraction* (not $\sim\!10$) is needed, so the benefit is placement-dependent too.

## Question / hypothesis

expC05 weights: as $|w_i|$ go Xavier $\to$ uniform $\gamma$, lstsq error degrades before recovering to the floor at the exact uniform endpoint (a span/approximation effect -- rank stays flat). Hypothesis: soft neurons at init, near-linear over $[-1,1]$, span a low-degree polynomial basis ($\tanh(w(x-c))=w(x-c)-\tfrac13 w^3(x-c)^3+\dots$) that cheaply fits smooth targets; pulling them to $\gamma$ destroys it $\Rightarrow$ the hump.

## Experiment design

Shared with expC05 weights (`common.py`: uniform grid, Glorot draw, neuron ordering, truncated-SVD lstsq, prime $2003$/$4001$ grids, rcond $10^{-13}$); tanh, fp64, $\lambda{=}0.25$ ($\gamma=N/8$). At $s{=}0$ weights are L1-rescaled to mean $\gamma$, so $|w(0)|\sim U(0,2\gamma)$.

- **Part 1 (histogram):** $|w_i(s)|=(1-s)\,|w^{\rm xav}_i|/\overline{|w^{\rm xav}|}+s$; 100 linear $s$; 2x2 $|w_i|$-histogram animation per $N$.
- **Part 2 (threshold):** freeze neurons with $|w_i(0)|<\tau$; interpolate the rest to $\gamma$ with the **non-protected set renormalized** so mean$|w|=\gamma$ (survivors stay frozen; on the L1 ball). Thresholds: 4 raw constants $\{0.125,0.25,0.75,1.5\}$ ($\sim\!1$–$12$ neurons) and 6 relative $\{0.0625\dots1.0\}\gamma$ ($\sim\!3$–$50\%$ of $W$), plus a $\tau{=}0$ baseline. **Placebo:** freeze the same *count* of *random* neurons. All curves use the same 6 seeds, **geometric mean** (so every curve shares the $s{=}0$ point). Targets: $e^x, x^2, -\cos(\pi x/2)$ (convex), $\sin(2\pi x)$ (non-convex control). $s$ on 26 points, dense near 0. Centers pinned **uniform** and **init** ($c_i=-b_i/w_i$) -- both geomean / 6-seed.

**Code & data.** `experiments/expC06_soft_neuron_interp/`. Outputs under `results/checkpoint_C_geometry/expC06_soft_neuron_interp/`: `histogram/{weight_hist_interp.mp4, weight_hist_snapshots.png}`, `{uniform,init}/{data.json, interp_{exp,x2,neg_cos,sin2pi}.png}`.

## Results

**`histogram/*`** -- $|w_i|$ is $\sim U[0,2]$ at $s{=}0$ and contracts to $[s,2-s]$: a symmetric squeeze onto a spike at 1, same shape at every $N$.

**`{uniform,init}/interp_*.png`** -- rel $L_2$ vs $s$; 4 rows ($N$) $\times$ 2 cols (threshold | placebo). Dashed black = $\tau{=}0$ baseline (hump); blue/green = constants; orange$\to$yellow = relative ($0.0625\gamma\to1.0\gamma$). Low-and-flat = hump prevented.

Peaks (rel $L_2$ over $s$), target $e^x$:

| | baseline | $0.5\gamma$ protect | $0.5\gamma$ placebo | count |
|---|---|---|---|---|
| uniform $N{=}256$ | $4.3\times10^{-10}$ | $8.7\times10^{-13}$ | $3.5\times10^{-12}$ | 114 |
| uniform $N{=}512$ | $3.3\times10^{-10}$ | $1.3\times10^{-12}$ | $8.1\times10^{-12}$ | 225 |
| init $N{=}256$ | $4.9\times10^{-7}$ | $5.3\times10^{-9}$ | $1.1\times10^{-7}$ | 114 |
| init $N{=}512$ | $8.3\times10^{-7}$ | $3.4\times10^{-8}$ | $1.7\times10^{-7}$ | 225 |

- **Hump tracks convexity.** The baseline hump is prominent for the convex targets ($e^x, x^2, -\cos(\pi x/2)$: peak tens-to-hundreds$\times$ above its $s{=}0$ start) but only $\sim\!2$–$5\times$ for non-convex $\sin(2\pi x)$ -- the soft-neuron polynomial basis matters for smooth/convex targets, not oscillatory ones.
- **Protect-soft flattens; placebo doesn't** -- both regimes, all targets (weakest on $\sin(2\pi x)$, which barely humps). The causal core.
- **Protecting a few smallest can *lower the floor*, not just flatten** -- at the uniform endpoint ($s{=}1$), keeping $\sim\!5$–$10$ soft neurons beats the unprotected cardinal floor by up to $\sim\!10\times$ (exp $N{=}256$: $4.6\times10^{-14}\to4.2\times10^{-15}$; $\sim\!6$–$10\times$ for $-\cos/\sin$ at $N{=}128$–$256$), fading to $\sim\!1$–$2\times$ by $N{=}512$. A trend worth chasing (see open questions); whether it survives more seeds is unconfirmed.
- **Convex vs non-convex differ in *how* uniformity helps.** A flattened (soft-protected) convex curve sits on a flat plateau through the whole interior and then drops sharply ($\sim\!2$–$4$ decades) to the floor *only at perfect uniformity* ($s{=}1$) -- interior weight-uniformity buys almost nothing, the exact cardinal endpoint buys everything. For non-convex $\sin(2\pi x)$ more uniformity is instead *monotonically* better through the interior (a gradual $\sim\!1$ decade gain) before the same endpoint drop. (Hesitant: one non-convex target, and its monotonic gain is modest vs the convex endpoint jump.)
- **Uniform:** even $\tau{=}1.5$ ($\sim\!10$ neurons) flattens ($4.3\times10^{-10}\to1.7\times10^{-12}$, $N{=}256$); $\tau\le0.25$ ($1$–$2$ neurons) doesn't -- a handful suffices, not one.
- **Init:** baseline explodes with $N$ ($\sim\!10^{-10}\to10^{-6}$); error climbs to its worst at perfect uniformity ($s{=}1$, no cardinal recovery). Protect-soft still helps ($\sim\!25$–$90\times$) but needs a *fraction* -- a handful of $\sim\!10$ barely dents it at $N\ge256$ -- so it's placement-dependent.
- **Placebo consistently fails to flatten** -- its peak stays $\sim\!4$–$13\times$ above protect-soft at every $N$, with no clear $N$-trend. (An apparent gap-grows-with-$N$ in an earlier arithmetic-mean pass was an outlier artifact -- it does not survive the robust geometric mean.)

## Additional details

- Histogram: $|w|/\overline{|w|}\sim U[0,2]$ because Glorot is uniform; the map is linear + mean-preserving, hence the exact $[s,2-s]$ support. (mp4 gitignored; snapshot PNG committed.)
- At $s{=}0$ every threshold gives identical weights ($\gamma\cdot$base), so all curves share that point and fan out only as the extreme weights get eaten.

## Conclusions

From the placebo contrast: the hump is caused by losing the **soft/low-bandwidth (small) neurons** -- freezing them, but not a random same-sized set, prevents it, across targets and both center regimes. The hump is prominent only for the convex targets and marginal for non-convex $\sin(2\pi x)$, as expected if the soft neurons supply a low-degree polynomial basis. Open: a *handful* suffices with uniform centers vs a *fraction* with init (placement-dependent).

## Open questions

- Fraction or raw count? The two regimes disagree; a fraction-vs-count sweep at matched placement would settle it.
- **Is the few-soft-neuron floor improvement real or a statistical artifact?** Protecting $\sim\!5$–$10$ smallest beating the cardinal floor by up to $\sim\!10\times$ is a small intervention; it needs many more seeds (and per-target/$N$ error bars) before we trust the trend, and an explanation for why it fades by $N{=}512$.
- **How does this relate to multistage residual fitting?** A few soft (low-bandwidth, $\approx$ low-degree-polynomial) neurons cheaply capture the coarse shape of a smooth/convex target, while the many sharp uniform-$\gamma$ neurons handle the fine structure -- i.e. one stage of residual fitting: a small low-order basis fits the bulk, the rest fits the residual. Does residual fitting get its precision from exactly this division of labor -- a handful of low-order modes doing most of the coarse work each stage, with the achievable precision set by how cleanly those coarse modes are captured? On that reading the hump is what happens when the sharp neurons are forced to fit the coarse shape themselves and do it poorly, and the floor improvement above is those coarse modes being captured better. If so, seeding/protecting a few soft modes per residual stage may be the right lever. Worth exploring directly.
- **Controlled multi-band ("cascaded") geometry.** The soft neurons here are an accident of the Xavier draw that we then protect -- instead, *design* the multi-scale basis. E.g. a uniform grid at the ideal $\gamma$ (the cardinal basis) plus a small, evenly-spaced sub-grid of low-bandwidth (soft) neurons, and optionally medium bands at intermediate $\gamma$ between them: a stack of uniformly-spaced magnitude regimes whose number could grow with $N$. This is "protect-the-soft-weights" made deterministic and even, and a natural way to realize the residual-fitting division of labor (each band owning a scale). Open: does such a cascaded geometry beat both pure-uniform and Xavier-protected, and how should the bands (count, spacing, bandwidth ratios) scale with $N$?
- **Precision vs generalization (data-poor regions).** Everything here optimizes precision *where there is data*. But the exact-kernel, perfectly-tuned-$\gamma$/centers construction may pay for that in-sample precision with poor behavior in data gaps and outside the sampled range: the uniform-$\gamma$ basis is all one (high) scale and discards the smooth, near-linear pieces that low-weight neurons supply -- exactly the pieces that extrapolate/interpolate-across-gaps gracefully, while a dense bank of sharp localized tanh steps can oscillate wildly off-data. Does channeling all the capacity into the precision-optimal geometry hurt generalization, and would the cascaded multi-band geometry recover it (its soft/low-order bands carrying the smooth global trend)? Easy to test: hold out a middle interval, fit lstsq on the rest with (a) a trained geometry, (b) our uniform construction, (c) the cascade, and compare error on the held-out gap.
