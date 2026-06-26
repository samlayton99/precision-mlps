# expC05 -- Geometry interpolation: what makes the QI geometry findable

**Status: draft -- Results are data-obvious (cross-checked numerically against every `data.json`). Conclusions and the interpretation are proposed, pending Sam.**

## TL;DR

- Bandwidth $\lambda$ is the dominant knob; sweeping it moves the error 10+ decades, and every geometry effect below is read at the viable band.
- **Center uniformity is monotone:** more uniform is strictly better, all the way to the QI grid.
- **Weight uniformity is conditional, not absolute:** at the viable $\lambda$ all-equal weights win by decades; at too-small $\lambda$ a Xavier spread wins. It also gets *worse before better* along the path (the "hump"), explained in expC06.
- **The weight sign pattern does not matter** (gelu $\approx$ gelu-positive; for odd tanh the readout absorbs it). The findable degrees of freedom are bandwidth, weight *magnitude* uniformity, and center placement.
- **gelu is tanh shifted by a constant in $\lambda$** -- same surface, same structure, viable band translated up.
- **A bimodal weight vector (mass on both signs, crossing near 0) does badly** -- the de-confounded run avoids it; the frozen crossing run shows the penalty.
- **The ingredients are asymmetric.** Center uniformity helps on its own; weight uniformity does not -- uniform weights *need* uniform centers (uniform weights with random centers is the worst corner). expC06 explains why, and gives a way around it (keep a spread of soft neurons / a cascaded multi-band geometry).

## Question

Between a random (Xavier) init and the exact QI inner layer, which ingredient matters how much -- bandwidth $\lambda$, center placement, per-neuron weight (bandwidth) uniformity -- and are those ingredients independent or coupled?

## Experiment design

The ideal QI inner layer is $\gamma(x-c_i)$: every weight equals $\gamma$ (the constant vector $\gamma\mathbf 1$), every bias is $-\gamma c_i$, and the centers $c_i$ sit on the uniform grid (with halo). Equivalently, in geometric coordinates the feature is $\tanh\!\big(w_i(x-c_i)\big)$ with $w_i=\gamma$, $c_i$ uniform. Starting from a genuine Glorot/Xavier draw $(w^\text{xav},b^\text{xav})$ -- centers $c^\text{xav}_i=-b^\text{xav}_i/w^\text{xav}_i$ -- we interpolate toward this ideal along three independent modes and read the lstsq fp64 relative-$L_2$ error at every grid point. All three share `common.py` (uniform geometry, neuron ordering, truncated-SVD readout, train/eval grids) so the surfaces are directly comparable.

**Normalization (shared rule).** Where a weight *pattern* is interpolated, magnitude is held to a fixed budget in the L1 (mean-abs) sense: $\operatorname{mean}|w|=\gamma$. L1 (not RMS) holds the *total bandwidth budget* $\sum_i|w_i|$ constant -- the uniform vector maximizes L1 at fixed L2, so RMS would silently hand the uniform endpoint ~15% extra budget and confound the comparison. **Bias is always derived, never interpolated raw:** $b=-w\,c$, so the center $-b/w=c$ holds exactly. Runtime asserts (all three modes): center drift $<10^{-9}$, budget error $<10^{-9}$, and zero origin crossings (sign of every weight preserved).

The three modes:

- **centers** (`run_centers.py`) -- weights pinned at the ideal $\gamma$; interpolate *positions* $c(t)=(1-t)\,c^\text{xav}+t\,c^\text{unif}$, $t\in[0,1]$; sweep $\gamma$. Axes $(t,\lambda)$ with $\lambda=\gamma h$, $h=2/N$. Anchor $t{=}0$ is the exact Xavier inner layer (centers $-b/w$, concentrated near 0 with heavy tails: density $\tfrac14$ for $|x|<1$ and $\tfrac1{4x^2}$ for $|x|\ge1$). A `uniform_init/` variant instead draws the $t{=}0$ centers uniformly over the span (matching the weight+bias mode's anchor), $N\in\{64,128,256\}$.
- **weights** (`run_weights.py`) -- centers pinned to the uniform grid, bias $=-w\,c^\text{unif}$; interpolate only the weight *pattern* on the L1 face, $u(s)=(1-s)\hat w+s\,\text{target}$ with $\hat w=w^\text{xav}/\operatorname{mean}|w^\text{xav}|$, then $w(s)=\gamma\,u(s)/\operatorname{mean}|u(s)|$; sweep $\gamma$. Axes $(s,\lambda)$, $\lambda=\operatorname{mean}|w|\,h$. The target octant-center is $\sigma\mathbf 1$ with $\sigma=\operatorname{sign}(w^\text{xav})$, so the path stays inside the Xavier orthant (no neuron's weight crosses zero). Three runs: `tanh` and `gelu` toward $\sigma\mathbf 1$, and `gelu_positive_init` toward $+\mathbf 1$ (a sign-pattern control; for the odd tanh the sign is absorbed by the readout, so it is only varied for gelu).
- **weightbias** (`run_weightbias.py`) -- interpolate *both* axes at a **fixed** bandwidth $\lambda^*=0.25$ ($\gamma=N/8$, neither swept nor selected). Weight uniformity $s$ moves $w(s)$ along the L1 face Xavier$\to\sigma\mathbf 1$ exactly as in the weights mode; center uniformity $t$ moves $c(t)=(1-t)\,c^\text{rand}+t\,c^\text{unif}$ from sorted uniform-random-over-span to the uniform grid; $b=-w(s)\,c(t)$. Axes $(s,t)$. Corners: $(1,1)$ = exact QI; $(1,0)$ = uniform weights / random centers; $(0,1)$ = Xavier weights / uniform centers; $(0,0)$ = the Xavier weight pattern at $\gamma$-scale with random centers (a controlled random start, not the literal small-magnitude Xavier init). Because $t{=}1$ forces uniform centers regardless of $s$, the $t{=}1$ edge reduces exactly to the weights mode at $\lambda=0.25$ (validated below).

**Common knobs.** 4 targets (sine, sine_8pi, runge, exp); 4 widths $N\in\{64,128,256,512\}$ (with halo, $W$ runs 205, 269, 461, 921; halo 70, 70, 102, 204); $21\times21$ grids; readout = truncated SVD with explicit bias column, rcond $10^{-13}$, fp64; metric = eval relative $L_2$ on a prime, train-misaligned grid ($N_\text{train}=2003$, $N_\text{eval}=4001$). centers/weights use 3 Xavier seeds; weightbias uses 1.

**Code & data.** `experiments/expC05_geometry_interpolation/` (`common.py`, `run_centers.py`, `run_weights.py`, `run_weightbias.py`, `config.yaml`). Data: `{centers,centers/uniform_init,weights,weights/crossing_0,weightbias}/data.json`. Figures: `centers/<seed>/interp_<target>.png`, `weights/{tanh,gelu,gelu_positive_init,crossing_0}/interp_<target>.png`, `weightbias/interp_<target>.png` (all 4x3 grids, rows = width). Notebook: `interp_viz.ipynb`. `weights/crossing_0/` is a frozen earlier (RMS, origin-crossing) run kept only for the bimodal-weight contrast below.

## Results

### Bandwidth $\lambda$ dominates

In both swept modes the bandwidth axis moves the error by 10+ decades; every geometry effect below is read at, or near, the viable band. That band is a robust horizontal valley consistent with expC02/expC03's $\lambda^*\!\approx\!0.25$. It deepens and widens with $N$ (more neurons -> lower floor until fp64 cuts it off). See the horizontal dark valley in col 1 of `weights/tanh/*` and `centers/*`.

### Center placement: monotone

- **More uniform is strictly better.** Across all targets and widths the error decreases monotonically as centers move Xavier $\to$ uniform grid; the QI grid is the best placement, with no intermediate dip (`centers/<seed>/interp_*.png`, col 3 -- every viable-$\lambda$ line slopes down to $t{=}1$).
- **The last bit of regularity matters most.** At the viable $\lambda$ the descent steepens sharply near $t{=}1$.
- **Size of the effect:** best-over-$\lambda$, the uniform grid beats Xavier centers by ~1-4 decades -- ~1.5 for sine, up to ~4 for sine_8pi/runge at $N{=}128$, only ~0.3 for exp (its smooth target is nearly center-insensitive).
- **It is a viable-band effect.** At small $\lambda$ uniformity gives no benefit (the lines are flat in $t$).
- **One exception:** runge at $N{=}64$, where random centers beat the uniform grid by ~2 decades -- consistent with curvature-clustering near runge's peak, but single-seed and on hold.
- **Seed-stable:** across 3 seeds the $t{=}1$ floor is identical; only the $t{=}0$ penalty varies, within its decade band.

### Weight uniformity: conditional and non-monotone

- **At the viable $\lambda$, uniform weights win by decades.** Equal weights ($s{=}1$) beat the Xavier spread ($s{=}0$) by 2-5 decades (sine $N{=}128$: $2.9\times10^{-10}\!\to\!6.4\times10^{-14}$; sine_8pi ~$10^5\times$). See `weights/tanh/*` col 1: the valley deepens toward $s{=}1$.
- **At too-small $\lambda$ the sign flips: the Xavier spread wins,** by up to ~6 decades (runge $N{=}512$, $\lambda\!\approx\!0.02$: uniform is $10^4\times$ worse). Reading: when the total bandwidth budget is too small, an uneven spread lets *some* neurons reach a usable scale; once the budget is right, evenness is what tiles the basis. (`weights/tanh/*` col 3: viable-$\lambda$ lines slope down to $s{=}1$, small-$\lambda$ lines slope up.)
- **Part of the fixed-$\lambda$ penalty is just a preferred-$\lambda$ shift.** Allowing each weight pattern to pick its own $\lambda$ (best-over-$\lambda$) shrinks the uniform advantage to under a decade for smooth targets, ~2-3 decades for sine_8pi.
- **Worse-before-better along the path (the "hump").** Even with $\lambda$ and centers ideal, error rises before recovering to the floor at the exact uniform endpoint. **expC06 explains this:** the hump is the loss of the soft (small) neurons, which span a low-degree polynomial basis that cheaply fits smooth/convex targets -- protecting them flattens it; the effect tracks target convexity.

### A bimodal weight vector (crossing zero) does badly

If the weight vector carries mass on *both* signs and the interpolation path pulls it through zero bandwidth, error spikes mid-path. The frozen `weights/crossing_0/*` run (RMS-normalized, interpolated through the origin) shows a clear pinch near $s\!\approx\!0.5$ (col 3) where ~half the Xavier weights cross zero, the effective rank dips ~15%, and the renorm inflates the survivors. The de-confounded run stays inside the Xavier orthant and has no pinch (`weights/tanh/*` vs `weights/crossing_0/*`). Practical reading: a usable geometry wants each neuron at a definite, nonzero bandwidth; a near-zero / bimodal weight is wasted capacity.

### Weight + bias together: weight uniformity needs center uniformity (at fixed $\lambda=0.25$)

- **The exact corner reaches the floor.** $(1,1)$ hits $\sim\!10^{-14}$ for every target/width, confirming $\lambda=0.25$ is the right bandwidth -- with both ingredients ideal, no $\lambda$ sweep is needed. The lone exception is runge $N{=}64$ ($3.3\times10^{-9}$), bandwidth-starved at small $N$.
- **Uniform weights give both the best and the worst.** $(1,1)$ is best, but $(1,0)$ -- uniform weights with random centers -- is the *worst* corner of the whole surface ($\sim\!10^{-3}$ to $10^{-5}$). Equal kernels tile perfectly when centers are regular and leave matching gaps when they are not.
- **The coupling is one-way.** Weight uniformity backfires without uniform centers: $(1,0)$ is several decades *worse* than the random start $(0,0)$. Center uniformity is not like this -- with Xavier weights, moving centers Xavier $\to$ uniform ($(0,0)\to(0,1)$) is flat-to-helpful, never harmful. So uniform weights *need* uniform centers, but not the reverse. **expC06** explains the mechanism (identical sharp kernels can only cover the interval when regularly placed) and a way around it (keep a spread of soft / low-bandwidth neurons -- a cascaded multi-band geometry -- which tolerates irregular placement).
- **Of the two single-ingredient corners, centers win.** $(0,1)$ (Xavier weights, uniform centers, $\sim\!10^{-9}$ to $10^{-13}$) beats $(1,0)$ by orders -- at fixed ideal $\lambda$, regular centers buy more than uniform weights. (A fixed-$\lambda$ statement; the ranking softens under best-over-$\lambda$.)
- See `weightbias/interp_*.png`: col 1 heatmap (dark top-right $(1,1)$, bright bottom-right $(1,0)$); col 2 high-$t$ lines descend to the floor while low-$t$ lines rise with $s$.

### The weight sign pattern does not matter

- **gelu vs gelu-positive-init are effectively identical** -- median $|\Delta\log_{10}\text{err}|\le0.05$ over the full $(s,\lambda)$ grid, every target/width; the figures are visually indistinguishable.
- **For odd tanh the sign is free by construction** -- flipping a neuron's $w$ negates its feature column, which the lstsq readout absorbs (a column sign the solve undoes exactly).
- So neither activation's accuracy depends on which signs the weights carry. The load-bearing degrees of freedom are bandwidth, weight *magnitude* uniformity, and center placement -- not sign.

### gelu is tanh translated by a constant in $\lambda$

gelu reproduces every qualitative feature of tanh -- the valley, the conditional sign flip, the monotone descent to $s{=}1$ at the viable $\lambda$ -- with comparable floors ($\sim\!10^{-14}$). The only systematic difference is a **constant additive shift of the viable band in $\lambda$** (gelu sits at roughly 0.5-0.7 vs tanh's 0.1-0.25); the surface is otherwise the same shape. Compare `weights/gelu/*` against `weights/tanh/*`: same picture, valley moved right.

### $t{=}1$ consistency check (validation)

Along weightbias $t{=}1$ (uniform centers) the surface must equal the weights-mode tanh run at $\lambda=0.25$. Where the weights $\gamma$-grid lands exactly on $\lambda=0.25$ (sine_8pi $N{=}256$) the two match to $|\Delta\log_{10}|=0.02$ (numerical noise); elsewhere they differ only because the weights grid's nearest column is at a slightly different $\lambda$ (0.225, 0.203). The construction is verified.

## Interpretation (proposed by Sam; implicated, not proven)

These surfaces are consistent with a single covering picture of what a good geometry must do, though they do not prove it:

- **Cover the interval with no gaps.** Every sub-interval of $[-1,1]$ (plus the halo) needs a neuron whose kernel sits there. Regular centers achieve this; random centers leave gaps, which is why uniform centers are monotonically better and why uniform *weights* over random *centers* is the worst corner (identical kernels make the gaps maximally visible).
- **Steep enough, but not too steep.** The kernels must be sharp enough ($\gamma$ large enough) to be placed precisely and resolve fine structure, but not so sharp that the QI theory breaks ($\lambda$ above its viable value) -- this is the $\lambda$ valley, and the upper wall is the QI aliasing limit.
- **Uniform steepness so no region is under-resolved.** Where weights are non-uniform, some regions get shallow kernels and are resolved worse -- so weight non-uniformity hurts once the budget is right. (At too-small budget the logic inverts: a spread is the only way any region gets a usable kernel -- the conditional sign flip.)
- **Why Xavier does decently with $\lambda$ held right.** A Xavier draw places centers roughly uniformly over $[-1,1]$ with fat tails that happen to cover the halo, so the coverage is already "good enough." Increasing center uniformity then mainly lets the geometry build *smaller, more precisely placed* kernels, which is what unlocks the last decades of precision.
- **The hump is the cost of giving up the coarse basis.** Early in the weight-uniformity path the soft neurons (a low-degree polynomial basis) are lost before the sharp uniform basis is fully in place, so smooth/convex targets get worse before better -- expC06 establishes this causally.

## Additional details

- **De-confounding history (why `crossing_0` exists).** The first weights rerun normalized by RMS and interpolated the *signed* direction toward $\gamma\mathbf 1$, dragging the negative-Xavier-weight cohort through zero bandwidth near $s{\approx}0.5$ -- the pinch above. The fix (L1 face toward the in-orthant center $\sigma\mathbf 1$, no crossing, mean-abs norm) removes it; `crossing_0/` is kept as the before-picture and as the bimodal-weight evidence.
- **Anchor difference between modes.** The primary centers mode anchors $t{=}0$ at the true Xavier centers $-b/w$ (concentrated near 0); weightbias and `centers/uniform_init` anchor at uniform-random-over-span. Both show the same direction (uniform is better); the random-uniform start is generally the harder one.
- **Best-over-$\lambda$ vs fixed-$\lambda$.** Quoting a penalty at one fixed $\lambda$ overstates it -- a non-ideal config usually prefers a smaller $\lambda$. Center placement's penalty survives re-optimizing $\lambda$ better than weight uniformity's does.

## Conclusions

*Proposed, pending Sam.*

- **Ordered, coupled ingredients.** Bandwidth $\lambda$ first (decades), then -- at the right $\lambda$ -- center regularity and weight uniformity, each worth decades but neither sufficient alone.
- **Center uniformity is monotone and robust; weight uniformity is conditional and non-monotone.** More-uniform centers are always better; uniform weights only help inside the viable band and get worse before better along the way (expC06).
- **The coupling is one-way:** weight uniformity must not be installed before center uniformity ($(1,0)<(0,0)$), while center uniformity is safe on its own. This still argues for moving the geometry *jointly* -- support for the $\gamma(x-c)$ / log-$\gamma$ reparameterization (expD03) and a joint / variable-projection objective (expD04) -- with expC06 offering a soft-neuron / cascaded workaround.
- **Sign is not a degree of freedom** for accuracy here (gelu $\approx$ gelu-positive; tanh sign absorbed by the readout).
- **gelu and tanh share one geometry, offset by a constant in $\lambda$** -- the findings are activation-independent up to that shift.

## Open questions

- Does the $\gamma(x-c)$ / log-$\gamma$ reparameterization remove the joint-movement barrier the multiply-not-add result implies? (expD03/expD04.)
- ~~Why does weight uniformity get worse before better even when $\lambda$ and centers are ideal? Are the small near-zero weights particularly useful? Is it worst for convex targets?~~ **Answered in expC06:** yes -- the hump is the loss of the soft (small-bandwidth) neurons, which span a low-degree polynomial basis; protecting them flattens it, and the effect tracks target convexity.
- Curvature-clustering (on hold): a deterministic test of whether clustering centers at high curvature beats the uniform grid (runge $N{=}64$ hint).
- Does the small-$N$ runge bandwidth starvation at $(1,1)$ close with a target-aware halo, or is it intrinsic to equal kernels on a peaked target?
