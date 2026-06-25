# expB02 -- Scaling laws for the fixed-geometry least-squares readout (1D)

**Status: conclusions approved by Sam.**

## TL;DR

- Clean scaling law: error follows a power-law descent in width (and in data) until it bottoms out at the fp64 precision floor. Some curves are flat at first, then the power law kicks in.
- The activation *and* the target set the slope and intercept -- not the floor (the floor is the common fp64 limit). relu shows the cleanest, most extended power law (slope ~$-2$ across all targets) but descends slowly, so it hasn't reached the floor in range; tanh/gelu descend far more steeply and hit the floor fast.
- The law survives at fixed $\lambda=0.25$ -- not an artifact of per-$N$ bandwidth selection, just a noisier descent.

## Question

How does the fixed-geometry lstsq error scale (1) as width grows, and (2) as the training-point count crosses the under- to over-determined threshold $W+1$?

## Experiment design

Geometry per $N$: uniform QI centers + halo (fixing $W$, span, $h=2/N$); $\gamma=\lambda/h$; feature entry $\Phi_{ik}=\text{act}(\gamma(x_i-c_k))$; readout via truncated-SVD lstsq on $[\Phi,\mathbf 1]$ (RCOND $=10^{-13}$ relative to $s_\max$), fp64. 6 targets x 3 activations (tanh, gelu, relu); metrics on a prime eval grid (misaligned with train): eval $L_\infty$ and rel $L_2$. **Size scaling:** $N$ on a log grid $16\to1024$ ($W$ from ~157 to 1843), error best-over-$\lambda$ at each $N$. **Data scaling (clean):** fixed geometry $N=128$ ($W=269$), $\lambda=0.25$, sweeping training points from underdetermined ($n<W$) through the threshold $W+1$ to overdetermined. **Fixed-$\lambda$ amendment:** the same width sweep at a single $\lambda=0.25$, decoupling bandwidth selection from the size scaling.

**Code & data.** `experiments/expB02_scaling_laws/` (`run.py`; `--plot` augments `data.json` with the fixed-$\lambda$ slice). Data: `data.json` (`width`, `width_fixed`, `data_clean`). Figures: `error_vs_width.png`, `fixed_lambda_scaling.png`, `error_vs_data_clean.png`. (A noise figure exists but the $1/\sqrt{n}$ law lives in expB01; dropped here.)

## Results

- **Width scaling:** every (activation, target) traces a power-law descent toward the *common* fp64 floor (~$5\times10^{-14}$, reached by tanh/gelu). The slope is set by the activation and the target: relu is a clean ~$N^{-2}$ across all targets, so it descends slowly and is only ~$10^{-6}$ by $N=1024$ -- a slow approach, not yet at the floor; tanh/gelu are far steeper (e.g. ~$N^{-12}$ on runge, or already at the floor on the smooth targets). `abs_cubed` ($C^1$) descends but is still ~$10^{-11}$ at the largest $N$. Some curves show an initial flat region before the power law begins.
- **Data scaling:** clean tanh error falls steeply across $W+1$ then floors (~$5\times10^{-15}$); at fixed geometry relu is resolution-limited (its bias dominates), so adding data doesn't help -- it plateaus above ~$10^{-5}$.
- **Fixed $\lambda=0.25$:** same law -- the floor sits within ~1 order of best-over-$\lambda$ and relu is unchanged; the only cost is a noisier, less monotone descent. Per-$N$ $\lambda$-selection was smoothing the curve, not creating the floor.

### Figures

- **`error_vs_width.png`** -- 3 rows (activations) x 2 cols (rel $L_2$, $L_\infty$), x = $N$, one line per target, reference line at $5\times10^{-14}$. tanh/gelu targets descend steeply onto the floor line; the relu rows are clean straight power laws (slope ~$-2$) that haven't reached the floor in range.
- **`fixed_lambda_scaling.png`** -- same layout at fixed $\lambda=0.25$; read side-by-side with the above -- same floor, same relu power law, descent noisier.
- **`error_vs_data_clean.png`** -- same layout vs training-point count with the $n=W+1$ threshold marked: tanh/gelu collapse to the floor at the threshold; relu stays flat above ~$10^{-5}$.

## Conclusions

The fixed-geometry lstsq readout has a clean power-law scaling law in both width and data: a log-linear descent that ends at the common fp64 precision floor. The activation and target set the slope and intercept (relu the cleanest, slowest power law; tanh/gelu steeper, reaching the floor sooner) -- the floor itself is the fp64 limit, not activation-dependent. The law holds at fixed $\lambda=0.25$.

## Open questions

- None
