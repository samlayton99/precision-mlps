# expC04 -- Center geometry: does placement have to be uniform?

**Status: conclusions data-obvious (no sign-off pending).**

## TL;DR

- Only uniform centers reach machine precision (~$10^{-13}$--$10^{-14}$, flat in width). Every non-uniform placement plateaus 2--3 orders above and never descends to the floor.
- With width, span, and $\gamma$ held equal, the gap is placement alone -- and it does not track $\mathrm{cond}(\Phi)$.

## Question

How much of the lstsq-to-machine-precision result depends on the centers being uniform?

## Experiment design

Five geometries place the same $W$ centers over the same span: **uniform** (the QI grid), **random** (uniform draws), **clustered** (Gaussian blobs around uniform meta-centers), **trained** (centers $x_0=-b/w$ extracted from a trained tanh net, everything else discarded), and **reg_clustered** (a regular grid broken into evenly-spaced clusters, each compressed toward its center by a geometric ratio $0.75$; ratio $1$ = uniform). For each base $N$ the uniform QI geometry fixes the three shared quantities -- total width $W=N+2R+1$, the center span, and $h=2/N$ -- and all five use $\gamma=\lambda/h$ at each swept $\lambda$, so at a given $(N,\lambda)$ the runs differ *only* in where the centers sit. Target sine; $N\in\{32,64,128,256\}$ ($W$ up to 461); $\lambda\in\{0.10,\dots,1.00\}$; 3 seeds for the stochastic geometries. Per cell: eval $L_\infty$, rel $L_2$, and $\mathrm{cond}([\Phi,\mathbf 1])$ of the train matrix.

**Code & data.** `experiments/expC04_center_geometry/` (`run.py`); geometries in `src/construction/center_geometry.py` (tests in `tests/test_center_geometry.py`). Data: `data.json`. Figures: `error_vs_width.png`, `centers_numberline.png`, `conditioning.png`.

## Results

- **Uniform is alone at the floor**, roughly flat in width. random/clustered plateau ~$10^{-11}$--$10^{-12}$; reg_clustered is the best non-uniform but still ~$4\times$ above uniform; trained is high-variance and degrades at large width. None descend to the floor as width grows.
- **Placement is the cause** ($W$, span, $\gamma$ identical), so machine precision is specific to uniform placement, not generic to lstsq.
- **Conditioning doesn't discriminate:** uniform, random, and reg_clustered all sit at $\mathrm{cond}(\Phi)\sim10^{19}$--$10^{20}$, yet random is 2--3 orders worse than uniform.

### Figures

- **`error_vs_width.png`** -- two panels (rel $L_2$, $L_\infty$), x = $W$, one curve per geometry (best-over-$\lambda$, mean over seeds, shaded min--max band). Uniform rides ~$10^{-13}$; the others plateau above.
- **`centers_numberline.png`** -- the seed-0 center layouts per geometry/width (uniform's even ticks, random's gaps, reg_clustered's periodic clusters, trained's concentration inside $[-1,1]$).
- **`conditioning.png`** -- $\mathrm{cond}(\Phi)$ at each geometry's best cell; a light diagnostic showing cond does not order the accuracy.

## Conclusions

Only uniform centers reach machine precision; all non-uniform placements plateau 2--3 orders above, attributable to placement alone, and the gap does not track $\mathrm{cond}(\Phi)$. (This extends the paper's "curvature does not explain the gap" to the feature matrix, from an independent direction.)

## Open questions

None -- *how forgiving* uniformity is, and the gamma/weight/bias interactions, are dissected in expC05.
