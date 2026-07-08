# expD02 -- Does Adam find a useful geometry? (init x training-regime cube)

**Status: the three wins are approved by Sam; finer single-seed trends pending.**

## TL;DR

- **Win #1:** QI init + train both layers reaches ~$10^{-5}$ -- 2--3 orders better than any standard-init trained net (~$10^{-2}$).
- **Win #2:** QI init + train + final lstsq refit recovers the construction floor (~$5\times10^{-14}$) from the *trained* geometry. Training doesn't destroy a good geometry; a final solve closes the gap.
- **Win #3:** scaled_xavier (right bandwidth, inexact centers) generalizes the gain -- getting $\gamma=O(N)$ is the transferable lever. All of this in fp32.

## Question

How does the achievable error depend on how the first layer is initialized and on what is trained vs frozen vs refit?

## Experiment design

One cube, sliceable along three axes plus width. 

**init (4)** -- the first layer, with centers pinned by $b=-\gamma c$:

- `xavier`: standard Glorot ($\gamma\approx0.1$, far too small), random centers;
- `scaled_xavier`: Xavier rescaled by $s=\gamma^*/\text{mean}|w|$ to bandwidth $\gamma^*=\lambda^*/h=N/8$ (at $\lambda^*=0.25$), heavy-tailed centers;
- `const_weight`: constant weight $\gamma^*$, Xavier centers clipped to span (clean coverage);
- `qi`: constant $\gamma^*$, uniform grid (the construction).

**regime (4):** `adam_both` (train both layers), `adam_first_lstsq` (first layer trained, readout re-solved by lstsq), `frozen_init_lstsq` (init frozen, readout solved -- the init's pure geometry quality), `qi_lstsq` (the init-independent floor reference). **function (6)**, **width (5)** $N\in\{32,\dots,512\}$. Adam runs in **float32 on MPS** (cosine schedule, best over two LRs); all refits and reported errors are fp64; metric eval rel $L_2$.

**Code & data.** `experiments/expD02_adam_geometry/` (`run.py`, `run_lambda_init.py`, `build_cube.py`, `plot_slices.py`). Data: `cube.json` (120 rows). Figures: `training_slice/`, `initialization_slice/`, `adam_geometry.png`. Sources: `data.json`, `data_lambda_init.json`, `stage1_tune.json`.

## Results

- **The three wins (sine, representative):** QI-init train-both ~$1.4\times10^{-5}$ vs Xavier's ~$4\times10^{-3}$; QI-init train + lstsq-refit ~$5\times10^{-14}$ (matching the no-training floor); scaled_xavier train-both ~$6\times10^{-4}$, beating Xavier without exact centers.
- **Geometry survives training; the readout is the barrier.** With QI init, `adam_first_lstsq` $\le10^{-13}$ (geometry intact), but the trained net's own readout (`adam_both`) drifts up to ~$10^{-5}$ -- Adam can't even hold machine precision from there.
- **Coverage gets you in; uniformity holds the floor.** Plain Xavier is useless (~$10^{-2}$); any covering geometry at the right bandwidth is far better at small $N$. But only `qi` (uniform) stays at the floor as $N$ grows -- covering-but-random inits start near the floor at $N=32$ and decay several orders by $N=512$.

### Figures

- **`training_slice/training_slice_<init>.png`** (4, one per init) -- lines = the 4 training regimes vs width. The qi panel shows wins #1/#2 (train-both ~$10^{-5}$, train-then-refit at the floor); the scaled_xavier panel shows win #3.
- **`initialization_slice/init_slice_<regime>.png`** (3, one per regime) -- lines = the 4 inits + QI baseline. The `frozen_init_lstsq` panel shows coverage-vs-uniformity (qi flat at the floor, others decaying with $N$); the `adam_both` panel shows qi-init's training advantage.
- **`adam_geometry.png`** -- original Xavier-baseline figure (kept for reference; superseded by the slices).

## Additional details

- Single seed per cell, so fine trends are noisy; the three wins are large and robust. Clean coverage (`const_weight`) did not beat heavy-tailed (`scaled_xavier`) here -- flagged.
- The HP-tuning stage surfaced the $\gamma$-init-scale finding (a ~$64\times$ init rescale lets even untrained random-center geometry hit the floor under lstsq) -- the seed of the future $\gamma$-init sweep. exp17 itself uses standard Xavier to keep the "does Adam learn geometry?" test honest.
- expD05 is the fp64 follow-up for that scale-init finding: it reruns scale-corrected and QI-scale initialization families in this repo's current PyTorch stack, with train/eval sanity, drift plots, and final lstsq refits. Its full matrix confirms that construction-scale initialization persists through Adam and beats standard affine baselines; the exact deployable default is still pending Sam review.

## Conclusions

*Three wins approved; finer orderings pending seed-averaged review.* A good init changes the trainable floor (win #1); a final lstsq refit recovers the construction floor from a trained geometry (win #2); and getting $\gamma=O(N)$ is the transferable lever (win #3). The combined recipe -- init in the right regime, train, then solve the readout -- is the most promising path to machine precision via training (achieved in fp32), and extends to $1\to\mathbb{R}^n$ via shared geometry + per-coordinate lstsq.

## Open questions

- **Did the coefficients move during training?** (Sam) With QI init, how much do the first-layer parameters change before the refit, and how does this look for runge?
- Seed-average the finer trends (coverage vs uniformity, scaled vs clean).
- **Promote the deployable scale-aware initializer** (expD05): decide whether scale-corrected Xavier, QI-scale grids, or low-QI multiscale grids should become the default recipe now that the full fp64 matrix confirms the scale story.
