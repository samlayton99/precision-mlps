# Open questions & proposed experiments -- master list

Every open question and proposed experiment, collected from each experiment's `results.md`, `thoughts.md`, and the prior roadmap. Grouped by checkpoint, deduped, sorted most-to-least relevant within each. Established findings live in `results/results.md` and the per-experiment `expX_results.md`.


- try it on harder functions (sin(1/x(+eps))) (more interesting functions in repo)
- 


## Checkpoint A -- numerics / method

- **Activation null-space regimes: bounded vs growing.** expA04 found tanh has a bounded (~constant) null space in $[\Phi,\mathbf 1]$ (rank $\approx N$) and reaches the floor, while gelu has an $O(N)$ (growing) null space (rank $\approx 0.4N$) and stalls 1-3 orders short -- with comparable condition numbers, so it's rank, not conditioning, that separates them. Open: *why* -- what property of an activation sets the bounded-vs-growing regime (the paper's kernel conditions K1-K4 on $K=\psi^{(r)}$ should predict it)? Is a bounded null space necessary to reach the floor, are there any correlations? Map other valid-kernel activations (sin, sech, softplus) onto the two regimes. (expA04; paper §3 kernel conditions)

## Checkpoint C -- geometry
- **Cascaded multi-band geometry.** Build a deliberate multi-bandwidth geometry by hand: the uniform grid at the ideal $\gamma$, plus a smaller evenly-spaced set of low-bandwidth ("soft") neurons, optionally a couple of bands at intermediate $\gamma$. Does it beat both plain-uniform and the accidental Xavier soft tail? How should band count, spacing, and bandwidth ratios scale with $N$? (expC06; thoughts)
- **Soft-neuron protection and the residual-fitting interpretation.** (a) Is freezing the smallest ("soft") neurons better than full weight uniformity? (Tentatively yes, especially convex targets -- it flattens the hump.) (b) Is the ~$10\times$ floor improvement from a few soft neurons at $N{=}256$ real or a seed fluke (it fades by $N{=}512$)? Needs many seeds with per-target, per-$N$ error bars; decide fraction-of-width vs fixed count. (c) Is multistage fitting just "a few soft low-order neurons do the coarse fit each stage, the sharp neurons fit the residual"? If so, seeding/protecting a few soft neurons per stage is the lever -- formalize and test. (expC06; thoughts)
- **Curvature-adaptive uniform centers (1D + 2D).** Increase center density in high-curvature regions while keeping placement locally uniform -- curvature-adaptive spacing, not random clustering. Does it beat the globally-uniform grid? Concrete test: can it get runge to machine precision at $N{=}32$, where the uniform grid falls short? Is that small-$N$ runge gap closable with a target-aware halo, or is it intrinsic to equal-width kernels on a peaked target? In 2D, place denser-but-locally-uniform coverage near the bumps / high-curvature regions instead of uniformly over the disk (`random_ridges`' runge win is likely incidental center-clustering) and check whether the scaling laws then descend cleanly. (expC05; expE01; thoughts)
- **Second bandwidth mode near $\lambda\approx0.05$.** At large $N$ a second near-floor region appears at small bandwidth (for runge it slightly beats $\lambda=0.25$). Is it aliasing, or does growing width genuinely open a usable small-bandwidth regime? Does it keep widening with $N$? (expC03; thoughts)
- **The last-mile step change.** Why does the final nudge to fully-uniform weights and centers give a sudden jump in precision, instead of gradual improvement? (thoughts; low priority)

## Checkpoint D -- optimizers
- **Reparameterization (expD03).** Does optimizing in natural coordinates let plain gradient descent reach the floor that raw $(w,b)$ can't? Head-to-head: log-$\gamma$ ($\gamma{=}e^{\eta}$); one global $\lambda$ with $\gamma{=}\lambda/h$; dimensionless centers $c_k=-1+h(k+\delta_k)$; a readout scaled as $\alpha{=}a\gamma$. Pair each with a final lstsq refit. Direct test of expC05's one-way coupling. (expC05; prior roadmap)
- **What does the trained geometry look like, and did it move?** Characterize a trained net's geometry: are centers uniform, what's their spread, what's $\gamma$? With QI init, how much do the first-layer params actually move before the refit, and what does that look like for runge? (expD02; thoughts)
- **Initialization Regimes in greater depth** Expanding on expD02, what properties should we test about the trained solutions in the init regime. Is there an optimizer and initialization regime that work better? this is the real question.
- **Variable projection (expD04).** Solve the readout exactly by lstsq at every step, and optimize only the nonlinear geometry $\theta=(\lambda,\delta_k)$ with Gauss-Newton / LM / quasi-Newton. Does it reach the floor where end-to-end Adam stalls? (prior roadmap)
- **Can any optimizer solve the lstsq readout?** First-order Adam can't solve the readout on the ill-conditioned $\Phi$ (the 4th barrier, expD01). Can a second-order / quasi-Newton method get there -- SSBroyden (the paper's two-stage Adam$\to$SSBroyden), LM, or a preconditioned solver? (expD01; thoughts)
- **Geometry-ladder levels 4-7 (expD01).** Continue the ladder: relax more constraints -- fixed $\gamma$ + free centers, then free $\gamma$ + free centers -- each with the readout either solved or trained. Pinpoint exactly which relaxation first loses precision. (expD01)
- **Seed-average the finer expD02 trends** (coverage vs uniformity; scaled vs clean init). (expD02)

## Checkpoint E -- 2D

- **Curvature-aware 2D coverage.** See the merged Checkpoint C curvature item -- the 2D half lives there.
- **Does optimal $\lambda$ drift with $N$ in 2D?** It looks like it may decrease slightly as $N$ grows. Confirm or rule out. (expE01; thoughts)
- **Do extended precision and larger $N$ push the other smooth 2D targets to the floor?** (expE01) More robust treatment, as it appears they will.

## Checkpoint F -- applications
- **1D and 2D real physics task** with the constructed geometry, end-to-end. (thoughts)
- **Depth.** Stack the construction across multiple layers (delay until the 1-hidden-layer case is fully understood/a good optimization and init strategy is found). Just try the initialization on multiple layers. (thoughts; frontier)
- **Higher output dimension ($\to\mathbb{R}^m$).** Emperical chec, works in theoery. Multi-output via shared geometry + per-coordinate lstsq is partly shown for $1\to\mathbb{R}^m$ (expD02); push it further -- more outputs, harder targets, and combined with higher input dim / depth. (expD02; frontier)
- **Higher input dimension ($\mathbb{R}^n\to$).** The harder open part: higher input dimension plus depth (the 2D Radon recipe is step 1). Domain matters -- the init works well over the relevant domain. (frontier)
- **Non-MSE losses.** How does the method behave under cross-entropy / non-MSE objectives? (thoughts)
- **Transformer init.** Initialize a transformer's first hidden layers with this construction (needs depth, higher input dimension, and domain solved first). (frontier)

## Checkpoint G -- generalization

Precision-vs-generalization of the construction and its variants (uniform, cascade multi-band, soft-weight protection), separate from the pure-precision question in Checkpoint C. First concrete experiment is the expG01 interactive explorer (below); this checkpoint will hold multiple.

- **expG01 -- interactive geometry / generalization explorer (built, live).** Local Dash web app: set $\lambda$, $N$, target function, #train/#test samples, and a slide-able hold-out mask; every change re-solves the SVD min-norm readout on the unmasked points and redraws -- target vs approximation, log-scale residual, and a 3x2 rel-$L_2$/$L_\infty$ table over the entire / unmasked / masked test range. Same halo setup as the batch experiments. Run: `~/.venvs/precisionMLPs/bin/python experiments/expG01_interactive_explorer/run.py`, then open http://127.0.0.1:8050. Writeup: `results/checkpoint_G_generalization/expG01_interactive_explorer/expG01_results.md`.
- **Precision vs generalization (mask the data).** The precision-optimal uniform-$\gamma$ geometry is all one sharp scale, so it may interpolate/extrapolate badly in data-poor regions. Mask out parts of the domain (a held-out middle interval, or scattered gaps) and compare held-out error head-to-head across three approaches: (1) an Adam-trained network, (2) the cascade multi-band geometry + lstsq (Checkpoint C), (3) the QI / uniform construction. Do the cascade's soft bands recover generalization where the single-scale uniform geometry fails? (expC06; thoughts)
- **Soft-weight tradeoff.** How does freezing / protecting the soft (low-bandwidth) neurons trade precision against generalization? (expC06; thoughts)
- **Data-poor regions.** Behavior under extrapolation and sparse coverage. (thoughts)

## Reference (conventions, not questions)

- **Success criterion.** Across $N\in\{32,64,128,256,\dots\}$ on the 6-category target family, over 3-5 seeds, error falls at the exponential-in-width rate (Corollary 1, $\sim e^{-\alpha W}$) and reaches eval relative $L_2\le 10^{-13}$ with $L_\infty$ at machine epsilon -- *without* initializing from the exact construction.
- **Standard logging.** train/eval $L_\infty$, eval rel $L_2$; $\gamma$ and $\lambda=\gamma h$ (mean/median/max); max $|$outer weight$|$ and $\|v\|_2$; feature-rank diagnostics (singular values, stable rank); seed-to-seed variance (3-5 seeds).
- **Target family (6 categories).** low-frequency analytic, high-frequency analytic, boundary-layer/steep, mixed-scale, polynomial/entire, one slightly-rough-but-smooth.
- **Answered / dropped.** Coefficient closeness QI vs lstsq (answered by expA03: same function, difference lives in the ~108-dim null space, lstsq is min-norm, absolute gap decays with width -- only a confirmatory cond/norm-vs-width plot remains, low priority). Also dropped or ruled out: $\Phi$-conditioning, objective mismatch, standalone noise studies, weight-blowup.
- **Stubs / deprioritized.** `expD03_reparameterization`, `expD04_varpro` (stubs, above). `exp13_solution_basins` deprioritized (curvature/conditioning ruled out as discriminators).
