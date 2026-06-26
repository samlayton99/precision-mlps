# Roadmap -- what to run next

The single place to pick up. For what's established, see `results/results.md` and the per-checkpoint consolidations (e.g. `results/checkpoint_C_geometry/expC_results.md`); for the math/code, `docs/explanation.md`. Sorted most to least important; each item carries its own open questions.

## Central question

Can a training/optimization strategy learn QI-like solutions, closing the gap between the explicit construction ($\sim 10^{-15}$) and standard training ($\sim 10^{-10}$)? The paper's three violations frame it: (1) $\gamma$ stays $O(1)$ instead of growing as $O(N)$, (2) outer weights blow up, (3) features rank-saturate.

**Success criterion.** A method works if, across $N\in\{32,64,128,256,\dots\}$ on the 6-category target family, over 3-5 seeds, error falls at $O(\log(1/\varepsilon))$ and reaches eval relative $L_2\le 10^{-13}$ with $L_\infty$ at machine epsilon -- **without** initializing from the exact construction.

## Where we are

The diagnostic checkpoints (A-E) localized the problem and produced a working recipe:

- **The readout is solved, not hard.** lstsq on a fixed correct geometry reaches the fp64 floor (Checkpoint A). The "weight blowup" violation is dissolved: the readout norm *decays* with width, and QI just sits ~$1.25\times$ off the min-norm representative (expA05).
- **Geometry is the whole game, and its $2N$ DOF collapse to ~one knob:** a shared bandwidth ($\lambda^*\approx0.25$, i.e. $\gamma=O(N)$) plus a uniform center grid; biases are derived; signs are irrelevant (Checkpoint C, `expC_results.md`). Bandwidth is the forgiving first-order knob; uniform centers are a hard structural requirement (non-uniform never reaches the floor). The failures are approximation-theoretic (coverage), not numerical (conditioning/curvature ruled out).
- **Raw first-order Adam cannot grow $\gamma$ or solve the readout on the ill-conditioned $\Phi$** (Checkpoint D), but **QI-init + train + final lstsq refit recovers the construction floor**, and scaled-Xavier (right bandwidth, inexact centers) generalizes the gain (expD02).
- **The recipe extends to 2D** (Checkpoint E).

So the open frontier: can an optimizer *discover* the precision-admitting geometry from a generic start, and does the recipe extend in dimension and depth?

## The decisive arc (items 1-3 close the paper)

One hypothesis -- *raw $(w,b)$ are the wrong optimization variables* -- tested from two angles, after one cheap design question. If either method reaches the floor from random init across the width ladder, the thesis becomes one sentence: *QI theory reveals the right coordinate system; in those coordinates, train + final solve reaches machine precision.* Resist re-mapping the landscape; the diagnostic phase is done. Point everything here.

**1. Resolve $\gamma$-only vs $\gamma$+uniformity (cheap, design-critical -- do first).**
A genuine tension: expC03 says $\gamma=O(N)$ is required; the $\gamma$-init finding says right-$\gamma$ + random centers + lstsq already hits the floor on smooth targets; but expD02/expC04 show random/scaled centers **decay with $N$ while uniform holds**. Likely resolution: **coverage + right $\gamma$ gets into the basin; uniformity holds the floor as $N$ grows.** Pin it down -- it decides whether the optimizer must learn *uniform centers* or *just $\gamma$*. If just $\gamma$, the problem collapses to a one-parameter barrier and the story is clean.

**2. Variable projection (`experiments/expD04_varpro`, stub) -- the cleanest shot.**
Eliminate the readout exactly (solve $v(\theta)$ by lstsq each step), optimize only the nonlinear geometry $\theta=(\lambda,\delta_k)$ with Gauss-Newton / LM / quasi-Newton. The nonlinear problem is tiny -- essentially one bandwidth plus small center deltas -- and a second-order method is immune to the vanishing $\mathrm{sech}^2$ gradient that blocks Adam from growing $\gamma$. Both a candidate method and a diagnostic: if VarPro reaches the floor where end-to-end Adam stalls, raw coordinates are confirmed as the wrong variables.

**3. Reparameterization (`experiments/expD03_reparameterization`, stub).**
Does optimizing in natural coordinates let first-order descent reach the floor that raw $(w,b)$ cannot? Test head-to-head on the same widths/targets/optimizer: log-scale $\gamma=\exp(\eta)$; global bandwidth $\gamma=\lambda/h$ with a single learnable $\lambda$; dimensionless centers $c_k=-1+h(k+\delta_k)$; an $\alpha=a\gamma$ readout. Pair with the final lstsq refit that already works. This is also the direct test of expC05's one-way coupling (weight uniformity needs center uniformity) -- does a joint reparameterization remove the joint-movement barrier?

## Supporting experiments (priority order)

**4. The $\gamma$-init-scale sweep.**
Accidental finding from expD02 stage-1 tuning: rescaling the init so $\gamma\approx\gamma^*=O(N)$ lets even untrained, random-center geometry hit the floor under lstsq, while at standard Xavier ($\gamma\approx0.1$) it is useless. Sweep the init $\gamma$-scale (global and per-neuron) across widths and the target family; map where untrained-init + lstsq stops reaching the floor as a function of $\gamma/\gamma_\text{ideal}$; test whether a steep-$\gamma$ init plus a short Adam pass (or the log-$\gamma$ reparameterization) closes the gap raw Adam cannot.

**5. Geometry-ladder levels 4-7 (`experiments/expD01_geometry_ladder`).**
expD01 covered through level 3 (frozen geometry, trained readout). Continue relaxing: fixed $\gamma$ + free centers, then free $\gamma$ + free centers, each with the readout solved or trained, to localize exactly where precision is lost as constraints come off.

**6. Scaling-law characterization (paper backbone).**
The expB02 fixed-$\lambda=0.25$ rerun confirmed the power-law-descent-then-floor law isn't a bandwidth-selection artifact. Remaining: what sets the descent slope (activation and target both matter; relu is a clean ~$N^{-2}$), and **does the number of width/data decades to reach the floor grow as $\log(1/\varepsilon)$** -- the form the success criterion needs, and possibly provable?

**7. Cascaded multi-band geometry (from expC06).**
expC06 showed the weight-uniformity "hump" is the loss of soft (low-bandwidth) neurons that span a low-degree polynomial basis; protecting them helps. Instead of protecting an accidental Xavier tail, *design* the multi-scale basis: a uniform grid at the ideal $\gamma$ plus a small evenly-spaced sub-grid of soft neurons, optionally medium bands between. Open: does it beat both pure-uniform and Xavier-protected, and how should band count/spacing/bandwidth-ratios scale with $N$? (Also the natural realization of the residual-fitting idea below.)

**8. Curvature-aware center placement (1D and 2D).**
expC05 (runge $N{=}64$) and expE01 (runge2d) both hint that clustering centers at high target curvature can beat uniform placement. Run a deterministic curvature-clustering experiment in 1D, and in 2D place uniform coverage near the bumps rather than uniformly over the disk; check whether the scaling laws then descend cleanly. Disentangle scale from placement (fix vector scale, vary only structure). Open: is the runge lead real or incidental center-clustering?

**9. The second bandwidth mode near $\lambda\approx0.05$ (expC03 tangent).**
At large $N$ a faint second near-floor region appears at small $\lambda$ (for runge it edges out $0.25$). Aliasing, or does width make the small-bandwidth regime attainable? Map whether scaling keeps opening it.

## Open threads (no dedicated experiment yet)

- **Precision vs generalization in data-poor regions (expC06).** The precision-optimal uniform-$\gamma$ geometry may extrapolate/interpolate-across-gaps poorly (all one sharp scale, no smooth global trend). Cheap test: hold out a middle interval, fit lstsq on the rest with (a) trained geometry, (b) uniform construction, (c) the cascade, compare held-out error. Does the cascade's soft bands recover generalization?
- **Residual-fitting division of labor (expC06, theory).** Is multistage residual fitting just "a few soft low-order modes do the coarse work each stage, sharp modes do the residual"? If so, seeding a few soft modes per stage may be the right lever. Worth formalizing.
- **Few-soft-neuron floor improvement -- real or artifact? (expC06).** Protecting ~5-10 smallest neurons beat the cardinal floor by up to ~10x; needs many more seeds + an explanation for why it fades by $N{=}512$. Also: fraction vs raw count (the uniform/init regimes disagree).
- **Coefficient closeness / cardinal-basis recovery (`explanation.md` §13).** expA02 showed lstsq and QI agree as *functions*; never as *coefficients*. Cheap plot: $\|a_\text{LS}-a_\text{QI}\|/\|a_\text{QI}\|$ and $\mathrm{cond}(\Phi)$ across widths, in both the tanh and cardinal bases, with a degraded-geometry control. Turns "geometry is the bottleneck" from hypothesis toward evidence.
- **Did the coefficients move during training? (expD02).** With QI init, how much do first-layer parameters change before the refit, and how does this look for runge?
- **Reproduce paper §4.1 "direct training fails" (paper completeness).** The infrastructure exists (`src/training/`, Adam->LBFGS loop, metric schema). Not yet run as a training experiment; the most direct "fill in Section 4.1" task, and it produces the $\gamma$/$\lambda$/weight-norm-vs-width plots (violations #1/#2) for free.

## The new frontier (Sam) -- after the optimizer arc

- **$\mathbb{R}^n\to\mathbb{R}^m$.** expD02 suggests $1\to\mathbb{R}^m$ is already solved (one shared geometry + per-coordinate lstsq). The open part is higher *input* dimension (the 2D Radon recipe of Checkpoint E is step one) combined with depth. Domain matters: the init works well over the relevant domain.
- **Depth.** Delay until the single-hidden-layer case is fully understood, then study stacking. Speculative payoff: initialize a transformer's first hidden layers with this construction -- which needs depth, domain, and higher input dimension solved first.

## Deprioritized / dropped

- **Deprioritized:** Hessian / solution-basin landscape (`experiments/exp13_solution_basins`, stub). The paper and expC03/expC04/expC05 already show curvature, conditioning, and landscape are not the discriminator (the failure is coverage/approximation). Low priority.
- **Dropped** (done or ruled out): $\Phi$-conditioning (doesn't discriminate -- expA03/expA04/expC04), objective mismatch (lstsq reaches the floor on the right geometry), standalone noise sensitivity (Checkpoint B), weight-blowup (dissolved -- expA05).

## Caution

The expD02 wins are **single-seed and fp32** -- enough to motivate, not to claim. Before "we found the method" goes in writing, it must clear the full protocol: 3-5 seeds, width ladder, 6 targets, fp64 eval, no construction init.

## Standard logging (every experiment)

- train $L_\infty$, eval $L_\infty$, eval relative $L_2$
- $\gamma$, $\lambda=\gamma h$ (mean/median/max)
- max absolute outer weight, $\|v\|_2$
- feature-rank diagnostics (singular values, stable rank)
- seed-to-seed variance (3-5 seeds)

Target family (6 categories): low-frequency analytic, high-frequency analytic, boundary-layer/steep, mixed-scale, polynomial/entire, one slightly-rough-but-smooth.
