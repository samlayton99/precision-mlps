# Future experiments / roadmap

Design spec for the open work. For what is already established, see `results/results.md` (the cross-experiment summary) and the per-experiment writeups under `results/checkpoint_*/`.

## Central question

Can we find a training/optimization strategy that learns QI-like solutions, closing the gap between the explicit construction ($\sim 10^{-15}$) and standard training ($\sim 10^{-10}$)? The paper's three violations in trained networks frame it: (1) $\gamma$ stays $O(1)$ instead of growing as $O(N)$, (2) outer weights blow up instead of staying $O(1)$, (3) features rank-saturate.

**Success criterion.** A method works if, across widths $N\in\{32,64,128,256,\dots\}$ on the 6-category target family, over 3-5 seeds, error falls at $O(\log(1/\varepsilon))$ and reaches eval relative $L_2\le 10^{-13}$ with $L_\infty$ at machine-epsilon precision -- **without** initializing from the exact constructive solution.

## Where we are

The diagnostic checkpoints (A-E) have localized the problem and produced a working recipe:

- The readout is a solved convex problem -- lstsq on a fixed geometry reaches the fp64 floor (Checkpoint A).
- Precision is controlled by the first-layer geometry, and mostly by bandwidth: $\lambda^*\approx 0.25$ (i.e. $\gamma=O(N)$) plus uniform centers. Bandwidth is first-order, center uniformity second-order (Checkpoint C).
- Raw first-order Adam cannot grow $\gamma$ or solve the readout on the ill-conditioned $\Phi$ (Checkpoint D), but **QI-init + train + final lstsq refit recovers the construction floor**, and scaled-Xavier (right bandwidth) generalizes the gain.
- The recipe extends to 2D (Checkpoint E).

So the open frontier is: can an optimizer *discover* the precision-admitting geometry from a generic start, and does the recipe extend in dimension and depth?

## Live experiments (priority order)

**1. Reparameterization (`experiments/expD03_reparameterization`, stub -- top priority).**
Motivated directly by expC05: in raw $(w,b)$ coordinates, weight and bias must reach $\gamma$-scale together (the diagonal-valley barrier), and the vanishing $\mathrm{sech}^2$ gradient blocks Adam from growing $\gamma$. Test natural coordinates head-to-head on the same widths/targets/optimizer: log-scale $\gamma=\exp(\eta)$; global bandwidth $\gamma=\lambda/h$ with a single learnable $\lambda$; dimensionless centers $c_k=-1+h(k+\delta_k)$; and an $\alpha=a\gamma$ readout. Question: does any of these let a standard optimizer reach the floor that raw coordinates cannot?

**2. The $\gamma$-init-scale sweep (top priority).**
Accidental finding from expD02 stage-1 tuning: rescaling the init so $\gamma\approx\gamma^*=O(N)$ lets even untrained, random-center geometry hit the floor under lstsq, while at standard Xavier ($\gamma\approx 0.1$) it is useless. Systematically sweep the init $\gamma$-scale (global and per-neuron) across widths and the target family; map where untrained-init + lstsq stops reaching the floor as a function of $\gamma/\gamma_\text{ideal}$; and test whether a steep-$\gamma$ init plus a short Adam pass (or the log-$\gamma$ reparameterization above) closes the gap raw Adam cannot.

**3. Variable projection (`experiments/expD04_varpro`, stub -- medium).**
Eliminate the readout exactly (solve $v(\theta)$ by lstsq each step) and optimize only the nonlinear geometry $\theta=(\lambda,\delta_k)$ with Gauss-Newton / LM / quasi-Newton. Both a candidate method and a diagnostic: if VarPro reaches the floor where end-to-end Adam stalls, the raw end-to-end coordinates are confirmed as the wrong optimization variables.

**4. Geometry-ladder levels 4-7 (`experiments/expD01_geometry_ladder`).**
expD01 covered level 3 (frozen geometry, trained readout). Continue relaxing: fixed $\gamma$ + free centers, then free $\gamma$ + free centers, each with the readout solved exactly or trained, to localize exactly where precision is lost as constraints come off.

**5. Coefficient-magnitude / weight-blowup study (Checkpoint A follow-up).**
Measure the norm/magnitude of the solved readout coefficients ($\max|v|$, $\|v\|_2$) for QI vs lstsq across width/target, and test whether trained optimizers select a high-norm null-space representative when a min-norm one fits identically. Directly characterizes the paper's "weight blowup" violation (expA03 showed the QI/lstsq difference is pure null-space).

**6. Scaling-law characterization.**
The expB02 fixed-$\lambda=0.25$ rerun is done (`fixed_lambda_scaling.png`): the power-law-descent-then-floor law survives without the best-over-$\lambda$ confound -- same floor (within ~1 order), relu unchanged, just a noisier descent. Remaining: characterize the law -- what sets the descent slope (activation and target both matter; relu is a clean ~$N^{-2}$), and does the number of width/data decades to reach the floor grow as $\log(1/\varepsilon)$ (the form the success criterion needs)?

**7. Curvature-aware center placement (1D and 2D).**
expC05 (runge) and expE01 (runge2d) both hint that clustering centers at high target curvature can beat uniform placement. Run a deterministic curvature-clustering experiment in 1D, and in 2D place uniform coverage near the bumps rather than uniformly over the disk; check whether the scaling laws then descend cleanly. (Also disentangle scale from placement: a follow-up that fixes vector scale and varies only structure.)

**8. The second bandwidth mode near $\lambda\approx 0.05$ (expC03 tangent).**
At large $N$ a faint second near-floor region appears at small $\lambda$ (for runge it edges out $\lambda=0.25$). Is it aliasing, or does increasing width make the small-bandwidth regime attainable? Map whether scaling keeps opening it.

## Deprioritized / dropped

- **Deprioritized:** Hessian / solution-basin landscape (`experiments/exp13_solution_basins`, stub). The paper and expC03/expC05 already indicate curvature/landscape is not the discriminator; kept low-priority.
- **Dropped** (done or ruled out): $\Phi$-conditioning (shown not to discriminate precision in expA03/expA04/expC04), objective mismatch (lstsq already reaches the floor on the right geometry), standalone noise sensitivity ($y$/$x$-noise and the $1/\sqrt{n}$ law are done in Checkpoint B).

## The new frontier (Sam)

Beyond the optimizer arc, the two structural extensions:

- **$\mathbb{R}^n\to\mathbb{R}^m$.** expD02 suggests $1\to\mathbb{R}^m$ is already solved via one shared geometry + a per-coordinate lstsq readout. The open part is higher *input* dimension (the 2D Radon recipe of Checkpoint E is the first step) and combining it with depth.
- **Depth.** Delay until the single-hidden-layer case is fully understood, then study stacking. Speculative payoff: initializing a transformer's first hidden layers with this construction -- which requires depth, domain, and higher-dimension input solved first.

## Standard logging (every experiment)

- train $L_\infty$, eval $L_\infty$, eval relative $L_2$
- $\gamma$, $\lambda=\gamma h$ (mean/median/max)
- max absolute outer weight, $\|v\|_2$
- feature-rank diagnostics (singular values, stable rank)
- seed-to-seed variance (3-5 seeds)

Target family (6 categories): low-frequency analytic, high-frequency analytic, boundary-layer/steep, mixed-scale, polynomial/entire, one slightly-rough-but-smooth.
