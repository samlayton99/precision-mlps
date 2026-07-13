# Results -- cross-experiment summary

Compilation of findings across all checkpoints. Per-experiment detail (design, figures, raw numbers) lives in each `results/checkpoint_*/exp*/exp*_results.md`. Conclusions here are either plainly visible in the data or signed off by Sam; pending items are flagged.

## The story so far

The research question is whether training can close the gap between the explicit QI construction ($\sim 10^{-15}$) and standard training ($\sim 10^{-10}$). Decomposing the MLP into geometry (centers, $\gamma$) $\to$ frozen $\tanh$ features $\Phi$ $\to$ linear readout, the work has localized the difficulty sharply and produced a working recipe:

- **The readout is solved, not hard.** On a fixed correct geometry, the readout is a convex least-squares solve that reaches machine precision in fp64 -- at least as accurate as the QI convolution formula in every tested case, and far better in fp64 (Checkpoint A). It is underdetermined (a large null space), but min-norm lstsq is a clean canonical choice. The "weight blowup" violation is dissolved: the readout norm decays with width, and QI sits only ~$1.25\times$ off min-norm (expA05).
- **Geometry is the whole game, and its degrees of freedom collapse.** Precision is controlled by the first-layer geometry, and Checkpoint C shows the $2N$ inner parameters reduce to essentially one free knob: all weights share a single bandwidth in a viable band ($\lambda^*\approx 0.25$, i.e. $\gamma=O(N)$), centers must lie on the uniform grid, and biases are derived. Bandwidth is the forgiving first-order knob (a wide basin); uniform centers are a hard structural requirement (non-uniform placement never reaches the floor). Sampling barely matters; $y$-noise sets a hard statistical floor (Checkpoint B).
- **Optimizers stall on raw coordinates, but a good init + final solve works.** First-order Adam cannot solve even the convex readout on the ill-conditioned $\Phi$ (Checkpoint D, expD01). But initializing in the right ($\gamma$, geometry) regime, training, and then re-solving the readout by lstsq recovers the construction floor -- the most promising path found (expD02). expD05 confirms the scale-aware story in the repo's fp64 stack: $\lambda=\gamma h$ persists through Adam, scale-aware families beat standard affine baselines across the full matrix, and the final lstsq refit is still the step that exposes near-construction precision.
- **It generalizes to 2D.** The fixed-geometry + lstsq recipe reaches the fp64 floor on a smooth 2D target under a Radon ridge geometry (Checkpoint E).

---

## Checkpoint A -- numerical validation and method justification

We confirm the tooling is sound and justify least squares as the readout method. `results/checkpoint_A_numerics/`.

- **expA01 numerics_sanity** -- the precision floor is real, not a numerics artifact: the construction hits its claimed floor (grid-independent), `lstsq`/`svd` recover the readout while $\Phi^\top\Phi$-forming solvers lose $\sim 7$ decades, the halo is necessary, and fp64 $\tanh$ is accurate to one ulp.
- **expA02 qi_vs_lstsq** -- the least-squares readout is empirically superior to the QI Toeplitz construction: at least as accurate at equal precision in all 48 cells, and $\sim 10^{-13}$ vs QI-fp64's $\sim 10^{-10}$ in fp64. The convolution machinery is unnecessary.
- **expA03 coeff_nullspace** -- QI and lstsq are the same function but different coefficients; the entire difference lives in the $\sim 108$-dim null space of $[\Phi,\mathbf 1]$. lstsq picks the min-norm representative. More data does not shrink the null space (it is geometry, not sampling).
- **expA04 activation_conditioning** -- tanh has an $O(1)$ null space (rank $\approx N$) and reaches the floor; GELU has an $O(N)$ null space (rank $\approx 0.4N$) and is $1$--$3$ orders worse. tanh is the right activation.
- **expA05 weight_blowup** -- the "weight blowup" violation is dissolved: once the target is resolved, the readout norm *decays* with width (power law) for both QI and lstsq; QI carries only ~$1.25\times$ the min-norm lstsq vector. The $10^6$--$10^7$ norms at small $N$ are resolution failures, not an optimizer pathology. *(Pending Sam.)*

**Open questions (Checkpoint A).**
- **Coefficient closeness / cardinal-basis recovery** (from expA02/explanation §13): expA02 showed lstsq and QI agree as *functions*; measure whether they agree as *coefficients* -- $\|a_\text{LS}-a_\text{QI}\|/\|a_\text{QI}\|$ and $\mathrm{cond}(\Phi)$ across widths, in both the tanh and cardinal bases, with a degraded-geometry control. Turns "geometry is the bottleneck" from hypothesis toward evidence.

---

## Checkpoint B -- scaling laws and noise robustness

We show the fixed-geometry + lstsq method scales predictably and is robust to noise in a quantified way. `results/checkpoint_B_scaling/`.

- **expB01 sampling_and_noise** -- precision is governed by the centers, not the sample points; $x$-jitter is harmless; $y$-noise sets a hard floor near the noise magnitude, recoverable only at the statistical $1/\sqrt{n}$ rate (infeasible in practice).
- **expB02 scaling_laws** -- a clean power-law scaling law: error descends as a power law in width (and in data, collapsing past the $W+1$ threshold) until it bottoms out at the common fp64 floor (~$5\times10^{-14}$). The activation and target set the slope and intercept, not the floor: relu is the cleanest, slowest power law (~$N^{-2}$, not yet at the floor in range); tanh/gelu descend far more steeply and reach the floor fast. Confirmed at fixed $\lambda=0.25$ (not a bandwidth-selection artifact).

**Open questions (Checkpoint B).**
- **Characterize the power-law scaling law**: the slope is set by activation and target -- does the number of width/data decades to reach the floor grow as $\log(1/\varepsilon)$ (the form the success criterion requires)?

---

## Checkpoint C -- how much does the geometry matter

We isolate what the geometry must satisfy and how precision fails as you leave it. `results/checkpoint_C_geometry/`. **Consolidated synthesis: `results/checkpoint_C_geometry/expC_results.md`** -- the $2N$ inner DOF collapse to one shared bandwidth plus a fixed grid, and the failures are approximation-theoretic (coverage), not numerical (conditioning/curvature ruled out).

- **expC01 lambda_tradeoff** -- the U-shaped error-vs-$\lambda$ curve holds for *both* QI and lstsq; QI's optimum is a narrow band, lstsq's is wide and flat.
- **expC02 lambda_vs_frequency** -- QI's optimal $\lambda\approx 0.30$ is constant across frequency/width; lstsq's wide flat bottom is the real finding (the earlier "lstsq optimum moves" reading was numerical jitter, corrected here).
- **expC03 lambda_basin** -- the robust bandwidth study: $\lambda^*\approx 0.25$, essentially constant in width, with $\gamma^*/N\approx 0.10$. The answer to "what magnitude should the inner weights be."
- **expC04 center_geometry** -- only uniform centers reach machine precision; every non-uniform placement plateaus $2$--$3$ orders above. The gap is placement alone and does not track $\mathrm{cond}(\Phi)$.
- **expC05 geometry_interpolation** -- perturbs each geometry axis off the QI point (de-confounded: L1/mean-abs weights, derived bias, no origin crossing). Center uniformity is monotone (more uniform is strictly better; non-uniform never reaches the floor). Weight uniformity is conditional and non-monotone (wins 2--5 decades at the viable $\lambda$, loses at small $\lambda$, and gets worse before better along the path). The two are coupled *one-way*: uniform weights *need* uniform centers (uniform weights + random centers is the worst corner, worse than the random start), while center uniformity helps on its own. The weight sign pattern carries no information (gelu $\approx$ gelu-positive; tanh sign absorbed by the readout), and gelu is tanh shifted by a constant in $\lambda$. Argues for joint movement / the $\gamma(x-c)$ / log-$\gamma$ reparameterization (expD03). *(Conclusions pending Sam.)*
- **expC06 soft_neuron_interp** -- explains expC05's weight-uniformity hump: it is the loss of the soft (small-bandwidth) neurons, which span a low-degree polynomial basis that cheaply fits smooth/convex targets. Protecting them -- but not a random same-size set -- flattens the hump (causal), and the effect tracks target convexity. Suggests a deliberate cascaded multi-band geometry. (Approved by Sam.)

**Open questions (Checkpoint C).**
- **The second mode near $\lambda\approx 0.05$** (expC03): at large $N$ a faint second near-floor region appears at small bandwidth (for runge it edges out $\lambda=0.25$). Aliasing, or does width make the small-bandwidth regime attainable?
- **Cascaded multi-band geometry** (expC06): a uniform grid at the ideal $\gamma$ plus an evenly-spaced sub-grid of soft (low-bandwidth) neurons -- deterministic "protect-the-soft." Does it beat both pure-uniform and Xavier-protected, and how should the bands scale with $N$?
- **Reparameterization test** ($\to$ expD03): does optimizing in $\gamma(x-c)$ / log-$\gamma$ coordinates remove the one-way-coupling barrier?
- **Curvature-clustering** (on hold): a deterministic test of whether clustering centers at high target curvature beats the uniform grid (the runge small-$N$ lead).

---

## Checkpoint D -- can optimizers find the geometry

We test whether training reaches the precision the geometry admits. `results/checkpoint_D_optimizers/`.

- **expD01 geometry_ladder** -- on the easiest rung (frozen ideal geometry, convex readout), Adam stalls at $\sim 10^{-3}$ while lstsq on the identical $\Phi$ reaches $\sim 10^{-13}$. Not weight blowup (weights stay $O(1)$) -- first-order descent failing on the ill-conditioned $\Phi$. The readout barrier is an optimization/conditioning problem; solve it directly. *(New writeup, pending sign-off.)*
- **expD02 adam_geometry** -- three wins (approved by Sam): (1) QI init + train both layers reaches $\sim 10^{-5}$, far better than standard-init training; (2) QI init + train + final lstsq refit recovers the construction floor ($\sim 5\times10^{-14}$) from the trained geometry; (3) scaled_xavier (right bandwidth, inexact centers) generalizes the gain. All in fp32. The recipe extends to $1\to\mathbb{R}^n$ via shared geometry + per-coordinate lstsq.
- **expD05 scale_init_story** -- full repo-local fp64 rerun of the initialization story: 1080 rows over 6 targets, 4 resolutions, 5 seeds, and 9 initializer families, with 0 train/eval sanity flags. Adam alone does not close the non-oracle precision gap (best non-oracle trained row $\sim 7.1\times10^{-5}$), but scale-aware initializers beat standard affine medians on all 24 target-resolution blocks. Final lstsq refits are the stronger geometry check: scale-corrected refit medians beat standard refit medians on all 24 blocks and reach $\le 10^{-10}$ on 9/24 blocks, with the best refit at $\sim 1.0\times10^{-14}$. *(Pending Sam review for which deployable initializer to promote.)*

**Open questions (Checkpoint D).**
- **Did the coefficients move during training?** (expD02): with QI init, how much do the first-layer parameters change before the refit, and how does this look for runge?
- **Promote the deployable scale-aware initializer** (expD05): decide between scale-corrected Xavier, QI-scale grids, and low-QI multiscale grids for the Checkpoint D recipe; the full matrix confirms the scale story, but the preferred deployable default is still a Sam-review choice.
- **Geometry-ladder levels 4--7** (expD01): relax the geometry (free centers, free $\gamma$) with the readout solved or trained, to localize where precision is lost.
- **Reparameterization** (`experiments/expD03_reparameterization`, stub): the natural test of the expC05 coupling finding -- optimize in $\gamma(x-c)$ / log-$\gamma$ coordinates.
- **Variable projection** (`experiments/expD04_varpro`, stub): eliminate the readout exactly, optimize only the nonlinear geometry -- a diagnostic for whether the geometry block is the real failure.

---

## Checkpoint E -- extending to 2D

`results/checkpoint_E_2d/`.

- **expE01 geometry_zoo_2d** -- the fixed-geometry + lstsq recipe carries to $\mathbb{R}^2\to\mathbb{R}$: a smooth target (`gauss_bump`) reaches the fp64 floor ($\sim 6\times10^{-14}$) under the Radon ridge geometry, which is best/tied on 3 of 4 targets at the largest width. Geometry placement decides precision, as in 1D. `runge2d` (central spike) stays resolution-limited. *(Conclusions pending sign-off; supersedes the hex-only study, now folded in as the `hex` geometry.)*

**Open questions (Checkpoint E).**
- **Concentrate coverage where the target needs it** (Sam): place uniform coverage near the bumps/high-curvature regions rather than uniformly over the disk (the `random_ridges` win on runge is likely incidental center-clustering), and check the scaling laws under that placement. Same curvature-clustering lead as expC05, in 2D.
- Whether extended precision and larger $N$ push the other smooth 2D targets to the floor.

---

## Future directions (beyond the current checkpoints)

The diagnostic experiments have localized the problem; these are the frontier:

- **Reparameterization (live, expD03)** and the **$\gamma$-init-scale sweep** -- the most direct follow-ups to the Checkpoint C/D findings.
- **Variable projection (medium, expD04)** -- reduced-coordinate training on the nonlinear geometry only.
- **Deprioritized:** Hessian / solution-basin landscape (`experiments/exp13_solution_basins`, stub) -- the paper and expC03/expC04/expC05 already indicate curvature, conditioning, and landscape are not the discriminator (the failure is coverage/approximation); kept as low-priority. Dropped from the roadmap entirely: $\Phi$-conditioning, objective mismatch, and standalone noise studies (done or ruled out in Checkpoints A/B/C).
- **The real new frontier (Sam):** **depth** (delay until 1-layer is fully understood) and **$\mathbb{R}^n\to\mathbb{R}^m$** (expD02 suggests $1\to\mathbb{R}^m$ is already solved via shared geometry + per-coordinate lstsq; the open part is higher input dimension and depth). A speculative application: initializing a transformer's first hidden layers this way -- which needs depth, domain, and higher-dimension input solved first.
