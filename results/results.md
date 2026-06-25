# Results -- cross-experiment summary

Compilation of findings across all checkpoints. Per-experiment detail (design, figures, raw numbers) lives in each `results/checkpoint_*/exp*/exp*_results.md`. Conclusions here are either plainly visible in the data or signed off by Sam; pending items are flagged.

## The story so far

The research question is whether training can close the gap between the explicit QI construction ($\sim 10^{-15}$) and standard training ($\sim 10^{-10}$). Decomposing the MLP into geometry (centers, $\gamma$) $\to$ frozen $\tanh$ features $\Phi$ $\to$ linear readout, the work has localized the difficulty sharply and produced a working recipe:

- **The readout is solved, not hard.** On a fixed correct geometry, the readout is a convex least-squares solve that reaches machine precision in fp64 -- at least as accurate as the QI convolution formula in every tested case, and far better in fp64 (Checkpoint A). It is underdetermined (a large null space), but min-norm lstsq is a clean canonical choice.
- **Geometry is the whole game, and it is mostly bandwidth.** Precision is controlled by the first-layer geometry: a bandwidth in the viable band ($\lambda^*\approx 0.25$, i.e. $\gamma=O(N)$) plus uniform centers. Bandwidth is the first-order knob; center uniformity is a softer, second-order requirement (Checkpoint C). Sampling barely matters; $y$-noise sets a hard statistical floor (Checkpoint B).
- **Optimizers stall on raw coordinates, but a good init + final solve works.** First-order Adam cannot solve even the convex readout on the ill-conditioned $\Phi$ (Checkpoint D, expD01). But initializing in the right ($\gamma$, geometry) regime, training, and then re-solving the readout by lstsq recovers the construction floor -- the most promising path found (expD02). The raw $(w,b)$ parameterization is itself a barrier, arguing for reparameterization.
- **It generalizes to 2D.** The fixed-geometry + lstsq recipe reaches the fp64 floor on a smooth 2D target under a Radon ridge geometry (Checkpoint E).

---

## Checkpoint A -- numerical validation and method justification

We confirm the tooling is sound and justify least squares as the readout method. `results/checkpoint_A_numerics/`.

- **expA01 numerics_sanity** -- the precision floor is real, not a numerics artifact: the construction hits its claimed floor (grid-independent), `lstsq`/`svd` recover the readout while $\Phi^\top\Phi$-forming solvers lose $\sim 7$ decades, the halo is necessary, and fp64 $\tanh$ is accurate to one ulp.
- **expA02 qi_vs_lstsq** -- the least-squares readout is empirically superior to the QI Toeplitz construction: at least as accurate at equal precision in all 48 cells, and $\sim 10^{-13}$ vs QI-fp64's $\sim 10^{-10}$ in fp64. The convolution machinery is unnecessary.
- **expA03 coeff_nullspace** -- QI and lstsq are the same function but different coefficients; the entire difference lives in the $\sim 108$-dim null space of $[\Phi,\mathbf 1]$. lstsq picks the min-norm representative. More data does not shrink the null space (it is geometry, not sampling).
- **expA04 activation_conditioning** -- tanh has an $O(1)$ null space (rank $\approx N$) and reaches the floor; GELU has an $O(N)$ null space (rank $\approx 0.4N$) and is $1$--$3$ orders worse. tanh is the right activation.

**Open questions (Checkpoint A).**
- **Characterize the weight blowup directly** (from expA03): measure the norm/magnitude of the solved readout coefficients ($\max|v|$, $\|v\|_2$) for QI vs lstsq across width/target, and test whether trained optimizers select a high-norm null-space representative when a min-norm one fits identically. (This is the coefficient-magnitude study slated for this checkpoint.)

---

## Checkpoint B -- scaling laws and noise robustness

We show the fixed-geometry + lstsq method scales predictably and is robust to noise in a quantified way. `results/checkpoint_B_scaling/`.

- **expB01 sampling_and_noise** -- precision is governed by the centers, not the sample points; $x$-jitter is harmless; $y$-noise sets a hard floor near the noise magnitude, recoverable only at the statistical $1/\sqrt{n}$ rate (infeasible in practice).
- **expB02 scaling_laws** -- a clean power-law scaling law: error descends as a power law in width (and in data, collapsing past the $W+1$ threshold) until it bottoms out at the common fp64 floor (~$5\times10^{-14}$). The activation and target set the slope and intercept, not the floor: relu is the cleanest, slowest power law (~$N^{-2}$, not yet at the floor in range); tanh/gelu descend far more steeply and reach the floor fast. Confirmed at fixed $\lambda=0.25$ (not a bandwidth-selection artifact).

**Open questions (Checkpoint B).**
- **Characterize the power-law scaling law**: the slope is set by activation and target -- does the number of width/data decades to reach the floor grow as $\log(1/\varepsilon)$ (the form the success criterion requires)?

---

## Checkpoint C -- how much does the geometry matter

We isolate what the geometry must satisfy: bandwidth band, center uniformity, and the coupling between them. `results/checkpoint_C_geometry/`.

- **expC01 lambda_tradeoff** -- the U-shaped error-vs-$\lambda$ curve holds for *both* QI and lstsq; QI's optimum is a narrow band, lstsq's is wide and flat.
- **expC02 lambda_vs_frequency** -- QI's optimal $\lambda\approx 0.30$ is constant across frequency/width; lstsq's wide flat bottom is the real finding (the earlier "lstsq optimum moves" reading was numerical jitter, corrected here).
- **expC03 lambda_basin** -- the robust bandwidth study: $\lambda^*\approx 0.25$, essentially constant in width, with $\gamma^*/N\approx 0.10$. The answer to "what magnitude should the inner weights be."
- **expC04 center_geometry** -- only uniform centers reach machine precision; every non-uniform placement plateaus $2$--$3$ orders above. The gap is placement alone and does not track $\mathrm{cond}(\Phi)$.
- **expC05 geometry_interpolation** -- at the viable bandwidth the geometry requirements are asymmetric: center placement is forgiving (~1--2 decades), but per-neuron bandwidth uniformity is strict (2--6 decades) -- the latter shown cleanly with the bandwidth scale and centers held fixed (weights mode rerun, rms-normalized). Provisional: in raw $(w,b)$ coordinates weight and bias appear to need $\gamma$-scale together (a diagonal valley -> argument for the $\gamma(x-c)$/log-$\gamma$ reparameterization), but the weight+bias mode still has the scale confound and is pending its own de-confounded rerun. *(Conclusions pending Sam.)*

**Open questions (Checkpoint C).**
- **The second mode near $\lambda\approx 0.05$** (expC03): at large $N$ a faint second near-floor region appears at small bandwidth (for runge it edges out $\lambda=0.25$). Aliasing, or does width make the small-bandwidth regime attainable?
- **Curvature-clustering** (expC05, on hold): a deterministic experiment to confirm whether clustering centers at high target curvature beats the uniform grid (the runge lead).
- **De-confound the weight+bias mode** (expC05): rerun it with the magnitude held and centers pinned (as the weights mode now is), to see whether the diagonal valley survives once scale is removed. (The weights mode is already de-confounded; this remains for weight+bias.)
- **Reparameterization test**: does optimizing in $\gamma(x-c)$ / log-$\gamma$ coordinates remove the diagonal-valley barrier? (Leads into Checkpoint D / expD03.)

---

## Checkpoint D -- can optimizers find the geometry

We test whether training reaches the precision the geometry admits. `results/checkpoint_D_optimizers/`.

- **expD01 geometry_ladder** -- on the easiest rung (frozen ideal geometry, convex readout), Adam stalls at $\sim 10^{-3}$ while lstsq on the identical $\Phi$ reaches $\sim 10^{-13}$. Not weight blowup (weights stay $O(1)$) -- first-order descent failing on the ill-conditioned $\Phi$. The readout barrier is an optimization/conditioning problem; solve it directly. *(New writeup, pending sign-off.)*
- **expD02 adam_geometry** -- three wins (approved by Sam): (1) QI init + train both layers reaches $\sim 10^{-5}$, far better than standard-init training; (2) QI init + train + final lstsq refit recovers the construction floor ($\sim 5\times10^{-14}$) from the trained geometry; (3) scaled_xavier (right bandwidth, inexact centers) generalizes the gain. All in fp32. The recipe extends to $1\to\mathbb{R}^n$ via shared geometry + per-coordinate lstsq.

**Open questions (Checkpoint D).**
- **Did the coefficients move during training?** (expD02): with QI init, how much do the first-layer parameters change before the refit, and how does this look for runge?
- **The $\gamma$-init-scale sweep** (future): map where untrained-init + lstsq stops reaching the floor as a function of $\gamma/\gamma_\text{ideal}$, and whether a steep-$\gamma$ init or log-$\gamma$ reparameterization closes the gap raw Adam cannot.
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
- **Deprioritized:** Hessian / solution-basin landscape (`experiments/exp13_solution_basins`, stub) -- the paper and expC05/expC03 already indicate curvature/landscape is not the discriminator; kept as low-priority. Dropped from the roadmap entirely: $\Phi$-conditioning, objective mismatch, and standalone noise studies (done or ruled out in Checkpoints A/B/C).
- **The real new frontier (Sam):** **depth** (delay until 1-layer is fully understood) and **$\mathbb{R}^n\to\mathbb{R}^m$** (expD02 suggests $1\to\mathbb{R}^m$ is already solved via shared geometry + per-coordinate lstsq; the open part is higher input dimension and depth). A speculative application: initializing a transformer's first hidden layers this way -- which needs depth, domain, and higher-dimension input solved first.
