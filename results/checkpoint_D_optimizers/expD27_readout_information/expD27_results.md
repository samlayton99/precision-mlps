# expD27 — Readout information and freezing — data-obvious observations; mechanism unresolved

## TL;DR

- None of the four requested variations recovers the QI approximation regime from poor geometry or repairs the noisy geometry within this constant-rate budget.
- Solving on Xavier geometry and then freezing produces destructive motion. Its large coefficients also amplify floating-point cancellation, limiting interpretation as a clean test of the note's mechanism.
- Transferring coefficients from an accurate QI fit avoids those enormous coefficients. Actual mixed-sine error improves from 1.287 to 0.963, but mean gamma stays near 1 and the refitted approximation barely improves.
- Clean and noisy QI starts mostly stop improving actual error when their readouts freeze. Clean QI retains its excellent approximation; noisy QI does not regain that accuracy.

## Question / hypothesis

Can the source of the readout coefficients, and the moment at which they stop changing, preserve useful information for geometry GD? Current error, mean scale, and separately refitted error distinguish ordinary fitting, parameter movement, and improved approximation.

## Experiment design

Sam confirmed freeze times $X\in\{2,10,50,150\}$, 500 geometry updates after each freeze, a gamma-1 reset at the same QI centers for variant 2, and 10% center-preserving scale noise for variant 4. Only the four requested variations were run; fixed-center and coefficient-shuffle controls remain proposals.

All targets use the matched finite-domain problem on $[-1,1]$: sine $\sin(2\pi x)$; mixed sine $m(x)=\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$; Runge $(1+25x^2)^{-1}$; and Gaussian envelope $e^{-x^2/(2(0.4)^2)}m(x)$. Training uses 1,024 uniform midpoints, evaluation uses 8,192 independent midpoints, and arithmetic is fp64. The model is

$$
f(x)=\sum_{j=1}^{177}c_j\tanh(a_jx+b_j)+d,\qquad L=\frac{1}{2n}\sum_i(f(x_i)-y_i)^2.
$$

The width convention is $N=128$ intervals, 129 reference centers, and 24 halo neurons per side. QI geometry has uniform centers $z_j$, $h=1/64$, $\gamma=16$, and $\lambda=\gamma h=0.25$. Ordinary GD uses rate 0.002 for every trainable parameter. Geometry always trains raw slopes and biases, so centers can move during training. Preserving centers in the reset/noise refers to the initialization operation.

1. **Solve, then freeze.** Begin at the same seeded Xavier geometry as the previous experiment. Solve the readout at step zero and after every geometry update, with no readout GD. Clone the state after the solve at $X$, then hold all readout coefficients, including output bias, fixed for 500 updates. The reference continues geometry GD followed by readout solves through step 650.
2. **QI readout transferred to gamma 1.** Solve on QI geometry, retain every coefficient, and change all scales to 1 while preserving the corresponding centers. Freeze the readout from step zero and perform 500 geometry updates. There is one trajectory per target; duplicating it across four artificial freeze times would not add a comparison.
3. **QI geometry, zero readout.** Start with all readout coefficients zero. Use joint GD until $X$, then freeze the readout for 500 updates. Continued joint GD is the reference.
4. **Noisy QI geometry, zero readout.** Multiply each QI slope and its bias by $1+0.1Z_j$, where the $Z_j$ are independent standard normal draws from a fixed seeded stream. No factors were clipped or redrawn; the same factors are used for every target. Centers are initially unchanged, and mean initial scale is 15.976391. Continue as in variant 3.

The experiment contains 12 continued references, 48 frozen-readout branches, and four permanently frozen transfer trajectories. Shared prefixes are cloned rather than independently retrained. The primary plots record actual independent-grid relative error and $\bar\gamma=177^{-1}\sum_j|a_j|$ at every state. Separate sparse readout solves evaluate the approximation available from saved geometry. Those diagnostic solves never alter training.

Every coefficient solve uses the normalized augmented feature matrix $[\tanh(a_jx_i+b_j),1]/\sqrt n$ and target $y/\sqrt n$, retaining singular values greater than $10^{-13}$ times the largest. Thus “optimal readout” here means the established numerical SVD reference, not an unlimited exact-arithmetic solution. Geometry gradients hold the newly solved coefficients fixed; the SVD is not differentiated through. With a changing truncated numerical rank, this should not be silently identified with the smooth constant-rank VarPro theorem.

**Code & data**

- [Implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD27_readout_information/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD27_readout_information/config.yaml), [tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD27_readout_information.py), [design and practicality check](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/expD27_plan.md).
- [Data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/data): 16 compressed files, approximately 54 MB, containing all trajectories, configurations, numerical solve diagnostics, timing, noise factors, and the original QI coefficient reference. The driver supports --resume with exact configuration checking and --plot-only.
- [Variant 2, all four targets](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_transfer/all_targets.png).
- Refitted geometry comparisons: [variant 1](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/refitted_geometry.png), [variant 3](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/refitted_geometry.png), [variant 4 with clean-QI reference](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/refitted_geometry.png).

| Target | 1. Solve, then freeze | 3. QI + zero readout | 4. Noisy QI + zero readout |
|---|---|---|---|
| Sine | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/sine.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/sine.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/sine.png) |
| Mixed sine | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/sine_mixture.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/sine_mixture.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/sine_mixture.png) |
| Runge | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/runge.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/runge.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/runge.png) |
| Gaussian envelope | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/solve_then_freeze/gaussian_envelope.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/qi_zero/gaussian_envelope.png) | [Plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD27_readout_information/figures/noisy_qi_zero/gaussian_envelope.png) |

## Results

**Variant 1: readout solving followed by freezing is unstable at the unchanged rate.** All frozen branches finish with worse actual error and worse refitted approximation than their continued-solve references. Sine and Runge undergo extreme growth in scale and error; mixed sine and Gaussian also deteriorate. For freezing at step 150 and evaluating at step 650:

| Target | Actual relative error | Refitted relative error | Mean gamma |
|---|---:|---:|---:|
| Sine | $2.46\times10^7$ | 0.860 | $1.27\times10^9$ |
| Mixed sine | 10.43 | 0.908 | 41.79 |
| Runge | $7.79\times10^8$ | 0.325 | $6.40\times10^8$ |
| Gaussian envelope | 11.57 | 0.988 | 34.59 |

The earliest Runge branch reaches mean gamma approximately $2.8\times10^{14}$. These values remain finite, so the runs finish rather than terminate with NaNs. Large gamma here accompanies a damaged geometry. The continued-solve reference itself also loses approximation accuracy on some targets, especially on its first step; this is not a successful baseline being hidden by the freeze comparison.

At a branch's first post-freeze geometry update, its slopes and biases match the continued reference bitwise. Their readouts then differ because only the reference updates/solves them. Sine frozen at step 10 goes from error approximately 0.00309 at the marker to 59.3 one step later, while the reference refits its coefficients. This localizes the initial prediction discrepancy to the missing readout adjustment on the same changed geometry. It does not establish a general instability law.

**Variant 2: an accurate QI readout does not restore QI scales under this schedule.** Before resetting geometry, the transferred coefficients achieve errors between $1.2\times10^{-14}$ and $4.4\times10^{-14}$ on the independent grid, with coefficient norms between 0.142 and 6.44. After the reset and 500 frozen-readout geometry updates:

| Target | Actual error at reset | Actual final error | Final mean gamma | Initial / final refit error |
|---|---:|---:|---:|---:|
| Sine | 1.1290 | 1.1177 | 0.999323 | $2.75\times10^{-9}$ / $2.40\times10^{-9}$ |
| Mixed sine | 1.2866 | 0.9635 | 0.998998 | 0.205962 / 0.205910 |
| Runge | 0.6654 | 0.6640 | 1.000032 | 0.0123889 / 0.0123884 |
| Gaussian envelope | 0.9920 | 0.9864 | 1.000029 | 0.223002 / 0.223267 |

Because the readout is fixed, the actual improvement comes from geometry updates. In particular, mixed sine does improve its fit appreciably. That improvement does not recover the accurate QI approximation regime or move the mean scale toward 16. The sine refit changes modestly; “no geometry improvement whatsoever” would overstate this negative result.

**Variant 3: freezing leaves an unfinished readout on an already accurate geometry.** Actual-error curves become nearly flat after freezing, while continued joint GD keeps improving. Mean scale changes by at most about $4.6\times10^{-5}$ among these branches. Offline readout refits remain near numerical precision. The poor actual errors therefore coexist with an excellent available approximation.

**Variant 4: the perturbed geometry is not repaired.** Initial noisy-geometry refit errors are approximately $3.68\times10^{-11}$, $6.26\times10^{-10}$, $1.71\times10^{-11}$, and $7.24\times10^{-10}$ for sine, mixed sine, Runge, and Gaussian. These errors are still small, but substantially above clean-QI accuracy. They remain near their initial levels after freezing, with small movements in either direction. The companion figure includes the clean-QI reference to make the missing recovery visible. Mean scale changes by at most about $4.8\times10^{-5}$.

All 48 scheduled frozen branches have worse final actual error than their matching continued references at the same absolute endpoint. This is a result at one rate, width, seed, perturbation level, and continuation length; it is not a universal verdict on readout freezing.

### Figures

The twelve primary scheduled-freeze figures share a 2×4 layout: columns freeze at 2, 10, 50, and 150; top is actual relative error, bottom is mean absolute slope; horizontal axes count total geometry updates. Colored solid lines are frozen branches, gray dashed lines are continued references, and vertical dotted lines mark freezing. Axes are shared within each row. Mean gamma uses logarithmic axes only when the scale range is enormous; otherwise its absolute linear range is enlarged to reveal small motion.

- **Variant 1, sine:** extreme scale and error growth after freezing; the reference continues coefficient solves.
- **Variant 1, mixed sine:** the first solved-geometry step already worsens approximation; frozen branches later worsen actual fitting.
- **Variant 1, Runge:** the largest scale explosion; gamma movement is destructive.
- **Variant 1, Gaussian envelope:** large initial scale jump and later deterioration under frozen coefficients.
- **Variant 1, refitted geometry:** four target rows and four freeze-time columns; diagnostic solves show that the damaged actual fits are accompanied by worse geometry.
- **Variant 2, all targets:** columns are functions, not freeze times. The readout is fixed for all 500 updates. Top compares actual error with evaluation-only refits; bottom shows mean gamma remaining near 1.
- **Variant 3, sine:** stopping readout GD arrests fitting while the mean scale changes only slightly.
- **Variant 3, mixed sine:** substantial actual error remains after every freeze time.
- **Variant 3, Runge:** geometry updates do little to close the readout gap.
- **Variant 3, Gaussian envelope:** the same finite-domain objective as the other functions; no whole-line constraint is introduced.
- **Variant 3, refitted geometry:** logarithmic limits from $10^{-16}$ to $10^{-12}$ put floor-level fluctuations in proportion.
- **Variant 4, sine:** the noisy initialization has almost the same ordinary fitting behavior as clean QI.
- **Variant 4, mixed sine:** readout freezing produces little subsequent error reduction.
- **Variant 4, Runge:** slightly different scale movement does not restore the clean approximation floor.
- **Variant 4, Gaussian envelope:** actual fitting remains far above its numerical refit.
- **Variant 4, refitted geometry:** limits from $10^{-16}$ to $10^{-8}$ show the separation between the noisy trajectories and the clean-QI reference; no recovery across that gap occurs.

## Additional details

**Numerical sensitivity of variant 1.** The initial Xavier coefficient solves have norms around $10^{10}$–$10^{11}$. Before some freezes they remain very large. The residual is then obtained by subtracting large contributions, and its rounding error is multiplied by large coefficients in the geometry gradient. A full NumPy recomputation versus the PyTorch forward pass gives a worst first-update discrepancy of about 0.213 in the Runge branches. This is a material limitation, not ordinary last-digit agreement.

To distinguish arithmetic sensitivity from an incorrect update formula, a separate analytic derivative uses exactly the forward values from the training implementation without differentiating them through autodiff. Its worst first-update discrepancy is $2.83\times10^{-12}$, within a conservative componentwise floating-point accumulation bound. Exact first-geometry-update agreement between each branch and its reference supplies another timing check. These validate the implemented intervention, but do not make the ill-conditioned trajectories clean evidence about exact-arithmetic VarPro or a Fourier barrier. No rate adjustment, clipping, or damping was introduced.

**Verification and cost.** Five focused tests pass, covering step-zero/per-step solves with no readout GD; exact branch timing; fixed coefficients including output bias; the zero-readout first geometry update; transfer/noise invariants; and evaluation nonmutation. Saved-data checks cover all 48 branch prefixes and first post-freeze geometry updates, all frozen coefficient arrays, and all zero-readout starts. All 64 trajectories finish with finite arithmetic, including the destructive large-value runs. Figures were checked for shared axes, readable labels, unclipped ranges, and numerical-floor presentation. Successful training runs took approximately 30 seconds total on this machine, and offline evaluations/refits approximately 158 seconds; these are different costs. Variant 1's repeated dense SVDs are an explicit diagnostic intervention, not a scalable optimizer claim.

## Conclusions

Under the requested constant-rate schedules, readout freezing does not supply a reliable route to recovering useful scale geometry. Readout provenance changes the outcome substantially: current-geometry solves can create highly sensitive large-coefficient states, while transferred QI coefficients permit some fitting progress without recovering the QI regime.

## Open questions

- **Proposed only: fix centers during training.** This would isolate gamma learning from the center movement present in every current raw-geometry run.
- **Proposed only: shuffle the transferred QI neuron coefficients.** Preserve their values and norm, but change their assignments to centers. This would help distinguish target-specific assignment information from coefficient magnitude. Output bias would remain unchanged.
- Whether any freezing schedule succeeds at another rate or longer horizon remains untested here. No such additional variations were run.
