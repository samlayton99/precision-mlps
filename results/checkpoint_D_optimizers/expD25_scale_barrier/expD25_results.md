# expD25 — Separating scale mobility, geometry quality, and readout fitting | Status: data-obvious results; broader interpretation pending discussion

## TL;DR

- Increasing only the geometry learning rate produces useful scale motion. With centers fixed, the independently refitted mixed-sine error falls from $0.206$ to $0.000342$ in 2,000 steps. The earlier near-constant refit curves do not establish that GD cannot improve geometry.
- The strongest tested escape starts at $\gamma=1$, holds uniform centers fixed, and trains one shared scale with Adam while keeping ordinary GD on the readout. After 10,000 steps, separate readout solves achieve approximately $10^{-14}$–$10^{-12}$ relative error on all four targets, with modest coefficient norms. No solved readout enters training.
- Moving gamma is insufficient by itself. The same adaptive control starting at $\gamma=16$ improves current fitting while worsening the mixed-sine and Gaussian refit errors to about $5\times10^{-8}$. Current loss does not protect the best approximation regime.
- These are structured, finite-domain, one-dimensional controls. They demonstrate an escape from the observed small-scale stall, not a general optimizer or elimination of the Fourier suppression mechanism.

## Question / hypothesis

Did gamma remain nearly fixed because useful scale directions were unavailable, or because ordinary GD took ineffective steps in those directions? When we increase mobility, does the attainable approximation improve, or does movement merely help an unfinished readout fit?

## Experiment design

### Model, targets, and matched samples

Every target uses the same standard tanh model, including an unconstrained output bias:

$$
f_{a,b,v}(x)=\sum_{j=1}^{177}c_j\tanh(a_jx+b_j)+d,
\qquad v=(c_1,\ldots,c_{177},d),
\qquad L=\frac1{2n}\sum_{i=1}^n(f(x_i)-y_i)^2.
$$

The targets on $[-1,1]$ are $\sin(2\pi x)$, the mixture $m(x)=\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$, Runge's function $(1+25x^2)^{-1}$, and $\exp[-(x/0.4)^2/2]m(x)$. The Gaussian row is a finite-domain problem here, matching the other rows. No whole-line loss or cancelling-tail constraint is used.

Training uses 1,024 uniform midpoints, fp64, and full-batch GD with readout rate $0.002$. Evaluation uses 8,192 different midpoints; the numerical audit increases this to 32,768. Width parameter $N=128$ means 129 interior grid centers plus 24 halo neurons on each side, for 177 neurons and 178 readout coefficients. Uniform centers have spacing $h=2/128$. Uniform-gamma starts use zero readouts; the Xavier arm preserves the earlier seeded random parameters and readout. There is one seed, with no per-target tuning. Uniform starts are deterministic.

Three geometry parameterizations separate possible causes. **Raw geometry** trains $a,b$ independently, with $\gamma_j=|a_j|$ and $z_j=-b_j/a_j$. **Independent scales with fixed centers** use $\tanh(a_j(x-z_j^0))$ and train only the slopes. **Shared scale with fixed centers** additionally constrains every slope to remain equal. Slopes remained positive in all fixed-center runs; positivity was not imposed by clipping or a logarithmic parameterization.

For shared-scale SGD, each slope receives the mean of the individual centered-scale gradients. At geometry rate $\eta_g$, this is scalar-$\gamma$ SGD with rate $\eta_g/177$, so tying parameters introduces no hidden 177-fold increase. Shared-scale Adam receives the same mean gradient; its moment normalization means that the SGD effective-rate statement does not apply to Adam. Adam uses rate $0.002$, betas $(0.9,0.999)$, and epsilon $10^{-8}$; readout updates remain ordinary GD at $0.002$.

### What the two error measurements mean

Let $A(\theta)$ be the training feature matrix with a constant column and let $r=A(\theta)v-y$. Each saved geometry receives an independent numerical least-squares readout solve, including at step zero. **All solves happen after training is complete.** They cannot change the trajectory. The displayed errors are

$$
E_{\rm current}=\frac{\|A_{\rm eval}(\theta)v-y_{\rm eval}\|_2}{\|y_{\rm eval}\|_2},
\qquad
E_{\rm refit}=\frac{\|A_{\rm eval}(\theta)v_*^{\rm num}-y_{\rm eval}\|_2}{\|y_{\rm eval}\|_2}.
$$

The numerical solve retains singular values above $10^{-13}$ times the largest singular value. Thus $E_{\rm refit}$ measures the accuracy attained by a specified stable numerical solve, not the infimum over unlimited exact-arithmetic coefficients. Retained rank and solved coefficient norm are recorded explicitly.

On the training grid, exact least squares gives the orthogonal loss split

$$
L(\theta,v)=L_*(\theta)+\frac1{2n}\|A(\theta)(v-v_*)\|_2^2.
$$

For the numerical diagnostic, the fitted residual is computed through the retained left-singular-vector projector. The measured discrepancy in this identity is at most $1.43\times10^{-14}$ in absolute loss across the saved states. The continuation figure shows square-root-normalized approximation and readout-gap components; their squares add to the squared training relative error up to that discrepancy. These components are not interchangeable with the independent-grid refit error.

### Sequence of controlled interventions

| Stage | Single change or measurement | Completed trajectories |
|---|---|---:|
| Landscape | Multiply slopes and biases together in a saved gamma-1 state; compare fixed and solved readouts | No training |
| Geometry rate | Raw geometry rates $0.002,0.2,20$; four targets and Xavier/$\gamma=1,4,16$; 2,000 steps | 48 |
| Fixed centers | Independent scale rates $0.002,20$; four targets and $\gamma=1,4$; 2,000 steps | 16 |
| Common scale | Independent versus shared scales at rate $20$; four targets and $\gamma=1,4,16$; 2,000 steps | 16 new, 8 reused |
| Longer trajectory | Shared-scale rates $0.002,20$ from $\gamma=1$; four targets; 10,000 steps | 8 |
| Adaptive control | Shared-scale SGD versus Adam at rate $0.002$; four targets and $\gamma=1,16$; 10,000 steps | 12 new, 4 reused |

There are 100 distinct completed training trajectories. The preliminary mixed-sine pilot belongs to the rate matrix, not an additional statistical replicate. Failures were to be retained without retuning; none occurred. Snapshots include every step from zero through ten, then logarithmically spaced states, step 2,000, and the final state. The longer fast-SGD runs reproduce their earlier 2,000-step parameters exactly.

**Code & data**

- [Experiment scripts](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD25_scale_barrier), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD25_scale_barrier/config.yaml), [implementation checks](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD25_scale_barrier.py), and [staged investigation plan with requirements check](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/expD25_plan.md).
- [Compressed data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/data): trajectories, parameters, configuration, gradients, losses, refit audits, and curvature diagnostics; approximately 41 MB.
- Figures: [F1 Landscape](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/landscape.png); [F2 Pilot errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/pilot/errors.png); [F3 Pilot motion](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/pilot/motion.png).
- Figures: [F4 Geometry-rate errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/rate_errors.png); [F5 Geometry-rate motion](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/rate_motion.png); [F6 Fixed-center errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_errors.png); [F7 Fixed-center motion](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_only_motion.png).
- Figures: [F8 Independent/shared errors](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/common_scale_errors.png); [F9 Independent/shared scales](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/common_scale_scales.png); [F10 Derivatives and curvature](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/derivatives.png); [F11 Longer shared-scale GD](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/continuation.png).
- Figures: [F12 Numerical readout audit](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/readout_sensitivity.png); [F13 Focused scale-escape comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/scale_escape.png); [F14 Adaptive scale from gamma 1](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_1.png); [F15 Adaptive scale from gamma 16](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_16.png).

## Results

### Mobility can improve approximation

In the saved gamma-1 states, increasing the common scale is initially a descent direction for the current loss on every target. The landscape also contains much better refitted geometries. Finite differences independently verify the signed scale-direction derivatives. This does not guarantee that an arbitrary large step will help, but it rules out a local uphill direction as the explanation in those particular states.

The geometry-rate sweep produces much more motion and substantial approximation gains from poor initial geometries. Holding centers fixed preserves the positive result: on mixed sine, the refit error changes from $0.206$ to $0.000342$; on Runge, from $0.0124$ to $8.43\times10^{-6}$; on Gaussian, from $0.223$ to $0.0390$. These are 2,000-step independent-scale runs at rate $20$, with the readout rate unchanged. Their original-rate counterparts barely improve the refit error. Useful changes therefore do not require center motion.

Constraining all scales to stay equal further improves the mixed-sine and Gaussian refits, but not Runge's, at the same per-neuron rate and horizon. Shared structure is helpful in some cases, not uniformly superior. Large raw slope/bias updates can damage already accurate initial geometry; that damage cannot be attributed to gamma alone because centers move as well.

### Adaptive scale steps give the strongest escape

Only the shared-scale optimizer changes between the matched SGD and Adam arms. Starting at gamma 1, all four Adam runs learn geometries with excellent independently evaluated numerical refits. The table is at 10,000 steps and the standard SVD cutoff; the last column is the error of the readout actually trained by GD.

| Target | Learned gamma | Refit relative error | Solved coefficient norm | Actual GD-readout relative error |
|---|---:|---:|---:|---:|
| Sine | 6.245 | $6.37\times10^{-14}$ | 0.674 | 0.0225 |
| Mixed sine | 13.880 | $7.79\times10^{-13}$ | 10.708 | 0.3214 |
| Runge | 7.738 | $3.33\times10^{-13}$ | 0.816 | 0.0535 |
| Gaussian envelope | 15.496 | $2.03\times10^{-14}$ | 4.148 | 0.2817 |

No target-specific gamma or stopping threshold was supplied. The retained uniform-center structure is a substantial prior, however. The small refit errors are not the trained model's current output errors. A terminal readout solve would realize the measured approximation; continued ordinary readout GD has not done so by step 10,000.

### A good approximation regime can still be lost

Starting the same adaptive method at gamma 16 gives an important counterexample to “more movement is better.” Mixed sine reaches gamma $32.605$, with refit error $5.74\times10^{-8}$; Gaussian reaches gamma $32.018$, with refit error $4.55\times10^{-8}$. Both started with refit errors near numerical precision. Their actual trained-readout errors improve during this motion, so current loss supplies no automatic warning that approximation quality is deteriorating. Sine and Runge retain small refit errors in the same control.

### Why the baseline rate is a plausible contributor

Measured at step 2,000 in the untouched baseline runs, the readout-block curvature norm exceeds the raw geometry-block norm by approximately $1.9\times10^3$–$4.0\times10^5$. Geometry gradients are also much smaller per parameter. These block Hessians are exact at each state with the other block fixed, including residual second-derivative terms; geometry curvature may be negative. Their different scales motivate separate step sizes, but do not constitute a global stability theorem. The causal evidence is the successful intervention that changes only the geometry rate.

The current-readout geometry gradient is overwhelmingly supplied by the residual component inside the numerically retained feature span. The ratio $\|J^Tr_\perp\|/\|J^Tr\|$ is around $10^{-11}$ or below in these measurements, frequently at floating-point resolution. This uses the Jacobian with the **current** readout. It is not the gradient after solving for new coefficients, which changes that Jacobian. A tiny current-readout out-of-span component therefore does not imply that larger geometry updates cannot improve the refitted approximation; the successful trajectories demonstrate that distinction.

### Figures

- **F1, Landscape:** Four target rows and three columns. The horizontal axis multiplies all slopes and biases in the same saved gamma-1 state, preserving centers. Columns show fixed/refitted error, solved coefficient norm, and retained rank. Look for accessible lower refit errors and whether the fixed readout initially agrees with that direction.
- **F2, Pilot errors:** Mixed sine from gamma 1, with three raw geometry rates. Solid curves show current-readout error and dashed curves show refits. This is the initial narrow test, reused in F4.
- **F3, Pilot motion:** The same pilot, showing mean absolute change in gamma against step. It checks that a changed error curve accompanies actual geometry motion.
- **F4, Geometry-rate errors:** Four function rows and four initialization columns. Purple, teal, and yellow-green represent geometry rates $0.002,0.2,20$; the readout rate is fixed. Solid current-error and dashed refit curves distinguish easier training from better attained approximation.
- **F5, Geometry-rate motion:** The same 4-by-4 comparison, with step on the horizontal axis and mean absolute net gamma displacement vertically. The statistic uses 128 fixed reference-interior slots, excluding halo and the extra endpoint. Xavier slots are not filtered by their current spatial positions.
- **F6, Fixed-center errors:** Four rows and initial gamma 1/4 columns. Compare rates $0.002$ and $20$, now permitting only scale changes. Improvement in the dashed curves establishes that the earlier benefit need not come from moving centers.
- **F7, Fixed-center motion:** The same cases' mean absolute gamma displacement. Read with F6 to separate mobility from approximation benefit.
- **F8, Independent/shared errors:** Four target rows and gamma 1/4/16 columns at geometry rate $20$. Blue is independent scales and orange is a shared scale; solid is actual error and dashed is refit error. Shared structure helps several cases and preserves the gamma-16 floors at this horizon, but is not always better.
- **F9, Independent/shared scales:** The same cases, plotting mean gamma; the independent-scale band spans the 10th–90th percentiles. This exposes deformation of the common-scale structure rather than hiding it behind an average.
- **F10, Derivatives and curvature:** Four rows with columns for exact readout/geometry block curvature norms, RMS gradients, and the current-readout out-of-span gradient ratio. Solid lines in the first two columns are readout quantities and dashed lines are geometry quantities. Colors denote starting gamma; ratios below $10^{-14}$ are displayed at that floor.
- **F11, Longer shared-scale GD:** Four rows, columns for gamma, numerical approximation component, and readout-gap component, comparing rates $0.002$ and $20$ to 10,000 steps. The vertical marker identifies the previous 2,000-step horizon. The last two columns show the orthogonal training-grid loss split, not two additive relative errors.
- **F12, Numerical readout audit:** Four rows, with cutoff on every horizontal axis; columns show independent-grid refit error, solved coefficient norm, and retained rank. Five colors denote saved geometries; solid and dotted lines use different SVD algorithms. The vertical line marks the production diagnostic cutoff. Tiny last digits vary, while the large gains and modest final Adam coefficient norms persist.
- **F13, Focused scale escape:** Four rows starting at gamma 1; columns show mean gamma, refit error, and actual trained-readout error. Purple to teal changes only the independent-scale rate; teal to orange changes only whether scales remain equal. This is the clearest summary of the sequential SGD controls.
- **F14, Adaptive scale from gamma 1:** Four rows; columns show shared gamma, independent-grid refit error, and actual trained-readout error. Purple is shared-scale SGD and orange is shared-scale Adam, both at nominal rate $0.002$. This is the strongest successful geometry-learning result. The large difference between the middle and right columns is the unfinished readout fit.
- **F15, Adaptive scale from gamma 16:** Identical layout and limits to F14, starting from an already accurate scale. The mixed-sine and Gaussian middle panels deteriorate while their right panels improve. This is the stopping/objective counterexample.

## Additional details

### Relation to the note's Fourier argument

The center-preserving tangent remains $c_j(x-z_j)\operatorname{sech}^2(\gamma_j(x-z_j))$ throughout the scale-only tests. Changing the learning rule does not remove its exponential high-frequency tail. Rather, these experiments show that the finite-domain training trajectories retain a usable gradient, and that the baseline update can be much too small to exploit it. They neither falsify the note's conditional Fourier inequality nor establish an exponential training-time barrier under its hypotheses.

Large gamma can also reduce the absolute centered-scale gradient even when its frequency response broadens. For fixed residual and readout, with uniform loss density $1/2$, a center away from the boundaries, and a residual smooth on the scale $1/\gamma$, changing variables $u=\gamma(x-z)$ gives

$$
g_\gamma=\frac{c}{2\gamma^2}\int e(z+u/\gamma)\,u\operatorname{sech}^2u\,du
\simeq \frac{c\,e'(z)}{2\gamma^3}\int u^2\operatorname{sech}^2u\,du
=\frac{c\pi^2e'(z)}{12\gamma^3}.
$$

The constant-residual term cancels because the tangent is odd. On a finite interval there are omitted-tail terms; at boundary/halo centers the symmetric approximation may not apply. During training the residual and coefficients also change, so this is a conditional local asymptotic, not a monotonic law for observed gamma travel. It explains why “a wider tangent must have a larger gradient” is not valid. The prior matched-residual probe illustrates this rise-then-fall mechanism; no additional asymptotic sweep was run here.

### Verification and limits

Six implementation tests pass, covering independent SGD updates and evaluation nonmutation; the center-preserving directional derivative; independent/shared scale updates without an extra width factor; the exact geometry Hessian against automatic differentiation; cache-configuration rejection; and Adam moments/bias correction with ordinary readout GD. An independent agent audited early updates, fixed-center preservation, shared-gradient averaging, and the Adam calculation. Fixed centers are preserved to roundoff; the corresponding runs have no slope sign crossings.

The readout audit uses two SVD algorithms and five relative cutoffs from $10^{-15}$ to $10^{-11}$, with the independent evaluation grid increased fourfold. Across all those settings, the final shared-Adam refit errors range from roughly $10^{-15}$ to $3.4\times10^{-11}$, depending on target and cutoff, and coefficient norms remain modest. Thus the substantial gains are robust, while a precise floor-level number is not an invariant claim. In particular, differences of a few $10^{-15}$ can be a large relative percentage of an error near machine precision.

Training has one forward/backward pass per step. SGD uses no moment state; Adam adds its usual moments, identical across the tied scales and compressible to two scalars. The SVDs, dense evaluations, and Hessians are offline measurement costs, not hidden optimizer operations. Shared-scale training explicitly uses the known uniform-center structure and is not an invertible reparameterization of an unrestricted MLP. No multi-width, multi-seed, random-center adaptive, higher-dimensional, or production-scaling claim has been established.

## Conclusions

In the tested uniform-center tanh networks, scale learning can materially improve the accuracy available after a numerical readout solve, and an adaptive shared-scale step escapes the gamma-1 stall. The same intervention can damage already excellent approximation while improving the current loss, so useful mobility and preservation of geometry quality are separate requirements.

## Open questions

1. Can a scalable readout update close the large optimization gap while allowing beneficial scale changes and preventing the gamma-16 deterioration? This is now a concrete reason to investigate the coupling in section 3, rather than infer it from a decomposition alone.
2. How much of the successful escape depends on fixed uniform centers and a shared scale, and does it persist across widths and other targets? Independent-scale Adam, random centers, and a learned center arrangement require separate controlled tests.
3. Can local scale normalization achieve comparable useful motion with a stable stopping criterion and without the hand-selected large SGD rate? The current comparisons establish a remedy in this restricted setting, not a universal rate law.
