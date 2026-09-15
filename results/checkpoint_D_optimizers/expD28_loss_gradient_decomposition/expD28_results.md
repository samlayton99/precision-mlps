# expD28 — Loss and geometry-gradient decomposition — draft-pending-Sam

## TL;DR

- Along ordinary GD, the numerical approximation-floor gradient is usually much smaller than the readout-gap gradient. Scaled Xavier makes that separation especially clear.
- Xavier mixed sine develops opposing gradients; Xavier Runge develops aligned gradients. These runs do not show a universal cancellation of two equally large signals.
- Some least-squares gradient directions cannot be resolved in fp64. Their cosines are omitted, including every QI case near the numerical approximation floor. Omission does not mean orthogonality or a zero gradient.
- This differentiates the existing **SVD-truncated numerical reference**, denoted (F_\tau), rather than claiming access to unrestricted exact-arithmetic VarPro derivatives.

## Question / hypothesis

Does ordinary GD produce small geometry improvement because the approximation and readout-gap gradients cancel, or because their magnitudes differ? Compare the loss decomposition, geometry-gradient norms, and angle between the component gradients along identical training protocols.

## Experiment design

For (m=177) tanh neurons, write

\[
 f_{\theta,v}(x)=\sum_{k=1}^{m}v_k\tanh(a_kx+b_k)+v_{m+1},
 \qquad \theta=(a_1,\ldots,a_m,b_1,\ldots,b_m).
\]

With (n=1024) training points, (A_{ik}=\tanh(a_kx_i+b_k)/\sqrt n), an appended constant column (1/\sqrt n), and (\bar y=y/\sqrt n), the training loss is

\[
 L(\theta,v)=\tfrac12\|Av-\bar y\|_2^2
             =\tfrac12\operatorname{mean}(f-y)^2.
\]

All runs use the same midpoint samples in ([-1,1]), fp64, full-batch ordinary GD, rate 0.002, and 2,000 updates. Readout coefficients, slopes, and biases train together. **All least-squares calculations occur after training**, so they do not affect subsequent updates. Saved states include step zero, every step through 20, and logarithmically spaced later states: 150 distinct diagnostic snapshots. Actual training loss is also saved at every update; an independent 8,192-point grid supplies supplementary actual-error diagnostics.

The four columns use sine (\sin(2\pi x)); mixed sine (\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x)); Runge (1/(1+25x^2)); and the same mixed sine multiplied by (\exp[-(x/0.4)^2/2]). **The Gaussian row uses the same finite training interval**, not the older whole-line objective.

Each figure uses one initialization:

- **Xavier:** the existing seed-0 Xavier slopes and biases with a separate deterministic random readout; output bias zero.
- **Scaled Xavier:** multiply both slopes and biases by the same positive factor so that (h\operatorname{mean}|a_k|=0.25), with (h=2/128). Centers (-b_k/a_k), signs, and the random readout remain unchanged. Individual scales are not equal.
- **QI:** uniform centers with every slope (a_k=16=0.25/h), and zero readout including output bias. The N=128 construction has 129 interior slots and 24 halo neurons on each side, giving 177 neurons total. Xavier uses the same width, though its random centers do not define a literal halo.

For the numerical readout reference, compute (A=U\Sigma V^T) and retain singular values (\sigma_i>10^{-13}\sigma_1). If (U_r) contains the retained left singular vectors, define

\[
 F_\tau(\theta)=\tfrac12\|(I-U_rU_r^T)\bar y\|^2,
 \qquad G_\tau(\theta,v)=L(\theta,v)-F_\tau(\theta).
\]

The reported gradients are

\[
 DL=\nabla_\theta L,\qquad DF_\tau=\nabla_\theta F_\tau,
 \qquad DG_\tau=DL-DF_\tau.
\]

They include **all slopes and biases, including halo slots, and exclude readout coordinates**. They are the raw coordinates used by GD, not the note's center-preserving scale-only derivative. All three norm curves use the same Euclidean metric. The bottom row is

\[
 \cos(DF_\tau,DG_\tau)=
 \frac{DF_\tau^TDG_\tau}{\|DF_\tau\|_2\|DG_\tau\|_2}.
\]

Negative values indicate opposition. Substantial cancellation additionally requires comparable magnitudes, as follows from

\[
 \|DL\|^2=\|DF_\tau\|^2+\|DG_\tau\|^2
 +2\|DF_\tau\|\|DG_\tau\|\cos(DF_\tau,DG_\tau).
\]

**Code & data**

- [Runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD28_loss_gradient_decomposition/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD28_loss_gradient_decomposition/config.yaml), and [five independent implementation tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD28_loss_gradient_decomposition.py).
- [Data directory](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/data): twelve compressed trajectory/diagnostic files, approximately 29 MB total, with configuration, states, full gradients, numerical ranks, and reliability checks. The accompanying JSON records the saved-state directional finite-difference audit.
- [Xavier figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/xavier.png).
- [Scaled-Xavier figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/scaled_xavier.png).
- [QI figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/figures/qi_zero.png).

## Results

**Scaled Xavier separates the gradient magnitudes most clearly.** At the final step, (\|DF_\tau\|/\|DG_\tau\|) ranges from approximately (4\times10^{-8}) to (2\times10^{-4}) across the four targets. Consequently, (DL\) and (DG_\tau\) almost coincide, whatever their angle with the much smaller (DF_\tau\). Every saved cosine passes the numerical resolution screen.

**Xavier shows different directional behavior across targets.** For mixed sine, the final norms are approximately 0.00102 for (DF_\tau\) and 0.00295 for (DG_\tau\), with cosine −0.377. The actual norm is 0.00274. There is opposition, but not near-total cancellation. For Runge, the final cosine is +0.398, indicating alignment. Mixed sine has 121 resolved snapshots out of 150; Runge has all 150. Sine and Gaussian have none under the stated resolution criterion.

The mixed-sine numerical floor changes from 0.0659306 to 0.0659689 while actual loss falls from 0.497219 to 0.273515. This illustrates why decreasing (L) need not decrease (F_\tau). Locally along gradient flow,

\[
 \frac{dF_\tau}{dt}=-DF_\tau^TDL
 =-\|DF_\tau\|^2-DF_\tau^TDG_\tau.
\]

An opposing (DG_\tau) can reverse progress on (F_\tau) without nearly canceling the entire vector (DL). For this ill-conditioned case, interpret the small floor changes alongside the numerical caveats below, rather than as high-precision effect-size estimates.

**QI already has a numerical approximation floor near fp64 precision.** Its squared losses (F_\tau) remain around (10^{-30})–(10^{-27}). The computed (DF_\tau) directions change substantially with rounding and are unresolved. The actual gradient is zero at step zero because every readout coefficient is zero. Readout learning then generates a nonzero geometry gradient, and (DG_\tau\) essentially coincides with (DL). This does not establish that exact-arithmetic (DF) is identically zero.

### Figures

- **Xavier:** a 3×4 grid with function columns. Top: actual loss (L) in blue, numerical floor (F_\tau) in teal, and gap (G_\tau) in dashed orange. Middle: corresponding geometry-gradient norms. Bottom: purple cosine curves, showing mixed-sine opposition and Runge alignment. Sine and Gaussian cosines are explicitly omitted.
- **Scaled Xavier:** identical layout, with vertical limits fitted to this initialization. Look for the large vertical separation between (DF_\tau) and the nearly overlapping (DL,DG_\tau), even where the cosine is negative. The smaller initial numerical floor also distinguishes this geometry from ordinary Xavier.
- **QI:** identical layout, with vertical limits fitted to the actual training curves. Look for the near-precision numerical floor, the explicitly marked zero initial geometry gradient, and the subsequent overlap of (DL,DG_\tau). Empty cosine panels report numerical uncertainty, not a cosine of zero.

The loss row has its own legend for L, F_tau, and G_tau. A separate legend immediately above the middle row explicitly labels the three gradient norms and the gray numerical-resolution threshold. Every horizontal axis is GD updates, linearly spaced. Loss and norm rows are logarithmic; cosine axes run from −1 to +1. Vertical limits are fitted separately for each initialization and shared across its four function columns. QI numerical-floor loss and gradient values are annotated with their estimated maxima below the displayed range; including them in the axis limits would compress the actual training curves. Losses retain their squared-residual definition, while the middle row retains unsquared gradient norms. Dotted teal segments are unresolved gradient estimates. The gray dotted norm is the numerical resolution threshold. Gray ticks below the cosine axis mark omitted snapshots; they are not values of −1.

## Additional details

### Differentiating the numerical least-squares reference

Using cutoff coefficients in the ordinary envelope formula is generally **not** the derivative of (F_\tau): the allowed singular subspace itself depends on geometry. The implementation differentiates that subspace explicitly, locally away from singular-value crossings or changes in retained rank.

For completeness, let (I,D) denote retained and discarded thin-SVD indices, (\alpha=U^T\bar y), (v_\tau=V_I\Sigma_I^{-1}\alpha_I), and (r_\tau=U_I\alpha_I-\bar y). Then

\[
 \nabla_A F_\tau=r_\tau v_\tau^T+U_D B V_I^T+U_I C V_D^T,
\]

where, for retained (i), discarded (l), and (\rho_{li}=\sigma_l/\sigma_i),

\[
 B_{li}=-\frac{\alpha_l\alpha_i}{\sigma_i}
 \frac{\rho_{li}^2}{1-\rho_{li}^2},\qquad
 C_{il}=-\frac{\alpha_i\alpha_l}{\sigma_i}
 \frac{\rho_{li}}{1-\rho_{li}^2}.
\]

Pull this matrix derivative through (A(\theta)) to obtain (DF_\tau). With no discarded nonzero directions, the correction vanishes and the usual envelope gradient is recovered. The correction is material for some Xavier cases.

Truncation also means (F_\tau) is not necessarily a lower bound on loss for unrestricted current coefficients: in principle (G_\tau) could be negative if they exploit discarded directions. Here it stays positive in every saved state. The retained ranks are 10, 99, and 140 for Xavier, scaled Xavier, and QI, respectively, and remain constant at the saved states. This numerical rank is not a claim about exact mathematical rank.

### Numerical reliability and verification

Each gradient is recomputed using two SVD algorithms and an alternative tanh implementation. The maximum change in (DF_\tau) supplies an empirical sensitivity measure. A direction is reported only when the retained ranks agree and its norm exceeds that variation by a factor of ten; the cosine also requires a resolved (DG_\tau). This is a screening rule, **not a rigorous error bound**. Shading shows the spread of the three reported cosines.

Five focused tests pass: truncated-subspace derivatives against independent autograd and finite differences; recovery of the full-rank envelope formula; direct directional checks of (L,F_\tau,G_\tau) through an actual tanh model; matched initialization and preserved centers; and uninterrupted training with exactly zero first geometry update from zero readout. Along all full runs, the ordinary analytic geometry gradient agrees with training autograd to maximum absolute discrepancy (4.1\times10^{-16}).

A supplementary saved-state audit checks initial and final mixed-sine/Runge Xavier and mixed-sine/Gaussian scaled-Xavier states. Directional finite differences broadly corroborate the resolved gradients, with appreciable perturbation-size sensitivity in Xavier. At displacement norm 0.003, finite-difference/analytic ratios range from 0.958 to 1.110 for those Xavier states and from 0.997 to 1.003 for scaled Xavier. Larger scaled-Xavier perturbations cross the cutoff boundary, demonstrating why local derivatives cannot be extrapolated through rank changes. All tried values are saved, including those crossings.

## Conclusions

These runs support a large imbalance between numerical approximation-floor and readout-gap gradient magnitudes, especially after scaling Xavier. They also show target-dependent alignment and opposition, rather than a universal cancellation explanation.

## Open questions

- Would an independently validated, higher-precision approximation reference resolve the omitted Xavier directions and preserve the retained-space conclusions?
- How much of the observed imbalance persists for center-preserving scale coordinates alone, and across seeds? The present figures measure the complete geometry block in ordinary GD coordinates.
