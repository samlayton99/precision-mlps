# What we know about the gamma barrier

Internal research synthesis, 14 September 2026. Scope: the handoff note and expD24–D30, including the completed loss-decomposition and weighting work in the separate Codex session **Junmi Optimization** (thread `01a08d13-ef01-7d51-b995-8ef9ed1f5632`). That session was read directly through the Codex app, rather than inferred from this conversation. Its new separate-Adam trials were still in progress when reviewed and are not counted as results here. No new training was run for this review. Independent audits checked the empirical evidence, the projection interpretation, and the drift argument below.

**We have not proved that useful gamma requires unstable training. We have evidence against that universal interpretation: controlled changes of geometry learning rate or parameterization already recover better approximation geometry without observed numerical failure. What remains plausible, and partly provable, is a finite-budget obstruction for ordinary GD in its original parameter coordinates.**

## The problem we are trying to explain

The construction gives a route to accurate functions. Its spacing is h, its inverse transition width is gamma, and lambda=gamma h measures how narrow transitions are relative to their spacing. The approximation work identifies a useful balance: very broad, overlapping features can be numerically difficult to combine; very narrow features create unwanted output between grid centers. The Fourier alias analysis makes the latter effect explicit. Maintaining a fixed positive lambda with h proportional to inverse width forces the construction's slopes to scale proportionally to width.

This establishes a construction and its scaling. It does not establish that every accurate tanh network must have the same slopes or centers. A universal failure-of-approximation theorem needs a necessity argument for the target/model class, not only the existence of an accurate construction outside the initialized regime. Also, the exact optimum is not universally lambda=0.25: the admissible balance depends on precision, resolution, target, and construction. The current tanh default is a useful regime, not an optimizer-independent constant forced on all solutions.

The continuous-domain approximation problem must also be separated from interpolation of finite samples. D30's gamma-128, center-sampled network admits least-squares interpolants at numerical precision while leaving much larger between-center errors. The executed GD runs did not reach that training floor. For this sampled training objective, gamma discovery is unnecessary: the initial geometry already has enough row rank.

## Two different reasons for a large training loss

For the exact finite-sample least-squares problem, let

\[
F(\theta)=\min_v L(\theta,v),\qquad G(\theta,v)=L(\theta,v)-F(\theta).
\]

Theta denotes geometry and v denotes readout. F is what an exact readout refit cannot remove; G is unfinished readout fitting. Both are nonnegative. This is the clean conceptual distinction behind the session's controls.

If F is already below the desired tolerance but G is large, the remaining task is to solve the readout. If F is too large, the geometry needs to improve too. A theory that attributes both situations to gamma being unable to grow will misdiagnose the first.

The measured refit uses a singular-value cutoff, so its F_tau is a numerical reference, not unrestricted exact-arithmetic F. This qualification is essential in small-gamma regimes with enormous fitted coefficients. In general L-F_tau is not guaranteed nonnegative, although it was positive in the saved D28 states. Independent-grid refits and coefficient norms help establish whether an apparent improvement is useful beyond a numerical rank change.

The separate **Junmi Optimization** discussion adds a useful directional statement. In a smooth region, write the geometry gradients as g_F=grad_theta F and g_G=grad_theta G. Joint gradient flow with geometry rate eta_theta gives

\[
\frac{dF}{dt}=-\eta_\theta\left(\|g_F\|^2+\langle g_F,g_G\rangle\right).
\]

Readout changes have no direct term here because F depends only on geometry. If the two gradients align, the readout-gap force helps approximation improve; if they oppose sufficiently, it prevents improvement or makes F increase. If g_F is tiny, its own descent contribution is tiny even without cancellation. Thus a training-loss decrease need not improve approximation, and a small total geometry gradient is not the only way approximation progress can stall. D28 provides examples of different signs, while D29 tests changing their relative influence. Neither establishes universal cancellation.

Exact VarPro retains the complete first-order derivative of F. Solving the readout does not omit another first-order derivative of that same objective. It changes the trajectory, and it may remove motion that would have been helpful later, but that longer-term possibility requires separate evidence. In addition, unrestricted F does not penalize enormous readout coefficients: it can reward a mathematically good but numerically difficult representation. This is a separate issue from whether its gradient has been computed correctly.

## What is quantitatively established

**Readout conditioning is a real optimization obstruction even with noiseless targets.** At fixed geometry, normalized A and residual r obey

\[
r_{k+1}=(I-\eta AA^T)r_k.
\]

A residual component in a singular direction contracts by 1-eta sigma_j squared. Stability for every residual direction requires 0<eta<2/sigma_max squared; weak directions can then decay extremely slowly. D26 and D30 verified target-weighted predictions against executed GD. D30's initial matrix has condition number about 10,621 and admits a modest-coefficient interpolant, but the larger tested constant rate predicts 225–655 million steps to relative training error 1e-12. This is a statement about that fixed readout problem and rate. It predicts eventual exact-arithmetic convergence, not a positive limiting loss, and is not an impossibility result for all step schedules.

The other session's whitening discussion supplies a clean way around this particular obstruction. Represent the retained feature span by an orthonormal matrix Q and use coefficients w, so the prediction is Qw. Then grad_w L=w-Q^T y and one GD step with eta=1 reaches the retained least-squares solution in exact arithmetic. Every retained direction has curvature one. This requires constructing the orthonormal basis, deciding numerical rank, and evaluating the resulting function reliably; it does not expand the feature span or discover good geometry. A schedule is unnecessary for the exactly whitened fixed quadratic. Whitening explains how a better metric can eliminate a genuine GD conditioning barrier without instability.

**Ordinary geometry motion is small, and the note's simple drift bound can certify that fact quantitatively.** Section 5 gives, for raw simultaneous GD and samples with |x_i|<=X,

\[
|a_{j,K}-a_{j,0}|\le X\sum_{k<K}\eta_k|c_{j,k}|\sqrt{2L_k}.
\]

Using the complete saved D26 histories, X=1, and the first 650 updates, the largest bound on any neuron's final gamma is 0.335 for sine, 0.359 for mixed sine, 0.254 for Runge, and 0.289 for Gaussian envelope. The target reference scale was 16. Thus the recorded residual/readout budget itself excludes reaching that scale over the recorded horizon, without invoking Fourier attenuation or a favorable sign. This is a retrospective certificate using measured coefficients and losses. It does not prove how those quantities evolve on all future runs.

**Fourier attenuation is real, but its dominance in stalled joint training has not been established.** The center-preserving tanh scale tangent has the transform derived in section 2. Controlled fixed-residual probes show its response weakening at small gamma, strengthening as gamma reaches the residual's frequency range, then weakening again at sufficiently large gamma. The high-gamma amplitude loss is included in the note. But low-frequency residual remains important in the small-gamma trajectories, finite sampling and boundaries matter, and raw slope/bias GD does not follow the same scale derivative as a center-preserving update. We have not verified the persistent spectral gap needed for the exponential escape-time theorem.

**Useful approximation gradients can be weak relative to readout-gap gradients without being absent.** D28 supports this magnitude imbalance in resolved numerical diagnostics. It also shows different alignment by target: there is no universal cancellation pattern. D29's completed weighting sweep is important counterevidence to an absence claim. Amplifying the numerical approximation objective improves independent-grid Xavier refits on all four targets at all three audited cutoffs. At weight 100,000, Runge's refit error falls from about 0.147 to 0.0128 and mixed sine from 0.448 to 0.213. Actual loss worsens for those two targets, showing that the ordinary objective and improved numerical approximation need not favor the same short-term motion. The intervention is costly and numerically sensitive; it is evidence that useful information exists, not a deployable method or proof of a particular depletion mechanism.

**Freezing did not recover the intended regime in the tested schedules.** This weakens the simple practical proposal that stopping readout fitting restores useful scale growth. It does not identify which directional or timescale premise failed. Retaining residual is insufficient if its geometry gradient is weak, points elsewhere, or cannot improve the desired approximation. Solve-then-freeze runs add the separate issue of enormous coefficients and rounding sensitivity.

## A direct polynomial drift argument

There is a general energy argument that can strengthen the finite-budget part of the story without a Fourier or bounded-readout assumption. It is an elementary deduction checked independently for this synthesis, not a new empirical result or novelty claim.

Let p be the complete vector of raw trainable parameters, including the slopes a, and let L be a nonnegative differentiable loss. Under unit-rate gradient flow,

\[
\dot p=-\nabla L,\qquad
\int_0^T\|\dot p(t)\|^2dt=L(0)-L(T).
\]

Cauchy–Schwarz therefore gives

\[
\|a(T)-a(0)\|^2\le T[L(0)-L(T)]\le TL(0).
\]

For plain simultaneous GD, assume an explicit sufficient-descent condition

\[
L_k-L_{k+1}\ge\alpha\eta_k\|\nabla L_k\|^2,\qquad\alpha>0.
\]

Weighted Cauchy–Schwarz applied to the slope updates yields

\[
\boxed{\|a_K-a_0\|^2\le
\frac{\sum_{k<K}\eta_k}{\alpha}(L_0-L_K).}
\]

If reaching a specified geometry set requires Euclidean slope displacement D, then constant-rate GD needs

\[
K\ge\frac{\alpha D^2}{\eta L_0}.
\]

If M slopes must rise from magnitudes at most g0 to at least Gamma, D squared is at least M(Gamma-g0) squared, regardless of sign changes. With width-independent initial loss and Gamma proportional to width W, this gives effective-time lower bounds of order W squared for one slope and W cubed for a positive fraction of the slopes. For a target mean absolute slope, the general bound is

\[
D^2\ge W(\bar\gamma_{\rm target}-\bar\gamma_0)^2.
\]

Restarting the argument at any late training state replaces L0 by the remaining actual training loss. Small remaining loss means that accumulating a large Euclidean displacement under descending raw GD requires a long time. An offline refit floor cannot be substituted for that actual loss.

This is a route to a clean finite-budget theorem, but its scope must remain explicit:

- The discrete result assumes sufficient descent along the relevant trajectory. That condition needs a step-size/curvature proof or an explicitly stated hypothesis.
- Violating a particular sufficient-descent inequality is not the same as instability. Even a scalar quadratic with eta=1.9/beta is stable and strictly decreasing while satisfying alpha=0.05 rather than alpha=0.5.
- Changing block learning rates, optimizer coordinates, or the metric changes the distance/time bound. A faster method can remain stable.
- A polynomial lower bound does not imply permanent nonreachability.
- A distance to a QI-like slope configuration is not automatically a distance to every accurate representation. That missing necessity statement is what prevents a geometry-travel theorem from becoming a universal accuracy-failure theorem.

## The escape routes already demonstrated

D25 directly contradicts the blanket claim that useful gamma changes necessarily require unstable training. Keeping centers fixed and readout rate 0.002, increasing only the independent scale rate to 20 improves mixed-sine refit error from 0.206 to 0.000342 and Runge from 0.0124 to 8.43e-6. None of its 100 distinct trajectories terminated in failure. This is evidence of useful controlled motion, not a global stability theorem for rate 20.

A stronger restricted escape fixes uniform centers and trains one shared scale with Adam, while ordinary GD trains readout. Starting from gamma=1, the 10,000-step runs reach gamma about 13.88 for mixed sine and 15.50 for Gaussian envelope, with evaluation-only refit errors 7.8e-13 and 2.0e-14 and modest fitted coefficients. No least-squares solves enter those training updates. Thus the model can learn scales giving excellent approximation in this restricted setting.

The limitation is substantial: the centers were already right, scales were tied, and the trained readout still had actual relative errors about 0.32 and 0.28. The same rule starting from good gamma can overshoot and damage approximation while reducing current loss. This is a demonstrated escape from scale immobility, not a complete optimizer or a proof of recovery from arbitrary initialization.

## What remains in the note and what should be claimed

Sections 2–5 contain correct mechanisms and conditional bounds, but their roles differ. Section 2 characterizes instantaneous frequency response and a conditional persistent-gap escape time. Section 3 supplies the exact readout projection split and a hypothesis about loss of useful force; the latter is not established by the decomposition. Section 4 relates local profiled convergence to residual-relevant projected sensitivity. Its toy example proves that compensation can slow a low-frequency problem; it does not identify the dominant cause in our networks. Section 5 supplies applicable drift inequalities, including the useful sampled certificate above. Section 6 explains how to separate raw tangent weakness from extra readout compensation; that separation remains incomplete in our actual training cases.

The L=F+G derivative split used in D28/29 is not identical to section 3's current-readout inside/outside-span split. Differentiating F uses solved coefficients, which also changes the geometry Jacobian. Therefore a large readout-gap gradient is not, by itself, evidence that readout removed a previously useful outside-span signal.

The next proof should specify the target accuracy and admissible representation class, identify a region where that accuracy cannot be achieved with acceptable coefficients, and bound how quickly standard GD can leave it. A finite-budget exclusion theorem is a realistic target. If the intended theorem concerns all unconstrained tanh representations from generic initialization, the missing approximation-necessity argument is much larger.

For the mechanism claim, identify a direction that improves refitted approximation, measure its raw and projected residual-normalized sensitivity, and show that the relevant suppression persists while the approximation remains inadequate. The measured residual/readout budgets can then quantify whether the resulting movement is enough. This work should isolate one implication at a time, rather than add another optimizer sweep.

The defensible position is that ordinary GD can spend its budget fitting the current features while making little progress toward more useful geometry, and it can also fit an already adequate geometry very slowly because of readout conditioning. We have conditional lower bounds and measured examples of both limitations, together with restricted ways around scale immobility. We do not have a theorem that accurate scales require unstable updates, or that standard GD can never find an accurate alternative representation.

## Evidence and independent reviews

- [Current lambda section](../lambda_theorem_compatibility/choosing_optimal_lambda/choosing_optimal_lambda.tex) and [construction compatibility audit](../lambda_theorem_compatibility/qi_proof_audit.md).
- [D25 scale mobility and restricted escape](../../results/checkpoint_D_optimizers/expD25_scale_barrier/expD25_results.md).
- [D26 readout freezing and fixed-matrix convergence](../../results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/expD26_results.md).
- [D27 freezing variations](../../results/checkpoint_D_optimizers/expD27_readout_information/expD27_results.md).
- [D28 loss-gradient decomposition](../../results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/expD28_results.md).
- [D29 approximation-gradient weighting and sweep](../../results/checkpoint_D_optimizers/expD29_weighted_profile/expD29_results.md).
- [D30 interpolatable center-sampled GD](../../results/checkpoint_D_optimizers/expD30_center_sampled_gd/expD30_results.md).
- Separate Codex session **Junmi Optimization**, thread `01a08d13-ef01-7d51-b995-8ef9ed1f5632`: completed discussions of exact F+G dynamics, readout whitening, D28, and the six-weight D29 sweep. Its subsequent Adam experiment remained pending at this review's cutoff.
- Independent audits: [empirical evidence](empirical_audit.md), [projection claims](projection_audit.md), [energy/drift bound](energy_bound_audit.md).
