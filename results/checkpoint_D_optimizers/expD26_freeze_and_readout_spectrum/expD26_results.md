# expD26 — Early readout freezing and fixed-geometry GD spectra — data-obvious observations; mechanism unresolved

## TL;DR

- Freezing the readout after 2, 10, 50, or 150 Xavier-GD updates does not produce mean scale growth over the next 500 updates. All 16 branches end with lower mean scale and higher actual error than continued joint GD at the same endpoint.
- Increasing fixed uniform $\lambda$ improves much of the readout singular spectrum. Mixed sine reaches 1% relative error in a predicted 1.92 million steps at $\lambda=0.25$, versus 75,102 at $\lambda=2$, at the same learning rate.
- Faster fitting comes with an approximation tradeoff. At $\lambda=2$, the numerical least-squares floor exceeds 0.1% for sine, mixed sine, and Gaussian envelope.

## Question / hypothesis

Does preventing further readout learning let ordinary GD move Xavier geometry toward larger scales? Separately, how does the fixed readout matrix explain the improvement in GD fitting when uniform scales increase beyond the QI reference?

## Experiment design

The model is $f(x)=\sum_{j=1}^{m}c_j\tanh(a_jx+b_j)+d$, with $m=177$ neurons and an output bias. The width convention is $N=128$ intervals, giving 129 reference centers and 24 halo positions on each side. Xavier uses the existing seeded random slopes, biases, and readouts with this neuron count; its centers are not forced onto the reference grid. All targets use the same finite domain $[-1,1]$, including Gaussian envelope:

$$
\begin{aligned}
y_{\mathrm{sine}}(x)&=\sin(2\pi x),\\
y_{\mathrm{mixed}}(x)&=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x),\\
y_{\mathrm{Runge}}(x)&=(1+25x^2)^{-1},\\
y_{\mathrm{Gaussian}}(x)&=e^{-x^2/(2(0.4)^2)}y_{\mathrm{mixed}}(x).
\end{aligned}
$$

Training uses 1,024 uniform midpoints, fp64, and $L=\sum_i(f(x_i)-y(x_i))^2/(2n)$. Independent evaluation uses 8,192 midpoints and reports $\|f-y\|_2/\|y\|_2$. The ordinary GD rate is $0.002$ throughout.

**Readout freezing.** One joint trajectory per target, seeded with seed 0, supplies branches at $X\in\{2,10,50,150\}$. State $X$ contains exactly $X$ completed joint updates. Each branch freezes all $c_j$ and $d$ for 500 additional updates; $a_j,b_j$ continue ordinary GD. Its control continues joint GD to the same endpoint $X+500$. The plotted scale is $\bar\gamma=m^{-1}\sum_j|a_j|$, including every neuron. This is mean scale, not displacement. Sparse numerical readout refits are performed after training to evaluate geometry independently of its current readout; they never enter training.

**Fixed-geometry spectrum.** Uniform centers remain fixed, with $h=2/128=1/64$, $a_j=\gamma=\lambda/h$, and $b_j=-\gamma z_j$. The sweep contains 77 equally spaced values $\lambda=0.1,0.125,\ldots,2$, corresponding to $\gamma=6.4,\ldots,128$, including the QI reference $\lambda=0.25$, $\gamma=16$. Define

$$
A=\frac{1}{\sqrt n}[\Phi,\mathbf1],\qquad \Phi_{ij}=\tanh(\gamma(x_i-z_j)),\qquad \widetilde y=\frac{y}{\sqrt n}.
$$

$A$ has 1,024 rows and 178 columns, retaining every halo feature and the constant feature. Convergence predictions start from zero readout, deliberately differing from the random readout in the freezing experiment. Two rates are reported: the common $\eta=0.002$ and the per-matrix reference $\eta=1/\sigma_1^2$, where $\sigma_1$ is the largest singular value. The latter is an offline comparison, not a change to the freezing runs. Data include thresholds 10%, 1%, and 0.1%, and predicted errors at 500, 2,000, and 10,000 updates.

**Numerical reference.** SVD refits retain $\sigma_j>10^{-13}\sigma_1$, matching prior diagnostics. Predictions describe this retained numerical feature space and assume exact arithmetic on the computed spectral model. A cutoff floor does not establish a mathematical null direction, and astronomical predictions do not promise that fp64 GD could attain the result.

**Code & data**

- [Configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/config.yaml), [freezing implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/freeze.py), [spectrum implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/spectrum.py). Run each script with the project Python; --plot-only redraws saved results.
- [Data folder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/data), [all predicted step counts, CSV](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/data/readout_spectrum.csv). Four compressed trajectories total approximately 36 MB and include configuration, all parameters, errors, scale statistics, and sparse refit diagnostics. Spectrum data add approximately 0.6 MB.
- Primary freezing figures: [sine](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine.png), [mixed sine](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_sine_mixture.png), [Runge](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_runge.png), [Gaussian envelope](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_gaussian_envelope.png).
- Companion figures: [readout-refitted geometry comparison](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/freeze_refit.png), [spectrum GIF](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_spectrum.gif), [approximation floors and convergence times](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/figures/readout_convergence.png).
- [Freezing tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD26_freeze.py), [spectral prediction tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD26_readout_spectrum.py).

- Small follow-up: [primary Runge diagnostic](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/late_freeze_runge.png), [saved-data linear-drift check](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/linear_drift.png), [data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data), [Runge script](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/late_freeze.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/late_freeze.yaml), [linear-drift calculation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/freeze_linear_drift.py).

- Paired-nudge check: [figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation.png), [implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/nudge_correlation.py), [data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/nudge_correlation.npz).

- Extended paired-nudge check: [5,000-step scatter](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000.png), [time evolution](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000_evolution.png), [data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/nudge_correlation_5000.npz).

- Raw-delta companion: [figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_correlation_5000_raw.png), [fit statistics](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/nudge_correlation_5000_raw_summary.json); uses the existing 5,000-step data.

- Dynamical prediction check: [observed versus predicted phase plot](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/nudge_flow_prediction.png), [implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/nudge_flow_prediction.py), [predictions](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/nudge_flow_prediction.npz), [discrepancies and rates](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/nudge_flow_prediction.json).

- Explicit joint-GD views: [actual simultaneous updates](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/joint_gd_raw_updates.png), [loss and parameter histories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/joint_gd_parameter_evolution.png), [implementation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD26_freeze_and_readout_spectrum/joint_gd_view.py), [summary](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum/freeze_mechanism/data/joint_gd_view_summary.json).

## Results

**Readout freezing.** Initial mean scale is $0.088044$. Every branch ends below its initial mean and below the matched joint control. Freezing earlier produces greater separation. All 16 frozen-readout branches finish with worse actual error. For freezing at step 2, the comparison at step 502 is:

| Target | Frozen-readout error | Joint-GD error | Frozen mean scale | Joint mean scale |
|---|---:|---:|---:|---:|
| Sine | 0.9920 | 0.9441 | 0.08490 | 0.08644 |
| Mixed sine | 0.9795 | 0.9345 | 0.08475 | 0.08654 |
| Runge | 0.8018 | 0.7388 | 0.08613 | 0.08669 |
| Gaussian envelope | 1.0258 | 1.0060 | 0.08632 | 0.08679 |

The companion refits show small changes, most visibly for sine, rather than a transition to a substantially different approximation regime. All saved refit ranks remain 10. Solved coefficient norms are approximately $8\times10^9$ to $2.1\times10^{11}$, so tiny differences in evaluated residuals require numerical qualification. This experiment does not establish that geometry quality is exactly invariant.

**Readout convergence.** The bulk singular spectrum becomes much flatter as $\lambda$ increases. For example, $\sigma_{128}$ rises from approximately $1.45\times10^{-8}$ at $\lambda=0.25$ to $0.0336$ at $\lambda=2$, while $\sigma_1$ remains near 9.8. The reference rate $1/\sigma_1^2$ stays near 0.0103: adjusting this global rate supplies about a fivefold speedup, whereas changes in target-relevant small singular values can produce much larger differences.

Predicted steps to **1% training relative error**, from zero readout at the common rate $0.002$:

| Target | $\lambda=0.1$ | $\lambda=0.25$ | $\lambda=1$ | $\lambda=2$ |
|---|---:|---:|---:|---:|
| Sine | 31,246 | 17,366 | 9,001 | 8,030 |
| Mixed sine | $1.67\times10^{11}$ | 1,915,567 | 89,573 | 75,102 |
| Runge | 606,285 | 6,372 | 2,299 | 2,193 |
| Gaussian envelope | $4.27\times10^{11}$ | 2,334,423 | 35,004 | 27,048 |

The improvement does not extend monotonically to arbitrary accuracy. At $\lambda=2$, numerical training floors are approximately 0.1565%, 0.4031%, 0.08956%, and 0.4164% for the four targets. Only Runge reaches 0.1% in the retained model at that scale. Mixed sine at $\lambda=1.45$ is a costly edge case: its training floor is barely below 0.1%, and reaching that threshold requires a predicted $5.0\times10^{23}$ common-rate steps because a tiny retained singular direction must be learned. Its independent-grid refit error already exceeds 0.1%. The static figure annotates this off-scale prediction; it does not claim feasible fp64 convergence.

### Figures

- **Sine freezing, 2×4:** columns freeze after 2, 10, 50, and 150 joint updates. Top: actual independent-grid relative error versus total updates. Bottom: mean absolute slope versus total updates. Solid colored branches continue 500 updates after the dotted freeze line; dashed gray lines continue joint GD to the same endpoint. Mean scale falls further after freezing.
- **Mixed-sine freezing, 2×4:** identical conventions. The joint mean scale eventually turns upward while frozen-readout branches remain below it; actual fitting stays better with joint GD.
- **Runge freezing, 2×4:** the same comparison on the localized rational target. Branches reduce actual error but remain behind the joint control.
- **Gaussian-envelope freezing, 2×4:** the same finite domain and model as the other functions. Errors remain near one; this is not the earlier whole-line constrained-model experiment.
- **Refitted geometry comparison, 4×4:** function rows and freeze-time columns. Every marker is a new offline readout solve; solid branches, dashed joint controls, and a dotted initial-geometry reference share an enlarged vertical range within each row. Sine shows the most visible change. Large solved coefficients limit the interpretation of fine differences.
- **Readout spectrum, 24-second GIF:** $\lambda$ advances uniformly across 77 frames. Left: sorted singular value versus direction index, with the QI-scale spectrum as a fixed dashed reference and the numerical cutoff marked. Right: updates for one e-fold decrease of residual amplitude in each retained singular direction, for the common and spectral reference rates. Axes stay fixed. Direction indices are not Fourier frequencies; the right panel is not yet a target-specific convergence time.
- **Approximation floor and convergence, 2×4:** target columns. Top: numerical training LS floor and independent-grid refit error against $\lambda$. Bottom: target-weighted predicted steps to 0.1% training error at both rates. Gray regions have a numerical floor above that threshold. The vertical dotted line marks $\lambda=0.25$; the exceptional mixed-sine count above the display range is annotated.

## Additional details

**Exact fixed-matrix dynamics.** With readout vector $v$ and normalized residual $r=Av-\widetilde y$, an update gives

$$
v_{k+1}=v_k-\eta A^Tr_k,\qquad r_{k+1}=(I-\eta AA^T)r_k.
$$

Write $A=U\Sigma V^T$, and let $u_j$ denote a left singular vector. Its residual component contracts by $1-\eta\sigma_j^2$. From $v_0=0$, squared relative error in the retained model is

$$
E_k^2=E_\perp^2+\frac{1}{\|\widetilde y\|^2}\sum_{j\in\mathrm{retained}}|u_j^T\widetilde y|^2(1-\eta\sigma_j^2)^{2k}.
$$

$E_\perp$ is the relative residual outside the retained numerical span. Singular values specify rates; target loadings $|u_j^T\widetilde y|^2$ specify which rates matter. Worst-case condition numbers alone therefore do not answer the target's step-count question. For a contraction strictly between zero and one, the residual-amplitude e-folding time is $-1/\log(1-\eta\sigma_j^2)$. The computation uses log1p to preserve tiny nonzero contractions. Counts above $2^{53}$ are approximate magnitudes, not unit-accurate floating-point integers. No count hit the reporting cap of $10^{32}$.

**Why narrow tanh transitions remain correlated features.** As $\gamma$ grows, $\tanh(\gamma(x-z))$ approaches a step, not a localized bump. For $z_i,z_j\in[-1,1]$, its uniform-density limiting Gram entry is

$$
\mathbb E[\operatorname{sign}(x-z_i)\operatorname{sign}(x-z_j)]=1-|z_i-z_j|.
$$

Neighboring features remain strongly correlated. Outside-domain halo features approach signed copies of the constant column. Differences between neighboring steps would form a different readout parameterization; that is not the raw tanh coefficient problem tested here.

**Verification and cost.** Seven focused tests passed together. Branch prefixes match exactly, every readout coefficient including bias remains bitwise fixed after freezing, and the first frozen update agrees with independent analytic gradients within $2.8\times10^{-17}$. Refits cannot mutate training states. Explicit fixed-geometry GD at $\lambda=0.1,0.25,1,2$, both rates, and all four targets through 2,000 steps agrees with the spectral prediction within $3.44\times10^{-15}$ in relative-error units. The 1,738 finite step-count brackets were checked; cutoff and zero-mode cases have separate tests. Figures and representative GIF frames were visually inspected; the GIF has 77 frames totaling 24 seconds. Independent review found no material implementation defect. Training the joint and branched runs costs about one second per target on this machine; offline evaluation and refits cost about 12.5 seconds per target. Those diagnostics are not training costs.

**Intervention limits.** This is one Xavier seed and a 500-update continuation, not a general exclusion of useful freezing schedules. Freezing changes future residuals and fixes current coefficient magnitudes; it does not isolate one term of the note's gradient decomposition. Raw $(a,b)$ updates also move centers. Mean $|a|$ cannot distinguish opposing per-neuron movements; saved data include individual slopes, displacement, and cumulative travel for subsequent analysis.

## Conclusions

For these runs, early readout freezing does not generate mean scale growth or better actual fitting than joint GD. Fixed-geometry SVD dynamics explain how larger uniform scales can accelerate ordinary readout GD while sacrificing high-accuracy approximation; neither observation alone identifies the cause of stalled useful geometry learning.

## Small follow-up: readout-size and scale trends

Sam proposed that, in the nearly linear regime, slopes may continue the effective-amplitude adjustment that readout coefficients were making. In the saved Xavier runs, mean absolute neuron readout size declines along with mean gamma. A local expansion around each bias, $\tanh(a_kx+b_k)\approx\tanh(b_k)+a_kx\operatorname{sech}^2(b_k)$, reproduces the observed scale drift closely for the step-2 branches. This identifies a correction of an initially incorrect linear trend; it is not a universal attraction of gamma to zero.

A single additional Runge check starts at uniform gamma 16 with zero readout and uses the unchanged GD rate. It reaches 1% independent error at step 6369 (best affine approximation: 71.98%), then branches into frozen and continuing readouts for 5000 updates. Mean readout magnitude is increasing before freezing. Mean gamma subsequently increases by $6.32\times10^{-7}$ when frozen, versus $3.93\times10^{-7}$ under joint GD. The direction is consistent with the proposed compensation at the level of these averages, but the motion is tiny: frozen error stays at 0.9993%, whereas joint GD reaches 0.5651%. This is a small hypothesis check; initialization and fitting stage both differ from the early Xavier case.

The first geometry update is identical across branches, frozen coefficients remain bitwise constant, analytic gradients match the saved first update, and the branch accuracy is independently checked. Readout size excludes output bias; the output bias is also frozen. All averages include 177 neurons. There are no coefficient solves or learning-rate changes.

- **Primary diagnostic — nonlinear Runge:** one figure with actual error, mean absolute neuron readout, and mean absolute slope; full history and a zoom around freezing. Gray is common warmup, purple is frozen readout, teal is continuing joint GD. The absolute gamma tick values show the small size of the movement.
- **Supporting saved-data check — linear drift:** four target columns show their overall linear trend, mean gamma, and observed versus small-slope-predicted gamma drift. This uses existing trajectories only.

### Paired one-step nudge check

**Interpretation correction:** The nudge scatters are not comparisons of prolonged frozen and unfrozen trajectories. Their base path is ordinary joint GD. Each plotted frozen-block probe lasts one step from a shared joint state, so its corresponding parameter update equals that component of the simultaneous joint update. Actual sustained freezing creates a different subsequent state and generally a different trajectory. The scatter evidence therefore cannot establish compensation after prolonged freezing; explanations suggesting it does were too strong. The earlier 500- and 5,000-step frozen-readout branches are separate experiments and do show distinct trajectories.

**Explicit joint view:** Actual differences of successive saved parameters now show completed simultaneous updates 1–5,000, with no frozen-block labels. Compared with the previous scatter's state-k probes, this includes update 0→1 and excludes the unexecuted probe 5,000→5,001. All overlapping 4,999 updates match bitwise, for both gamma and readout. A companion plot shows training relative error and mean absolute gamma/readout over states 0–5,000, averaging the same 128 selected slots. All 177 neurons and all parameter blocks train. No additional training was needed.

A second small diagnostic uses 200 joint-GD states from early Xavier/Runge and 200 states immediately after the nonlinear Runge model reaches 1% error. At each state, two probes start from identical parameters: geometry steps with readout frozen, and readout steps with geometry frozen. Exactly 128 reference slots give 25,600 points per panel. Following Sam's clarification, the plotted quantities are signed changes in magnitudes, $X=|c_j|\Delta|\gamma_j|$ and $Y=|\gamma_j|\Delta|c_j|$, with $\gamma_j=|a_j|$. The base trajectory remains joint GD; these are single-step probes, not diverging frozen continuations.

Pearson correlation is 0.245 early and 0.429 after fitting, with intercept-inclusive linear-fit $R^2$ of 0.060 and 0.184. Among pairs with both nudges nonzero, growth/shrinkage directions agree about 58% and 66%, respectively. The later geometry nudges remain much smaller than the rescaled readout nudges. This does not establish strong one-to-one compensation, nor test whether compensation emerges over multiple frozen updates. Independent PyTorch forks at three states in each window exactly reproduce the probe updates and leave the frozen blocks unchanged.

- **Nudge-correlation figure:** early and fitted panels; linear axes, small dots colored by local training step 1–200 using viridis, a black best-fit line, and a dashed equal-nudge reference. Each axis has its own range and explicitly printed power of ten. Color tracks the shared joint-training state. Apparent colored curves consist of repeated small dots from the same neuron over time.


**5,000-step extension.** Both windows now contain 640,000 pairs, with the original 200-step results preserved and reproduced bitwise. Full-window correlations are 0.304 (Xavier) and 0.436 (fitted); the last 1,000 steps give 0.707 and 0.444. Per-step correlation rises from 0.160 to 0.709 in Xavier, while training relative error approaches 0.71794. The fitted run improves from 0.009997 to 0.005651, with correlation staying near 0.43–0.45. Its rescaled readout nudges have roughly $10^5$ times the RMS strength of its rescaled geometry nudges. Despite the higher late-Xavier Pearson correlation, only half the nonzero late pairs agree in growth/shrinkage direction; covariance weights large nudges more strongly than a sign count does.

The extension includes an all-steps scatter with last-1,000-step zooms, plus time curves for error, per-step correlation and nonzero sign agreement, and each block's RMS nudge divided by its own initial value. Independent frozen-block forks at steps 1, 100, 200 and 5,000 reproduce the probe updates exactly. No frozen branch is carried forward in this diagnostic.

**Mechanism check.** Put $E_0=\langle e\rangle$ and $E_1=\langle xe\rangle$. Expanding around each current bias gives $\partial_{a_j}L\approx c_j\operatorname{sech}^2(b_j)E_1$ and $\partial_{c_j}L\approx\tanh(b_j)E_0+a_j\operatorname{sech}^2(b_j)E_1$. These approximations match the early Xavier gradient vectors within 2.5% and 0.6%, respectively, over the first 200 states. Offset error shrinks faster than trend error, changing the relationship between the two probes and bending the neuron tracks. After approximately 2,000 steps, the tiny nonlinear gradient terms dominate, so the affine approximation no longer explains the gradient. In the fitted run, the readout feature $\tanh(ax+b)$ and slope feature $cx\operatorname{sech}^2(ax+b)$ are not proportional, and the geometry stays almost fixed while readout fitting continues.

Even the bias-free linear toy model $f_j=c_ja_jx$ does not predict a shared equality line: with positive coefficients, $\Delta a_j=-\eta c_jE_1$ and $\Delta c_j=-\eta a_jE_1$, so the plotted nudges satisfy $Y_j/X_j=a_j^2/c_j^2$, differing by neuron. The requested rescaling compares contributions to the product $a_jc_j$ in that toy model; it does not make ordinary GD use equal functional step sizes. For nonlinear neurons, these product changes are not the full changes in prediction space.

**Raw-coordinate check.** Removing the multipliers, the axes are the signed magnitude changes $\Delta|\gamma_j|$ and $\Delta|c_j|$. Both inward contraction and curved neuron tracks persist. In Xavier, RMS raw gamma nudges fall to 1.08% of their step-1 size and RMS raw readout nudges to 0.105%, while the RMS parameter magnitudes change by only approximately 1.5% and 1.7%. Thus shrinking multipliers are not the main source of the contraction. In the fitted window, raw nudge RMS falls to 48.2% and 42.7%. Full-window raw-coordinate Pearson correlations are 0.276 and 0.487; last-1,000-step values are 0.655 and 0.497. These are paired update coordinates, not parameter positions; curved paths do not establish oscillatory spiraling in parameter space. The raw arrays reproduce the original weighted coordinates exactly when their recorded multipliers are restored. No training was repeated.

### Dynamics behind the paired-nudge plots

These are projections of one high-dimensional trajectory into update coordinates, one projection per neuron. They are not independent initial conditions of a common autonomous two-dimensional system. Away from parameter sign crossings, the raw plotted deltas equal the respective gradient-flow velocities, multiplied by the step size and the signs of the current parameters. Sign crossings fold the coordinates and require the finite absolute-difference definition used by the code.

For normalized residual $r=(f-y)/\sqrt n$ and normalized Jacobian $J=\partial_\theta f/\sqrt n$, full gradient flow obeys

$$
\dot\theta=-J^Tr,\qquad \dot r=-JJ^Tr.
$$

Writing $w=\dot\theta$, differentiation gives $\dot w=-H(\theta)w$, where $H=\nabla^2L$ is the full loss Hessian. It includes the residual-weighted second derivatives, not only $J^TJ$. The selected two velocity coordinates are generally forced by all the other coordinates; they need not form a closed two-variable ODE. A locally constant positive semidefinite Hessian produces sums of real decaying exponentials, rather than a linearized focus with complex eigenvalues. Projecting those modes into two coordinates can nevertheless bend a path toward the origin. Time variation and sign changes allow more complicated shapes.

**Predicting early Xavier dynamics.** Use empirical inner product $\langle u,v\rangle=n^{-1}\sum_i u_iv_i$, and the orthonormal functions $u_0=1$, $u_1=x/\sqrt{\langle x,x\rangle}$. Define $m_\ell=\langle e,u_\ell\rangle$ and $B_{\ell p}=\langle u_\ell,\partial_{\theta_p}f\rangle$. Discarding the gradient contribution from residuals outside these two modes gives $\dot\theta\approx-B^Tm$, hence $\dot m\approx-BB^Tm$. Freezing $B$ at the initial saved state closes the model. The predicted GD recurrence is $m_{k+1}=(I-\eta BB^T)m_k$; parameter and nudge predictions follow by integrating $-\eta B^Tm_k$ and taking signed magnitude differences.

The initial matrix is

$$
BB^T=\begin{pmatrix}5.140279&-0.009590\\-0.009590&1.289142\end{pmatrix}.
$$

Its two eigenvalues predict amplitude e-folding times of 96.77 and 387.36 updates at rate 0.002. The nearly decoupled fast direction is offset error; the slow direction is slope error. Both rates come from the initial Jacobian, not a fit to time traces. Over the first 200 states, relative Euclidean discrepancies are 0.61% for the two residual moments jointly, 7.70% for gamma nudges, and 5.26% for readout nudges. This is an initial-state prediction, unlike the earlier small-slope calculation that used each actual current residual and parameters. The model fails for the late small nonlinear forces: its last-1,000-step nudge discrepancies are approximately 100%. Whole-window errors would hide this failure by weighting the much larger early nudges.

**Predicting the fitted model with fixed geometry.** Fix all initial slopes and biases, and let $A=[\tanh(ax+b),1]/\sqrt n$. With the complete readout $v$, $r=Av-y/\sqrt n$ satisfies

$$
\dot v=-A^Tr,\qquad \dot r=-AA^Tr.
$$

If $A=U\Sigma V^T$, each readout residual mode contracts in discrete GD by $(1-\eta\sigma_\ell^2)^k$. The implementation evaluates the finite-time coefficient change from the initial state with stable expm1 expressions; no coefficient refit, singular-value cutoff, or fitted decay rate enters this prediction. Its largest $\eta\sigma^2$ is 0.1931, so every nonzero mode contracts without alternating sign.

For fixed $a_j,b_j$, put $\psi_j=x\operatorname{sech}^2(a_jx+b_j)$. At every predicted readout state, the hypothetical raw-slope velocity is

$$
\dot a_j^{\rm probe}=-c_j(t)\langle e(t),\psi_j\rangle.
$$

It is calculated but never applied. Readout velocities are sums of decaying modes; slope probes multiply a changing readout coefficient by another residual-mode sum. Their paths can therefore bend and change sign even though geometry is fixed. The residual outside the readout span is constant and may leave a nonzero asymptotic geometry probe, so this model does not automatically imply convergence of every probe to zero.

Over all 5,000 states and 128 neurons, this initial-state prediction matches the observed joint-GD raw gamma and readout deltas with relative Euclidean discrepancies of **0.0282% and 0.0294%**, respectively. At each individual step the vector discrepancies remain below 0.056% and 0.058%. Independent direct-gradient steps verify the spectral recurrence and residual pairing. The initial selected gamma/readout magnitudes match saved states bitwise. A two-panel figure uses identical axes and time colors to compare the observed and predicted phase patterns.

This identifies evolving readouts and residuals on almost stationary geometry as sufficient to explain this fitted-run pattern. It does not prove a general long-time barrier, explain every late-Xavier force, or establish neuronwise compensation after prolonged readout freezing.

## Open questions

1. In saved freezing states, how do the signed geometry-gradient contributions from residuals inside and outside the current numerical readout span combine? Measure their directions, not only their norms, using the current-readout Jacobian.
2. Does fixed-center scale training exhibit the same freezing response, or is the raw-slope motion coupled chiefly to center movement? Keep the learning rate unchanged when isolating this factor.
3. How much readout slowdown remains in a local difference basis spanning the same functions? This would test coefficient parameterization without changing the approximation space; it has not been run here.
