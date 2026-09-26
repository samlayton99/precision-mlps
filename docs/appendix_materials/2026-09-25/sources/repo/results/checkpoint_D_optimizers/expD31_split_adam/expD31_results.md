# expD31 — Separate Adam streams with an outside approximation multiplier · draft-pending-Sam

## TL;DR

- **10,000-step Xavier extension:** completed the original VarPro split at $\mu=50,100,250,500$, with constant and cosine-decaying common learning rates. The two new figures have matched axes, refitted relative $L_2$ error in the middle row, and linear mean-gamma axes. This extension is recorded without a new interpretation, as requested.

- **Smaller Xavier multipliers:** the new $\mu\in\{100,500,1000,2000\}$ sweep uses the original trainer and records explicit independent-grid relative refit error at every step. At $\mu=500$, Runge finishes at $8.38\times10^{-10}$, mixed sine at $3.42\times10^{-7}$, and Gaussian at $1.83\times10^{-5}$. None reaches the QI reference near $10^{-14}$.

- All 36 split-Adam runs and 12 ordinary-Adam controls completed 500 updates. Multipliers act after normalization, with the unchanged base rate. Seven new implementation tests and five inherited profile-gradient tests pass.
- From Xavier, the split update at $\mu=1000$ improves actual fitting on all four targets compared with ordinary Adam and the matching weighted GD. It improves the final refitted geometry on mixed sine, Runge, and Gaussian envelope, but worsens it on sine. Runge reaches independent-grid refitted relative $L_2$ error $2.35\times10^{-6}$, with mean gamma $10.33$.
- Larger outside multipliers generate much larger slopes and frequently worse geometry. All four Xavier/$\mu=1000$ runs briefly reach very small refitted errors early; most lose that improvement later. The normalization and multiplier therefore enable motion without ensuring that useful geometry is preserved.
- Large slopes also expose sampling failures. Xavier/Runge at $\mu=100000$ has sampled refitted relative error about $0.1$, but independent-grid refitted error about $413$. Initial Xavier directions are numerically sensitive, and the method still requires a dense SVD each step. This is not a demonstrated stable or scalable recipe.

## Question / hypothesis

Does separately normalizing the approximation and readout-gap gradients with Adam, then amplifying the approximation direction outside normalization, improve the previous weighted-GD experiment?

## Experiment design

The network is $f_{\theta,v}(x)=\sum_{k=1}^{m}v_k\tanh(a_kx+b_k)+v_{m+1}$, with geometry $\theta=(a,b)$ and reported scale $\gamma_k=|a_k|$. Both slopes and biases move; these are not center-preserving, gamma-only updates. The normalized feature matrix is $A=[\tanh(x_i a_k+b_k),\mathbf 1]/\sqrt n$, and $\bar y=y/\sqrt n$. The actual loss is $L=\frac12\|Av-\bar y\|^2$.

We retain the previous numerical reference $F_\tau=\frac12\|(I-U_rU_r^T)\bar y\|^2$, keeping singular values above $10^{-13}\sigma_1$, and define $G_\tau=L-F_\tau$. The inherited derivative differentiates the retained singular subspace, including its correction to the untruncated VarPro envelope gradient. It is a local derivative between cutoff crossings. Numerical $F_\tau$ is not unrestricted exact-arithmetic VarPro.

For $Q\in\{F,G\}$, form $g_t^Q=\nabla_\theta Q_\tau$ at the same current state. Maintain independent moment streams:

$$m_t^Q=\beta_1m_{t-1}^Q+(1-\beta_1)g_t^Q,\qquad s_t^Q=\beta_2s_{t-1}^Q+(1-\beta_2)(g_t^Q)^2,$$

$$u_t^Q=\frac{m_t^Q/(1-\beta_1^t)}{\sqrt{s_t^Q/(1-\beta_2^t)}+\epsilon_A},\qquad \theta_{t+1}=\theta_t-\eta(\mu u_t^F+u_t^G).$$

All moment operations are coordinatewise. We use $\beta=(0.9,0.999)$, $\epsilon_A=10^{-8}$, $\eta=0.002$, and $\mu\in\{1000,10000,100000\}$. The readout receives ordinary Adam using $\nabla_vL=\nabla_vG_\tau$, computed before the geometry update. Solved coefficients are never installed. No clipping, learning-rate schedule, acceptance test, or numerical-gradient gating is used.

This is not Adam applied to $\mu F_\tau+G_\tau$: the two geometry gradients have separate histories, and multiplication occurs after normalization. Likewise, summing two Adam directions at $\mu=1$ would not recover ordinary Adam. The control is therefore an independent ordinary-Adam run on $L$, not a split run labeled $\mu=1$. Matching weighted-GD curves reuse expD29 data and retain their original readout GD; comparing those curves changes both optimizers, as requested, and does not isolate geometry normalization alone.

All runs use seed zero, float64, $N=128$ intervals, 129 interior neurons and 24 halo neurons per side (177 neurons total), 1,024 midpoint training samples in $[-1,1]$, and 500 updates. Targets are sine, mixed sine, Runge, and Gaussian-envelope mixed sine, with the same formulas and samples as expD29. Initializations are ordinary Xavier; Xavier slopes and biases multiplied together to reach mean $\gamma=16$ while preserving centers; and uniform QI geometry at $\gamma=16$ with zero readout. The latter two start at $\lambda=0.25$ in the same respective mean/uniform senses as before.

Every update records $L,F_\tau,G_\tau$, mean gamma, gradient norms, proposed update norms, retained rank, and coefficient norms. Early and logarithmically spaced states are saved. Endpoint validation uses 8,192 independent midpoint samples, refits at three SVD cutoffs, and 65,536-point checks for the largest multiplier. A separate check evaluates the best saved $F_\tau$ state of each Xavier/$\mu=1000$ trajectory. It checks an evaluation-only refit of that saved geometry, not an early-stopped trained model.

The 10,000-step extension retains Xavier and compares $\mu\in\{50,100,250,500\}$ under two common-rate policies. One holds $\eta_t=0.002$ throughout; the other uses

$$\eta_t=0.002\left[0.001+0.999\frac{1+\cos(\pi t/10000)}{2}\right],\qquad 0\le t\le10000.$$

Decay starts at the beginning and spans the entire run, ending at $2\times10^{-6}$. The same rate multiplies both geometry streams and the ordinary-Adam readout step; the outside multiplier, Adam epsilon, and moment histories are otherwise unchanged. There is no warm phase, clipping, restart, coefficient installation, or choice of schedule based on early results. Four ordinary-Adam controls are matched to each policy; the constant controls reuse the completed identical 10,000-step expD33 controls. The extension records explicitly reconstructed training-refit relative error at every split step, evaluates saved geometries on 8,192 independent samples, and checks final and best saved refits on 65,536 samples and at alternative cutoffs. The best sampled projection state is also saved; the best independent refit reported is the best among saved states, not necessarily among all 10,001 states.

The dynamic-ratio extension replaces the fixed outside multiplier with $\mu_t=r\|u_G(t)\|_2/\|u_F(t)\|_2$, for $r\in\{0.01,0.1,1,10,100\}$. Thus the post-Adam geometry contributions satisfy $\|\mu_tu_F\|_2/\|u_G\|_2=r$ before their sum is applied. Both Adam histories are formed from the unscaled gradients, and the readout update remains ordinary Adam. The four Xavier targets and both 10,000-step learning-rate policies are inherited. Effective multipliers, achieved ratios, and existing numerical-sensitivity diagnostics are saved. Alternative-backend audits recompute the controller from each alternative direction while retaining the same preceding moment histories. No clipping or confidence gate changes the requested ratio; a zero or nonfinite direction makes it undefined and stops the run with a recorded reason. This is the requested diagnostic of fixed relative update strength, with the same unresolved production-cost and numerical-sensitivity qualifications as the original split.

**Code & data**

- [Dynamic-ratio constant-rate figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/figures/constant.png), [dynamic-ratio cosine-decay figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/figures/cosine.png), and [recorded ratios and endpoint checks](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/data/summary.json).

- [Dynamic-ratio runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/dynamic_ratio.py), [dynamic-ratio tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD31_dynamic_ratio.py), and [dynamic-ratio data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/dynamic_ratio/data). Configurations are embedded in each trajectory.


- [Saved-data balance plotting](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/plot_balance.py), [constant-rate ratios](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/balance_constant.png), and [cosine-decay ratios](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/balance_cosine.png). These use the existing trajectory arrays only; no training or coefficient solves are repeated.

- [10,000-step runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/long_run.py), [long-run data](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/data), [constant-rate figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/constant.png), [cosine-decay figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/long_run/figures/cosine.png), and [schedule implementation tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD31_schedule.py). The runner inherits the original training function and adds the common-rate policy and saved-state checks.

- [Smaller-multiplier runner](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/mu_refit_sweep.py), [500-step relative-refit figure](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/figures/xavier.png), [first-30-step view](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/figures/xavier_early.png), [sweep data and independent checks](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/mu_refit_sweep/data). This runner reuses the original training function and inherits its configuration; only Xavier, the four requested multipliers, and denser state recording are selected.

- [Run and analysis](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/run.py), [configuration](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expD31_split_adam/config.yaml), [implementation tests](/Users/sam/my-repos/research/collaborations/precisionMLPs/tests/test_expD31_split_adam.py), [requirements checklist and status](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/expD31_status.md).
- [Data folder](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/data): 48 compressed trajectories with embedded configurations, endpoint summary, three-cutoff audit, early-refit check, and dense-grid check. The ordinary-Adam control uses file suffix zero; this is a control identifier, not a split weight.
- [Xavier trajectories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/xavier.png), [scaled-Xavier trajectories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/scaled_xavier.png), [QI trajectories](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/qi_zero.png), [independent-grid verification](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_D_optimizers/expD31_split_adam/figures/independent_evaluation.png).
- Reproduce using the run script; `--plot-only` redraws saved data, while `--analyze-only` recomputes checks and plots without training.

## Results

Xavier/$\mu=1000$ produces the clearest favorable results. This table uses independent-grid relative $L_2$ errors at update 500. The actual model retains its trained coefficients; the refitted model replaces coefficients only for evaluation.

| Target | Ordinary Adam: actual | Split Adam: actual | Ordinary Adam: refitted | Split Adam: refitted | Split Adam: mean gamma |
|---|---:|---:|---:|---:|---:|
| Sine | 0.921 | 0.519 | 0.00528 | 0.0859 | 6.91 |
| Mixed sine | 0.913 | 0.397 | 0.448 | 0.183 | 13.85 |
| Runge | 0.643 | 0.149 | 0.0587 | $2.35\times10^{-6}$ | 10.33 |
| Gaussian envelope | 0.998 | 0.332 | 0.492 | 0.0409 | 13.48 |

The Runge gain is also large compared with the matching $\mu=1000$ weighted-GD run: its actual error was $0.738$ and refitted error $0.0975$. The new refit remains between $2.30\times10^{-6}$ and $3.36\times10^{-6}$ over the three checked cutoffs. The solved coefficient norm is still large, about $4.0\times10^5$ at the primary cutoff; the gain is not a bounded-readout or machine-precision result.

Xavier sine illustrates why training gain and approximation gain must remain separate. Its actual error improves, yet its final refit is worse than the ordinary-Adam geometry. At update one, that same split run briefly has independent refitted error $6.84\times10^{-9}$. Mixed sine and Gaussian likewise briefly reach about $10^{-7}$ refitted error at early saved states, before losing much of the gain. The dip in the middle row is therefore not merely a plotting artifact.

Larger multipliers drive Xavier mean gamma to roughly $90$–$120$ for $\mu=10000$ and $1100$–$1800$ for $\mu=100000$. They do not yield monotonically better training or approximation. The largest-weight Runge refit is especially misleading on the training samples: the independent error is about $413$, confirmed by the denser grid, even though its sampled relative projection error is about $0.1$. This failure survives the stricter cutoff; relaxing the cutoff worsens it further.

For scaled Xavier, ordinary Adam has the lowest final actual loss on all four targets among these Adam variants. Moderate splitting improves the refitted sine and Runge geometry, but strongly degrades the mixed-sine and Gaussian refits. At $\mu=100000$ the final refitted geometry is much worse on all four targets. For QI, changing the split multiplier makes comparatively small changes to the trained fit; the major improvement over GD is also present in ordinary Adam. All Adam variants move away from QI's initial numerical approximation floor.

### Smaller Xavier multipliers: relative refit error

All 16 runs completed 500 updates with the unchanged $\eta=0.002$, Adam settings, numerical VarPro derivative, samples, and seed. The four repeated $\mu=1000$ histories match the original losses and gamma values bitwise. Every state is saved. At each state we solve coefficients using the training samples and reconstruct predictions on 8,192 independent midpoint samples. Thus the plotted middle row is $\|f_{\theta,v_*}-y\|_2/\|y\|_2$, not squared loss and not merely a projection-norm conversion. Coefficients are never installed into the trained readout; the geometry still uses the original VarPro-derived stream.

Final relative refit errors:

| Target | $\mu=100$ | $\mu=500$ | $\mu=1000$ | $\mu=2000$ |
|---|---:|---:|---:|---:|
| Sine | $6.37e-08$ | $6.29e-07$ | $0.0859$ | $0.177$ |
| Mixed sine | $0.0492$ | $3.42e-07$ | $0.183$ | $0.213$ |
| Runge | $5.51e-05$ | $8.38e-10$ | $2.35e-06$ | $0.0124$ |
| Gaussian envelope | $0.0223$ | $1.83e-05$ | $0.0409$ | $0.0962$ |

Smaller multipliers improve all four final refits relative to the old $\mu=1000$ setting: sine prefers 100 among this sweep, while the other three prefer 500. Runge at 500 improves final relative error by about 2,800 times versus 1000. Its final trained-readout relative error is still $0.0568$: this is primarily a geometry result, with a remaining readout optimization gap.

Best geometries observed on the independent evaluation grid, across the four multipliers:

| Target | Multiplier | Step | Best relative refit error | Mean gamma at that step |
|---|---:|---:|---:|---:|
| Sine | 100 | 454 | $1.28e-09$ | 1.052 |
| Mixed sine | 500 | 48 | $1.16e-07$ | 5.273 |
| Runge | 500 | 29 | $3.18e-10$ | 5.147 |
| Gaussian envelope | 1000 | 5 | $2.42e-07$ | 5.075 |

The best-state errors survive evaluation on 65,536 samples. Three-cutoff checks remain above machine precision; for example the best Runge state's error is approximately $5.49\times10^{-9}$, $3.18\times10^{-10}$, and $1.12\times10^{-10}$ as the relative SVD cutoff changes from $10^{-12}$ to $10^{-14}$. The matching QI reference is about $7.6\times10^{-15}$. Multiplier tuning therefore produces substantial gains, but this sweep does not demonstrate convergence to the desired numerical floor.

The same initial numerical sensitivity remains: alternative backends or SVD drivers change the initial proposed step by roughly 25%–118% of its norm across targets. The checks above validate the attained predictions; they do not establish reproducibility of the learned trajectory across implementations or seeds. No learning-rate schedule or further tuning was run in this sweep.

### Figures

- **Dynamic ratio, constant learning rate:** four function columns, with actual squared training loss, refitted relative $L_2$ error, and linear mean gamma as the three rows. Viridis identifies $r=0.01,0.1,1,10,100$. Black dotted curves reuse the matched ordinary-Adam control; middle-row circles score saved refits independently, and the dashed reference is QI. The controller chooses $\mu_t$ after Adam so the geometry-contribution norm ratio is the selected $r$.
- **Dynamic ratio, cosine decay:** the same quantities and axis limits, with the common learning rate decreasing from $0.002$ to $0.000002$. Across the two versions, all 40 trajectories complete 10,000 steps. Recorded post-Adam ratios match their prescribed values to at most $1.12\times10^{-15}$ relative discrepancy. Effective multipliers and numerical-sensitivity audits are retained in the trajectory files. The 17 selected implementation tests pass, including independent multi-step Adam comparisons under both schedules. No new performance interpretation is added.

- **Constant-rate loss and update balance:** four function columns; top is the unweighted scalar loss ratio $F_\tau/G_\tau$, with $G_\tau=L-F_\tau$. Bottom is $\|\eta_t\mu u_F\|_2/\|\eta_tu_G\|_2$, the ratio of the recorded geometry-update contributions after Adam and outside scaling. This bottom row matches the kind of ratio used in the current-readout projected-split figure. Viridis colors identify the four multipliers; the dashed line marks equal magnitudes. Update ratios omit the final state's unapplied proposed update.
- **Cosine-decay loss and update balance:** the same ratios, colors, and corresponding axis limits for the saved decay trajectories. Both rows use logarithmic vertical axes and all 10,000 applied steps. Checks confirm finite positive operands and $F_\tau+G_\tau=L$ in every plotted trajectory. Ratios of norms do not encode angles or cancellation.

- **10,000 steps, constant rate:** four function columns and four viridis multiplier curves. The rows show actual squared training loss, explicitly reconstructed refitted relative $L_2$ error, and mean $|a_k|$. Black dotted curves are ordinary Adam at the same common rate. Middle-row circles evaluate saved refits on independent samples; the gray dashed line is the QI refit reference.
- **10,000 steps, cosine decay:** identical layout, multipliers, and corresponding axis limits. The common rate decreases smoothly from $0.002$ to $0.000002$ over the entire run; black dotted controls use that same schedule. Across both versions, all 32 split trajectories and all eight controls finish at step 10,000. Four constant controls are reused and four scheduled controls are new. Data include final and best saved refit checks; no trained coefficients are replaced with solved coefficients.

- **Smaller Xavier multipliers, 500 steps:** four function columns; actual squared loss, explicit independent-grid relative refit error, and mean gamma. Viridis identifies 100, 500, 1000, and 2000; black dotted curves are ordinary Adam, and the gray dashed middle-row reference is the corresponding QI refit. Middle-row axes match across targets and reach $10^{-16}$; gamma uses linear axes.
- **Smaller Xavier multipliers, first 30 steps:** the same quantities and lines, with each early refit marked, showing the rapid initial gains and continuing scale growth.

- **Xavier trajectories:** four target columns; rows are actual loss, sampled refitted loss, and mean absolute slope. Solid viridis curves are split Adam at the three multipliers; same-color dashed curves are matching weighted GD; black dotted is ordinary Adam. The second row shows both the strong Runge improvement and the early minima lost on other targets. The third row makes the very large outside-weight motion visible.
- **Scaled-Xavier trajectories:** same layout and legend. Ordinary Adam fits the trained readout well, while large outside multipliers can damage already useful geometry. Moderate weights improve the sine/Runge refit without improving their actual fit over ordinary Adam.
- **QI trajectories:** same layout. The actual-loss curves are close across Adam variants, while the refitted-loss row shows departure from the initial high-precision geometry. Read the absolute gamma tick labels: its movement is much smaller than Xavier's despite occupying a full panel.
- **Independent-grid verification:** initialization rows and target columns; categorical horizontal positions are ordinary Adam and the three split weights. Black is the actual trained model's independent error, blue the relative training projection reference, and orange the evaluated refit on the independent grid. The striking orange/blue separation in Xavier/Runge at the largest weight exposes the failure hidden by the sampled refit metric.

## Additional details

Machine epsilon is a relative spacing near one, not an absolute loss threshold. A squared loss can be well below epsilon while its derivative is much larger: $F(\theta)=\theta^2/2$ at $\theta=10^{-10}$ gives $F=5\times10^{-21}$ and $F'=10^{-10}$. These values are far from float64 underflow. The separate numerical questions are whether the computed residual and projected derivative are accurate, and whether the resulting update can alter the stored parameters. Enforcing an update-norm ratio controls relative magnitude; it does not certify the accuracy of the direction being amplified.

The long-run extension passes the 15 selected implementation checks across the split-Adam, schedule, and current-readout suites. The schedule tests check the endpoints and monotonicity, then compare three scheduled updates against an independently differentiated scalar objective and three independent PyTorch Adam optimizers. They verify that the common rate reaches the readout and both geometry streams, with the multiplier outside normalization. The earlier numerical-sensitivity and diagnostic-cost qualifications still apply.

At the first update, an individual $F$ coordinate contributes $\eta\mu g_F/(|g_F|+\epsilon_A)$. When $|g_F|\gg\epsilon_A$, its magnitude is approximately $2$, $20$, or $200$. This is the literal requested outside scaling; it is not the same physical step size as multiplying the original tiny gradient by the same $\mu$. Adam's epsilon also makes scale invariance approximate, especially near the QI numerical floor.

Momentum can retain a large proposed update after the current approximation gradient becomes small. For example, at Xavier/sine's update-one refit minimum, $\|DF_\tau\|$ is about $1.18\times10^{-12}$, but the next proposed $F$ displacement has norm about $23.5$. This comes from the recorded Adam moments and outside multiplier. It is evidence that the current small gradient does not stop motion; this experiment does not isolate momentum from coordinate scaling, $DG$, or numerical sensitivity as the cause of later degradation.

Alternative SVD drivers and tanh evaluation backends flag the initial Xavier directions as sensitive. Holding the preceding moment history fixed, their initial proposed-step discrepancies are about 25%–118% of the primary step norm across targets. These checks do not prove the trajectory would reproduce under another numerical implementation. Relative sensitivity near QI's initial zero signal must also be read with its tiny absolute step size. The checks record uncertainty without clipping or gating the requested update.

All seven new tests pass: two signed/tiny-gradient stream comparisons against PyTorch Adam, three outside-multiplier/independent-history checks, an exact ordinary-Adam trajectory and unchanged-first-readout check, and a first split update checked against an independently differentiated scalar SVD objective. The five inherited projector-gradient tests also pass. All 48 trajectories remain finite; finiteness does not imply a stable optimization recipe. The dense SVD still violates the project's intended production cost and architecture constraints.

## Conclusions

These runs demonstrate that separately normalized and amplified approximation updates can move Xavier geometry substantially and sometimes improve its independently evaluated approximation. They also demonstrate loss of useful geometry, strong numerical sensitivity, and a severe between-sample refit failure at large scale; the tested settings do not provide a dependable way to recover the desired regime.

## Open questions

- Can the early useful geometries be retained without the continuing large motion? A targeted intervention would need to distinguish Adam's carried moments from the two current gradient directions.
- Can the Xavier improvement reproduce with numerically better resolved approximation directions, additional seeds, and a method that meets the computational budget?
