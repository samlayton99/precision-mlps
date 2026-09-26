# expD33 — Amplify the projected signal at the current readout · data-obvious

## TL;DR

- **10,000-step extension:** all 20 split runs and four ordinary-Adam controls complete and reproduce their first 500 steps exactly. The multiplier curves separate later. The smallest tested multiplier, 100, yields the best final refit of each target among split runs and improves each final refit relative to ordinary Adam; ordinary Adam still has the lowest final trained-model error on every target.

- In the original short sweep, all 20 Xavier runs complete 500 updates. Amplifying the current-readout projected signal does not reproduce expD31's rapid VarPro-driven approximation improvements.
- Sine, mixed sine, and Gaussian trajectories almost overlap ordinary Adam across the requested multipliers. Runge has more visible differences, without a substantial final refit gain.
- The initial projected signal is extremely small, including numerical uncertainty comparable with the sine and Gaussian signals. Adam's epsilon attenuates these directions before the outside multiplier acts.

## Question / hypothesis

Does the current-readout signal $J^T r_\perp$, amplified through a separate Adam stream, produce the geometry improvements previously seen when amplifying the VarPro derivative?

## Experiment design

Use $A=[\tanh(x_i a_k+b_k),\mathbf 1]/\sqrt n$, $r=Av-y/\sqrt n$, and $P_\tau=U_rU_r^T$ retaining singular values greater than $10^{-13}\sigma_1$. The amplified direction is $h=J_{\rm current}^T(I-P_\tau)r$, where the Jacobian uses the currently trained readout. Its companion is $g_{\rm rest}=\nabla_\theta L-h$. Neither component is labeled $DF_\tau$ or $DG_\tau$. Two independent Adam histories produce $u_h,u_{\rm rest}$ and the update $\theta^+=\theta-\eta(\mu u_h+u_{\rm rest})$. The readout receives ordinary Adam from the same pre-update state. Solved coefficients are used only for diagnostics.

The settings match expD31: Xavier seed zero; sine, mixed sine, Runge, and Gaussian-envelope mixed sine on 1,024 midpoint samples in $[-1,1]$; $N=128$, 177 neurons including 24 halo neurons per side; float64; 500 updates; $\eta=0.002$; Adam $\beta=(0.9,0.999)$ and $\epsilon_A=10^{-8}$. The only optimizer change is the split direction, with the requested $\mu\in\{100,500,1000,5000,25000\}$. Both slopes and biases train; reported gamma is $|a_k|$.

Record actual loss, numerical refit loss $F_\tau=\frac12\|(I-P_\tau)y/\sqrt n\|^2$, gamma, component/update norms, and the difference between the projected current residual and the refitted residual. Check best and final geometries on 8,192 independent samples and at three SVD cutoffs. Dense SVD is an explicitly requested diagnostic, outside the production optimizer budget.

The duration extension repeats the same twenty current-readout cases for 10,000 updates and adds four ordinary-Adam controls of the same length. Only duration and diagnostic recording change. Runs start from the original initial state, so optimizer moments evolve continuously; completed trajectories are checked against every original first-500-step loss and mean-gamma value, plus the refit squared loss for split runs. Solved coefficients are never installed into the trained readout.

For the extended split runs, reconstruct the fitted model on the training samples at every step and report its relative $L_2$ error. At about 200 saved states, including the best sampled projection-loss state, refit on the training grid and score predictions on 8,192 separate midpoint samples. Score the final and best saved independent-grid refits again on 65,536 samples. The companion diagnostic plots $\|\eta\mu u_h\|_2/\|\eta u_{\rm rest}\|_2$, the ratio of the two actual proposed geometry-step norms after Adam and outside multiplication. Its value does not determine the angle between the streams or whether their sum improves approximation.

The matched $J$ versus $J_*$ follow-up uses four separate function figures, with columns $\mu=100,250,500,1000$. All paired runs have identical Xavier arrays, samples, Adam settings, outside multiplier, and cosine schedule from $0.002$ to $0.000002$ across 10,000 updates. Twelve existing $J_*$ trajectories are reused; sixteen current-$J$ trajectories and four $J_*/1000$ trajectories fill the missing settings. The schedule is applied to both geometry streams and the readout. The top row is trained relative $L_2$, computed exactly as $\sqrt{2L/\mathrm{mean}(y^2)}$; the middle row uses the explicitly reconstructed least-squares relative residual; both use the same training samples. The bottom row is mean gamma. Each panel has exactly the two method curves. Eleven selected implementation checks pass, including independently computed scheduled updates. Initial and final predictions independently verify the relative-error conversion for all 32 trajectories. No performance interpretation is added.

**Code & data**

- Matched cosine comparison: [plotting](../../../experiments/expD33_current_readout_split/compare_jstar.py), [missing-run fill](../../../experiments/expD33_current_readout_split/fill_comparison.py), [source manifest](j_vs_jstar/data/sources.json), [additional trajectories](j_vs_jstar/data/trajectories/), and four figures: [sine](j_vs_jstar/figures/sine.png), [mixed sine](j_vs_jstar/figures/sine_mixture.png), [Runge](j_vs_jstar/figures/runge.png), [Gaussian envelope](j_vs_jstar/figures/gaussian_envelope.png).

- [Runner](../../../experiments/expD33_current_readout_split/run.py), [configuration](../../../experiments/expD33_current_readout_split/config.yaml), [tests](../../../tests/test_expD33_current_readout_split.py), [requirements and status](../../../docs/expD33_status.md).
- [Figure](figures/xavier.png), [trajectories and evaluation checks](data/).
- [10,000-step extension runner](../../../experiments/expD33_current_readout_split/long_run.py), [extended training/refit/gamma figure](long_run/figures/training.png), [relative update strength](long_run/figures/stream_strength.png), [logarithmic gamma view](long_run/figures/gamma_log.png), [extended trajectories and checks](long_run/data/). The extension runner accepts `--targets` and `--train-only` for separate target jobs, `--analyze-only` for completed data, and `--plot-only` for redrawing.
- Reproduce the original short sweep with the original runner; `--plot-only` uses cached results.

## Results

The final independent-grid refit errors are approximately $0.00527$ for sine, $0.448$ for mixed sine, and $0.492$ for Gaussian across the five multipliers, essentially the ordinary-Adam results. Mean gamma also follows the ordinary-Adam path. Runge's final refit errors range from $0.0571$ to $0.0624$, compared with approximately $0.0587$ for ordinary Adam. Its final mean gamma ranges from $0.117$ to $0.132$. This is far from the earlier VarPro-weighted Runge result of $2.35\times10^{-6}$ relative refit error and mean gamma $10.33$.

### 10,000-step extension

The short-run overlap is temporary. Later projected updates can rival or exceed the companion stream, and trajectories separate. This statement concerns relative post-Adam step norms; it does not establish that Adam's epsilon ceases to dominate every gradient coordinate.

Final independent-grid relative refit errors:

| Target | Ordinary Adam | $\mu=100$ | $\mu=500$ | $\mu=1000$ | $\mu=5000$ | $\mu=25000$ |
|---|---:|---:|---:|---:|---:|---:|
| Sine | $6.25e-06$ | $1.41e-07$ | $3.81e-07$ | $3.43e-07$ | $0.303$ | $0.0506$ |
| Mixed sine | $0.203$ | $0.193$ | $0.408$ | $0.306$ | $0.446$ | $0.475$ |
| Runge | $0.00334$ | $1.17e-05$ | $0.0356$ | $0.228$ | $0.137$ | $0.201$ |
| Gaussian envelope | $0.194$ | $0.122$ | $0.2$ | $0.49$ | $0.355$ | $0.531$ |

At multiplier 100, sine's final refit improves by about 44 times over ordinary Adam, Runge by about 285 times, Gaussian by about 1.6 times, and mixed sine by about 5%. These are geometry improvements measured after fitting coefficients. The actual trained-model errors are worse than ordinary Adam on all four targets; that holds for every tested split multiplier at update 10,000.

Final mean gamma:

| Target | Ordinary Adam | $\mu=100$ | $\mu=500$ | $\mu=1000$ | $\mu=5000$ | $\mu=25000$ |
|---|---:|---:|---:|---:|---:|---:|
| Sine | 0.359 | 0.426 | 0.396 | 0.427 | 7.27 | 60.4 |
| Mixed sine | 0.554 | 0.526 | 2.28 | 2.42 | 7.68 | 25.7 |
| Runge | 0.239 | 0.32 | 1.53 | 3.18 | 10.2 | 86.5 |
| Gaussian envelope | 0.551 | 1.25 | 1.33 | 4.17 | 18.4 | 161 |

The largest multiplier produces substantial scale movement without good final refits. For example, Runge ends at mean gamma about 86.5 and refit error 0.201, whereas multiplier 100 ends near gamma 0.320 and refit error $1.17\times10^{-5}$. Gamma movement alone is not a success criterion. These comparisons do not isolate whether poorer results arise from the projected direction, step magnitude, optimizer memory, or numerical sensitivity.

All endpoint refits were checked again on 65,536 points. The largest relative change from the 8,192-point score is about 0.90% for Runge at multiplier 25,000; the conclusions above are unchanged. Best saved refits were also checked. Best-saved refers to the recorded geometry states, rather than an assertion that every intermediate independent-grid error was evaluated. No run reaches the QI reference near $10^{-14}$.

### Figures

- **10,000-step training, refit, and gamma:** the same four function columns and five viridis multiplier colors; black dotted curves are matched ordinary Adam. Top: actual squared loss. Middle: explicit reconstructed training-refit relative $L_2$ error, with open-circle independent-grid checks and a gray QI reference. Bottom: mean gamma on linear axes.
- **Relative update strength:** four function panels with the norm ratio of the amplified projected contribution to the companion contribution, after both Adam histories and the outside multiplier. Crossing one means their norms exchange order; it does not determine cancellation or approximation improvement.
- **Logarithmic gamma view:** the same mean-gamma trajectories on common logarithmic axes, so the smaller-multiplier curves remain visible alongside the large excursions.

- **Current-readout split, Xavier:** four function columns; actual squared loss on top, numerical least-squares squared loss in the middle, and mean gamma on the bottom. Viridis colors identify the five multipliers, black dots mark ordinary Adam, and gray dashes show the previous VarPro-$\mu=1000$ reference in the loss rows. Gamma has expanded linear axes per function so the small movement is visible.

### Matched comparison figures

- **Sine:** four multiplier columns; trained relative L2, refitted relative L2, and linear mean gamma. Purple solid is the original VarPro split; teal dashed is the current-readout split. Axes match within each row.
- **Mixed sine:** the same two methods, columns, quantities, and common cosine schedule.
- **Runge:** the same layout, with both error rows evaluated on identical training samples.
- **Gaussian envelope:** the same layout and finite-interval training objective as the other functions.

## Additional details

At initialization $\|h\|$ ranges from roughly $3\times10^{-16}$ to $6\times10^{-15}$. For a first-step coordinate much smaller than Adam's epsilon, the amplified update is approximately $-\eta\mu h_k/\epsilon_A$. Multiplying after Adam does not remove that attenuation. Some initial signals are at the variation seen across SVD implementations; the experiment does not establish an exact zero derivative. The literal projected current residual differs from the refitted residual by at most roughly $7\times10^{-14}$ in these runs.

Six new tests verify the current-readout VJP, decomposition, zero-readout behavior, and independent multistep Adam updates; seven inherited Adam tests pass. Saved initial states and first readout updates match the controls. These checks establish what was implemented, not the numerical robustness of every tiny projected signal.

## Conclusions

In the 500-step sweep, the current-readout split does not reproduce expD31's rapid VarPro-driven geometry improvements. The 10,000-step extension establishes later geometry improvement at the smallest tested multiplier, but stronger amplification often produces worse approximation and every split run ends with worse trained-model error than ordinary Adam.

## Open questions

- How much of the difference is due to the current versus solved readout weighting, versus the numerical resolution of the two signals?
- Can smaller outside multipliers in the original VarPro split retain its early approximation improvements?
