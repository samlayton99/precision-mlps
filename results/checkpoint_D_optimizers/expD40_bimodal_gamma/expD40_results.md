# expD40: bimodal gamma initialization — Status: data-obvious, complete

## TL;DR

- Completed 56 validation-only pilots and 45 three-seed, 20k-step confirmations using two width-512 tanh hidden layers. Tested equal low/high groups at both 10:1 and 30:1 gamma separation, in layer 2 alone or both layers.
- Selected mixtures have higher mean validation and test MSE than their single-scale, matched-weight-norm controls on all four tasks. Test error is higher in all 12 seed pairs; final-step comparisons retain the same mean ordering.
- The low/high row-norm groups remain fully separated after unconstrained training in the four audited seed-0 models. Low rows rotate and grow more relative to their starting size, while high rows move farther in absolute weight distance.
- The models learn features: all 45 new confirmations beat their frozen-initial-feature ridge controls on validation and test at their validation-selected checkpoints. The negative comparison concerns bimodality, not a failure to use the MLP.

## Question / hypothesis

Can broad, low-gamma features and sharper, high-gamma features complement each other during ordinary training, improving fitting without the validation deterioration seen with uniformly large bandwidth in D39? A successful mixture should be compared against both its component scales and an intermediate single scale, not merely against the historical initializer.

## Experiment design

Start with the complete D39 corrected centered-bank reference: two width-512 tanh hidden layers, 24 banks of 21/22 endpoint-inclusive centers, train-input projection quantiles, exact gamma-times-spacing .25, and the same seed-specific Xavier readout. Construct both reference layers before changing any row. In each layer define $g$ as the median reference row norm. Preserve the reference unit directions and zero-crossing centers across every arm. To set row norm $\gamma_i$, scale both weight and bias by $\gamma_i/\|w_i\|_2$; this preserves the hyperplane $u_i^Tx=c_i$.

The five tested gamma distributions are all-low $.1g$, all-middle $g$, all-high $3g$, a 50/50 mixture of $.1g$ and $3g$, and the uniform RMS control $\sqrt{(.1^2+3^2)/2}\,g$. Both mixture modes are spread along each direction bank by alternating centers with randomized phase; the odd-size banks allocate the extra high neuron in exactly half the banks. A separate RNG gives exactly 256 rows per mode without changing reference geometry, readout or minibatch draws. These are literal physical row norms, not two values of the dimensionless QI bandwidth.

The RMS control matches total squared weight norm exactly. It does not match bias norm, preactivation or activation variance, or derivative statistics; those are separately recorded. With these well-separated modes, high-gamma rows carry about 99.9% of the mixture's squared weight norm. Both arms still have the same number of trainable parameters. All parameters train freely, so the final distribution need not remain bimodal.

Each shape is tested in two scopes: only hidden layer 2, or both hidden layers. Layer-2-only is the cleaner distribution-shape comparison because its input features are identical at initialization. In the both-layer comparison, changing layer 1 also changes the distribution seen by layer 2, while its centers remain those of the common reference. This is an explicit interaction, not a claim of matched second-layer activation statistics. Initial output variance is recorded; readout rescaling is not introduced.

The first stage is forty seed-0, 10k-step runs on Airfoil, Kin8nm, corrected SARCOS and Superconductivity. D39's four centered-reference pilots are reused after checking recipe/data identity and reproducing initial predictions. Only fitting and validation errors are evaluated. All arms use D38's fixed per-task learning rates, fp64 Adam, batch size 256, no weight decay, a flat first 20% of the budget then cosine decay to 1% of the initial learning rate, and the same replacement minibatch sequence. Architecture, precision and training recipes are unchanged.

A recorded adaptive second stage adds sixteen runs with the same low mode $.1g$ and milder high mode $g$: mixture and matched RMS control $\sqrt{(.1^2+1^2)/2}\,g$, each in both scopes on all four tasks. The all-low and all-high endpoints are already present as the first-stage low and middle controls. This extension was specified after the first task and two Kin8nm arms had completed, before any second-stage run or new test evaluation. It tests whether the very strong high mode, rather than mixing itself, explains a poor outcome. Both ranges and all their outcomes are retained.

Per task, the best of the four mixture configurations (two mode ranges by two layer scopes) is selected by minimum validation MSE over the common checkpoint schedule. Its matched RMS control, best single-scale control in the same scope, and centered reference are included in confirmation, with duplicate controls removed. The homogeneous selection includes the milder RMS control. The mixture is retained for confirmation even if it loses the screen. Choices, all 56 new screening outputs and training source hashes are locked before the new test evaluation. Confirmation uses 20k steps and seeds 0/1/2; preserved standard/original-QI and prior D39 task-specific results supply historical context.

The fitting and validation errors are reported from the same model state: either the final checkpoint or each run's validation-selected checkpoint. An earlier best validation score cannot be combined with later, lower fitting error and described as one improved model. No noninferiority threshold or significance test is claimed. Three seeds measure initialization/minibatch variability on one fixed split; repeated exploration of the same validation split remains exploratory.

The single-scale winner is conditional on the mixture-selected layer scope, not the best homogeneous initializer across all scopes. Confirmation reuses the validation examples, and seed 0 was part of screening; its validation results are descriptive replications under a longer schedule, not independent validation evidence. Seeds 1/2 add new initialization/minibatch repetitions.

The benchmark test splits have also appeared in earlier experiments. Locking the new choices before evaluating their test performance prevents choosing among these candidates using new test scores; it does not make the longstanding benchmark a newly blinded dataset.

Confirmation records the actual trained readout, observational centered-SVD least squares at cutoff $10^{-12}$, validation-selected ridge on frozen initial features, and input affine regression. Diagnostic solves never update the model or optimizer. Saved final models permit original-group gamma/cosine/saturation measurements using train inputs and trace-reproduction checks on train/validation only.

**Code & data**

- Code: experiments/expD40_bimodal_gamma, using the unchanged D38 engine and D39 reference factory; tests/test_expD40_bimodal_gamma.py.
- Design and provenance: config.yaml, STATUS.md, implementation_manifest.json; selection.json was locked after the complete screen.
- Traces and final checkpoints: data/pilot, followup/data/pilot and data/compare; short sanity runs live separately in smoke.
- Analysis and provenance: confirmation_summary.json, verification_summary.json, gamma_diagnostics_final.json and mode_diagnostic_note.md in this output directory. Diagnostics are seed 0 at the final checkpoint; summary performance uses all three seeds.
- Principal performance figures under figures/: matched_scale_generalization.png, confirmation_train.png, confirmation_val.png, confirmation_test.png, paired_fit_validation_selected.png, paired_fit_validation_final.png, readouts_train.png, readouts_val.png, readouts_test.png and feature_learning.png.
- Screening figures under figures/: screen_validation.png, screen_validation_strong.png, screen_mode_ranges_both.png, screen_mode_ranges_last.png, screen_both_trajectories.png and screen_last_trajectories.png.
- Final geometry figures under figures/: gamma_movement_{task}.png and gamma_direction_cosines_{task}.png for each of the four task identifiers, plus group_movement_summary.png, group_displacement.png and group_mean_replacement_sensitivity.png. The separate files ending in _pilot.png describe 10k pilots; initial gamma and saturation figures describe the original 30:1 construction.

## Results

The 22 focused D38/D39/D40 tests passed. All 101 new training runs completed with saved checkpoints. The final audit checked 78 comparison records including historical controls, their finite errors, and 780 observational least-squares measurements; every solve lowers or preserves fitting error within tolerance. Training sources and the screening/selection hashes remain unchanged. Initial and final predictions of the four selected seed-0 models were independently reconstructed before the geometry audit.

The original 30:1 mixture worsens screening validation substantially on Airfoil and Kin8nm; their both-layer errors are 53% and 78% above the centered reference. Reducing the high mode from $3g$ to $g$ removes much of this penalty. The selected mixtures use $.1g/g$ in layer 2 on Airfoil/SARCOS, $.1g/g$ in both layers on Kin8nm, and $.1g/3g$ in both layers on Superconductivity. Each task has a homogeneous configuration somewhere in the full screen that beats its best mixture; confirmation retains the predeclared same-scope controls rather than substituting broader winners.

The decisive longer-run comparison is against the matched single scale: $.711g$ for the mild mixtures and $2.123g$ for the wide mixture, in exactly the same layer scope. The following are means over three seeds, with each run evaluated at its own validation-selected checkpoint. A positive change means the mixture has higher error.

| Task | Bimodal validation MSE | Matched single-scale validation MSE | Validation change | Test change |
|---|---:|---:|---:|---:|
| Airfoil | 0.043701 | 0.041987 | +4.1% | +4.1% |
| Kin8nm | 0.061952 | 0.059876 | +3.5% | +4.9% |
| SARCOS | 0.010882 | 0.010759 | +1.1% | +2.3% |
| Superconductivity | 0.099845 | 0.096602 | +3.4% | +5.3% |

The single-scale control has lower test error in every one of the 12 paired runs and lower validation error in 10/12. These are observed seed comparisons, not a significance claim. Relative to the centered reference, the mixture improves mean validation by 2.7% on Airfoil and 7.1% on Kin8nm, but worsens it by 0.9% on SARCOS and 2.9% on Superconductivity. The Airfoil/Kin8nm improvements therefore do not establish a benefit from two modes: their matched single scale does better still. The small Superconductivity screening lead over all-high initialization does not persist in mean confirmation validation.

At the common 20k endpoint, mixture/control fitting MSE ratios are nearly one, while validation and test remain worse on every task:

| Task | Fitting ratio | Validation ratio | Test ratio |
|---|---:|---:|---:|
| Airfoil | 0.999 | 1.037 | 1.069 |
| Kin8nm | 0.996 | 1.052 | 1.041 |
| SARCOS | 1.015 | 1.012 | 1.022 |
| Superconductivity | 0.982 | 1.035 | 1.051 |

Validation-based stopping changes fitting comparisons, especially on Airfoil: its selected mixture has 3.95 times the selected RMS control's fitting error, whereas their final fitting errors are virtually equal. Airfoil also illustrates why the two checkpoint conventions must remain separate: versus the centered reference, its mixture's selected test error is 4.8% lower, but its final test error is 15.4% higher. No earlier validation minimum is paired with a later fitting minimum.

Historical controls remain relevant. D39's softer Kin8nm banks have selected test MSE .06031, compared with .06276 for the new RMS control and .06586 for the mixture. D39's second-layer-only SARCOS has .01095, compared with .01155 and .01182. These experiments do not establish a better task recipe than those prior choices.

All 45 new trained models beat their own validation-selected ridge fit on frozen initial features on both validation and test at their validation-selected checkpoints. For the selected mixtures, mean test MSE improves over frozen-feature ridge by 3.5–13.4 times and over input affine regression by 3.1–11.4 times. Thus representation learning is active. Conversely, an unregularized final LS readout worsens validation in 44/45 runs despite improving fitting. Superconductivity's wide mixture gives final LS validation MSE 29.10 and 5.68 in seeds 0 and 2, versus trained-head MSE .098 and .103. These ill-conditioned diagnostic solves never enter training or select a new candidate.

The geometry audit uses seed 0 at 20k steps, separately from the three-seed performance comparison. In every mixed layer, the largest final low-group norm is still below the smallest high-group norm. The modes broaden without merging. Superconductivity layer 1 is the strongest example of adaptation: low-row norms grow by a median 6.70 times and have median cosine .201 to their initial directions; high rows grow by 1.112 times and retain cosine .950. Yet the high rows move farther in absolute distance: median $.881$ versus $.538$. This ordering of absolute displacement holds in all six mixed layers; relative displacement has the opposite ordering. Rotation alone does not measure which group learns more.

Small physical gamma does not guarantee a neuron remains unsaturated as its bias and upstream representation change. At 20k, the low group's layer-2 saturated-activation fractions are 35.5%, 0%, 46.4%, and 34.9% across Airfoil, Kin8nm, SARCOS, and Superconductivity, using $|\tanh(z)|>.99$ on the fixed fitting-input probe. Replacing either final layer-2 group's activations by their fitting-data means worsens validation for all four audited predictors, but the low-group effect on Superconductivity is only about 1%. These fixed-readout perturbations establish sensitivity of those fitted predictors; they do not demonstrate that two groups are better than a retrained single-scale model.

### Figures

All trajectory panels use logarithmic gradient steps, logarithmic MSE and a shared upper limit of one. Step zero is omitted from logarithmic axes, and triangles mark clipped values. Shading is the observed three-seed range, not a confidence interval. Gamma histograms share physical norm bins and limits across tasks, with percentages measured against all 512 rows. Originally low/high rows remain blue/orange even after training; an unmixed reference layer is gray.

- **Matched-scale generalization:** validation/test panels show mixture-to-RMS error ratios at each run's validation-selected checkpoint. Dots are paired seeds and bars are ratios of means; the common horizontal line is equal error. All test dots lie above it.
- **Confirmation fitting:** four task panels show trained-head fitting trajectories for standard (blue), centered reference (gray), RMS (purple), selected homogeneous (green) and bimodal (orange). Purple and green coincide on Kin8nm, so green is omitted there.
- **Confirmation validation:** the same four panels, colors and shared scales show the small final differences relative to the full learning process.
- **Confirmation test:** the same layout provides held-out trajectories for the locked candidates, without selecting new configurations from these curves.
- **Selected-checkpoint fitting/validation ratios:** two panels place mixture/control fitting and validation errors at the same selected model states. Gray, purple and green identify the three controls; dots are seed pairs and bars are ratios of means.
- **Final fitting/validation ratios:** the corresponding two-panel comparison fixes every model at 20k, separating stopping effects from endpoint differences.
- **Fitting readouts:** four task panels compare standard blue and bimodal orange; solid lines are learned heads and dotted lines are observational LS. The horizontal gray line is affine regression on inputs.
- **Validation readouts:** the same layout makes the wide Superconductivity mixture's unstable LS validation behavior visible through off-scale markers.
- **Test readouts:** the same layout shows the trained nonlinear models' large improvement over the affine baseline and the absence of a reliable benefit from final LS refitting.
- **Feature learning:** validation/test panels show trained-mixture error divided by the frozen-feature ridge control (purple) or affine control (gray). Every paired point is below one.
- **Complete screening heatmap:** task columns and initialization rows show minimum validation MSE relative to the centered reference, including both mode ranges and both layer scopes. Blue/red indicate lower/higher errors on a common logarithmic color scale.
- **Original screening heatmap:** the same layout isolates the pre-follow-up 30:1 stage; its large Airfoil/Kin8nm penalties motivate the separately recorded milder stage.
- **Both-layer mode ranges:** four task panels compare reference, middle scale, wide mixture, mild mixture and mild RMS validation trajectories; only initialization changes.
- **Layer-2 mode ranges:** the same four-panel comparison leaves initial first-layer features paired, providing the cleaner distribution-shape control.
- **Original both-layer screening trajectories:** four task rows and fitting/validation columns display all five original gamma shapes and the centered reference.
- **Original layer-2 screening trajectories:** the corresponding eight panels retain the common first-layer initialization.
- **Airfoil gamma movement:** a two-by-two layout places layers in rows and initialization/final state in columns; only layer 2 has mode labels.
- **Kin8nm gamma movement:** the same layout shows both mild-mixture layers and their retained norm separation.
- **SARCOS gamma movement:** the same layout leaves layer 1 gray and compares the two layer-2 groups.
- **Superconductivity gamma movement:** the same layout shows large relative movement of originally low layer-1 rows without merging with the high group.
- **Airfoil direction cosines:** left/right panels show layers 1/2, comparing each row only with its own initial direction; the first layer has no low/high labels.
- **Kin8nm direction cosines:** the two panels separate the strong low/high alignment difference in layer 1 from the smaller changes in layer 2.
- **SARCOS direction cosines:** the two panels retain the unmixed first-layer distinction and the original second-layer group labels.
- **Superconductivity direction cosines:** the two panels show the low first-layer group's broad reorientation and the high group's much stronger alignment.
- **Group movement summary:** layers form two rows, with norm growth and direction alignment in the two columns. Tasks lie on each horizontal axis; original-group medians and middle-half ranges distinguish relative growth from retained alignment.
- **Group displacement:** panels separate absolute row-weight displacement from displacement divided by initial norm. Their opposite low/high ordering prevents reading rotation as greater absolute movement.
- **Group mean-replacement sensitivity:** fitting/validation error-ratio panels show the effect of replacing low or high layer-2 activations by their fitting means, using the same fixed readout. This is a representation perturbation, not a refitted capacity comparison.

## Additional details

This mixture deliberately abandons the constant single-bank product $\gamma h=\lambda$. It is a multiscale empirical hypothesis motivated by the prior bandwidth tradeoff, not a more faithful implementation of the scalar constructive theorem. The original and corrected QI controls remain distinct. All data-overlap/split caveats documented in D38 continue to apply: SARCOS uses corrected disjoint splits, while Superconductivity remains a random-row benchmark with repeated-input caveats.

The 20k confirmations restart from the paired initial seeds with their own 20k cosine schedule; they are not continuations of the 10k pilot checkpoints. Geometry changes between those endpoints therefore do not isolate an additional 10k of training under an otherwise fixed schedule. Physical gamma is a row norm, not an activation-scale invariant; layer-2 directions are measured in hidden-neuron coordinates whose upstream representation also trains.

Three seeds share one split, the validation set has been reused for screening, and seed 0 participates in both stages. Results describe the tested 50/50 mixtures, two ranges and fixed training recipes. A different fraction, continuous distribution, center recalibration, bias/output matching or learning-rate retuning would be a different experiment. The matched RMS control equalizes squared weight norm only and cannot isolate every activation or optimization effect of bimodality.

## Conclusions

The tested bimodal initializations retain distinct low/high norm groups but have higher mean validation and test error than the matched single-scale controls on all four tasks. Feature learning remains strong; retaining two gamma populations alone does not predict improved held-out performance in these comparisons.

## Open questions

- Would a smaller fraction of very low-gamma neurons change the tradeoff relative to a matched single-scale control?
- Would controlling preactivation/output scale, in addition to row norms, alter the comparison?
