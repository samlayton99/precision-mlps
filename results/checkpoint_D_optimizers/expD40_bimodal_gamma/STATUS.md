# D40 coordinator record — complete

Owner: current Codex task. 2026-09-29. Sam requests a bimodal distribution of actual weight-row L2 norms, with very low and high gamma neurons, to test improved fitting without lost validation performance. No historical results or training code will be overwritten.

## Frozen first-stage design

Use two width-512 tanh hidden layers, the unchanged D38 Adam/LR/batch/schedule/data recipes, and paired seed-specific readout and batch RNG. Four diagnostic tasks: Airfoil, Kin8nm, corrected SARCOS and Superconductivity. Forty new seed-0, 10k-step runs: five gamma shapes in each of two layer scopes. Reuse D39's four centered-reference pilots after identity/data checks.

Build the complete D39 corrected centered-bank reference first. For each layer, let g be its median row norm. Retain its unit directions and zero-crossing centers across every D40 arm. Modify each row by scaling its weight AND bias. Shapes: all .1g; all g; all 3g; all sqrt((.1^2+3^2)/2)g; and exactly half .1g/half 3g. The RMS arm matches total squared weight norm to the mixture, not bias, preactivation, activation or derivative energy. Gamma here means actual row norm, so the mixture has two literal peaks, not merely two values of gamma times center spacing. These are multiscale hypotheses, not the single-band QI construction.

Scopes: both hidden layers; or second hidden layer only. The latter is the primary controlled distribution-shape comparison because first-layer features are unchanged. In the both-layer arm, second-layer centers stay fixed to the common reference; changes in the upstream activation distribution are an explicit interaction. No arm recalibrates centers after changing gamma. Readout weights are identical across arms; initial output variance is recorded, not normalized away.

Mode assignment alternates along the center grid within each bank, with independent randomized phase and exactly balanced total counts. Odd-size banks split evenly between floor/ceiling high counts. The assignment uses a separate RNG and cannot alter directions, reference geometry, readout or minibatch sequence. All parameters train freely afterward; no gamma freezing or persistent mode constraint.

## Selection and confirmation plan

Screen only train and validation. Select the better of the two mixture scopes per task by minimum validation MSE on the common schedule, whether or not it beats a control. Lock these identities and screening/source hashes before held-out confirmation. For each task confirm the selected mixture, its corresponding RMS control, the best single-scale control within that scope, and the centered reference, deduplicating identical controls. All confirmations use 20k steps and seeds 0/1/2. Also compare preserved D38 standard/original-QI controls and D39's prior task-specific recipes as historical context, without retuning on test errors.

Report fitting and validation at the SAME checkpoint: both final endpoints and each run's validation-selected checkpoint. Show whole training/validation trajectories, paired seed changes, and test errors for the locked choices. No arbitrary noninferiority margin or significance claim. Record diagnostic LS readouts and frozen initial-feature ridge in confirmation. Reconstruct initial/final gamma distributions by original mode labels, own-row cosine, saturation and low/high contribution ablations. If a further mixture range or fraction is explored after this screen, record it as a separate adaptive stage before its runs.

## Implementation checks and cost

Tests must verify exact physical row norms and counts; paired directions, centers, readout and untouched layer 1; RMS weight-norm equality; deterministic independent masks; and finite forward/backward updates. Training uses the existing one-forward/one-backward Adam step with its usual O(P) state. The only new work is O(P) initialization and bounded offline diagnostics. No new optimizer, per-example persistent state, loss-based control, or precision-floor claim is introduced. LS remains observational and does not feed training. The already checked scalar constructor is unchanged.

## Earlier progress

Design recorded before new training. Twenty-two D38/D39/D40 tests pass, including the four new paired geometry/mode tests. Two short real-data runs completed with finite gradients/losses. The centered reference reproduces the saved initial D39 train/validation errors on all four tasks to 1e-12. An independent implementation review found no blocking issue; a half-mixture configuration guard was added before the campaign. Forty validation-only runs are running in four independent shards. Training sources are frozen in implementation_manifest.json.

The separately recorded followup_plan.md adds sixteen runs for a 10:1 mode separation, with the same low mode and high gamma reduced to the reference median. Only mixtures and their own RMS controls are new; the pure endpoints already exist. This adaptive extension was specified after Airfoil and the first two Kin8nm pilots, before its own training or test evaluation. Both mode ranges enter the same validation-only selection; no original training source was changed. Initial diagnostics verify twelve reference/mixture reconstructions across all four tasks and plot actual physical row norms.

## Confirmation launch

All 56 new screening runs are complete. The four reused centered references also pass strict identity, source, schedule and data checks. Selection is locked in selection.json, including all 60 screening-file hashes and both training source manifests. Forty-five confirmation runs cover the following mixtures and their preselected centered/single-scale/RMS controls, seeds 0/1/2, 20k steps:

| Task | Selected mixture | Best homogeneous control in that scope | Mixture / homogeneous screening validation MSE |
|---|---|---|---:|
| Airfoil | .1g/g in layer 2 only | g in layer 2 only | 1.044 |
| Kin8nm | .1g/g in both layers | Matched RMS .711g in both | 1.045 |
| SARCOS | .1g/g in layer 2 only | g in layer 2 only | 1.019 |
| Superconductivity | .1g/3g in both layers | 3g in both | .976 |

The milder range avoids much of the wide mixture's validation penalty. The only mixture beating its best homogeneous control in the same scope in screening is Superconductivity; allowing the homogeneous control to choose either scope removes even that lead. This is a one-seed exploratory result, not the completed confirmation conclusion. No additional training choices will use confirmation test scores.


## Completed 2026-09-29

All 56 new pilots and 45 confirmations completed with their final checkpoint files. The 45 new confirmations plus 33 preserved historical controls supply 78 comparison records and 780 valid observational least-squares measurements. All frozen training source, screening, selection, identity and data checks passed. The 22 focused implementation tests passed before the campaign; only analysis/plot/report files changed afterward.

Bimodal validation-selected mean errors are higher than the matched single-scale RMS controls on all four tasks: validation +4.1%, +3.5%, +1.1%, +3.4%; test +4.1%, +4.9%, +2.3%, +5.3% for Airfoil, Kin8nm, SARCOS and Superconductivity. The mixture loses all 12 paired test comparisons and 10/12 validation comparisons. Common-20k comparisons preserve the same mean validation/test ordering; fitting differences versus RMS are within 2%. No further candidate was chosen from these results.

All 45 trained confirmations improve over their frozen initial-feature ridge controls on validation and test at validation-selected checkpoints. Unregularized final LS worsens validation in 44/45 despite improving fitting. Readout diagnostics never affect training.

Final seed-0 geometry diagnostics passed independent checkpoint and initial/final train/validation prediction reconstruction. Original low/high norm ranges remain separated in all six mixed layers. Low groups rotate and move more relative to their initial norms; high groups move farther in absolute weight distance. The low groups can become saturated as bias and upstream features train. Fixed-readout mean-replacement probes are reported only as sensitivity checks.

Final outputs: expD40_results.md, confirmation_summary.json, verification_summary.json, gamma_diagnostics_final.json, mode_diagnostic_note.md, and screening/confirmation/readout/feature-learning/gamma figures. No training or analysis remains running. Existing historical and unrelated work was preserved; no commit or publication was made.
