> Supplement update (November 13, 2026): the displayed chirp is replaced by the earlier 15-block inverse square root on [1e-4,1]. See [table2.md](table2.md). This added curve uses direct multiscale construction with a C17 extension, not the sample-fitted uniform recipe; the original chirp data below remain archived results.

> Update (November 12, 2026): the current main figure uses the verified NPU Adam-to-L-BFGS results in [npu_adam_lbfgs/table1.md](npu_adam_lbfgs/table1.md). Adam and QUILL results are unchanged. The original Adam-only protocol and results are documented below.

# Hard analytic functions — completed

## Findings

- First figure: four functions, six matched widths, Adam and NPU trained for exactly 50k total steps, plus tuned sample-fitted QUILL.
- Second figure: QUILL alone on eighteen functions: the original sixteen, shifted Gamma, and the compact bump. No additional Adam/NPU training is part of this figure.
- QUILL tuning stays within the uniform tanh-feature/least-squares framework. Pivoted rank-revealing QR and a smaller explicit cutoff materially improve the numerical floor.

## Question

Can ordinary sample-fitted QUILL achieve approximately 1e-14 relative error on challenging analytic special functions, implicit functions and oscillatory quotients, and how do trained networks compare on four representative cases?

## Procedure

The target catalog starts from the recent 24-case study in “Rebuttal: valid theory.” Airy, erf, chirp and localized oscillation are retained verbatim. Nonanalytic cases are excluded. New cases add implicit inversion and special-function/quotient examples. All targets are on [-1,1], and all have analytic continuation across the real halo. That older study was not a QUILL difficulty ranking.

The first figure uses W=105,169,297,559,1089,2141. At each shared width every method receives the same max(2003,4(N+1)) covering sample values. No derivatives or exterior target labels are used. Adam/NPU settings come from saved validation-ranked pilots; there are no new baseline hyperparameter sweeps. Continue the 10k optimizer/RNG states through 50k total updates. Keep the rate constant through 25k, then cosine-decay to 1e-6. Batch size 100, Float64, seed 0, standard target-independent initialization. No L-BFGS, final readout refit, or extra seeds are included in the final baseline comparison.

QUILL uses the audited fit_qi implementation: uniform centers, tanh features, and rank-revealing least squares. Tune lambda within [.2,.3], explicit cutoffs 1e-14/1e-15/machine epsilon, and gelsd/gelsy/gelss. Halo comparisons hold total W and the training points fixed. Selection uses validation only. The second figure adds W=4225 to resolve the particularly sharp elliptic-E target. Stop further tuning once that validation error reaches the requested range.

Physical-dx relative L2 is the common metric. NPU validation and scoring refine logarithmic panels around possible input zero crossings; this resolves spikes that ordinary quadrature can miss. Re-ranking existing pilots changes neither the trained weights nor their optimization state. Independent final quadrature is finer than validation.

## Main comparison

Function and signed-residual rows show each method's best validation-selected fit among the six shared widths. The scaling row shows all six comparisons. Selected widths are documented in the results table, and training sample counts grow with width.

| Function | Method | Width | Relative L2 |
|---|---|---:|---:|
| airy | QUILL | 2141 | 6.1262e-15 |
| airy | Adam | 297 | 0.73724 |
| airy | NPU | 105 | 0.7353 |
| kepler | QUILL | 2141 | 3.4961e-15 |
| kepler | Adam | 297 | 0.00052648 |
| kepler | NPU | 559 | 0.0013677 |
| elliptic_k | QUILL | 1089 | 6.6878e-16 |
| elliptic_k | Adam | 169 | 0.0014473 |
| elliptic_k | NPU | 559 | 0.010113 |
| chirped_division | QUILL | 559 | 1.6439e-15 |
| chirped_division | Adam | 297 | 0.10224 |
| chirped_division | NPU | 559 | 0.10349 |

## QUILL-only suite

Best validation-selected model from the displayed width sweep; independent test error shown.

| Function | Width | Solver | Lambda | Relative L2 |
|---|---:|---|---:|---:|
| airy | 2141 | gelsy | 0.25 | 6.1262e-15 |
| kepler | 2141 | gelsy | 0.275 | 3.4961e-15 |
| elliptic_k | 1089 | gelsy | 0.2 | 6.6878e-16 |
| chirped_division | 559 | gelsy | 0.2 | 1.6439e-15 |
| erf | 169 | gelsy | 0.225 | 4.0298e-15 |
| chirp | 1089 | gelsy | 0.25 | 6.2258e-15 |
| localized | 559 | gelsy | 0.2 | 4.2011e-15 |
| dawson | 297 | gelsy | 0.2 | 1.3741e-15 |
| bessel0 | 559 | gelsy | 0.25 | 2.7287e-15 |
| bessel6 | 559 | gelsy | 0.25 | 2.9998e-15 |
| sine_integral | 559 | gelsy | 0.275 | 9.4982e-16 |
| voigt | 559 | gelsy | 0.25 | 1.2835e-15 |
| kummer | 559 | gelsy | 0.225 | 1.2478e-15 |
| elliptic_e | 4225 | gelsy | 0.225 | 3.0216e-15 |
| fresnel | 1089 | gelsy | 0.25 | 2.1933e-15 |
| log_ratio | 559 | gelsy | 0.25 | 1.3872e-15 |
| gamma_shifted | 1089 | gelsy | 0.25 | 6.6291e-15 |
| smooth_bump | 2141 | gelsy | 0.25 | 7.6021e-14 |

The added Gamma target is Gamma(0.05+2.475(x+1)), as in the earlier 24-function study. It is analytic on the requested interval; its nearest real pole is just outside it. Halo features do not require exterior labels for sample-fitted QUILL.

The compact bump is exp(-1/(1-(x/0.35)^2)) for |x|<0.35, and zero otherwise. It is C-infinity but **not analytic** at the support boundaries. It is an additional smooth-function example, not evidence for an analytic-only theorem.

Both additions use the same seven displayed widths, automatic halo, lambda=.25, gelsy, and machine-epsilon rank cutoff. The fourteen measured models and independent quadrature checks are in quill_extension_verified.json; selected dense predictions and reference checks are in gamma_bump/.

## Verification

18 implementation and NPU regression checks passed. The training resume audit is bitwise exact. Refined NPU pilot ranking preserves optimizer/model/RNG states exactly. Target values were independently checked against 70-decimal mpmath; derivative checks audit the target implementation only and supply no training information.

Reloaded and checked 160 saved scalar networks with ordinary Float64 Torch inference. Five low-width QUILL fits have <=0.04% relative spread across integration rules. All selected high-precision fits and all trained baselines pass the stricter check. Raw flags are retained. The selected-fit reference spot audit, shared training-grid hashes, actual step counts and final learning rates are saved alongside the report.

## Interpretation and limits

These are measured results for a declared finite training budget, not optimizer impossibility claims. The NPU family includes native, positive-input complex, and compositional candidates; the compositional candidate has more parameters at the same hidden width. All settings and parameter counts are saved. One seed is used. The function list was fixed before observing trained outcomes.

Analyticity does not imply uniform difficulty. The elliptic-E composition needs more resolution than most other cases. Near the floating-point floor, reference evaluation and matrix arithmetic matter; quadrature checks and high-precision reference spot checks bound the practical interpretation. The method remains a uniform feature bank plus least squares, with no target-specific nonlinear input features or multiscale partition.

## Artifacts

- hard_functions_3x4.png/.pdf: four-function comparison.
- additional_functions_1x2.png/.pdf: two-column function key and QUILL scaling on all eighteen functions.
- main50k/: the 48 trained runs; quill_tuned/: tuning candidates; predictions/: plotted arrays.
- scaling.csv, verified_summary.json, selected_fits.json, verification.json, reference_spot_audit.json, figure_provenance.json.
- specification_main50k.json, target_audit.json, training_audit.json, refinement_state_audit.json, source snapshot and environment versions.

The superseded 100k plan, its pilots, and any completed old full runs are retained for provenance but excluded from the final figures. The initial derivative-based prototype is also archived and excluded.

## Next decision

The requested two figures and saved data are complete. No additional baseline training is pending.
