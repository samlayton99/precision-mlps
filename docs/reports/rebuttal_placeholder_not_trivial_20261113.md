# Rebuttal placeholder audit: R1, R4, R5, R6

Read-only audit of saved experiments, 2026-11-13. No experiments rerun and no manuscript edited. Paths below are relative to the precisionMLPs repository. Appendix/figure numbers in PDF57 are **unknown here**; do not infer them from filenames.

## R1: four hard functions

Authoritative results: `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/npu_adam_lbfgs/selected_fits.json`, `selected_comparison.json`, `table1.md`, `display_selection.json`, `figure_caption.txt`. The draft's four numeric rows match these current results.

| Target | Adam W / parameters | Gated NPU W / parameters | QUILL W / parameters |
|---|---:|---:|---:|
| Airy | 297 / 892 | 105 / 532 | 2141 / 6424 |
| Kepler | 297 / 892 | 105 / 317 | 2141 / 6424 |
| Elliptic K | 169 / 508 | 105 / 317 | 1089 / 3268 |
| Oscillatory quotient | 297 / 892 | 169 / 509 | 559 / 1678 |

- Six shared candidate widths: **105, 169, 297, 559, 1089, 2141**. This is width matching, not equal parameter counts, and table winners need not have the same width. Scalar tanh MLP has 3W+1 stored parameters. Native complex gated NPU followed by biased linear readout has 3W+2; Airy's selected composed model has a learned two-channel affine input map, NPU(2,W), and biased linear readout, totaling 5W+7.
- **Float64 construction/fitting, training, and inference** for this suite. QUILL uses fixed uniform tanh features and function-value least squares; no derivative training or exterior labels. 70-digit reference spot checks are verification, not the arithmetic used to fit/infer these networks.
- Baselines: **50,000 Adam updates**, minibatch 100 sampled with replacement; constant rate for 25k, then cosine decay to 1e-6 at 50k. Gated NPU additionally gets **up to 1,000 full-batch L-BFGS iterations**, strong-Wolfe line search, no readout refit. These are iterations, not closure calls. Selected NPU attempted iterations: Airy 11 (best 10), Kepler 1000, K 1000, quotient 964 (best 960; next step encountered nonfinite loss and retained the earlier valid checkpoint).
- Training grid at six widths respectively: **2003, 2003, 2003, 2052, 4100, 8196 uniform endpoint-inclusive points on [-1,1]**. Same function-value locations at each matched width, not same count across the independently selected table winners. All selected trained table rows use 2003 points. Selected QUILL rows use 8196, 8196, 4100, 2052.
- Adam rates/seeds, in row order: .03/0, .01/0, .01/0, .03/0; all selected Adam initializations PyTorch default. NPU: composed .01/0, native .005/1, native .005/0, native .01/0. Original 48-run six-width comparison used seed 0; later native NPU W105 candidates used seeds 0 and 1. **No mean or median over seeds is reported.**
- Selection: Adam and QUILL use smallest validation L2 among candidate widths/settings. NPU starts from the validation-best saved candidate at each width, retains validation-best L-BFGS checkpoints, then the displayed model is chosen across widths by **smallest sampled maximum absolute residual**, on 4097 uniform points plus exact input-zero locations and logarithmic probes. This post-audit rule was requested to avoid narrow spikes. The reported NPU L2 is therefore **not minimum achieved L2**. Do not describe all three columns as best L2 across runs.
- Error is physical-domain relative L2, sqrt(integral residual squared / integral target squared). Adam/QUILL final values use **16,384 nodes** (64 panels x 256 Gauss-Legendre nodes), cross-checked at 8192 nodes. NPU final values use 256-node quadrature on panels additionally refined around actual input zeros: **50,688 nodes for Airy; 34,816 each for Kepler, K, quotient**, cross-checked at half order. These counts were recalculated from the saved selected checkpoints without training. Validation is lower-order independent quadrature. Dense plotting/audit points are not the L2 quadrature rule.
- Target definitions in the draft are correct; K is the complete elliptic integral of the first kind with **parameter m**, not modulus k. Kepler is the unique real root (derivative >= .1).

Sources: `experiments/expK06_hard_analytic_suite/{common.py,targets.py,train.py,models.py,verify.py,integration.py,npu_lbfgs.py,finalize_npu_lbfgs.py}` and `results/.../expK06_hard_analytic_suite/specification_main50k.json`. The original specification predates L-BFGS; use current NPU protocol/caption for the update.

Authoritative figure: `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/npu_adam_lbfgs/hard_functions_3x4.png` (and PDF). Top-level copy is also regenerated, but the NPU subdirectory makes the revision explicit.

## R4: default / QUILL initialization

- Each cell is the **median of 361 unsmoothed held-out error measurements in one seed-0 trajectory**, logged every 25 updates from **1000 through 10,000 inclusive**. It is not a median across seeds, not a median across evaluation points, and not all 9001 update errors.
- Domain [-1,1]; targets x², sin(2πx), and sin(2πx)+.5sin(6πx)+.25sin(14πx).
- Width **269**, **808 parameters**, one hidden tanh layer, Float64 training/inference. Default is unmodified PyTorch Linear initialization, not Xavier.
- **10,000 full-batch Adam updates** on 1024 uniform midpoint training points. Evaluation uses a disjoint 2048-point uniform midpoint grid; independent 2048-point Gauss-Legendre checks only at initial/final endpoints. Adam betas (.9,.999), epsilon 1e-8, no weight decay.
- Warmup: eta_t=eta_peak min(t/1000,1), t starts at 1; first rate is eta_peak/1000. Peak held constant after warmup. First update magnitude is not necessarily eta_peak/1000 because Adam also normalizes gradients.
- Initial QUILL midpoint relative L2 errors, in table order: **1.0883234156777923e-15; 1.8550908189661897e-15; 7.206223094969709e-14**.
- QUILL initialization is a direct cardinal construction using function/derivative samples, **30-decimal mpmath offline construction**, N128, halo70 per side, Kc160, lambda .25. Exported parameters and subsequent optimization/inference are Float64. This is distinct from R1's function-value least-squares recipe.
- Full experiment contains 11 rates (1e-2 through 1e-12), 3 functions, 2 initializations: 66 trajectories. Table shows seven selected rates, with no further aggregation across rates. Figure smoothing does not enter table values.

Sources: `experiments/expD55_qi_adam_drift/{config.json,run.py,expanded_sweep.py,plot_comparison.py}`; `src/data/targets.py`; `results/checkpoint_D_optimizers/expD55_qi_adam_drift/warmup1000/{square,sine,sine_mixture}/initial.json`; expanded sweep `manifest.json`, `summary.csv`.

Authoritative figure: `results/checkpoint_D_optimizers/expD55_qi_adam_drift/expanded_lr_1e-12_10k/comparison_2x3.png` (and PDF).

## R5: additional functions

**Draft is stale:** retain erf and replace the final chirp row by **1/sqrt(x), W4815, relative L2 5.0e-14**. Latest copyable table: `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/table2.md`. The corresponding figure has 18 curves: four main functions plus 14 additional functions. Inverse root also appears in R6, so R6 is a methodological elaboration of the same result, not another unique target.

- All domains [-1,1], **except inverse root on [0.0001,1]**.
- The 13 other rows use Float64 sample-fitted QUILL; select **smallest validation L2 across the saved width/configuration sweep**, report independently evaluated test L2. Candidate widths 105,169,297,559,1089,2141,4225. These are not minimum widths satisfying a preset tolerance. No random-seed aggregation for deterministic construction.
- Final L2 for those 13 rows uses 16,384 Gauss-Legendre nodes on 64 panels, checked at 8192; validation uses 2048. Training samples follow max(2003,4(N+1)), up to16388 at N4096/W4225. Halo tuning can change construction N while holding width and original training grid fixed.
- Inverse-root row is a **separate previously selected 15-block direct cardinal-QI construction**; not validation-minimum over the new suite, and not raw uniform sample fitting. See R6.
- Bump: exp(-1/(1-(x/.35)^2)) when |x|<.35, zero otherwise; C-infinity but **nonanalytic** at +/- .35. Do not invoke the analytic theorem for this row.
- S is Fresnel sine integral S(z)=integral_0^z sin(pi*t²/2)dt. D is Dawson integral exp(-z²) integral_0^z exp(t²)dt. Si is sine integral. J0/J6 are Bessel first kind, orders0/6. V uses Gaussian sigma=.025 and Lorentzian gamma=.01. E uses elliptic parameter m. Kummer is 1F1(1/2;1;-150x²). erf is error function, not the Gaussian CDF itself. Gamma argument is2.475x+2.525. All formulas are preserved in current table2.md and targets.py.

Sources: `experiments/expK06_hard_analytic_suite/{targets.py,report.py,add_gamma_bump.py,verify.py,plot.py}`; results `verified_summary.json`, `quill_extension_verified.json`, `gamma_bump/selected.json`, `inverse_sqrt_multiscale_import.json`. Verified the listed table widths match validation minima for the 13 sample-fitted rows.

Authoritative figure: `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/additional_functions_1x2.png` (and PDF), **linear width axis**, latest chirp-to-inverse-root replacement. Current methodological caption is `supplementary_caption.txt`.

## R6: inverse root

- **4815 hidden neurons =15*(256+2*32+1)**. Scalar ordinary tanh MLP contains **14,446 scalar weights/biases (3W+1)**, excluding fixed input-normalization metadata. Fifteen blocks are concatenated into one hidden layer.
- Domain [1e-4,1], centered at .50005 with scale .49995. Fifteen dyadic scales s_j=(1e-4/2)*2^j, j=0,...,14. N256 per block, halo32 each side, lambda .25, Kc160, cutoff order16 giving C17 extension. Direct cardinal construction uses derivative samples of a shared prototype; no Adam, LS fitting, stochastic seeds, or standard training-set size.
- **30-decimal mpmath cardinal-stencil construction; Float64 derivative/extension evaluations and exported network inference.** Do not imply every offline operation is 30 digits. Float64 NumPy value in current table **4.9531453618859716e-14**, rounded5.0e-14; refined quadrature gives5.011666100943239e-14. Approximately1e-13 is conservative but less precise.
- **3584 quadrature nodes =14 logarithmic panels x256**, cross-check **7168 nodes =14x512**. This is physical-dx relative L2, not a log-domain metric: f²dx=d(log x), allowing the quadrature to integrate relative-error squared in log x with normalization log(1/eps).
- Selected W4815 is the previously verified construction, not the lowest empirical error in the width sweep. Current figure imports20 saved points in the historic displayed W1000–5000 window. Do not describe its width as validation-minimum or a theoretical minimum.
- Raw uniform standard-halo sample-fitted rerun is a different experiment, `expK07_raw_inverse_sqrt`: W4225 reaches1.0753e-4, **not** the5e-14 used here. Do not substitute its recipe into R6.

Sources: `results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension/{selected_config.json,verification.json,scaling_1k_5k.json}`; `experiments/expK02_inverse_sqrt_extension/{construction.py,verify.py}`; K06 `inverse_sqrt_multiscale_import.json` records source/model hashes and the reloaded metrics. Standalone historic figure: `results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension/inverse_sqrt_scaling.png` (and PDF); current integrated curve is in the R5 figure above.

## R3 / R7 / appendix numbering

Not owned or verified in this chat. No trustworthy mapping of their reported numerical rows, experiment budgets, or owners established here. Leave these placeholders **unknown pending the corresponding experiment owner**. Likewise PDF57 appendix/figure numbering has not been inspected here. Do not guess.
