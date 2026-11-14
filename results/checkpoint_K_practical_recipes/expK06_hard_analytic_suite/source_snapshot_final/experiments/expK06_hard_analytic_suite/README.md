# Hard analytic functions — final experiment

The final scope is **four functions × six widths × two trained methods, 50k steps per run**, plus a **QUILL-only scaling figure for 18 functions (the original 16 plus Gamma and a compact bump)**. There are 48 final Adam/NPU runs. Superseded 100k runs and the larger pilot screen are retained but excluded from the final figures.

QUILL is the **sample-fitted** recipe: uniform tanh centers and a rank-revealing least-squares readout. No method receives derivatives or exterior target labels. The old direct-derivative prototype is archived and excluded.

## Targets and analyticity

The “Rebuttal: valid theory” thread supplied its recent 24-target catalog. It was a GD scale-acquisition study, not a QUILL difficulty ranking. Its analytic Airy, erf, chirp and localized-oscillation cases are retained exactly. Nonanalytic cases are excluded. The new suite adds implicit Kepler inversion, elliptic integrals, Dawson/Bessel/Fresnel/Voigt/Kummer functions, and transcendental quotients. Definitions are frozen in `targets.py` and were selected before inspecting trained outcomes.

All targets use [-1,1]. Entire functions have no finite singularities. The Kepler inverse is real-analytic because its defining derivative is at least 0.1. Elliptic arguments remain strictly between zero and one on any finite real interval. Quotient denominators and the logarithm's argument are positive on the real line. Thus every target has an analytic neighborhood of the finite real halo. Nearby complex singularities still limit resolution. For sample-fitted QUILL, the halo adds features, not exterior labels.

Our K(m), E(m) use the parameter m, not the modulus. D is Dawson's integral, S the Fresnel sine integral, and V the normalized Voigt profile. Definitions and analytic properties: [Airy](https://dlmf.nist.gov/9.2), [error/Dawson/Fresnel](https://dlmf.nist.gov/7.2), [Bessel](https://dlmf.nist.gov/10.2), [Kummer](https://dlmf.nist.gov/13.2), [elliptic integrals](https://dlmf.nist.gov/19.2).

## Data and baseline training

The first figure uses W=105,169,297,559,1089,2141. These correspond to N=64,128,256,512,1024,2048 and R=max(20,ceil(sqrt(N))). At each W all methods receive the same `linspace(-1,1,max(2003,4*(N+1)))` function values. Target mean/amplitude normalization uses training values only. No target-specific geometry or frequency initialization is given to trained models.

Reasonable Adam/NPU settings are selected from already-saved pilots, separately by target and width. Adam uses ordinary PyTorch or Xavier initialization. NPU candidates include the native gated model, a positive affine-input complex model, and an NPU with a small learned affine input layer. Exact selected configurations and parameter counts are saved. NPU composition may have more parameters at the same W.

Resume the saved 10k-step optimizer/RNG state and train through **50k total steps**. The first 10k updates are identical under the old and new schedules: constant rate through 25k, then cosine decay to 1e-6 by 50k. Both families use batch 100, Float64, seed 0, and best-validation checkpoint retention. **No L-BFGS, final readout polishing, or additional seeds are included in the final 50k comparison.**

## QUILL tuning

`tune_quill.py` retains the audited `fit_qi` framework. It checks bandwidths within [.2,.3], explicit rank cutoffs 1e-14/1e-15/machine epsilon, and rank-revealing `gelsd`, `gelsy`, and `gelss` solves. Halo tests hold total W and training samples fixed, adjusting the number of interior intervals accordingly. All target outputs share a factorization but have independent readouts.

Pivoted QR (`gelsy`) and the smaller cutoff improve the numerical floor. Parameters are selected by validation, never by test error. No adaptive centers, nonlinear input features, multiscale partitions, derivative fitting, or alternate architecture are introduced. A QUILL-only W=4225 point resolves the sharp elliptic-E composition. Further tuning stops after that reaches the requested validation precision.

## Measurement and figures

Every saved network is scored with ordinary Float64 Torch inference and physical-dx relative L2. NPU validation refines logarithmic integration panels around possible input zero crossings, avoiding missed spikes. Final integration uses independent higher-order rules. Re-ranking pilots preserves model, optimizer and RNG state exactly.

- `hard_functions_3x4.png/.pdf`: four columns. Rows show function fits, signed residual/RMS on a symmetric-log axis, and all six matched-width scaling points. The top two rows use each method's best validation-selected fit among the six widths; widths are recorded in the results table.
- `additional_functions_1x2.png/.pdf`: a compact two-column key of all eighteen functions and their QUILL-only width curves. Color identifies the function. No baseline curves are included.

No smoothing is applied. White circles show measured widths. A few poorly resolved low-width QUILL fits have small metric variation from floating-point cancellation; raw quadrature comparisons remain saved. The selected high-precision fits pass refinement checks and independent 70-decimal reference spot checks.

18 implementation/NPU regression tests pass. They cover model gradients, literal PyTorch initialization, fitted QUILL geometry, the unique implicit root, physical integration, and a sharply peaked integrand with a known exact integral. Additional audits check exact training continuation, every run's 50k step count and final learning rate, saved-network reloads, and target values against mpmath.

## Reproduction and saved data

From the repository root, using the saved pilot states:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/main50k.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/tune_quill.py probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/tune_quill.py sweep
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/tune_quill.py extend
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/reference_check.py
~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/plot.py
~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/report.py
```

Outputs: `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/`. The current specifications, selected settings, all final checkpoints, every QUILL tuning candidate, test arrays, CSV measurements, source snapshots, and audits are retained. `sweep.py` describes the superseded larger plan; do not use it to reproduce the final scoped experiment.

## Figure style

Keep figures simple: functions, axes, units, and compact legends only. Put descriptions, training budgets, selection rules, and model-width details in the external caption or results text. Use consistent typography and arrange long function keys in compact columns rather than one tall list. Short function names are included beneath formulas in the supplementary key, as requested.

## Added Gamma and compact bump

Run `add_gamma_bump.py` to reproduce the two extra QUILL-only curves. They use the definitions from expD58: Gamma(0.05+2.475(x+1)) and exp(-1/(1-(x/0.35)^2)) for |x|<0.35, zero otherwise. Gamma is analytic on the requested interval. The bump is smooth but nonanalytic, so the 18-function supplement is not an analytic-only collection. Neither addition changes the four-function trained comparison. Seven widths, lambda=.25, automatic halo, and gelsy at machine-epsilon cutoff are used; results and checks are saved in gamma_bump/ and quill_extension_verified.json.

## NPU fairness audit and Adam-to-L-BFGS update

The original Adam-only results remain preserved. The requested update uses complex NPU only,
with 50k Adam steps followed by up to 1,000 full-batch L-BFGS iterations, retaining the best
validation checkpoint including the starting model. At each of the six widths, the starting
checkpoint is selected by validation from the original 50k models and, at width 105, extra
native NPU runs at learning rate .005 with seeds 0 and 1. This is the released paper code's
simple-task rate, added in the audit. Same training grids, function values and Float64.
No derivative labels, readout refitting or RealNPU are included in the updated comparison.

`audit_npu_fairness.py` and `audit_readout_sweep.py` separately investigate numerical readout
fitting; `audit_npu_fresh.py` checks NPU/RealNPU sensitivity. Their results are diagnostics,
not silently included in the final NPU curves. The MLP and QUILL results stay unchanged.

Run `npu_lbfgs.py`, then `finalize_npu_lbfgs.py`, then `plot.py`. Outputs and verification are
in `npu_adam_lbfgs/`; the renderer uses its verified results when present. The old main
figure and provenance are preserved in `npu_adam_lbfgs/prior_adam_only/`. The updated
copyable table is `npu_adam_lbfgs/table1.md`. Figure layout, colors and typography are unchanged.

After inspecting spikes near NPU input zeros, the user requested pointwise-stable
displayed fits for all four functions. `finalize_npu_lbfgs.py` now selects the
NPU model with the smallest sampled maximum absolute error across six widths,
using `npu_spike_probe/pointwise_candidate_audit.json`, and verifies it on a denser
grid plus random/zero-neighborhood probes. Its measured L2 error is reported in
the table; the six-point scaling curves are unchanged. This post-audit selection
is disclosed in `npu_adam_lbfgs/display_selection.json` and the captions.
Airy uses the composed gated NPU; the other selected models are native gated
NPUs. The prior minimum-L2 display is retained in `prior_l2_only_selection/`.

## Supplement: inverse square root replaces the chirp

The displayed supplement now replaces the simple quadratic chirp with the earlier
15-block inverse-square-root construction from expK02. The error function remains.
The original chirp results are preserved; only the displayed target list changes.
`inverse_sqrt_multiscale_curve.json` imports the previously displayed 1k--5k-width
window; `inverse_sqrt_multiscale_import.json` records source/model hashes and the
rechecked selected model. Table 2 uses W=4815 and relative L2 approximately 5e-14.
The plot extends to 5k and includes the coarse multiscale errors up to about 3e3.

This is explicitly a different recipe: 15 scaled cardinal-QI blocks with C17
extension, derivative samples, and 30-decimal construction, exported for Float64
inference on [1e-4,1]. It is not the raw uniform sample-fitted expK07 run. The other
17 plotted targets retain their sample-fitted uniform networks on [-1,1]. The
current table is `table2.md`; old figures are in `prior_chirp_supplement/`.
