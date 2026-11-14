# Hard analytic functions: sample-fitted QUILL, Adam, and NPU

This benchmark follows Sam's explicit coefficient-route instruction: **QUILL fits its readout by least squares to function-value samples**. No learner receives derivatives or target labels outside [-1,1]. The initial direct-derivative prototype is preserved under `archive_direct_prototype/` and excluded from all comparisons.

## Selection and provenance

Sam asked for four detailed functions and twelve additional hard analytic functions. The thread **Rebuttal: valid theory** (01a1c878-69ed-7b71-87bd-059876ee4097) supplied its 24-case catalog. That was a GD scale-acquisition study, not a QUILL or NPU accuracy ranking. Its analytic Airy, erf, chirp, and localized-oscillation targets are retained verbatim. Its five nonanalytic cases are excluded. The shifted Gamma is omitted in favor of functions with real-analytic continuation across the whole finite halo.

The four main targets are Airy Ai(-30x), the unique real inverse of y-0.9 sin(y)=pi x, K(0.99/(1+9x²)), and exp(-2x²)cos(30x+12x²)/(1+100(x-0.2)²). These span oscillation, implicit evaluation, a special-function composition near a branch singularity, and an oscillatory quotient. The other twelve are listed exactly in targets.py. They were selected for these structural difficulties **before inspecting trained-baseline outcomes**, not ranked by how poorly NPU performs. There is no claim that a finite list contains the mathematically hardest analytic functions.

Special-function conventions and analytic properties follow [DLMF Airy](https://dlmf.nist.gov/9.2), [error/Dawson/Fresnel](https://dlmf.nist.gov/7.2), [Bessel](https://dlmf.nist.gov/10.2), [Kummer](https://dlmf.nist.gov/13.2), and [elliptic integrals](https://dlmf.nist.gov/19.2). Our K(m), E(m) take the **parameter m**, not the modulus sqrt(m). V is SciPy's normalized Voigt profile with Gaussian standard deviation sigma and Lorentzian half-width gamma.

## Analyticity and halo

Airy, erf, the chirps and Gaussian products, Dawson, integer-order J, Si, Fresnel, and the stated Kummer function have entire extensions. The real Voigt profile has an entire extension formed from the appropriate two conjugate-shifted Faddeeva functions; taking a real part is only the real-axis evaluation formula.

The Kepler inverse exists uniquely on the real line because 1-0.9 cos(y)>=0.1. The analytic implicit-function theorem gives an analytic neighborhood of every point, hence a neighborhood of any compact real halo. The elliptic arguments stay strictly between 0 and 1 on every finite real interval; their closest complex branch points still limit resolution. In particular, the supplementary elliptic-E example is intentionally sharp. The quotient denominators are positive on the entire real line; the logarithm's argument is positive. Thus all 16 extend analytically across the real halo, although they have different complex singularity distances.

For this **sample-fitted** recipe, halo centers add features and do not require evaluating the target outside the data interval. Analytic continuation is a property of the chosen suite, not an extra source of training labels.

## Matched data and width

N=64,128,256,512,1024,2048; R=max(20,ceil(sqrt(N))); W=N+2R+1 gives 105,169,297,559,1089,2141 hidden neurons. Every method at a given N receives exactly `linspace(-1,1,max(2003,4*(N+1)))` and its function values. Target amplitude and mean normalization use these training values only. Physical relative L2 integrates uniform dx on [-1,1]. Validation starts with 64 panels of order-32 Gauss-Legendre. NPU validation adds logarithmically spaced panels around zero crossings of its raw or learned-affine inputs, down to 1e-18 offsets, to resolve narrow spikes. Independent scoring refines the rule to orders 64/128 (and 256 if needed). Test data never select hyperparameters or checkpoints.

QUILL calls the audited `docs/qi_practical_guide/reference.py:fit_qi`: a uniform tanh bank, automatic halo, explicit `gelsd` cutoff 1e-14. Lambda .2, .25, .3 is selected per target and width on validation. All candidates are retained. Geometry factorizations are shared across the sixteen independent output columns.

## Favorable trained baselines

At every target and matched width, each family gets six 10k-step pilots:

- Adam/tanh: PyTorch default or Xavier-uniform (hidden gain 5/3), rates .001, .01, .03.
- NPU: rates .001 and .01 for each of three variants. `native` uses the reference gated complex NPU on raw x, with zero imaginary initialization; `positive` applies the fixed affine map u=1.5+0.5x, fixes gates to one, and uses Glorot-random real and imaginary weights; `composed` uses a learned 2x2 NAU on [x,1], then NPU(2,W), then a linear output. Every NPU readout has a learnable bias. These are deliberately favorable options, not a claim of reproducing one exact paper configuration.

All initializations are target-independent. There are no QUILL-derived centers, fitted initial weights, or target-specific frequency scales in Adam/NPU. NPU composition can use more parameters than a tanh network at the same hidden width; counts are recorded, so this is not an equal-parameter or equal-compute study.

Before selection, re-score each NPU pilot's saved best and final states using the resolved integral. This repairs under-resolution found in a pilot near zero, without changing any trained weights or optimizer/RNG state. Select the pilot by validation separately at each target/method/width, then resume its exact optimizer/RNG state through 100k total Adam updates. Batch size is 100 for both families. Hold the rate for 50k steps and cosine-decay to 1e-6 by 100k; no warmup, clipping, or weight decay. Keep the best validation checkpoint, including any initial checkpoint that genuinely remains best.

Offer the retained model up to 500 full-batch L-BFGS iterations, then a `gelsd` readout solve at the same 1e-14 cutoff as QUILL. Retain either polish only when validation improves. Save pre-polish Adam, readout-fit, final optimizer, and selected states separately. The figure labels these as **polished Adam** and **polished NPU** rather than attributing the entire result to Adam alone. One fresh seed repeats the largest-width recipes for the four main functions. Seed-1 results are reported separately, not substituted opportunistically for seed 0.

## Figures and validation

The main figure is 3x4: function and approximations, signed residual on a symmetric-log axis, and relative-L2 width scaling. Function/residual rows show each method's best validation-selected fit across the declared widths, with its width annotated. The bottom row retains all matched-width comparisons. Training sets are identical between methods at a given width; their sample count grows with N. The main fit rows therefore show the best result within the shared sweep, not one identical-width experiment. Residuals are divided by RMS(f), with signs retained and a linear region around zero. No smoothing is applied.

The supplemental figure is 1x2: a centered color-keyed list of twelve definitions, then their individual width curves. Color identifies the function and line style identifies the method; all curves remain available individually in the data.

Seven basic tests cover target definitions, the implicit root, the actual sample-fitted geometry, native-model gradients, literal PyTorch initialization, physical integration weights, and an analytically integrable sharp spike. The original NPU regression tests also pass (18 checks total). Independent target evaluation checks use 70-decimal mpmath, including derivative checks solely to audit the target implementations—not as learner inputs. A separate training audit verifies bitwise continuation and that accepted polishing cannot worsen validation. A refinement audit confirms that re-ranking preserves the saved optimization state exactly. The initial regular-grid NPU scores are retained for provenance and are not treated as resolved evidence. Saved models and independent dense/quadrature predictions are audited before the final figures are produced.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/audit_targets.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/construction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK06_hard_analytic_suite/sweep.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python -m pytest -q tests/test_hard_analytic_suite.py
```

All output lives in `results/checkpoint_K_practical_recipes/expK06_hard_analytic_suite/`. The sweep is resumable and preserves every candidate. `status.json` records its current stage; `specification.json` freezes the numerical source hashes and protocol. Later plotting and audit edits do not invalidate numerical checkpoints.
