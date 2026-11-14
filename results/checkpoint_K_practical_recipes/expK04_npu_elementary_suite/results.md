# Small NPU screen near the singularity

Four functions on **[0.0001,1]**, evaluated with physical-dx relative L2. No halo or continuation is imposed on NPU. This is an illustrative optimization screen, not a proof of an NPU approximation limit or a success-rate estimate.

| Function | Selected architecture | Seed 0, Adam | Seed 0, best after polish | Same recipe, seed 1 |
|---|---|---:|---:|---:|
| 1/sqrt(x) | fixed_gate_random (1 parameters) | 1.79673e-13 | 1.79673e-13 | 1.79653e-13 |
| 1/(sqrt(x)+x) | shallow_32 (97 parameters) | 0.000123366 | 0.000103711 | 0.000128298 |
| exp(-x)/sqrt(x) | shallow_32 (97 parameters) | 0.00108358 | 0.000122795 | 0.000668616 |
| -log(x)/sqrt(x) | compact_stack (5 parameters) | 0.00256106 | 0.000188225 | 0.119303 |

The pure power reproduces near-machine precision. None of the three modified functions reached 1e-12 in this screen. Errors near 1e-4 are good ordinary approximations, but do not establish the high-precision behavior sought here. The log/root example is particularly initialization-sensitive. A separate shallow-six-unit repeat reached about 1.66e-4 on it, while the selected five-parameter compact architecture scored about 0.119 on its fresh-seed repeat. All runs are retained.

## Protocol

- Initial screen: shallow 6-unit and 32-unit NPU-to-linear models, plus NPU(1,8) → linear(8,4) → NPU(4,4) → linear(4,1). Standard Glorot real/readout weights, imaginary weights zero, learned gates initialized at 0.5.
- Two Adam rates (0.001 and 0.01), 20k updates, batch 100. Hold the rate for 10k, then cosine decay to 1e-6. Physical inputs remain x; log coordinates are used for sampling only.
- Log sampling uses the importance weight x*log(1/a)/integral(f^2 dx), so expected loss is exactly physical relative-L2 squared. It does not silently replace the objective with unweighted log-domain MSE.
- Polish the validation-best Adam model with ordinary full-batch L-BFGS, up to 500 iterations, strong-Wolfe search, history 50. Train on order-32 composite log-panel quadrature, validate on order 64; report independent order-256/512 scores.
- Three scalar inverse-root controls use uniform sampling: fixed gate with zero/random exponent, and learned gate with random exponent. Exponents are learned; the correct value is not used to initialize training.
- A follow-up compact stack learns five parameters in a sum-of-two-powers followed by a power. It uses random Glorot weights and fixed input gates at 1. The architecture is generic and was tested on all four functions.
- Recipe selection uses seed-0 validation scores. One fresh seed checks each selected recipe. The expanded initial plan was reduced at the user's request; the other target functions in problem.py were not run.

Positive-input shallow NPUs with zero imaginary initialization remain on their RealNPU branch. The larger stack can activate imaginary parameters on signed intermediate inputs. No target-specific feature engineering or function-value construction initializes any trained model.

## Interpreting the algebraic example

1/(sqrt(x)+x) is exactly representable (up to stabilization/rounding) by the small compositional NPU architecture. A test verifies this by assigning its known parameters; those values are NOT used in training. Its observed training error therefore demonstrates optimization difficulty under these budgets, not a representational impossibility.

## Checks and limitations

Six implementation tests passed: the general physical-L2 metric, importance weight identity, native power representation, compositional representation, finite initial gradients, and an end-to-end training smoke test. Every saved selected checkpoint was reloaded. The reported primary and fresh-seed results agree across quadrature orders 256/512. Some nonselected large stacked fits show quadrature sensitivity and are flagged in validation.json; their apparent low sampled losses are not accepted as high-precision solutions.

This modest search does not rule out better NPU results with other starts, architectures or optimizers. QUILL has not yet been run on the three new functions; no QUILL superiority result on them is claimed. The natural transcendental examples exp(-x)/sqrt(x) and -log(x)/sqrt(x) are useful next candidates for such a comparison.

The paper's published benchmarks are different: joint arithmetic/extrapolation tasks and Float32 in the released code. The tiny fixed-gate Float64 power control estimates one exponent, which explains its much better precision without contradicting the paper's aggregate errors.

## Saved artifacts

all_runs.csv, illustrative_results.csv, findings.json, validation.json, specification.json, original_source_manifest.json and per-run configs, initial/final/selected checkpoints, histories and dense predictions. The compact follow-up has its own specification.
