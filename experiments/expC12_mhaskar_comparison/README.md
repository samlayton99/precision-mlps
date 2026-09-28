# Mhaskar versus QUILL: current precision contract

The current comparison is `strict.py`: construction and inference use correctly rounded $p$-bit operations, with an FP64 linear-solve exception. `run.py` and `construction.py` retain the earlier parameter/output-only quantization experiment as a historical control; do not use its curves as the current precision experiment.

Run from the repository root with the project's NumPy/SciPy environment and `gmpy2==2.3.1`:

```sh
.venv/bin/python experiments/expC12_mhaskar_comparison/strict.py
```

For the current session, the previously installed MPFR dependency is available through `PYTHONPATH=/private/tmp/precisionmlps-strict-arithmetic`. The dependency is also pinned in `experiments/expC09_bandwidth_figures/requirements-strict.txt`.

The output goes to `results/checkpoint_C_geometry/expC12_mhaskar_comparison/strict/`, preserving the previous experiment. Figures are PNGs. The shared renderer preserves panels (a) and (b), replacing panel (c) with the newly measured precision comparison.

## Arithmetic boundary

- External sampling coordinates, target observations, and configuration-grid values are rounded when they enter model calculations.
- QUILL computes spacing, centers, the predicted bandwidth, and affine hidden weights/biases at $p$ bits. It generates its feature matrix at $p$ bits, passes that matrix and rounded labels to the FP64 SVD, and immediately rounds the resulting readout.
- Mhaskar computes its discrete Chebyshev projection, conversion to monomials, scaled tanh derivatives, finite-difference weights, and merged network parameters at $p$ bits. Its projection is evaluated directly through a rounded Chebyshev recurrence and rounded sums; there is no FP64 DCT or coefficient-construction exception.
- Both infer using the same affine form: separate rounded multiplication and addition, correctly rounded tanh, and sequential rounded readout products and sums. No fused multiply-add is used.
- Reference values, relative-error measurements, and candidate selection metrics use FP64. MPFR implements the specified arithmetic; this is not a claim of native low-precision hardware execution.
- Completed parameters are saved in FP64 containers only if their $p$-bit values can be represented exactly. Nonrepresentable constructions are recorded as invalid candidates.

## Exact candidate pruning

The original degree/step grid and full validation grid are retained. To reduce cost, every candidate is first evaluated on every eighth validation point. Its partial residual norm divided by the **full** target norm is a lower bound on its full relative error. Reject a candidate only when this lower bound exceeds the current best full error, with a conservative numerical margin. This preserves the full-grid minimizer; it is not approximate shortlist selection. Reporting uses a separate grid.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_mhaskar_pbit.py tests/test_strict_precision.py tests/test_mhaskar_comparison.py
```

The independent test oracle computes operations at higher precision and rounds after each specified operation, solely to verify the kernels. Production construction and evaluation do not use this oracle. Tests include a case where FP64-then-round would double-round incorrectly. Every experimental model is reloaded and replayed bit for bit; all saved parameters and predictions are checked for exact $p$-bit representability.

## Numerical rescue study

`rescue.py` tests sorted, pairwise, Neumaier, and compensated Dot2 readouts on all 46 saved Mhaskar models, then retunes symmetric/compensated construction at seven representative precisions. `extrapolation.py` additionally tests three Richardson levels at 24 and 53 bits. These variants retain one $p$-bit value per stored parameter and use only $p$-bit primitives, including their correction registers. They cost extra arithmetic and temporary storage. Richardson also adds neurons, counted against the same width budget.

The measured rescue variants are saved under `results/checkpoint_C_geometry/expC12_mhaskar_comparison/rescue/`; they do not overwrite the strict sequential baseline. `rescue_diagnostics.py` uses an independent 256-bit reference dot product exclusively to identify accumulation error in already-rounded data. That reference is never used to construct, select, or evaluate a deployed model.
