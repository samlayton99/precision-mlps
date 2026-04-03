# precisionMLPs

## Research Question

Can we find a training/optimization strategy that learns QI-like solutions, closing the gap between explicit construction (~10^-15) and training (~10^-10)?

Three violations in trained networks explain the gap:
1. **Gamma scaling**: gamma stays O(1) instead of growing as O(N)
2. **Weight blowup**: outer weights diverge instead of staying O(1)
3. **Rank saturation**: features collapse instead of uniform utilization

## Architecture

```
src/                          Core library (PyTorch, all computation in float64)
  config/
    schema.py                 ExperimentConfig and sub-configs (dataclasses)
    loader.py                 YAML load/save, sweep expansion
  models/
    layers.py                 GammaLinear, GammaExpLinear, StandardLinear
    mlp.py                    QIMlp: single-hidden-layer tanh MLP
    freeze.py                 requires_grad freezing utilities
  construction/
    qi_mpmath.py              High-precision QI via mpmath Toeplitz solve
    readout.py                Feature matrix Phi, exact readout solve (numpy/scipy)
    initialize.py             Project construction into model params
  data/
    targets.py                TargetFn registry (6 categories)
    sampling.py               Sampling functions (equispaced, uniform, Chebyshev, QI grid)
    dataset.py                build_dataset() -> Dataset dataclass
  training/
    optimizers.py             Optimizer dispatch (Adam, LBFGS, SGD)
    losses.py                 MSE, Lp, hybrid boundary
    train_loop.py             Multi-stage training orchestration
    metrics.py                MetricsCollector: uniform metric set across experiments

experiments/                  One folder per experiment, each with config.yaml + run.py
  exp00_sanity/               Numerics sanity checks
  exp01_lambda_tradeoff/      U-shaped error curve in lambda
  exp02_basin_stability/      QI basin width and recovery
  exp03_geometry_ladder/      Progressive constraint relaxation (7 levels)
  exp04_hessian/              Hessian eigenspectrum comparison
  exp05_phi_conditioning/     Feature matrix conditioning
  exp06_objective_mismatch/   Loss function comparison
  exp07_noise_sensitivity/    Y-noise and X-noise robustness
  exp08_reparameterization/   Log-gamma, global bandwidth, dimensionless centers
  exp09_varpro/               Variable Projection reduced objective

tests/                        Unit tests
```

## Key Abstractions

**QIMlp** (`src/models/mlp.py`): Single-hidden-layer tanh MLP (`nn.Module`). Always `inner_layer -> tanh -> readout`. Exposes `features(x)` for the Phi matrix and accessors for gamma, centers, readout weights.

**QIResult** (`src/construction/qi_mpmath.py`): Immutable dataclass holding construction output. Pure data -- never references a model.

**MetricsCollector** (`src/training/metrics.py`): Logs a fixed set of metrics at every eval step: train/eval loss, L_inf, relative L2, gamma/lambda stats, outer weight norms, feature rank. Writes JSONL.

**ExperimentConfig** (`src/config/schema.py`): Top-level dataclass. Every field has a default. YAML files only override what they need.

## Conventions

- **PyTorch with fp64.** `torch.set_default_dtype(torch.float64)` in `src/__init__.py`.
- **Device selection.** CUDA -> MPS -> CPU. Set in `src/__init__.py` as `DEVICE`.
- **Single hidden layer only.** No depth experiments until 1-layer is understood.
- **Freezing via requires_grad.** `param.requires_grad_(False)` to freeze.
- **Construction uses numpy/mpmath, training uses PyTorch.** Readout solve uses numpy/scipy. Model forward passes and gradients use PyTorch.
- **Multi-stage training.** Adam -> LBFGS is the default. LBFGS uses PyTorch's built-in closure pattern.
- **Analysis is per-experiment.** No pre-built analysis module. Each experiment's `run.py` implements its own analysis using PyTorch directly.
- **Results format.** JSONL for metrics. Config YAML saved alongside.

## Experiment Workflow

Each experiment's `run.py` is a self-contained script that imports from `src/`:

```python
from src.config import load_config, expand_sweep
from src.data import build_dataset
from src.models import QIMlp
from src.construction import construct_qi, initialize_from_construction
from src.models.freeze import freeze_gamma, freeze_centers
from src.training import run_training

config = load_config("experiments/exp01/config.yaml")
for cfg in expand_sweep(config):
    for seed in cfg.seeds:
        for width in cfg.widths:
            dataset = build_dataset(cfg)
            model = QIMlp(cfg.model)
            # ... construct, freeze, train, analyze, save
```

## Dependencies

PyTorch, mpmath, numpy, scipy, PyYAML.

## Success Criterion

A method works if, across widths N in {32, 64, 128, 256}, on the target-family matrix (6 categories), over 3-5 seeds, it reaches eval relative L2 <= 1e-13 and eval L_inf consistent with construction-level precision, without initialization from the exact constructive solution.
