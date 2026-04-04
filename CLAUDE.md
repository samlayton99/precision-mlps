# precisionMLPs

## Research Question

Can we find a training/optimization strategy that learns QI-like solutions, closing the gap between explicit construction (~10^-15) and training (~10^-10)?

Three violations in trained networks explain the gap:
1. **Gamma scaling**: gamma stays O(1) instead of growing as O(N)
2. **Weight blowup**: outer weights diverge instead of staying O(1)
3. **Rank saturation**: features collapse instead of uniform utilization

## How to Succeed
1. Always get context of the research! The papers/theory guiding the experiments are in the /papers/ folder. It has both the pdf and latex versions. Read the main_paper.tex and Section_3_rewrite.tex. We are trying to complete this paper by finding an optimization strategy.
2. Use the additional repo 'continuous-mlps' (next door neighbor to this repo in the file structure) as inspiration or a resource when you need it (it is a correct implementation of the paper), but not as something to just copy exactly.
3. read future_experiments.md every time. This is our main design spec doc that I will be working with you through.
4. When implementing new machinery or experiments, always write and clearly communicate to me the tests that verify your implementation actually matches the research (e.g. show me the QI construction reaches machine eps precision after being first built, etc.) 

## Architecture

```
papers/                       Source material guiding everything
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
results/                      Experiment results output
```

## QI Construction: Critical Facts

The QI construction in `src/construction/qi_mpmath.py` has two precision regimes.
**Read `papers/practical_implementation.tex` before touching construction code.**

- **fp64 path** (default, `precision="fp64"`, lambda=0.30): ~10ms, L_inf ~ 1e-12.
  Limited by fp64 cancellation in the convolution (c_j reach |c_0|~338 with alternating signs).
- **mpmath path** (`precision="mpmath"`, lambda=0.25): ~55s cold, ~0.25s cached, L_inf ~ 3e-15 (machine eps).
  Required because fp64 Toeplitz solve is ill-conditioned at lambda=0.25.

**Cardinal coefficients `c_j` are target-independent and cached to disk** at
`results/qi_cache/`, keyed by `(lambda_star, Kc, N, precision, mp_dps)`.
Second call at same config completes in ~0.25s even for mpmath.

**Use mpmath for baseline experiments (exp02 basin, exp03 ladder, exp04 Hessian)**
where QI is a fixed reference point. **Use fp64 everywhere else**
(training runs, sweeps, initialization). Both paths produce valid fp64 coefficients.

**mpmath does NOT violate fp64 assumptions.** The construction is an offline
precomputation that produces fp64 coefficients. The model, training loop, and
evaluation all run in fp64. Analogy: `numpy.pi` is computed at high precision
once and stored as a fp64 constant.

**Parameter warnings:**
- `lambda_star=1.5` does NOT work (intrinsic aliasing too large).
- `Kc=12` does NOT work (cardinal coefficients don't decay that fast).
- Use `Kc=160` (matches continuous-mlps) and `halo=default_halo(N, lambda_star)`.

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

An optimization method has been found if, as widths N increase (e.g. {32, 64, 128, 256, ...}) on the target-family matrix (6 categories), over 3-5 seeds, it's error falls at O(log(1/eps)) and eventually reaches eval relative L2 <= 1e-13 and eval L_inf consistent with machine epsilon precision, without initialization from the exact constructive solution.
