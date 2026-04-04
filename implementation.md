# Implementation Plan

Build order matters. Each phase depends on the previous one. Test each phase before moving on.

## Stack

- **PyTorch** for models, training. Default dtype float64.
- **mpmath** for high-precision QI construction (Toeplitz solve).
- **numpy/scipy** for readout solves and construction projection.
- **PyYAML** for config loading.

---

## Phase 1: Config + Data

### `src/config/schema.py` (done -- just dataclasses)

### `src/config/loader.py`
- `load_config`: yaml.safe_load -> recursive merge into ExperimentConfig defaults
- `expand_sweep`: itertools.product over config.sweep axes, deepcopy + _set_nested
- `_set_nested`: split on ".", getattr chain, setattr final

### `src/data/targets.py`

```python
@dataclass(frozen=True)
class TargetFn:
    fn: Callable           # f(x), works on torch tensors
    deriv: Callable        # f'(x), torch
    fn_numpy: Callable     # f(x), works on numpy arrays (for construction)
    deriv_numpy: Callable  # f'(x), numpy (for construction)
    name: str
    category: str
```

Populate TARGET_REGISTRY with ~10 functions across 6 categories:
- sine, cosine (low_freq)
- sine_8pi (high_freq)
- runge, tanh_steep (boundary_layer)
- sine_mixture (mixed_scale)
- exp, poly5 (polynomial)
- abs_cubed (rough)

### `src/data/sampling.py`
Plain functions, all returning `[n_points, 1]` float64 tensors:
```python
def equispaced(n_points, domain=(-1.0, 1.0)) -> torch.Tensor
def uniform_random(n_points, domain=(-1.0, 1.0), seed=0) -> torch.Tensor
def chebyshev(n_points, domain=(-1.0, 1.0)) -> torch.Tensor
def qi_grid(N, domain=(-1.0, 1.0), halo=0) -> torch.Tensor
def get_sampling_fn(name) -> Callable
```

### `src/data/dataset.py`
```python
@dataclass
class Dataset:
    x_train: torch.Tensor  # [n_train, 1]
    y_train: torch.Tensor  # [n_train, 1]
    x_eval: torch.Tensor   # [n_eval, 1]
    y_eval: torch.Tensor   # [n_eval, 1]
    target_fn: TargetFn

def build_dataset(config: ExperimentConfig) -> Dataset:
    # 1. get_target(config.target)
    # 2. sample x_train via get_sampling_fn(config.sampling)(config.n_train, config.domain)
    # 3. y_train = target.fn(x_train)
    # 4. add noise if config.y_noise_std > 0 or config.x_noise_std > 0
    # 5. x_eval = equispaced(config.n_eval, config.domain)
    # 6. y_eval = target.fn(x_eval)
```

**Test:** load config, build dataset, verify shapes and dtypes.

---

## Phase 2: Models

### `src/models/layers.py`
All `nn.Module`. Input `[batch, 1]`, output `[batch, width]`.

```python
class GammaLinear(nn.Module):
    def __init__(self, width, gamma_init=None, center_init=None):
        self.gamma = nn.Parameter(gamma_init or torch.ones(1, width))
        self.centers = nn.Parameter(center_init or torch.zeros(1, width))
    def forward(self, x):
        return self.gamma * (x - self.centers)

class GammaExpLinear(nn.Module):
    def __init__(self, width, h=1.0, log_gamma_init=None, center_init=None):
        self.log_gamma = nn.Parameter(log_gamma_init or torch.zeros(1, width))
        self.centers = nn.Parameter(center_init or torch.zeros(1, width))
        self.h = h
    def forward(self, x):
        gamma = torch.exp(self.log_gamma) / self.h
        return gamma * (x - self.centers)
    def get_gamma(self):
        return (torch.exp(self.log_gamma) / self.h).detach()

class StandardLinear(nn.Module):  # wraps nn.Linear(1, width)

def get_layer(layer_type, width, **kwargs) -> nn.Module
```

### `src/models/mlp.py`
```python
class QIMlp(nn.Module):
    def __init__(self, config: ModelConfig, **layer_kwargs):
        self.inner_layer = get_layer(config.layer_type, config.width, **layer_kwargs)
        self.readout = nn.Linear(config.width, 1)
    def forward(self, x):           # [batch, 1] -> [batch, 1]
        return self.readout(torch.tanh(self.inner_layer(x)))
    def features(self, x):          # -> [batch, width]
        return torch.tanh(self.inner_layer(x))
    def get_gamma(self):             # -> [width]
    def get_centers(self):           # -> [width]
    def get_readout_weights(self):   # -> [width]
    def get_lambda(self, h):         # -> [width]
```

### `src/models/freeze.py`
```python
def freeze_params(model, filter_fn):
    for name, param in model.named_parameters():
        if filter_fn(name, param):
            param.requires_grad_(False)

def unfreeze_params(model, filter_fn): ...
def freeze_inner_layer(model): ...    # "inner_layer" in name
def freeze_gamma(model): ...          # "gamma" or "log_gamma"
def freeze_centers(model): ...        # "centers"
def freeze_readout(model): ...        # "readout"
def unfreeze_all(model): ...
def get_trainable_param_count(model): ...
```

**Test:** forward shapes, freeze/unfreeze gradient flow, accessors across layer types.

---

## Phase 3: Construction (DONE)

The construction is implemented in `src/construction/qi_mpmath.py` and has two
precision regimes. See `papers/practical_implementation.tex` for the full
discussion. The short version:

### Two precision regimes

**fp64 (default, fast, ~10ms, achieves ~1e-12):**
- `precision="fp64"`, `lambda_star=0.30`, `Kc=160`, halo scaling
- Toeplitz solve via `np.linalg.solve`
- Convolution via fp64 Kahan summation
- Plateau at ~1e-12 from cancellation error in the c_k (|c_0|~338, alternating signs)

**mpmath (slow, ~55s cold / ~0.25s cached, achieves ~3e-15):**
- `precision="mpmath"`, `lambda_star=0.25`, `Kc=160`, halo scaling
- Toeplitz solve via `mp.lu_solve`
- Convolution via mpmath Kahan summation
- Reaches true fp64 machine epsilon

### API

```python
from src.construction import construct_qi, evaluate_qi, default_halo

# Fast fp64 (default)
qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64)

# Machine-epsilon baseline (for exp02, exp03, exp04)
qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64, precision="mpmath")

# Evaluate
y = evaluate_qi(qi, x_array)
```

### Cache

Cardinal coefficients `c_j` are **target-independent** and cached to disk at
`results/qi_cache/` keyed by `(lambda_star, Kc, N, precision, mp_dps)`. This
amortizes the 55s mpmath cost across all targets and seeds that share the
same construction configuration. Set `QI_CACHE_DIR` env var or pass
`cache_dir=Path(...)` to override; pass `use_cache=False` to skip.

### Per-experiment precision recommendations

| Experiment | Precision | Why |
|-----------|-----------|-----|
| exp00 (numerics) | both | explicit comparison |
| exp01 (lambda tradeoff) | fp64 | sweeps many params |
| exp02 (basin stability) | **mpmath** | QI is the reference point |
| exp03 (geometry ladder) | **mpmath** | measures precision loss vs. construction |
| exp04 (Hessian) | **mpmath** | Hessian at true QI solution |
| exp05-09 (training-heavy) | fp64 | fast iteration |

Rule: use mpmath when QI is a fixed reference to compare against; use fp64
when training / sweeping / initializing.

### Other construction modules

**`src/construction/readout.py`** (DONE, numpy/scipy):
- `build_phi(x, gamma, centers) -> Phi[n, W]`
- `solve_readout(Phi, y, method, ridge_alpha) -> (v, info)`  (methods: lstsq, qr, svd, ridge)
- `solve_readout_with_bias(Phi, y, ...) -> (v, bias, info)`

**`src/construction/initialize.py`** (DONE):
- `initialize_from_construction(model, qi_result, construct_gamma, construct_centers, construct_readout)`
- `initialize_and_freeze(model, qi_result, freeze_config, init_config)`
- `initialize_with_readout_solve(model, x_train, y_train, method, ridge_alpha)`

Models can be initialized with full width `N + 2*halo + 1` (all centers) OR
interior-only width `N + 1` (drops halo). `initialize_from_construction`
auto-detects based on `model.config.width`.

### CRITICAL: parameter warnings

1. **lambda_star=1.5 does NOT work.** At lambda=1.5 the intrinsic aliasing
   error is ~1e-3. We need lambda ≤ 0.30. Old default was wrong.
2. **Kc=12 does NOT work.** The cardinal coefficients decay much slower than
   the simplified rate formula `exp(-2*lambda*Kc)` predicts. At Kc=12 the
   edge coefficients are O(1e-5), causing ~1e-5 truncation error. Must use Kc~=160.
3. **Halo sizing is lambda-dependent.** Use `default_halo(N, lambda_star)`:
   - Lower bound: `ceil(35/(2*lambda))` to drive halo-tail below 1e-15
   - Lower bound: `0.4*N` to match continuous-mlps empirical rule

---

## Phase 4: Training

### `src/training/losses.py`
```python
def mse(model, x, y):
    return ((model(x) - y) ** 2).mean()

def lp_loss(model, x, y, p=6.0):
    return (torch.abs(model(x) - y) ** p).mean()

def hybrid_boundary_loss(model, x, y, boundary_weight=5.0, boundary_fraction=0.1): ...

def get_loss_fn(name, **kwargs):  # returns partial with (model, x, y) signature
```

### `src/training/optimizers.py`
```python
def build_optimizer(config: OptimizerStageConfig, model) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    # "adam" -> Adam, "lbfgs" -> LBFGS(strong_wolfe), "sgd" -> SGD

def build_scheduler(optimizer, config, total_steps):
    # cosine annealing or None
```

### `src/training/metrics.py`
```python
class MetricsCollector:
    def __init__(self, model, dataset): ...
    @torch.no_grad()
    def collect(self, step) -> dict:
        # train_loss, eval_loss, eval_linf, eval_rel_l2
        # gamma stats, lambda stats, readout weight stats
        # feature rank via SVD of Phi
    def to_jsonl(self, path): ...

def compute_eval_metrics(model, x, y) -> dict:  # standalone
```

### `src/training/train_loop.py`
```python
@dataclass
class TrainResult:
    loss_history: list[float]
    eval_metrics: dict[str, list[float]]
    eval_steps: list[int]
    stage_boundaries: list[int]
    final_metrics: dict[str, float]

def train_step(model, optimizer, x, y, loss_fn) -> float:
    # zero_grad -> backward -> step

def train_step_lbfgs(model, optimizer, x, y, loss_fn) -> float:
    # closure pattern

def run_training(config, model, dataset) -> TrainResult:
    # iterate stages, build optimizer per stage, eval at eval_interval
```

**Test:** train QIMlp on sine 1000 steps, verify loss decreases.

---

## Phase 5: Experiments

Each experiment's `run.py` is a self-contained script. No runner class. Pattern:

```python
from src.config import load_config, expand_sweep
from src.data import build_dataset, get_target
from src.models import QIMlp
from src.construction import construct_qi, initialize_from_construction
from src.models.freeze import freeze_gamma
from src.training import run_training

config = load_config("experiments/exp01/config.yaml")
for cfg in expand_sweep(config):
    for seed in cfg.seeds:
        for width in cfg.widths:
            torch.manual_seed(seed)
            cfg.model.width = width
            dataset = build_dataset(cfg)
            model = QIMlp(cfg.model)
            # ... construct, freeze, train, save results
```

Analysis (Hessian, Phi conditioning, perturbation, etc.) is written directly
in each experiment's run.py using PyTorch. No pre-built analysis module.

---

## Phase 6: Tests

- `test_construction.py` (DONE, 37 tests): construct_qi precision, readout solvers,
  cache roundtrip, precision flag, lambda sensitivity, scaling. Marked `slow` tests
  exercise the mpmath path.
- `test_models.py`: forward shapes, layer types, accessors
- `test_freeze.py`: freeze/unfreeze, gradient flow, param counts
- `test_training.py`: loss decreases, metrics collector keys

Run fast: `PYTHONPATH=. python3 -m pytest tests/ -v -m "not slow"` (~1s)
Run all:  `PYTHONPATH=. python3 -m pytest tests/ -v` (~60s, includes mpmath)

---

## Notes

**Target function dual interface:** Each TargetFn has torch and numpy versions.
The torch version is for training. The numpy version is for QI construction
(both fp64 and mpmath paths call it on scalar floats).

**LBFGS closure:** PyTorch LBFGS needs a closure that does zero_grad + loss + backward.
This is the only difference from Adam/SGD in the training loop.

**Device:** CUDA -> MPS -> CPU, set in src/__init__.py. MPS has limited float64 support.

**Readout solve bridge:** detach Phi to numpy, solve with scipy, copy weights back
with torch.no_grad(). Clean separation between numpy linear algebra and torch gradients.

**mpmath is offline, not a fp64 violation.** The QI construction is a one-time offline
computation producing fp64 coefficients. The trained model, training loop, and evaluation
all run in fp64. Computing coefficients in mpmath is analogous to computing `numpy.pi`
offline at high precision and storing as an fp64 constant.

**Construction is a reference point, not a training target.** Training experiments
(exp05-09) always use fp64 construction. Baseline experiments (exp02-04) use mpmath
construction as the "true QI solution" to compare training against.
