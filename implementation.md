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

## Phase 3: Construction

### `src/construction/qi_mpmath.py`

The most complex module. Adapt from continuous-mlps.

```python
@dataclass(frozen=True)
class QIResult:
    N: int; h: float; gamma: float; lambda_val: float
    centers: np.ndarray; interior_centers: np.ndarray
    a_coeffs: np.ndarray; interior_a_coeffs: np.ndarray
    c0: float; toeplitz_residual: float; halo: int; Kc: int

def construct_qi(target_fn, target_deriv, N, *,
                 lambda_star=1.5, Kc=12, halo=8, mp_dps=50) -> QIResult:
```

Algorithm:
1. h = 2/N, gamma = lambda_star / h
2. Grid: x_k = -1 + k*h for k in range(-halo, N + halo + 1)
3. Toeplitz matrix T[r,j] = h * gamma * sech^2(gamma * (r-j) * h). Use mpmath.
4. Solve T @ c = e_Kc for cardinal coefficients
5. Evaluate g'(x_k) at all grid points
6. Convolve: a_n = sum_j c_j * g'(x_{n+j}) * h (Kahan summation)
7. Bias: c0 = g(-1) - sum_n a_n * tanh(gamma * (-1 - x_n))
8. Project to float64 numpy arrays

Kahan summation is ~10 lines, inline it here.

### `src/construction/readout.py`
numpy/scipy only:
```python
def build_phi(x, gamma, centers):        # -> [n, width] numpy
def solve_readout(Phi, y, method="lstsq", ridge_alpha=0.0):       # -> (v, info)
def solve_readout_with_bias(Phi, y, method="lstsq", ridge_alpha=0.0):  # -> (v, bias, info)
```

### `src/construction/initialize.py`
```python
def initialize_from_construction(model, qi_result, *,
                                  construct_gamma=True, construct_centers=True, construct_readout=True):
    # torch.no_grad(), copy numpy arrays into model params

def initialize_and_freeze(model, qi_result, freeze_config, init_config):
    # init then freeze per config

def initialize_with_readout_solve(model, x_train, y_train, method="lstsq", ridge_alpha=0.0):
    # detach Phi to numpy, solve, copy back
```

**Test:** construct_qi on sine N=64, verify L_inf < 1e-13. Initialize model, verify output matches.

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

- `test_construction.py`: construct_qi on sine, verify precision, readout solvers agree
- `test_models.py`: forward shapes, layer types, accessors
- `test_freeze.py`: freeze/unfreeze, gradient flow, param counts
- `test_training.py`: loss decreases, metrics collector keys

Run: `python -m pytest tests/ -v`

---

## Notes

**Target function dual interface:** Each TargetFn has torch and numpy versions.
The torch version is for training. The numpy version is for mpmath construction.

**LBFGS closure:** PyTorch LBFGS needs a closure that does zero_grad + loss + backward.
This is the only difference from Adam/SGD in the training loop.

**Device:** CUDA -> MPS -> CPU, set in src/__init__.py. MPS has limited float64 support.

**Readout solve bridge:** detach Phi to numpy, solve with scipy, copy weights back
with torch.no_grad(). Clean separation between numpy linear algebra and torch gradients.
