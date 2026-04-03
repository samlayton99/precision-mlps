"""Experiment configuration as composable Python dataclasses.

Every field has a default. YAML files only override what they need.

Usage:
    config = ExperimentConfig(
        name="lambda_tradeoff",
        widths=[16, 32, 64, 128, 256],
        construction=ConstructionConfig(enabled=True, lambda_star=1.5),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """MLP architecture specification. Always float64."""
    width: int = 64
    layer_type: str = "gamma_linear"    # "gamma_linear" | "gamma_exp" | "standard"
    activation: str = "tanh"


@dataclass
class ConstructionConfig:
    """QI construction parameters."""
    enabled: bool = False
    mp_dps: int = 50                    # mpmath decimal places
    Kc: int = 12                        # Toeplitz half-width
    halo: int = 8                       # ghost nodes
    gamma: Optional[float] = None       # if None, computed as lambda_star / h
    lambda_star: float = 1.5


@dataclass
class FreezeConfig:
    """Which parameter groups to freeze after initialization."""
    gamma: bool = False
    centers: bool = False
    readout: bool = False


@dataclass
class InitConfig:
    """How to initialize model parameters."""
    from_construction: bool = False     # whether to use QI construction values
    construct_gamma: bool = True        # which groups to copy from construction
    construct_centers: bool = True
    construct_readout: bool = True
    readout_solve: str = "none"         # "none" | "lstsq" | "ridge"
    ridge_alpha: float = 0.0


@dataclass
class OptimizerStageConfig:
    """Single training stage. Default two-stage: Adam -> LBFGS."""
    name: str = "adam"                  # "adam" | "sgd" | "lbfgs"
    learning_rate: float = 1e-3
    steps: int = 50000
    use_cosine_schedule: bool = True
    min_learning_rate: float = 1e-6
    kwargs: dict = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Multi-stage training configuration."""
    stages: list[OptimizerStageConfig] = field(default_factory=lambda: [
        OptimizerStageConfig(name="adam", learning_rate=1e-3, steps=30000),
        OptimizerStageConfig(name="lbfgs", learning_rate=1.0, steps=5000),
    ])
    loss: str = "mse"                   # "mse" | "lp" | "hybrid_boundary"
    loss_kwargs: dict = field(default_factory=dict)  # e.g., {"p": 6.0} for lp
    readout_solve_every: int = 0        # 0 = disabled
    eval_interval: int = 100


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "unnamed"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    widths: list[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Target function
    target: str = "sine"                # registry key
    domain: tuple[float, float] = (-1.0, 1.0)

    # Data
    n_train: int = 256
    n_eval: int = 2048
    sampling: str = "equispaced"        # "equispaced" | "uniform" | "chebyshev" | "qi_grid"
    y_noise_std: float = 0.0
    x_noise_std: float = 0.0

    # Sub-configs (keep grouped where there are many related fields)
    model: ModelConfig = field(default_factory=ModelConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    init: InitConfig = field(default_factory=InitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    output_dir: str = "results"
    sweep: dict = field(default_factory=dict)
