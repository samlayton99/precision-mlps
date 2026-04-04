"""Tests for training loop, optimizers, losses, and metrics."""

import math

import pytest
import torch

from src.config.schema import (
    ExperimentConfig, ModelConfig, TrainingConfig, OptimizerStageConfig,
)
from src.data.dataset import build_dataset
from src.models.mlp import QIMlp
from src.training.losses import mse, lp_loss, get_loss_fn
from src.training.optimizers import build_optimizer, build_scheduler
from src.training.metrics import MetricsCollector, compute_eval_metrics
from src.training.train_loop import run_training, train_step


EXPECTED_METRIC_KEYS = {
    "step", "train_loss", "eval_loss", "eval_linf", "eval_rel_l2",
    "gamma_mean", "gamma_median", "gamma_max", "gamma_min",
    "lambda_mean", "lambda_median", "lambda_max",
    "readout_weight_max", "readout_weight_l2",
    "feature_rank", "feature_stable_rank",
    "phi_smin", "phi_smax", "phi_cond",
}


def _tiny_cfg(steps_adam=100, steps_lbfgs=0, width=16, lr=1e-2):
    stages = [OptimizerStageConfig(name="adam", learning_rate=lr, steps=steps_adam,
                                    use_cosine_schedule=False)]
    if steps_lbfgs > 0:
        stages.append(OptimizerStageConfig(name="lbfgs", learning_rate=1.0,
                                            steps=steps_lbfgs, use_cosine_schedule=False,
                                            kwargs={"max_iter": 20}))
    return ExperimentConfig(
        target="sine", n_train=64, n_eval=128,
        model=ModelConfig(width=width, layer_type="gamma_linear"),
        training=TrainingConfig(stages=stages, eval_interval=50),
    )


def _init_model(m, width):
    with torch.no_grad():
        m.inner_layer.gamma.data.fill_(5.0)
        m.inner_layer.centers.data.copy_(torch.linspace(-1, 1, width).reshape(1, width))
        m.readout.weight.data.normal_(0.0, 0.1)
        m.readout.bias.data.zero_()


def test_mse_matches_manual():
    cfg = ModelConfig(width=4, layer_type="gamma_linear")
    m = QIMlp(cfg)
    x = torch.linspace(-1, 1, 10).reshape(-1, 1)
    y = torch.sin(x)
    loss = mse(m, x, y)
    manual = ((m(x) - y) ** 2).mean()
    assert torch.allclose(loss, manual)


def test_lp_loss_p2_equals_mse():
    cfg = ModelConfig(width=4, layer_type="gamma_linear")
    m = QIMlp(cfg)
    x = torch.linspace(-1, 1, 10).reshape(-1, 1)
    y = torch.sin(x)
    loss_mse = mse(m, x, y)
    loss_lp2 = lp_loss(m, x, y, p=2.0)
    assert torch.allclose(loss_mse, loss_lp2)


def test_get_loss_fn_dispatch():
    assert callable(get_loss_fn("mse"))
    assert callable(get_loss_fn("lp", p=6.0))
    assert callable(get_loss_fn("hybrid_boundary"))
    with pytest.raises(ValueError):
        get_loss_fn("bogus")


def test_build_optimizer_dispatch():
    cfg = ModelConfig(width=4, layer_type="gamma_linear")
    m = QIMlp(cfg)
    adam = build_optimizer(OptimizerStageConfig(name="adam", learning_rate=1e-3), m)
    assert isinstance(adam, torch.optim.Adam)
    sgd = build_optimizer(OptimizerStageConfig(name="sgd", learning_rate=1e-3), m)
    assert isinstance(sgd, torch.optim.SGD)
    lbfgs = build_optimizer(OptimizerStageConfig(name="lbfgs", learning_rate=1.0), m)
    assert isinstance(lbfgs, torch.optim.LBFGS)


def test_build_scheduler():
    cfg = ModelConfig(width=4, layer_type="gamma_linear")
    m = QIMlp(cfg)
    stage = OptimizerStageConfig(name="adam", learning_rate=1e-3, use_cosine_schedule=True)
    opt = build_optimizer(stage, m)
    sched = build_scheduler(opt, stage, total_steps=100)
    assert sched is not None
    stage2 = OptimizerStageConfig(name="adam", use_cosine_schedule=False)
    opt2 = build_optimizer(stage2, m)
    assert build_scheduler(opt2, stage2, 100) is None


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    cfg = ModelConfig(width=16, layer_type="gamma_linear")
    m = QIMlp(cfg)
    _init_model(m, 16)
    x = torch.linspace(-1, 1, 64).reshape(-1, 1)
    y = torch.sin(2 * math.pi * x)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss0 = train_step(m, opt, x, y, mse)
    for _ in range(200):
        loss = train_step(m, opt, x, y, mse)
    assert loss < loss0


def test_metrics_collector_keys():
    torch.manual_seed(0)
    cfg = _tiny_cfg(steps_adam=20, width=8)
    ds = build_dataset(cfg, seed=0)
    m = QIMlp(cfg.model)
    _init_model(m, 8)
    collector = MetricsCollector(m, ds)
    metrics = collector.collect(step=0)
    assert EXPECTED_METRIC_KEYS.issubset(set(metrics.keys()))


def test_compute_eval_metrics():
    torch.manual_seed(0)
    cfg = ModelConfig(width=8, layer_type="gamma_linear")
    m = QIMlp(cfg)
    x = torch.linspace(-1, 1, 32).reshape(-1, 1)
    y = torch.sin(x)
    em = compute_eval_metrics(m, x, y)
    assert set(em.keys()) == {"mse", "linf", "rel_l2"}
    assert em["linf"] >= 0


def test_run_training_result_structure():
    torch.manual_seed(0)
    cfg = _tiny_cfg(steps_adam=50, width=16)
    ds = build_dataset(cfg, seed=0)
    m = QIMlp(cfg.model)
    _init_model(m, 16)
    result = run_training(cfg, m, ds)
    assert len(result.loss_history) == 50
    assert len(result.stage_boundaries) == 1
    assert len(result.final_metrics) > 0
    assert "eval_linf" in result.final_metrics
    assert result.loss_history[-1] < result.loss_history[0]


def test_training_jsonl_roundtrip(tmp_path):
    import json
    torch.manual_seed(0)
    cfg = _tiny_cfg(steps_adam=20, width=8)
    ds = build_dataset(cfg, seed=0)
    m = QIMlp(cfg.model)
    _init_model(m, 8)
    collector = MetricsCollector(m, ds)
    collector.collect(step=0)
    collector.collect(step=10)
    path = tmp_path / "metrics.jsonl"
    collector.to_jsonl(str(path))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["step"] == 0
    assert rows[1]["step"] == 10
