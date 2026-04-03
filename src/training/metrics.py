"""Uniform metrics collection across all experiments.

MetricsCollector computes a fixed set of metrics at every eval step,
ensuring cross-experiment comparability.

Metrics logged:
- train_loss, eval_loss, eval_linf, eval_rel_l2
- gamma_mean/median/max/min, lambda_mean/median/max
- readout_weight_max, readout_weight_l2
- feature_rank, feature_stable_rank, phi_condition_number
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from src.data.dataset import Dataset


class MetricsCollector:
    """Collects and stores metrics during training."""

    def __init__(self, model: nn.Module, dataset: Dataset):
        # TODO: implement
        # self._model = model
        # self._dataset = dataset
        # self._history = defaultdict(list)
        raise NotImplementedError

    @torch.no_grad()
    def collect(self, step: int) -> dict[str, float]:
        """Compute all metrics at current model state.

        Returns dict with all metric values for this step.
        """
        # TODO: implement
        raise NotImplementedError

    def to_dict(self) -> dict[str, list]:
        """Return full metric history."""
        # TODO: implement
        raise NotImplementedError

    def to_jsonl(self, path: str) -> None:
        """Write metric history as JSON-lines file."""
        # TODO: implement
        raise NotImplementedError


@torch.no_grad()
def compute_eval_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Standalone eval metrics: mse, linf, rel_l2."""
    # TODO: implement
    raise NotImplementedError
