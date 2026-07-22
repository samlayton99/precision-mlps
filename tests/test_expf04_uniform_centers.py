import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "experiments"
    / "expF04_qi_init_real_data"
    / "all20_2layers_uniform"
    / "run.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("expf04_uniform_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self):
        self.fc1 = object()
        self.fc2 = object()

    def hidden1(self, x):
        return x


@pytest.mark.parametrize(
    "scheme,expected_layers",
    [
        ("baseline", []),
        ("qi1", ["fc1"]),
        ("qi2", ["fc1", "fc2"]),
    ],
)
def test_uniform_runner_initializes_expected_layers_with_uniform_centers(
    monkeypatch, scheme, expected_layers
):
    runner = _load_runner()
    model = _FakeModel()
    layer_names = {model.fc1: "fc1", model.fc2: "fc2"}
    calls = []

    def record_init(layer, x_input, **kwargs):
        calls.append((layer_names[layer], kwargs))

    monkeypatch.setattr(runner, "qi_ridge_init_layer_", record_init)
    runner.apply_init(model, scheme, torch.zeros(8, 3), p=4)

    assert [name for name, _ in calls] == expected_layers
    assert all(kwargs["uniform_centers"] is True for _, kwargs in calls)
