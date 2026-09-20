"""expD17 sanity gates: the QI-init geometry probe reaches the floor on 1-D sine,
and the drift metric is exactly zero at iteration 0 (before any Adam step)."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "expD17_run", REPO_ROOT / "experiments" / "expD17_geometry_motion" / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)


def test_qi_init_probe_reaches_floor_1d_sine():
    b = _run.build_interp1d("sine", "qi")
    pre = b["probe_fn"]()
    assert pre < 1e-12, f"QI-init geometry probe should hit the fp64 floor, got {pre:.3e}"


def test_drift_zero_at_init():
    b = _run.build_interp1d("sine", "qi")
    g0 = _run.geom_flat(b["geom_params"])
    g1 = _run.geom_flat(b["geom_params"])
    assert float(torch.linalg.norm(g1 - g0)) == 0.0


def test_readout_zeroed_both_arms():
    for arm in ("qi", "standard"):
        b = _run.build_interp1d("sine", arm)
        m = b["model"]
        assert float(m.readout.weight.abs().max()) == 0.0
        assert float(m.readout.bias.abs().max()) == 0.0
