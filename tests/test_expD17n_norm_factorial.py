"""Gate tests for expD17/norm_factorial.

The three properties the factorial's validity rests on:

1. `rms_center` adds NO parameters, so arms differ only in the reparameterization
   (expD19's BatchNorm/LayerNorm arms carried 2W extra parameters and that
   confounded every comparison there).
2. `rms_center` is a pure reparameterization at init: it does not change what
   the geometry can express, only the gradient geometry. Measured as the
   pre-train lstsq probe being unchanged.
3. The QI init still reaches the fp64 floor, i.e. the geometry under test is
   the correct one.

Loaded by explicit path: several experiments define a `run.py`, and a bare
`import run` binds whichever is already in sys.modules (ORIENTATION 7b).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN = (REPO_ROOT / "experiments" / "expD17_geometry_motion"
        / "norm_factorial" / "run.py")


def _load():
    spec = importlib.util.spec_from_file_location("expD17n_run", _RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expD17n_run"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def NF():
    return _load()


def test_arms_are_the_six_expected(NF):
    assert NF.ARMS == ["tanh_none_std", "tanh_none_qi",
                       "gelu_none_std", "gelu_none_qi",
                       "gelu_rmsc_std", "gelu_rmsc_qi"]
    # tanh x rms_center is deliberately absent (expD21: 0.79x, 32x worst case)
    assert not any(a.startswith("tanh") and "rmsc" in a for a in NF.ARMS)


def test_normalization_adds_no_parameters(NF):
    """Within an activation, every arm has an identical parameter count."""
    counts = {}
    for arm in NF.ARMS:
        b = NF.build("interp1d", "sine", arm, 64)
        counts[arm] = sum(p.numel() for p in b["model"].parameters())
    assert counts["gelu_none_std"] == counts["gelu_none_qi"] \
        == counts["gelu_rmsc_std"] == counts["gelu_rmsc_qi"]
    assert counts["tanh_none_std"] == counts["tanh_none_qi"]


def test_rms_center_is_a_reparameterization(NF):
    """Normalization must not change what the geometry can express: the
    pre-train lstsq probe is the same with and without it."""
    plain = NF.build("interp1d", "sine", "gelu_none_qi", 64)["probe_fn"]()
    normed = NF.build("interp1d", "sine", "gelu_rmsc_qi", 64)["probe_fn"]()
    assert plain < 1e-12 and normed < 1e-12
    assert abs(plain - normed) < 1e-12


def test_rms_center_flattens_the_column_spread(NF):
    """The pathology it targets: GELU + QI init gives a column-norm spread of
    many orders; rms_center takes it to exactly 1."""
    out = {}
    for arm in ("gelu_none_qi", "gelu_rmsc_qi"):
        b = NF.build("interp1d", "sine", arm, 64)
        sc = NF.WS.preact_sign_class(b["inner"], b["x_ref"])
        out[arm] = NF.col_norm_stats(b["model"], b["x_ref"], sc)["max_over_min_live"]
    assert out["gelu_none_qi"] > 1e6
    assert out["gelu_rmsc_qi"] == pytest.approx(1.0, abs=1e-9)


def test_qi_geometry_reaches_the_floor(NF):
    """The QI init under test is the correct geometry: an lstsq readout on it
    reaches the fp64 floor, while the standard init does not."""
    qi = NF.build("interp1d", "sine", "tanh_none_qi", 128)["probe_fn"]()
    std = NF.build("interp1d", "sine", "tanh_none_std", 128)["probe_fn"]()
    assert qi < 1e-12, f"QI geometry did not reach the floor: {qi:.3e}"
    assert std > 1e-6, f"standard init unexpectedly at the floor: {std:.3e}"


def test_tabular_is_two_layers_with_layer1_geometry(NF):
    """Tabular must be depth 2, with only the first layer tracked as geometry."""
    b = NF.build("tabular", "kin8nm", "gelu_none_qi", 128)
    m = b["model"]
    assert hasattr(m, "hidden2"), "tabular arm is not two hidden layers"
    assert b["extra"]["depth"] == 2
    # geometry = first layer only
    assert b["geom_params"][0] is m.inner.weight
    assert b["geom_params"][1] is m.inner.bias
    n_geom = sum(p.numel() for p in b["geom_params"])
    assert n_geom < sum(p.numel() for p in m.parameters())


def test_tabular_metric_is_variance_normalized(NF):
    """expD20's prep standardizes the target on train, so the zero-readout
    model (which predicts the train mean) scores ~1.0 -- a mean-predictor no
    longer gets free credit for the offset."""
    b = NF.build("tabular", "kin8nm", "tanh_none_qi", 128)
    assert b["eval_fn"]() == pytest.approx(1.0, rel=0.1)
