"""Gate tests for expD21: the five normalization variants are a fair comparison.

The point of expD21 is a selection decision, so the comparison has to be clean:
identical trainable parameter counts, identical geometry, and each variant a
pure reparameterization of the same represented function at init. expD19's
BN/LN arms failed the first of those (2W extra parameters), which is what these
tests pin down so it cannot regress.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "experiments" / "expD21_norm_probe" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("expD21_run_undertest", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()


def _built(variant, act="gelu", N=64, seed=0):
    b = R.build_interp1d("sine", act, variant, N, seed)
    model = b["model"]
    if variant == "rms_nocenter":
        R.set_static_(model, b["x_ref"], center=False)
    elif variant == "rms_center":
        R.set_static_(model, b["x_ref"], center=True)
    R.prime_bn_(model, b["x_ref"])
    return b, model


@pytest.mark.parametrize("act", ["tanh", "gelu"])
def test_identical_parameter_counts(act):
    """Every variant must train the same number of parameters."""
    counts = {}
    for v in R.VARIANTS:
        _, m = _built(v, act=act)
        counts[v] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert len(set(counts.values())) == 1, counts


def test_norm_layers_have_no_parameters():
    for v in ("batchnorm_noaffine", "layernorm_noaffine"):
        _, m = _built(v)
        assert m.norm is not None
        assert sum(p.numel() for p in m.norm.parameters()) == 0, v


def test_identical_geometry_across_variants():
    """The halo is held fixed, so the inner layer is bit-identical everywhere."""
    ref = None
    for v in R.VARIANTS:
        _, m = _built(v)
        w = torch.cat([m.inner.weight.reshape(-1), m.inner.bias.reshape(-1)])
        if ref is None:
            ref = w
        else:
            assert torch.equal(ref, w), v


def test_all_variants_represent_the_same_function_at_init():
    """Readout zeroed -> every variant outputs exactly zero at init, so each is
    a pure reparameterization and the comparison isolates gradient geometry."""
    for v in R.VARIANTS:
        b, m = _built(v)
        m.eval()
        with torch.no_grad():
            out = m(b["x_ref"])
        assert torch.allclose(out, torch.zeros_like(out), atol=0.0), v


def test_rms_center_normalizes_and_nocenter_does_not_shift():
    """rms_center produces zero-mean unit-variance columns on the live set;
    rms_nocenter leaves the mean alone but sets unit RMS."""
    b, m = _built("rms_center")
    with torch.no_grad():
        F = m.features(b["x_ref"])
        live = m.raw_features(b["x_ref"]).std(0) > 1e-300
    assert F[:, live].mean(0).abs().max() < 1e-10
    assert (F[:, live].pow(2).mean(0).sqrt() - 1).abs().max() < 1e-10

    b, m = _built("rms_nocenter")
    with torch.no_grad():
        F = m.features(b["x_ref"])
        live = m.raw_features(b["x_ref"]).pow(2).mean(0).sqrt() > 1e-300
    assert (F[:, live].pow(2).mean(0).sqrt() - 1).abs().max() < 1e-10
    assert m.col_shift.abs().max() == 0.0


def test_batchnorm_eps_floor_is_why_it_leaves_spread():
    """The mechanism behind expD19's puzzle (BN won the PINN row while barely
    reducing the column-norm spread): BN divides by sqrt(var + eps), so columns
    with var < eps are scaled but never normalized. rms_center divides by the
    true std and flattens them."""
    b, m_bn = _built("batchnorm_noaffine")
    _, m_rc = _built("rms_center")
    sign = R.preact_sign(m_bn.inner, b["x_ref"])
    s_bn = R.col_norm_stats(m_bn, b["x_ref"], sign)["max_over_min_live"]
    s_rc = R.col_norm_stats(m_rc, b["x_ref"], sign)["max_over_min_live"]
    with torch.no_grad():
        var = m_bn.raw_features(b["x_ref"]).var(0, unbiased=False)
    assert (var < m_bn.norm.eps).any(), "no sub-eps columns: test is vacuous here"
    assert s_rc < 1.0 + 1e-9 < s_bn, (s_rc, s_bn)


def test_seed_axis_changes_the_data():
    """Seeds vary the DATA realization; without this the runs are deterministic
    and a robustness claim over seeds would be vacuous."""
    x0 = R.build_interp1d("sine", "gelu", "baseline", 64, 0)["x_ref"]
    x1 = R.build_interp1d("sine", "gelu", "baseline", 64, 1)["x_ref"]
    assert not torch.equal(x0, x1)
    assert x0.min() >= -1.0 and x0.max() <= 1.0
    assert x1.min() >= -1.0 and x1.max() <= 1.0
