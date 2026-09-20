"""expD16 sanity: the QI init geometry is correct (lstsq readout hits the floor),
and the Xavier control is not.

Loads run.py by explicit path (never `import run` -- module-name collision bug,
see docs/ORIENTATION.md section 7b).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def d16():
    path = REPO_ROOT / "experiments" / "expD16_optimizer_zoo" / "run.py"
    spec = importlib.util.spec_from_file_location("expD16_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refit_rel_l2(model, bundle):
    """Freeze the model's geometry; solve the readout by fp64 truncated-SVD
    lstsq with a bias column; return eval rel L2."""
    X_tr, y_tr, X_ev, y_ev, y_norm = bundle
    with torch.no_grad():
        Phi_tr = model.features(X_tr).numpy()
        Phi_ev = model.features(X_ev).numpy()
    Aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(s > 1e-13 * s[0], 1.0 / np.where(s > 0, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv * (U.T @ y_tr.numpy().ravel()))
    pred = Phi_ev @ sol[:-1] + sol[-1]
    return float(np.linalg.norm(pred - y_ev.numpy().ravel()) / y_norm)


def test_qi_init_geometry_reaches_floor(d16):
    """QI init + exact readout solve must reach ~1e-13 rel L2 on sine."""
    model = d16.build_model("qi", 128, seed=0)
    bundle = d16.data_bundle("sine")
    err = _refit_rel_l2(model, bundle)
    print(f"QI init + lstsq readout, sine N=128: rel L2 = {err:.3e}")
    assert err < 1e-12, f"QI geometry broken: lstsq floor {err:.3e}"


def test_xavier_init_geometry_does_not(d16):
    """The Xavier control geometry must NOT be at the floor (sanity contrast)."""
    model = d16.build_model("xavier", 128, seed=0)
    bundle = d16.data_bundle("sine")
    err = _refit_rel_l2(model, bundle)
    print(f"Xavier init + lstsq readout, sine N=128: rel L2 = {err:.3e}")
    assert err > 1e-10


def test_qi_init_values(d16):
    """gamma = lambda*/h on every neuron; centers are the uniform grid + halo."""
    N = 64
    model = d16.build_model("qi", N, seed=0)
    W, c_uniform, gamma, h, halo = d16.geometry_for_N(N)
    w = model.inner_layer.linear.weight.detach().numpy().ravel()
    b = model.inner_layer.linear.bias.detach().numpy().ravel()
    assert np.allclose(w, gamma)
    assert np.allclose(-b / w, c_uniform)
    assert W == N + 2 * halo + 1
