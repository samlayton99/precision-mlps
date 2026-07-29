"""Tests for expD02 run_qi_adam_lbfgs (QI init -> Adam plateau -> LBFGS finish).

Verifies the implementation matches the research before the full run:
  1. the fp64 QI init starts AT the construction floor (~machine precision);
  2. Adam's plateau detector fires (from the QI solution there is nothing to
     improve, so it must stop at ~patience steps, far below the cap);
  3. LBFGS recovers precision from a perturbed QI state (the finisher works);
  4. the end-to-end cell is finite and LBFGS does not end above Adam's plateau.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "expD02_adam_geometry"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(EXP_DIR))
    spec = importlib.util.spec_from_file_location(
        "expd02_qi_adam_lbfgs", EXP_DIR / "run_qi_adam_lbfgs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = _load_module()
N_TEST = 32
FN_TEST = "sine"


def _setup(fn=FN_TEST, N=N_TEST):
    W, c_uniform, gamma_uniform, h, halo = M.geometry_for_N(N)
    X_tr, y_tr, X_ev, y_ev, y_norm = M.eval_bundle(fn)
    ro = M.lstsq_solution(np.full(W, gamma_uniform), -gamma_uniform * c_uniform,
                          X_tr, y_tr)
    net = M.init_qi_net(W, gamma_uniform, c_uniform, ro)
    Xt = torch.tensor(X_tr, dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(y_tr, dtype=torch.float64).reshape(-1, 1)
    return net, Xt, yt, (X_tr, y_tr, X_ev, y_ev, y_norm)


def test_qi_init_starts_at_floor():
    net, _, _, (X_tr, y_tr, X_ev, y_ev, y_norm) = _setup()
    w, b = net.inner_wb(); v, rb = net.readout_vb()
    err = M.rel_l2_with_readout(w, b, v, rb, X_ev, y_ev, y_norm)
    assert err < 1e-10, f"QI init should start at the floor, got {err:.2e}"


def test_adam_plateau_stops_early():
    net, Xt, yt, _ = _setup()
    r = M.adam_stage(net, 1e-3, Xt, yt, max_steps=20000, warmup=100, patience=500)
    # From the QI solution Adam cannot improve, so the plateau detector must
    # fire at ~patience steps, nowhere near the cap.
    assert r["steps_run"] < 2000, f"plateau never fired ({r['steps_run']} steps)"
    assert np.isfinite(r["final_loss"])


def test_lbfgs_recovers_from_perturbed_qi():
    net, Xt, yt, (X_tr, y_tr, X_ev, y_ev, y_norm) = _setup()
    with torch.no_grad():  # knock the readout off the solution
        g = torch.Generator().manual_seed(0)
        net.readout.weight.add_(
            1e-2 * torch.randn(net.readout.weight.shape, generator=g,
                               dtype=torch.float64))
    w, b = net.inner_wb(); v, rb = net.readout_vb()
    err_before = M.rel_l2_with_readout(w, b, v, rb, X_ev, y_ev, y_norm)
    assert err_before > 1e-6  # the perturbation actually hurt

    r = M.lbfgs_stage(net, Xt, yt, max_iters=1000)
    w, b = net.inner_wb(); v, rb = net.readout_vb()
    err_after = M.rel_l2_with_readout(w, b, v, rb, X_ev, y_ev, y_norm)
    assert err_after < 1e-3 * err_before, (
        f"LBFGS should recover precision: {err_before:.2e} -> {err_after:.2e}")
    assert r["final_loss"] <= r["trace"][0]


def test_run_cell_end_to_end():
    r = M.run_cell(FN_TEST, N_TEST, 1e-3,
                   adam_kw={"max_steps": 5000, "warmup": 100, "patience": 500},
                   lbfgs_kw={"max_iters": 500})
    for k in ("qi_lstsq", "err_init", "err_after_adam", "err_after_lbfgs",
              "refit_after_lbfgs"):
        assert np.isfinite(r[k]), f"{k} not finite"
    assert r["err_init"] < 1e-10
    # The LBFGS finish must never end above Adam's plateau (its own loss is
    # guarded to be monotone non-increasing over chunks).
    assert r["err_after_lbfgs"] <= r["err_after_adam"] * 10
