"""expD07 suite verification: generators produce the requested spectra/ranks,
the metric plumbing reports the floor for a direct solve (oracle), the phi
floor is at the known fp64 construction precision, and the harness runs.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "expD07_lstsq_optim_suite"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO_ROOT))

import build_problems as bp  # noqa: E402
from common import digits, rel_excess  # noqa: E402
import run as harness  # noqa: E402


def test_synthetic_conditioning_and_consistency():
    A, y = bp.make_synthetic("illcond8_iso", bp.CELLS["illcond8_iso"],
                             N=32, R=64, seed=0)
    s = np.linalg.svd(A, compute_uv=False)
    assert abs(np.log10(s[0] / s[-1]) - 8.0) < 0.05
    v, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.mean((A @ v - y) ** 2) < 1e-20  # y is in the range of A


def test_rankdef_rank_and_inconsistency():
    Ac, yc = bp.make_synthetic("rankdef_consistent",
                               bp.CELLS["rankdef_consistent"], 32, 64, 1)
    s = np.linalg.svd(Ac, compute_uv=False)
    assert (s > 1e-10 * s[0]).sum() == int(0.6 * 32)
    v, *_ = np.linalg.lstsq(Ac, yc, rcond=None)
    assert np.mean((Ac @ v - yc) ** 2) < 1e-20

    Ai, yi = bp.make_synthetic("rankdef_inconsistent",
                               bp.CELLS["rankdef_inconsistent"], 32, 64, 1)
    v, *_ = np.linalg.lstsq(Ai, yi, rcond=None)
    assert np.mean((Ai @ v - yi) ** 2) > 1e-8  # off-range part stays


def test_phispec_matches_measured_phi_shape():
    """phispec cell = exponential decay throughout + rank cliff at 0.6m
    (the measured [Phi,1] shape from analyze_phi_regime.py)."""
    A, y = bp.make_synthetic("phispec_iso", bp.CELLS["phispec_iso"], 32, 64, 5)
    s = np.linalg.svd(A, compute_uv=False)
    r = int(0.6 * 32)
    assert (s > 1e-12 * s[0]).sum() == r
    assert abs(np.log10(s[0] / s[r - 1]) - 10.0) < 0.1
    v, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.mean((A @ v - y) ** 2) < 1e-18  # y stays in the row space


def test_solution_alignment():
    cfg = bp.CELLS["illcond8_tail"]
    A, y = bp.make_synthetic("illcond8_tail", cfg, 32, 64, 2)
    # y = A v* with v* only in the smallest singular directions -> tiny ||y||
    assert np.linalg.norm(y) < 1e-5
    cfg = bp.CELLS["illcond8_head"]
    A, y = bp.make_synthetic("illcond8_head", cfg, 32, 64, 2)
    assert np.linalg.norm(y) > 1e-2


def test_oracle_reaches_floor_through_metric():
    """An independent direct solve must report ~all digits via the harness
    metric -- validates the L*, L0, rel_excess plumbing."""
    A, y = bp.make_synthetic("illcond8_iso", bp.CELLS["illcond8_iso"],
                             32, 64, 3)
    v_ref, *_ = np.linalg.lstsq(A, y, rcond=None)
    l_star = np.mean((A @ v_ref - y) ** 2)
    l0 = np.mean(y ** 2)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # independent solver
    v2 = Vt.T @ ((U.T @ y) / s)
    ex = rel_excess(np.mean((A @ v2 - y) ** 2), l_star, l0)
    assert digits(ex) > 12


def test_phi_floor_machine_precision():
    arrays, meta = bp.make_phi("sine", N=64, ratio=2)
    assert meta["floor_eval_rel_l2"] < 1e-10
    assert meta["cols"] == arrays["centers"].size + 1
    assert meta["R"] == 2 * meta["cols"]


def _toy_entry(cell="wellcond_iso", seed=4):
    A, y = bp.make_synthetic(cell, bp.CELLS[cell], 32, 64, seed)
    v_ref, *_ = np.linalg.lstsq(A, y, rcond=None)
    entry = {"id": "t", "kind": "synthetic", "cls": cell,
             "N": 32, "R": 64,
             "L_star": float(np.mean((A @ v_ref - y) ** 2)),
             "L0": float(np.mean(y ** 2)),
             "smax": float(np.linalg.svd(A, compute_uv=False)[0])}
    return torch.from_numpy(A), torch.from_numpy(y), entry


def test_harness_optimize_smoke():
    A, y, entry = _toy_entry()
    spec = {"id": "adam", "family": "adam", "lr": 1e-2}
    rec = harness.optimize(A, y, entry, spec, steps=500)
    assert rec["digits"] > 2                      # Adam makes real progress
    assert rec["traj"]["step"][0] == 0
    assert rec["traj"]["step"][-1] == 500
    assert len(rec["traj"]["step"]) == len(rec["traj"]["rel_excess"])
    # rel L2 plumbing: starts at 1 (v0 = 0 -> ||y||/||y||), decreases, and
    # matches sqrt(train_mse / L0) by construction
    assert abs(rec["traj"]["rel_l2"][0] - 1.0) < 1e-12
    assert rec["final_rel_l2"] < rec["traj"]["rel_l2"][0]
    assert rec["floor_rel_l2"] < 1e-6


def test_cgls_reaches_floor():
    """CGLS is the matched Krylov method -- on a well-conditioned consistent
    system it must reach ~machine-precision rel L2 within ~rank iterations."""
    A, y, entry = _toy_entry(seed=6)
    rec = harness.optimize(A, y, entry, {"id": "cgls", "family": "cgls"},
                           steps=500)
    assert rec["final_rel_l2"] < 1e-12


def test_custom_families_run_and_descend():
    A, y, entry = _toy_entry(seed=7)
    for spec in [
        {"id": "nesterov", "family": "nesterov", "lr_rel": 1.0},
        {"id": "gd_bb", "family": "bb"},
        {"id": "lbfgs", "family": "lbfgs", "history": 20, "max_iter": 20},
        {"id": "adam_epsc", "family": "adam", "eps": 1e-3,
         "lr_mode": "eps_coupled"},
        {"id": "amsgrad", "family": "adam", "lr": 1e-3, "amsgrad": True},
        {"id": "bb_rms", "family": "bb_rms", "warmup": 50},
    ]:
        rec = harness.optimize(A, y, entry, spec, steps=300)
        assert np.isfinite(rec["final_rel_l2"]), spec["id"]
        assert rec["final_rel_l2"] < 0.5, spec["id"]
        assert rec["traj"]["step"][0] == 0, spec["id"]


def test_krylov_family_reaches_floor():
    """NLCG with exact line search == CG on a quadratic; LSQR is the stable
    bidiagonalization -- both must hit ~machine precision on a
    well-conditioned consistent system within the budget."""
    A, y, entry = _toy_entry(seed=8)
    for spec in [{"id": "nlcg_pr", "family": "nlcg"},
                 {"id": "lsqr", "family": "lsqr"},
                 {"id": "nlcg_restart", "family": "nlcg", "restart": 50},
                 {"id": "cgls_refine", "family": "cgls_refine", "cycle": 60},
                 {"id": "cgls_reortho", "family": "cgls_reortho"}]:
        rec = harness.optimize(A, y, entry, spec, steps=500)
        best = min(rec["traj"]["rel_l2"])
        assert best < 1e-12, (spec["id"], best)
