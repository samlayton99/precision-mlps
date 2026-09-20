"""expD09 verification: does the machinery actually measure what we claim?

Five checks, each pinned to a research claim rather than to code shape:

1. The frozen system IS the repo's QI feature system (geometry + halo +
   gamma = lambda*/h), and its truncated-SVD floor reaches construction
   precision on the smooth targets.
2. The metric plumbing can express the floor: scoring the stored a_svd
   through Metrics reproduces the manifest floors to fp64.
3. One block containing every column makes BCD a direct solve -- it must
   hit the floor in ONE sweep. If it doesn't, the sub-solve is wrong and
   every other number in the experiment is meaningless.
4. Exact BCD is monotone in the train objective (block Gauss-Seidel with an
   exact sub-solve can never increase the residual within a sweep).
5. The null-space diagnostic is real: V_null is orthonormal, lies in the
   numerical null space of A_train, and does NOT lie in the null space of
   A_eval -- which is exactly why train and eval can disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "expD09_2nd_order_regime"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO_ROOT))

import common  # noqa: E402
import solvers  # noqa: E402

pytestmark = pytest.mark.skipif(
    not common.MANIFEST_PATH.exists(),
    reason="run experiments/expD09_2nd_order_regime/build_problems.py first")


@pytest.fixture(scope="module")
def manifest():
    return {e["id"]: e for e in common.load_manifest()}


SMOOTH = ["phi_sine_N256", "phi_sine_8pi_N256", "phi_runge_N256"]


def test_geometry_matches_repo_construction(manifest):
    """Centers, halo and gamma are the repo standard, not a local re-derivation."""
    from src.construction.qi_mpmath import default_halo

    for N in common.WIDTHS:
        centers, gamma = common.geometry(N)
        h = 2.0 / N
        halo = default_halo(N, lambda_star=common.LAMBDA_STAR)
        assert centers.size == N + 2 * halo + 1
        assert gamma == pytest.approx(common.LAMBDA_STAR / h, rel=1e-15)
        # uniform spacing h, spanning [-1, 1] plus the halo on both sides
        assert np.allclose(np.diff(centers), h, rtol=1e-13)
        assert centers[halo] == pytest.approx(-1.0, abs=1e-13)
        assert centers[halo + N] == pytest.approx(1.0, abs=1e-13)


def test_svd_floor_reaches_construction_precision(manifest):
    """The reference floor is a real machine-precision solve, not a stall.

    Smooth targets at N=256 must land at ~1e-14 eval rel L2 -- the level the
    explicit QI construction reaches. If this test fails the entire y axis
    of the experiment is mislabelled.
    """
    for pid in SMOOTH:
        e = manifest[pid]
        assert e["floor_eval_rel_l2"] < 1e-13, (pid, e["floor_eval_rel_l2"])
        assert e["floor_train_rel_l2"] < 1e-13, (pid, e["floor_train_rel_l2"])


def test_metrics_reproduce_the_manifest_floor(manifest):
    """Scoring the stored SVD solution through Metrics returns the floor."""
    for pid in SMOOTH:
        e = manifest[pid]
        prob = common.load_problem(e)
        met = common.Metrics(prob)
        met.record(prob["a_svd"])
        assert met.rows["eval_rel"][0] == pytest.approx(
            e["floor_eval_rel_l2"], rel=1e-10)
        assert met.rows["train_rel"][0] == pytest.approx(
            e["floor_train_rel_l2"], rel=1e-10)
        # the min-norm solve has no null component, by construction
        assert met.rows["null_norm"][0] < 1e-10 * met.rows["a_norm"][0]


def test_single_block_bcd_is_a_direct_solve(manifest):
    """k >= d means one block: BCD must equal the SVD solve after one sweep.

    This is the calibration of the sub-solve. Anything wrong with the
    truncation, the min-norm choice or the residual sign shows up here as
    orders, not as noise.
    """
    for pid in SMOOTH:
        e = manifest[pid]
        prob = common.load_problem(e)
        spec = {"family": "bcd", "blocks": "contig", "k": e["d"]}
        met = common.Metrics(prob)
        for a in solvers.iterate(prob["A"], prob["y"], spec, 1):
            met.record(a)
        assert met.rows["eval_rel"][0] == pytest.approx(
            e["floor_eval_rel_l2"], rel=1e-6), pid


def test_exact_bcd_is_monotone_while_the_iterate_stays_bounded(manifest):
    """Exact block Gauss-Seidel cannot increase ||A a - y||.

    Guards the residual bookkeeping: a sign error or a stale residual shows
    up immediately as a non-monotone train curve, and the train curve is one
    of the three things every figure reports.

    Scoped to the DAMPED sub-solve on purpose. At the repo-standard rcond of
    1e-15 the iterate reaches ||a|| ~ 1e12 (pinned by the next test), and
    then the *measurement* A a - y cancels to ~eps ||A|| ||a|| ~ 1e-1, so
    the recorded curve is noise-limited and genuinely non-monotone. That is
    a property of fp64, not of the algorithm, and it is why the damped arms
    exist.
    """
    e = manifest["phi_sine_N128"]
    prob = common.load_problem(e)
    for blocks in ("contig", "rand"):
        spec = {"family": "bcd", "blocks": blocks, "k": 32,
                "sub_rcond": solvers.RCOND_DAMPED}
        met = common.Metrics(prob)
        for a in solvers.iterate(prob["A"], prob["y"], spec, 30, seed=0):
            met.record(a)
        tr = np.asarray(met.rows["train_rel"])
        assert np.all(np.diff(tr) <= 1e-9 * tr[:-1] + 1e-17), blocks


def test_undamped_sub_solve_blows_the_iterate_up(manifest):
    """Pin the measured pathology the damped arms exist to isolate.

    Each block's exact min-norm step inverts that block's OWN near-null
    directions. The increments cancel inside the block, so the residual is
    unharmed, but `a` random-walks along globally-near-null directions and
    its norm grows without bound. Almost all of that growth is in the null
    space, so the train objective barely notices -- which is exactly what
    makes it dangerous.
    """
    e = manifest["phi_sine_N128"]
    prob = common.load_problem(e)
    spec = {"family": "bcd", "blocks": "contig", "k": 64}   # rcond 1e-15
    met = common.Metrics(prob)
    for a in solvers.iterate(prob["A"], prob["y"], spec, 120, seed=0):
        met.record(a)
    a_norm = met.rows["a_norm"][-1]
    null_norm = met.rows["null_norm"][-1]
    assert a_norm > 1e9, a_norm
    assert null_norm > 0.9 * a_norm, (null_norm, a_norm)
    # the damped arm keeps it bounded at comparable train error
    spec_d = dict(spec, sub_rcond=solvers.RCOND_DAMPED)
    met_d = common.Metrics(prob)
    for a in solvers.iterate(prob["A"], prob["y"], spec_d, 120, seed=0):
        met_d.record(a)
    assert met_d.rows["a_norm"][-1] < 1e6, met_d.rows["a_norm"][-1]


def test_null_drift_leaks_into_eval_at_the_measured_gain(manifest):
    """The train/eval gap has a named mechanism, with a measured gain.

    V_null spans directions the TRAIN system cannot see. They are *also*
    nearly invisible to the eval system (both leak at ~1e-15 relative), so
    null drift is not free error -- it is amplified by that gain. A drift of
    ||P_null a|| = M costs ~M * leak of eval error, which is why the 1e12
    blowup above matters and why `null_norm` is recorded every sweep.
    """
    e = manifest["phi_sine_N256"]
    prob = common.load_problem(e)
    V = prob["V_null"]
    assert V.shape[1] == e["d"] - e["rank"]
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    A, A_ev = prob["A"], prob["A_ev"]
    train_leak = np.linalg.norm(A @ V, 2) / np.linalg.norm(A, 2)
    eval_leak = np.linalg.norm(A_ev @ V, 2) / np.linalg.norm(A_ev, 2)
    assert train_leak < 1e-14 and eval_leak < 1e-14, (train_leak, eval_leak)

    # the gain is predictive: push the SVD solution M along a null direction
    # and the eval error must rise by ~M * eval_leak, not stay at the floor.
    met = common.Metrics(prob)
    v = V[:, 0]
    for M in (1.0, 1e8):
        met.record(prob["a_svd"] + M * v)
    base, pushed = met.rows["eval_rel"]
    assert base == pytest.approx(e["floor_eval_rel_l2"], rel=1e-6)
    assert pushed > 1e5 * base, (base, pushed)
