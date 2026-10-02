"""Tests for the Claude geometry reader (experiments/expG05_geometry_reader_claude).

What they pin down:
  * the standard geometry + least squares reaches the fp64 floor (the app shows the real QI result);
  * the analytic MSE gradient is exact (complex-step check) and Adam matches torch.optim.Adam;
  * geometry ops: resample keeps the arrangement and the mean lambda exactly, clean/jitter
    interpolate as documented;
  * recording: pause/resume and fork-from-a-saved-frame reproduce an uninterrupted run bit for bit;
  * staged edits inject only the touched entries and are recorded as flagged frames.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "experiments" / "expG05_geometry_reader_claude"
sys.path.insert(0, str(APP))

import engine as E  # noqa: E402
import presets  # noqa: E402
from app import Session, Store  # noqa: E402


def prob(**kw):
    return E.Problem(**kw).build()


# ---------------------------------------------------------------- numerics
def test_standard_geometry_reaches_fp64_floor():
    p = prob(target="sin(2*pi*x)", n_train=2048)
    P = E.uniform_geometry(137, p.domain, "sqrt", 0.25)
    a, b, info = E.solve_ls(p.x, p.y, P.c, P.g, p.rcond)
    m, _ = E.metrics(E.Params(P.c, P.g, a, b), p)
    assert m["rel_l2"] < 1e-13 and m["linf"] < 1e-12, m   # L_inf peaks at x=-1 (3e-13)
    lam = E.local_lambda(P.c, P.g)
    assert np.allclose(lam, 0.25)
    h = np.diff(np.sort(P.c))
    assert np.allclose(h, h[0], rtol=1e-10) and np.isclose(P.c[P.c >= -1 - 1e-12].min(), -1)


def test_gradient_is_exact_complex_step():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 40)
    y = np.sin(3 * x)
    P = E.Params(rng.uniform(-1, 1, 7), rng.uniform(1, 5, 7), rng.normal(size=7), 0.3)
    _, g = E.loss_and_grads(x, y, P)

    def loss(c, gg, a, b):
        r = np.tanh(gg[None, :] * (x[:, None] - c[None, :])) @ a + b - y
        return np.sum(r * r) / x.size

    h = 1e-30
    for key in ("c", "g", "a"):
        for k in range(7):
            args = {"c": P.c.astype(complex), "g": P.g.astype(complex), "a": P.a.astype(complex)}
            args[key][k] += 1j * h
            cs = loss(args["c"], args["g"], args["a"], P.b).imag / h
            assert abs(cs - g[key][k]) <= 1e-13 * max(1, abs(cs)), (key, k)
    cs = loss(P.c, P.g, P.a, P.b + 1j * h).imag / h
    assert abs(cs - g["b"][0]) < 1e-13


def test_adam_matches_torch():
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(1)
    x = np.linspace(-1, 1, 64)
    y = np.sin(2 * np.pi * x)
    P = E.Params(np.linspace(-1, 1, 9), np.full(9, 2.0), rng.normal(size=9) * 0.1, 0.05)
    cfg = E.AdamConfig(lr=3e-3, eps=1e-15)
    tp = {k: torch.tensor(np.atleast_1d(getattr(P, k)).astype(float), requires_grad=True) for k in "cgab"}
    topt = torch.optim.Adam(list(tp.values()), lr=3e-3, eps=1e-15)
    opt = E.Adam(9)
    xt, yt = torch.tensor(x), torch.tensor(y)
    for _ in range(200):
        topt.zero_grad()
        f = torch.tanh(tp["g"][None, :] * (xt[:, None] - tp["c"][None, :])) @ tp["a"] + tp["b"][0]
        ((f - yt) ** 2).mean().backward()
        topt.step()
        _, g = E.loss_and_grads(x, y, P)
        opt.step(P, g, cfg, cfg.lr)
    for k in "cga":
        np.testing.assert_allclose(getattr(P, k), tp[k].detach().numpy(), rtol=1e-10, atol=1e-12)
    assert abs(P.b - tp["b"].item()) < 1e-12


def test_snapshot_schedules():
    s = E.SnapConfig(mode="geometric", first=1, ratio=1.5, max_gap=100)
    st = s.steps(0, 50_000)
    gaps = np.diff([0] + st)
    assert st[-1] == 50_000 and np.all(gaps[:-1] >= 1)
    assert np.all(np.diff(gaps[:-1]) >= 0) and gaps[:-1].max() == 100
    assert len(st) < 600
    e = E.SnapConfig(mode="every", every=7)
    assert e.steps(0, 30) == [7, 14, 21, 28, 30]


# ---------------------------------------------------------------- geometry ops
def test_resample_keeps_arrangement_and_mean_lambda():
    rng = np.random.default_rng(2)
    P = E.uniform_geometry(60, (-1, 1), "none", 0.25)
    P = E.jitter_gamma(E.jitter_centers(P, 0.3, rng), 0.5, rng)
    lam0 = E.local_lambda(P.c, P.g).mean()
    for W in (30, 60, 121, 240):
        Q = E.resample(P, W)
        assert Q.W == W
        assert np.isclose(E.local_lambda(Q.c, Q.g).mean(), lam0, rtol=1e-12)
        assert np.isclose(Q.c.min(), P.c.min()) and np.isclose(Q.c.max(), P.c.max())
    Q = E.resample(P, 240)
    assert np.median(np.abs(Q.g)) > 3 * np.median(np.abs(P.g))       # gamma grows as spacing shrinks
    R = E.resample(P, 60)
    np.testing.assert_array_equal(R.c, P.c)


def test_clean_and_jitter():
    rng = np.random.default_rng(3)
    P0 = E.uniform_geometry(81, (-1, 1), "sqrt", 0.25)
    P = E.jitter_gamma(E.jitter_centers(P0, 0.5, rng), 0.7, rng)
    Q = E.clean(P, 1.0, "both", (-1, 1), "sqrt", 0.25)
    np.testing.assert_allclose(np.sort(Q.c), P0.c, atol=1e-14)
    np.testing.assert_allclose(E.local_lambda(Q.c, Q.g), 0.25, rtol=1e-12)
    np.testing.assert_array_equal(E.clean(P, 0.0, "both", (-1, 1), "sqrt", 0.25).c, P.c)
    half = E.clean(P, 0.5, "centers", (-1, 1), "sqrt", 0.25)
    o = np.argsort(P.c)
    np.testing.assert_allclose(half.c[o], (P.c[o] + P0.c) / 2, atol=1e-14)
    gq = E.clean(P, 1.0, "gamma", (-1, 1), "sqrt", 0.25)            # ideal gamma at the jumbled centers
    np.testing.assert_allclose(E.local_lambda(gq.c, gq.g), 0.25, rtol=1e-12)


def test_presets_build():
    for item in presets.catalog():
        P, has_readout, target = presets.build(item["key"], 81, (-1, 1), 0.25)
        assert np.all(P.g >= 0) and P.c.size == P.a.size
    P, has_readout, target = presets.build("xavier_adam_gn_best", 81, (-1, 1), 0.25)
    assert has_readout and P.W == 559
    p = prob(target=target, n_train=8193)
    m, _ = E.metrics(P, p)
    assert m["rel_l2"] < 1e-10           # the recorded trained model (L2RE 3.2e-11 in expD06)


# ---------------------------------------------------------------- session / recording
def make_session(tmp_path, W=41):
    s = Session(Store(tmp_path))
    s.dispatch("set_problem", {"n_train": 256, "n_eval": 300})
    s.dispatch("op", {"op": "preset", "key": "uniform_sqrt", "W": W})
    s.dispatch("readout_init", {"mode": "xavier", "seed": 1})
    return s


def wait_idle(s, timeout=60):
    t0 = time.time()
    while s.running and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert not s.running


def play(s, steps, **snap):
    return s.dispatch("play", {"steps": steps, "adam": {"lr": 1e-3},
                               "snap": {"mode": "every", "every": 10, **snap}})


def test_pause_resume_and_fork_are_bit_identical(tmp_path):
    s1 = make_session(tmp_path / "a")
    play(s1, 200)
    wait_idle(s1)
    ref = s1.run.P.copy()

    s2 = make_session(tmp_path / "b")
    play(s2, 100)
    wait_idle(s2)
    s2.dispatch("play", {"steps": 100})          # extends a finished run
    wait_idle(s2)
    np.testing.assert_array_equal(s2.run.P.c, ref.c)
    np.testing.assert_array_equal(s2.run.P.a, ref.a)

    # replay the first run, scrub to step 100, fork (by editing nothing: stop at frame), continue 100
    s3 = make_session(tmp_path / "a")
    rid = s3.store.list()[0]["id"]
    s3.dispatch("load", {"id": rid})
    i100 = int(np.flatnonzero(s3.replay["arrays"]["step"] == 100)[0])
    s3.dispatch("seek", {"i": i100})
    s3.dispatch("stop", {})                      # ends replay, state = frame, fork keeps Adam moments
    assert s3.fork and s3.fork["step"] == 100
    play(s3, 100)
    wait_idle(s3)
    np.testing.assert_array_equal(s3.run.P.g, ref.g)
    np.testing.assert_array_equal(s3.run.P.a, ref.a)
    assert s3.run.parent["run_id"] == rid


def test_staged_edit_injects_only_touched_entries(tmp_path):
    s = make_session(tmp_path)
    play(s, 60)
    wait_idle(s)
    before = s.run.P.copy()
    s.dispatch("edit", {"g": {"5": before.g[5] * 2}, "a": {"3": 0.123}})
    assert s.staged is not None
    assert s.dispatch("play", {"steps": 10}).get("needs_decision")   # play refuses with staged edits
    n0 = len(s.run.frames)
    s.dispatch("inject", {})
    assert s.staged is None
    iv = s.run.interventions[-1]
    assert iv["kind"] == "geometry+readout" and iv["changed"]["g"] == [5] and iv["changed"]["a"] == [3]
    assert len(s.run.frames) == n0 + 1
    assert s.run.P.g[5] == before.g[5] * 2 and s.run.P.a[3] == 0.123
    np.testing.assert_array_equal(np.delete(s.run.P.g, 5), np.delete(before.g, 5))
    # inject least squares: readout replaced, geometry untouched
    s.dispatch("inject_ls", {})
    assert s.run.interventions[-1]["kind"] == "readout"
    # a knob ends the run and saves it with its interventions
    s.dispatch("set_problem", {"noise": 0.0, "n_train": 300})
    assert s.run is None
    meta, arrays = s.store.load(s.store.list()[0]["id"])
    assert meta["n_interventions"] == 2 and arrays["c"].shape[1] == 41


def test_view_builds_in_every_mode(tmp_path):
    s = make_session(tmp_path)
    v = s.build_view()
    assert v["mode"] == "idle" and len(v["geom"]["c"]) == 41
    play(s, 30)
    wait_idle(s)
    s.dispatch("edit", {"c": {"2": float(s.run.P.c[2]) + 0.01}})
    v = s.build_view()
    assert v["mode"] == "paused" and v["has_staged"] and v["changed"]["c"] == [2]
    s.dispatch("stop", {})
    s.dispatch("load", {"id": s.store.list()[0]["id"]})
    v = s.build_view()
    assert v["mode"] == "replay" and v["run"]["n_frames"] >= 4


# ---------------------------------------------------------------- regressions from the review
def test_divergence_pauses_cleanly(tmp_path):
    s = make_session(tmp_path)
    s.dispatch("play", {"steps": 500, "adam": {"lr": 1e6, "gamma_param": "log"}})
    wait_idle(s)
    assert s.run is not None and not s.running
    assert "diverged" in s.message or "error" in s.message
    s.dispatch("stop", {})                       # still controllable
    assert s.run is None
    with pytest.raises(ValueError):
        s.dispatch("edit", {"g": {"0": float("nan")}})


def test_undo_redo_in_a_run_keeps_the_staged_base(tmp_path):
    s = make_session(tmp_path)
    play(s, 40)
    wait_idle(s)
    s.dispatch("edit", {"c": {"4": float(s.run.P.c[4]) + 0.003}})
    s.dispatch("undo", {})
    assert s.staged is None
    s.dispatch("play", {"steps": 20})           # the run advances past the staging frame
    wait_idle(s)
    s.dispatch("redo", {})
    s.dispatch("inject", {})
    ch = s.run.interventions[-1]["changed"]
    assert ch["c"] == [4] and ch["g"] == [] and ch["a"] == [] and not ch["b"]


def test_sort_and_scrubbed_edits_refused_during_run(tmp_path):
    s = make_session(tmp_path)
    play(s, 40)
    wait_idle(s)
    with pytest.raises(ValueError):
        s.dispatch("op", {"op": "sort"})
    s.dispatch("seek", {"i": 1})
    with pytest.raises(ValueError):
        s.dispatch("edit", {"g": {"0": 1.0}})
    assert s.staged is None


def test_full_reset_ends_run_and_restores_defaults(tmp_path):
    s = make_session(tmp_path)
    s.dispatch("set_problem", {"target": "exp(x)"})
    s.dispatch("play", {"steps": 100000, "adam": {"lr": 1e-3}})
    time.sleep(0.2)
    s.dispatch("reset", {})
    assert s.run is None and not s.running and s.replay is None and s.staged is None
    assert s.prob.target == "sin(2*pi*x)" and s.working.W == 81
    assert len(s.store.list()) == 1                         # the interrupted run was saved, not lost
    time.sleep(0.1)
    assert s.run is None                                    # the old training thread did not come back


def test_default_halo_is_max_10_sqrt_n():
    for N in (16, 64, 100, 265, 1000):
        R = max(10, int(np.ceil(np.sqrt(N))))
        P = E.uniform_geometry(N + 1 + 2 * R, (-1, 1), "sqrt", 0.25)
        assert int((P.c < -1 - 1e-12).sum()) == R and int((P.c > 1 + 1e-12).sum()) == R
        np.testing.assert_allclose(np.diff(P.c), 2 / N, rtol=1e-9)
