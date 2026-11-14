"""Checks for the two failure modes that could manufacture apparent QI damage."""
import json
import numpy as np
import torch

from experiments.expD55_qi_adam_drift.run import (
    HERE, construct_qi, evaluation_steps, lr_at, make_model, make_standard_model, run_target, target,
)
from src.construction import evaluate_qi


def test_transfer_preserves_constructed_function():
    torch.set_num_threads(1)
    f, df = target("square")
    q = construct_qi(f, df, N=128, precision="mpmath")
    net = make_model(q)
    x = np.linspace(-1, 1, 127)
    with torch.no_grad():
        pred = net(torch.tensor(x[:, None], dtype=torch.float64)).numpy().ravel()
    np.testing.assert_allclose(pred, evaluate_qi(q, x), atol=3e-15, rtol=0)
    assert np.linalg.norm(pred - f(x)) / np.linalg.norm(f(x)) < 1e-13


def test_no_update_control_and_fresh_start(tmp_path):
    c = json.loads((HERE / "config.json").read_text())
    c.update(steps=3, warmup_steps=2, learning_rates=[0., 1e-3], train_points=64, eval_points=128)
    run_target("square", c, tmp_path)
    control = json.loads((tmp_path / "lr_0e+00.json").read_text())["history"]
    trained = json.loads((tmp_path / "lr_1e-03.json").read_text())["history"]
    assert control[0] == trained[0]
    assert [p["step"] for p in trained] == [0, 1, 2, 3]
    assert all(p["eval_rel_l2"] == control[0]["eval_rel_l2"] for p in control)
    initial = torch.load(tmp_path / "initial.pt", weights_only=True)
    final = torch.load(tmp_path / "lr_0e+00_final.pt", weights_only=True)
    assert all(torch.equal(initial[k], final[k]) for k in initial)
    grids = np.load(tmp_path / "qi.npz")
    assert np.intersect1d(grids["train_x"], grids["eval_x"]).size == 0


def test_schedule_and_logged_endpoint():
    for warmup in (1000, 2000):
        assert lr_at(1, .01, warmup) == .01 / warmup
        assert lr_at(warmup, .01, warmup) == .01
        assert lr_at(10000, .01, warmup) == .01
    assert evaluation_steps(10001)[-1] == 10001


def test_display_smoothing_preserves_onset_and_uses_time_spacing():
    from experiments.expD55_qi_adam_drift.plot import smooth_error
    x = np.arange(0, 301, 25)
    y = np.full(x.shape, 1e-6)
    y[0], y[8] = 1e-15, 1e-1
    result = smooth_error(x, y)
    np.testing.assert_array_equal(result[:5], y[:5])
    np.testing.assert_allclose(result[8], 1e-5)  # geometric mean over steps 150..250
    np.testing.assert_allclose(result[-1], 1e-6)  # truncated endpoint window
    # Adding early logging points must not change the smoothed later trajectory.
    dense_x = np.r_[np.arange(0, 101), x[5:]]
    dense_y = 10 ** np.interp(dense_x, x, np.log10(y))
    dense_result = smooth_error(dense_x, dense_y)
    np.testing.assert_allclose(dense_result[101:], result[5:])


def test_hybrid_step_axis_spacing_and_inverse():
    from experiments.expD55_qi_adam_drift.plot import step_axis_forward, step_axis_inverse
    x = np.r_[0., .5, 1., np.geomspace(2., 1000., 101), np.linspace(1001., 10000., 101)]
    z = step_axis_forward(x)
    assert np.all(np.diff(z) > 0)
    np.testing.assert_allclose(step_axis_inverse(z), x, atol=1e-10)
    log_gaps = np.diff(step_axis_forward([1., 10., 100., 1000.]))
    np.testing.assert_allclose(log_gaps, log_gaps[0])
    np.testing.assert_allclose(step_axis_forward([0., 1000., 10000.]), [0., .25, 1.])
    linear_gaps = np.diff(step_axis_forward([1000., 4000., 7000., 10000.]))
    np.testing.assert_allclose(linear_gaps, linear_gaps[0])


def test_standard_initialization_is_unmodified_torch_and_shared(tmp_path):
    c = json.loads((HERE / "config_standard.json").read_text())
    c.update(steps=3, warmup_steps=2, learning_rates=[0., 1e-3], train_points=64, eval_points=128)
    run_target("square", c, tmp_path)
    torch.manual_seed(c["seed"])
    expected = torch.nn.Sequential(
        torch.nn.Linear(1, c["width"], dtype=torch.float64), torch.nn.Tanh(),
        torch.nn.Linear(c["width"], 1, dtype=torch.float64),
    ).state_dict()
    initial = torch.load(tmp_path / "initial.pt", weights_only=True)
    control = torch.load(tmp_path / "lr_0e+00_final.pt", weights_only=True)
    assert all(torch.equal(initial[k], expected[k]) and torch.equal(control[k], expected[k]) for k in expected)
    zero = json.loads((tmp_path / "lr_0e+00.json").read_text())["history"]
    trained = json.loads((tmp_path / "lr_1e-03.json").read_text())["history"]
    assert zero[0] == trained[0]
    assert zero[0]["eval_rel_l2"] > 1e-3  # standard initialization is not QI-overwritten
    assert not (tmp_path / "qi.npz").exists()


def test_full_checkpoint_preserves_adam_continuation(tmp_path):
    from experiments.expD55_qi_adam_drift.run import train_step, mse, tensor
    c = json.loads((HERE / "config_standard.json").read_text())
    c.update(steps=5, checkpoint_every=2, warmup_steps=3,
             learning_rates=[1e-3], train_points=64, eval_points=128)
    run_target("square", c, tmp_path)
    saved = torch.load(tmp_path / "lr_1e-03_step_000002.pt", weights_only=True)
    assert saved["step"] == 2 and len(saved["pre_update_mse"]) == 2
    net = make_standard_model(c["width"])
    net.load_state_dict(saved["model"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, **c["adam"])
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng_state"])
    data = np.load(tmp_path / "dataset.npz")
    for step in range(saved["step"] + 1, c["steps"] + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, 1e-3, c["warmup_steps"])
        train_step(net, opt, tensor(data["train_x"]), tensor(data["train_y"]), mse)
    expected = torch.load(tmp_path / "lr_1e-03_final.pt", weights_only=True)
    assert all(torch.equal(value, expected[key]) for key, value in net.state_dict().items())
