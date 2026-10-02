"""Scientific protocol checks for the activation-lens comparison."""
import copy
import json

import numpy as np
import torch

from experiments.expD41_activation_lens import run
from src.data.targets import get_target


def small_config():
    cfg = json.loads((run.HERE / "config.json").read_text())
    cfg.update(interior_centers=16, halo_per_side=5, cardinal_half_width=24,
               n_train=101, n_eval=257, n_validation=129, steps=30, warmup_steps=5,
               activations=["tanh", "notch", "sinc"])
    return cfg


def test_constructor_uses_derivative_filter_and_anchors_boundary(monkeypatch):
    cfg = small_config()
    def forbidden(*args, **kwargs):
        raise AssertionError("Function-value LS must not initialize constructive QI")
    monkeypatch.setattr(run, "readout_solve", forbidden)
    for activation in cfg["activations"]:
        params, info = run.construct(activation, "runge", 1.0, cfg)
        model = run.Network(activation, params)
        x = np.linspace(-1, 1, 101)
        pred = model(torch.tensor(x)).detach().numpy()
        expected = run.features(activation, *params[:2], x) @ params[2] + params[3]
        np.testing.assert_allclose(pred, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(pred[0], get_target("runge").fn_numpy(-1.0), atol=2e-14)
        assert info["readout_construction"].startswith("derivative cardinal")


def test_tanh_constructor_matches_repository_reference():
    from src.construction.qi_mpmath import construct_qi, evaluate_qi
    cfg = json.loads((run.HERE / "config.json").read_text())
    target = get_target("sine")
    # Use a well-conditioned cardinal solve to isolate the construction formula
    # from fp64 cancellation at narrower lambda (covered by the bandwidth sweep).
    params, _ = run.construct("tanh", "sine", 0.5, cfg)
    reference = construct_qi(target.fn_numpy, target.deriv_numpy,
        N=cfg["interior_centers"] - 1, halo=cfg["halo_per_side"],
        lambda_star=0.5, Kc=cfg["cardinal_half_width"], precision="fp64")
    x = run.grid(1001)
    pred = run.features("tanh", *params[:2], x) @ params[2] + params[3]
    np.testing.assert_allclose(pred, evaluate_qi(reference, x), atol=5e-12, rtol=0)


def test_observational_solve_does_not_change_next_adam_update():
    cfg = small_config()
    width = cfg["interior_centers"] + 2 * cfg["halo_per_side"]
    model = run.Network("sinc", run.xavier(width, 0, cfg))
    control = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    control_opt = torch.optim.Adam(control.parameters(), lr=0.002)
    x = torch.tensor(run.grid(cfg["n_train"]))
    y = torch.tensor(get_target("sine").fn_numpy(x.numpy()))
    rng_before = torch.get_rng_state().clone()
    run.evaluate("sinc", model.arrays(), "sine", cfg)
    assert torch.equal(rng_before, torch.get_rng_state())
    run.train_step(model, opt, x, y, run.mse)
    run.train_step(control, control_opt, x, y, run.mse)
    for actual, expected in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_recording_and_disjoint_grids():
    assert run.recording_steps(10000)[0] == 0
    assert run.recording_steps(10000)[-1] == 10000
    assert len(set(run.grid(1024)) & set(run.grid(8192))) == 0
    assert len(set(run.grid(8192)) & set(run.grid(32768))) == 0


def test_xavier_is_paired_and_zero_bias():
    cfg = small_config()
    first, second = run.xavier(26, 2, cfg), run.xavier(26, 2, cfg)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert np.all(first[1] == 0) and first[3] == 0
