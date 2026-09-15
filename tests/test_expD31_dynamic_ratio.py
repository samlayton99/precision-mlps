"""Verify the post-Adam controller and its integration with the split trainer."""
import numpy as np
import pytest
import torch

from experiments.expD31_split_adam import run as exp


@pytest.mark.parametrize("ratio", [.01, .1, 1, 10, 100])
def test_balance_preserves_direction_and_requested_norm(ratio):
    f = np.array([1., -2., 3.]) * 1e-120
    g = np.array([-4., 2., 1.])
    direction, mu = exp.balanced_geometry_direction(f, g, ratio)
    np.testing.assert_allclose(np.linalg.norm(mu*f) / np.linalg.norm(g), ratio, rtol=1e-14)
    np.testing.assert_allclose(direction, mu*f+g, rtol=1e-14)
    assert mu > 0


def test_undefined_ratio_is_reported_without_silent_floor():
    for f, g in [(np.zeros(2), np.ones(2)), (np.ones(2), np.zeros(2)),
                 (np.array([np.inf, 1.]), np.ones(2))]:
        with pytest.raises(FloatingPointError):
            exp.balanced_geometry_direction(f, g, .1)


@pytest.mark.parametrize("scheduled", [False, True])
def test_dynamic_trainer_against_independent_adam_histories(monkeypatch, scheduled):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    ratio = .1
    cfg = exp.config() | dict(n_train=41, steps=4, diagnostic_snapshots=5,
                              readout_rcond=.02, target_update_ratio=ratio)
    if scheduled:
        cfg["lr_schedule"] = dict(kind="cosine", start_step=0, min_factor=.1)
    initial = dict(a=np.array([.5, 1., 2., 3.]), b=np.array([-.2, .1, .4, -.6]),
                   v=np.random.default_rng(5).normal(size=5))
    monkeypatch.setattr(exp.profile, "initial_state", lambda arm, cfg: {k: v.copy() for k, v in initial.items()})
    x = torch.tensor(exp.profile.previous.midpoint_grid(cfg["n_train"]))
    y = torch.tensor(exp.profile.previous.matched.target_values("sine_mixture", x.numpy(), cfg)) / np.sqrt(len(x))
    p = {k: torch.nn.Parameter(torch.tensor(v)) for k, v in initial.items()}
    virtual = [torch.nn.Parameter(torch.zeros(8)) for _ in range(2)]
    opts = [torch.optim.Adam([q], lr=1., betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"]) for q in virtual]
    readout = torch.optim.Adam([p["v"]], lr=.002, betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
    expected = {k: [v.detach().numpy().copy()] for k, v in p.items()}
    multipliers = []
    for step in range(cfg["steps"]):
        h = torch.tanh(x[:, None]*p["a"] + p["b"])
        A = torch.cat([h, torch.ones((len(x), 1))], dim=1) / np.sqrt(len(x))
        U, s, _ = torch.linalg.svd(A, full_matrices=False)
        rank = int((s > cfg["readout_rcond"]*s[0]).sum())
        rstar = U[:, :rank] @ (U[:, :rank].T @ y) - y
        gf = torch.cat(torch.autograd.grad(.5*rstar.square().sum(), [p["a"], p["b"]], retain_graph=True))
        grads = torch.autograd.grad(.5*(A@p["v"]-y).square().sum(), list(p.values()))
        gg = torch.cat(grads[:2]) - gf
        moves = []
        for q, opt, gradient in zip(virtual, opts, (gf, gg)):
            before = q.detach().clone()
            q.grad = gradient
            opt.step()
            moves.append(q.detach() - before)
        eta = .002*(.1 + .9*.5*(1+np.cos(np.pi*step/4))) if scheduled else .002
        mu = ratio * torch.linalg.vector_norm(moves[1]) / torch.linalg.vector_norm(moves[0])
        multipliers.append(float(mu))
        with torch.no_grad():
            delta = eta*(mu*moves[0] + moves[1])
            p["a"].add_(delta[:4])
            p["b"].add_(delta[4:])
        readout.param_groups[0]["lr"] = eta
        p["v"].grad = grads[2]
        readout.step()
        for k, v in p.items():
            expected[k].append(v.detach().numpy().copy())
    actual = exp.train("sine_mixture", "xavier", 1, cfg)
    for k in p:
        np.testing.assert_allclose(actual[k], expected[k], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual["effective_mu"][:-1], multipliers, rtol=1e-9)
    np.testing.assert_allclose(actual["F_step_norm"][:-1] / actual["G_step_norm"][:-1], ratio, rtol=1e-13)
