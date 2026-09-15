"""Independent checks of moments, outside weighting, and actual training."""
import numpy as np
import pytest
import torch

from experiments.expD31_split_adam import run as exp


@pytest.mark.parametrize("epsilon", [1e-8, 1e-12])
def test_stream_matches_pytorch_adam_with_tiny_and_changing_gradients(epsilon):
    stream = exp.AdamStream(5, epsilon=epsilon)
    p = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=1., eps=epsilon)
    gradients = np.random.default_rng(7).normal(size=(30, 5))*np.array([1., 1e-8, 1e-13, 10., 0.])
    gradients[4:8] = 0
    for g in gradients:
        before = p.detach().numpy().copy()
        p.grad = torch.tensor(g, dtype=torch.float64)
        optimizer.step()
        m_before, s_before, t_before = stream.m.copy(), stream.s.copy(), stream.t
        preview = stream.direction(g)
        np.testing.assert_array_equal(stream.m, m_before)
        np.testing.assert_array_equal(stream.s, s_before)
        assert stream.t == t_before
        actual = stream.direction(g, advance=True)
        np.testing.assert_array_equal(actual, preview)
        np.testing.assert_allclose(actual, before-p.detach().numpy(), rtol=2e-12, atol=3e-15)


@pytest.mark.parametrize("mu", [1000, 10000, 100000])
def test_combined_update_matches_independent_adams_with_exterior_weight(mu):
    streams = [exp.AdamStream(4), exp.AdamStream(4)]
    eta = .002
    parameters = [torch.nn.Parameter(torch.zeros(4, dtype=torch.float64)) for _ in range(2)]
    optimizers = [torch.optim.Adam([p], lr=eta*w) for p, w in zip(parameters, (mu, 1))]
    total = np.zeros(4)
    rng = np.random.default_rng(3)
    for _ in range(20):
        gradients = [rng.normal(size=4)*scale for scale in (1e-5, 1.)]
        directions = [s.direction(g, advance=True) for s, g in zip(streams, gradients)]
        total -= eta*(mu*directions[0]+directions[1])
        for p, optimizer, g in zip(parameters, optimizers, gradients):
            p.grad = torch.tensor(g, dtype=torch.float64)
            optimizer.step()
        expected = sum(p.detach().numpy() for p in parameters)
        np.testing.assert_allclose(total, expected, rtol=3e-13, atol=1e-11)
    # Inputs/moment histories are unweighted. Moving mu inside Adam gives a
    # radically different first displacement for a resolved gradient.
    g = np.ones(4)*1e-5
    outside = mu*exp.AdamStream(4).direction(g)
    inside = exp.AdamStream(4).direction(mu*g)
    assert np.linalg.norm(outside)/np.linalg.norm(inside) > .99*mu


def small_config():
    return exp.config() | dict(resolution=8, halo=2, n_train=41, steps=3,
                               diagnostic_snapshots=4, readout_rcond=.02)


def independent_ordinary_adam(cfg, target, arm):
    state = exp.profile.initial_state(arm, cfg)
    p = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for k, v in state.items()}
    x = exp.profile.previous.midpoint_grid(cfg["n_train"])
    y = exp.profile.previous.matched.target_values(target, x, cfg)
    tx, ty = torch.tensor(x), torch.tensor(y)
    opt = torch.optim.Adam(p.values(), lr=cfg["learning_rate"],
                           betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
    states = {k: [v.detach().numpy().copy()] for k, v in p.items()}
    for _ in range(cfg["steps"]):
        opt.zero_grad(set_to_none=True)
        output = torch.tanh(tx[:, None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
        (.5*(output-ty).square().mean()).backward()
        opt.step()
        for k, v in p.items():
            states[k].append(v.detach().numpy().copy())
    return states


def test_ordinary_control_and_first_readout_update():
    torch.set_default_dtype(torch.float64)
    cfg = small_config()
    control = exp.train("sine_mixture", "xavier", 0, cfg)
    reference = independent_ordinary_adam(cfg, "sine_mixture", "xavier")
    for k in ("a", "b", "v"):
        np.testing.assert_array_equal(control[k], reference[k])
    for mu in cfg["mu_values"]:
        split = exp.train("sine_mixture", "xavier", mu, cfg | dict(steps=1))
        np.testing.assert_array_equal(split["v"][1], control["v"][1])
        assert np.linalg.norm(split["a"][1]-control["a"][1]) > .01


def test_first_split_step_against_independent_scalar_svd_autograd(monkeypatch):
    torch.set_default_dtype(torch.float64)
    cfg = small_config() | dict(steps=1)
    initial = dict(a=np.array([.5, 1., 2., 3.]), b=np.array([-.2, .1, .4, -.6]),
                   v=np.random.default_rng(5).normal(size=5))
    monkeypatch.setattr(exp.profile, "initial_state", lambda arm, cfg: {k: v.copy() for k, v in initial.items()})
    x = exp.profile.previous.midpoint_grid(cfg["n_train"])
    y = exp.profile.previous.matched.target_values("sine_mixture", x, cfg)
    q = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in initial.items()}
    h = torch.tanh(torch.tensor(x)[:, None]*q["a"]+q["b"])
    A = torch.cat((h, torch.ones((len(x), 1))), dim=1)/np.sqrt(len(x))
    yn = torch.tensor(y)/np.sqrt(len(x))
    U, s, _ = torch.linalg.svd(A, full_matrices=False)
    rank = int((s > cfg["readout_rcond"]*s[0]).sum())
    r = U[:, :rank]@(U[:, :rank].T@yn)-yn
    F = .5*r.square().sum()
    L = .5*(A@q["v"]-yn).square().sum()
    f = np.concatenate([v.detach().numpy() for v in torch.autograd.grad(F, [q["a"], q["b"]], retain_graph=True)])
    all_l = torch.autograd.grad(L, list(q.values()))
    l = np.concatenate([v.detach().numpy() for v in all_l[:2]])
    mu = 1000
    expected = np.r_[initial["a"], initial["b"]] - cfg["learning_rate"]*(
        mu*f/(abs(f)+cfg["adam_epsilon"]) + (l-f)/(abs(l-f)+cfg["adam_epsilon"]))
    actual = exp.train("sine_mixture", "xavier", mu, cfg)
    np.testing.assert_allclose(np.r_[actual["a"][1], actual["b"][1]], expected, rtol=1e-10, atol=2e-10)
    expected_v = initial["v"] - cfg["learning_rate"]*all_l[2].numpy()/(abs(all_l[2].numpy())+cfg["adam_epsilon"])
    np.testing.assert_allclose(actual["v"][1], expected_v, rtol=1e-14, atol=1e-14)
