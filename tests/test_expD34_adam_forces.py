"""Independent checks of full-residual decomposition and shared Adam history."""
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from experiments.expD34_readout_race import adam_forces as af, targets, transport


def example():
    x = targets.grid(71)
    p = np.r_[np.random.default_rng(20).normal(size=21)*.6, .17]
    return p, x, np.sin(2*np.pi*x)+.3


def test_force_split_matches_autodiff_and_complete_orthogonal_basis():
    p, x, y = example()
    g, r, j, ec = map(np.asarray, af.field(jnp.array(p), jnp.array(x), jnp.array(y)))
    pt = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    xx = torch.tensor(x)
    def predict(v):
        a, b, c = v[:-1].reshape(3, -1)
        return torch.tanh(xx[:, None]*a+b) @ c+v[-1]
    output = predict(pt)
    reference = torch.autograd.grad(.5*((output-torch.tensor(y))**2).mean(), pt)[0].numpy()
    q = np.c_[np.ones_like(x), x/np.sqrt(np.mean(x*x))]
    jt = torch.autograd.functional.jacobian(lambda v: torch.tensor(q.T) @ predict(v)/len(x), pt).numpy()
    np.testing.assert_allclose(g, reference, atol=3e-16, rtol=3e-14)
    np.testing.assert_allclose(j, jt, atol=3e-16, rtol=3e-14)
    channels, info = af.split(jnp.array(g), jnp.array(j), jnp.array(ec))
    assert info['resolved']
    np.testing.assert_allclose(np.sum(channels, axis=0), g, atol=5e-16)
    np.testing.assert_allclose(j @ channels[0], 0, atol=6e-16)
    qfull = transport.basis(x, 33)
    z = p[:-1].reshape(3, -1)
    row, arr = transport.modal_diagnostics(z, p[-1], x, y, np.ones(7)/7, 7, 33)
    B = np.linalg.solve(sum(arr['K_'+k] for k in 'abcd')[:2, :2],
                        sum(arr['K_'+k] for k in 'abcd')[:2, 2:])
    retained = (arr['J_a'][2:].T-arr['J_a'][:2].T @ B) @ arr['residual_modes'][2:]
    tail = r-qfull @ arr['residual_modes']
    gt, _, _, _ = af.field(jnp.array(p), jnp.array(x), jnp.array(y+ r-tail))
    tail_effective = af.split(gt, jnp.array(j), jnp.array(q.T @ tail/len(x)))[0][0]
    np.testing.assert_allclose(channels[0, :7], retained+tail_effective[:7], atol=1e-15)


def test_preconditioned_balance_and_unresolved_channel():
    p, x, y = example()
    g, _, j, ec = af.field(jnp.array(p), jnp.array(x), jnp.array(y))
    mobility = jnp.geomspace(.01, 100., len(p))
    channels, info = af.split(g, j, ec, mobility)
    np.testing.assert_allclose(j @ (mobility*channels[0]), 0, atol=3e-14)
    np.testing.assert_allclose(channels.sum(axis=0), g, atol=5e-16)
    broken, info = af.split(g, jnp.zeros_like(j), ec)
    assert not info['resolved']
    np.testing.assert_array_equal(broken[2], g)
    np.testing.assert_array_equal(broken[:2], 0.)


@pytest.mark.parametrize('epsilon', [1e-8, 1e-12])
@pytest.mark.parametrize('beta1', [.9, 0.])
def test_channel_moments_reconstruct_actual_pytorch_adam(epsilon, beta1):
    rng = np.random.default_rng(9)
    streams = rng.normal(size=(50, 2, 5))*np.array([1., 1e-9, 1e-13, 10., 0.])
    streams[7:12, 1] = -streams[7:12, 0]
    streams[25:30] = 0
    p = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=.002, betas=(beta1, .999), eps=epsilon)
    m = v = jnp.zeros(5); cm = jnp.zeros((2, 5))
    for n, channels in enumerate(streams, 1):
        g = channels.sum(axis=0)
        before = p.detach().numpy().copy()
        p.grad = torch.tensor(g)
        optimizer.step()
        m, v, cm, mh, ch, inv = af.moments(jnp.array(g), jnp.array(channels), m, v, cm, n,
                                          beta1, .999, epsilon, True)
        np.testing.assert_allclose(ch.sum(axis=0), mh, atol=3e-15, rtol=2e-12)
        np.testing.assert_allclose(-.002*inv*mh, p.detach().numpy()-before, atol=3e-16, rtol=2e-11)
        np.testing.assert_allclose((-.002*inv*ch).sum(axis=0), p.detach().numpy()-before,
                                   atol=3e-16, rtol=2e-11)


def test_targets_preserve_anchors_and_matched_moments():
    x = targets.grid(2048)
    mapping = targets.polynomial_map(x)
    q = np.polynomial.legendre.legvander(x, 9) @ mapping
    for name in af.TARGETS:
        xx, y, _, scale = af.data(name)
        np.testing.assert_array_equal(xx, x)
        if name in targets.TARGETS:
            np.testing.assert_array_equal(y, targets.values(name, x, mapping))
        else:
            assert np.mean(y*y) == pytest.approx(1., abs=2e-15)
        if name in af.BLENDS or name == 'moment4':
            np.testing.assert_allclose(q[:, :2].T @ y/len(x), [.3, .4], atol=3e-15)
    assert len(af.TARGETS) == 13


@pytest.mark.parametrize('optimizer', ['gd', 'adam', 'momentum', 'adaptive_only'])
def test_actual_training_and_resume(optimizer):
    from experiments.expD34_readout_race import adam_run as ar
    p, _, _ = example(); x = targets.grid(64); y = np.sin(2*np.pi*x)+.3
    beta1 = .9 if optimizer in ('adam', 'momentum') else 0.
    adaptive = optimizer in ('adam', 'adaptive_only')
    setting = np.array([.002, beta1, .999, 1e-8, adaptive])[None]
    state = jax.tree.map(lambda v: v[None], ar.initial(p))
    advance = ar.advance_factory(64)
    full, row, lo, hi = advance(state, jnp.array(y[None]), jnp.array(setting), 37)
    part, *_ = advance(state, jnp.array(y[None]), jnp.array(setting), 13)
    # Serialization round trip of all parameter, optimizer and diagnostic state.
    part = jax.tree.map(lambda v: jnp.asarray(np.asarray(v).copy()), part)
    resumed, *_ = advance(part, jnp.array(y[None]), jnp.array(setting), 24)
    for key in full:
        np.testing.assert_array_equal(full[key], resumed[key])
    assert row.shape == (1, len(ar.METRICS))
    assert np.isfinite(row).all()
    pt = torch.nn.Parameter(torch.tensor(p, dtype=torch.float64)); xx = torch.tensor(x)
    opt = torch.optim.Adam([pt], lr=.002, betas=(beta1, .999), eps=1e-8) if adaptive else None
    moment = np.zeros_like(p)
    for n in range(1, 38):
        pt.grad = None
        a, b, c = pt[:-1].reshape(3, -1)
        prediction = torch.tanh(xx[:, None]*a+b) @ c+pt[-1]
        (.5*((prediction-torch.tensor(y))**2).mean()).backward()
        if adaptive:
            opt.step()
        else:
            moment = beta1*moment+(1-beta1)*pt.grad.numpy()
            with torch.no_grad(): pt -= torch.tensor(.002*moment/(1-beta1**n))
    np.testing.assert_allclose(full['p'][0], pt.detach().numpy(), atol=5e-15, rtol=5e-13)
    width = 7
    np.testing.assert_allclose(full['positive'][0]-full['negative'][0], abs(full['p'][0, :width])-abs(p[:width]), atol=1e-15)
    growth = np.mean(abs(full['p'][0, :width])-abs(p[:width]))
    assert full['signed_channels'][0].sum()+full['crossing'][0] == pytest.approx(growth, abs=3e-15)
    assert float(full['identity_max'].max()) < 3e-15


def test_preconditioned_coarse_velocity_and_finite_step_defect():
    p, x, y = example()
    g, _, j, ec = af.field(jnp.array(p), jnp.array(x), jnp.array(y))
    mobility = jnp.geomspace(.1, 2., len(p))
    lag = jnp.sin(jnp.arange(len(p)))*.01
    channels, info = af.split(g, j, ec, mobility)
    velocity = -mobility*(g+lag)
    prediction = -info['C'] @ info['z']-j @ (mobility*lag)
    np.testing.assert_allclose(j @ velocity, prediction, atol=2e-15)
    defects = []
    for step in (.001, .0005, .00025):
        _, _, _, en = af.field(jnp.array(p)+step*velocity, jnp.array(x), jnp.array(y))
        defects.append(np.linalg.norm(en-ec-step*prediction))
    assert defects[-1] < defects[0]/14


def test_runner_checkpoint_resume_preserves_case_and_trace(monkeypatch, tmp_path):
    from argparse import Namespace
    from experiments.expD34_readout_race import adam_run as ar
    selected = ar.cases('primary', 0)[:2]
    monkeypatch.setattr(ar, 'cases', lambda stage, index: selected)
    monkeypatch.setattr(ar, 'verify_gpu', lambda output: None)
    def execute(folder, end):
        ar.run(Namespace(output=folder, stage='primary', index=0, samples=64, end_step=end, max_seconds=100))
    execute(tmp_path/'resume', 3)
    execute(tmp_path/'resume', 7)
    execute(tmp_path/'whole', 7)
    for filename in ('state.npz', 'trace.npz'):
        actual = np.load(tmp_path/'resume/primary_0'/filename)
        expected = np.load(tmp_path/'whole/primary_0'/filename)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key])
