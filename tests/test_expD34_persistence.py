"""Independent model gradients and actual simultaneous forecast increments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from experiments.expD34_readout_race import persistence as pe, plateau, transport


@pytest.mark.parametrize('model', pe.MODELS)
def test_model_gradient_and_coarse_jacobian(model):
    rng = np.random.default_rng(89); w = 7; x = np.linspace(-.99, .99, 96)
    p0 = rng.normal(0, .3, 3*w+1); p = p0+rng.normal(0, .015, len(p0)); y = np.sin(7*x)+.3
    columns = [0, 1, 2, 3, 9] if model == 'five_mode' else list(range(10))
    q = transport.basis(x, 9)[:, columns] if model in ('five_mode', 'ten_mode') else np.empty((len(x), 0))
    pt = torch.tensor(p, requires_grad=True); xx = torch.tensor(x); pp0 = torch.tensor(p0)
    def output(v):
        a, b, c = v[:-1].reshape(3, w)
        if model == 'linear_features':
            a0, b0, _ = pp0[:-1].reshape(3, w); h0 = torch.tanh(xx[:, None]*a0+b0)
            h = h0+(1-h0*h0)*(xx[:, None]*(a-a0)+(b-b0))
        else: h = torch.tanh(xx[:, None]*a+b)
        return h @ c+v[-1]
    r = output(pt)-torch.tensor(y)
    active = torch.tensor(q) @ (torch.tensor(q).T @ r/len(x)) if q.shape[1] else r
    expected = torch.autograd.grad(active.square().mean()/2, pt)[0].detach().numpy()
    g, _, jc = pe.field(jnp.asarray(p), jnp.asarray(p0), jnp.asarray(x), jnp.asarray(y), jnp.asarray(q), model)
    np.testing.assert_allclose(g, expected, rtol=2e-11, atol=2e-13)
    qc = np.stack((np.ones_like(x), x/np.sqrt(np.mean(x*x))))
    expected_jc = torch.autograd.functional.jacobian(lambda v: torch.tensor(qc) @ output(v)/len(x), pt)
    np.testing.assert_allclose(jc, expected_jc, rtol=2e-11, atol=2e-13)


def test_linear_features_retains_readout_interaction():
    rng = np.random.default_rng(31); p0 = rng.normal(0, .2, 22); x = np.linspace(-1, 1, 31)
    displacement = rng.normal(0, .01, len(p0)); p = p0+displacement
    h, _, c, d = pe.features(jnp.array(p), jnp.array(p0), jnp.array(x), True)
    linear = np.asarray(plateau.prediction(p0, x))+np.asarray(plateau.tangent(p0, x)) @ displacement
    (a0, b0, _), _ = plateau.af.unpack(p0)
    expected = (1-np.tanh(x[:, None]*a0+b0)**2)*(x[:, None]*displacement[:7]+displacement[7:14])
    np.testing.assert_allclose(np.asarray(h @ c+d)-linear, expected @ displacement[14:21], atol=2e-16)


@pytest.mark.parametrize('model', pe.MODELS)
def test_steps_and_outward_accounting(model):
    rng = np.random.default_rng(77); x = np.linspace(-.99, .99, 48)
    p0 = rng.normal(0, .2, (2, 22)); p0[:, 0] = 1e-8
    y = np.broadcast_to(3*x+np.sin(7*x), (2, len(x)))
    columns = [0, 1, 2, 3, 9] if model == 'five_mode' else list(range(10))
    q = transport.basis(x, 9)[:, columns] if model in ('five_mode', 'ten_mode') else np.empty((len(x), 0))
    eta = .002; advance = pe.advance_factory(jnp.array(x), jnp.array(q), model, eta)
    got = advance(pe.initial(jnp.array(p0)), jnp.array(p0), jnp.array(y), 13)
    expected = p0.copy(); positive = np.zeros((2, 7)); negative = positive.copy()
    for _ in range(13):
        for i in range(2):
            g = np.asarray(pe.field(expected[i], p0[i], x, y[i], q, model)[0])
            nxt = expected[i]-eta*g; delta = abs(nxt[:7])-abs(expected[i, :7])
            positive[i] += np.maximum(delta, 0); negative[i] += np.maximum(-delta, 0)
            expected[i] = nxt
    np.testing.assert_allclose(got['p'], expected, rtol=1e-12, atol=2e-14)
    np.testing.assert_allclose(got['positive'], positive, atol=2e-15)
    np.testing.assert_allclose(got['negative'], negative, atol=2e-15)
    np.testing.assert_allclose(got['positive']-got['negative'], abs(expected[:, :7])-abs(p0[:, :7]), atol=2e-15)
