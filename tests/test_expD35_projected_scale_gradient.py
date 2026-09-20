import numpy as np
import torch
from experiments.expD35_projected_scale_gradient import run as exp


def test_projection_terms_match_literal_formula():
    rng = np.random.default_rng(31)
    A, r, J = rng.normal(size=(30, 7)), rng.normal(size=30), rng.normal(size=(30, 5))
    for cutoff in (1e-13, .65):
        out = exp.split(A, r, J, cutoff)
        inverse = np.linalg.pinv(A, rcond=cutoff)
        P = A@inverse
        C, Z = inverse@J, J-P@J
        np.testing.assert_allclose(out['left'], C.T@(A.T@r), atol=1e-13)
        np.testing.assert_allclose(out['right'], Z.T@((np.eye(30)-P)@r), atol=1e-13)
        np.testing.assert_allclose(out['left']+out['right'], J.T@r, atol=1e-13)


def test_signed_slope_gradient_matches_autodiff_and_zero_readout():
    x = np.linspace(-1, 1, 63)
    y = np.sin(4*x)
    state = dict(a=np.array([-.7, 1.3, -2.1]), b=np.array([.4, -.2, .6]), v=np.array([.3, -.7, .8, .1]))
    out = exp.measure(state, x, y, 1e-13)
    a = torch.tensor(state['a'], dtype=torch.float64, requires_grad=True)
    pred = torch.tanh(torch.tensor(x)[:, None]*a+torch.tensor(state['b']))@torch.tensor(state['v'][:-1])+state['v'][-1]
    loss = .5*(pred-torch.tensor(y)).square().mean()
    expected = torch.autograd.grad(loss, a)[0].numpy()*np.sign(state['a'])
    np.testing.assert_allclose(out['total'], expected, atol=2e-15)
    state['v'] = np.zeros(4)
    out = exp.measure(state, x, y, 1e-13)
    np.testing.assert_array_equal(out['norms'], 0)
    assert np.isnan(out['cosine'])


def test_diagnostics_do_not_change_training_state():
    cfg = exp.config() | dict(steps=2, n_train=32, diagnostic_snapshots=3)
    torch.set_default_dtype(torch.float64)
    case = exp.baseline.train('sine', 'qi_zero', cfg)
    before = {k: case[k].copy() for k in ('a', 'b', 'v')}
    out = exp.diagnose(case, cfg)
    for k in before:
        np.testing.assert_array_equal(before[k], out[k])
    np.testing.assert_array_equal(out['norms'][0], 0)
    np.testing.assert_array_equal(out['a'][0], out['a'][1])
    np.testing.assert_array_equal(out['b'][0], out['b'][1])
