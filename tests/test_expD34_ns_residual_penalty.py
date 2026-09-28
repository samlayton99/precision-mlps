"""Check the spectral meaning, differentiability, and combined Adam update."""
import numpy as np
import pytest
import torch

from experiments.expD34_ns_residual_penalty import run as exp


@pytest.mark.parametrize("shape", [(8, 4), (4, 8)])
@pytest.mark.parametrize("refinements", [0, 4])
def test_spectral_identity_including_null_space(shape, refinements):
    rng = np.random.default_rng(12)
    U, _, Vh = np.linalg.svd(rng.normal(size=shape), full_matrices=False)
    s = np.array([1., .1, .001, 0.])
    A = (U*s)@Vh
    y = rng.normal(size=shape[0])
    response = exp.scalar_response(s/np.linalg.norm(A), refinements=refinements)
    actual = exp.ns_factor(torch.tensor(A), refinements=refinements).numpy()
    np.testing.assert_allclose(actual, (U*response)@Vh, atol=3e-13, rtol=1e-11)
    retained = U[:, :3]
    F = .5*np.linalg.norm(y-retained@(retained.T@y))**2
    expected = F + .5*np.sum((1-response[:3]**2)**2*(retained.T@y)**2)
    result = exp.residual_penalty(torch.tensor(A), torch.tensor(y), refinements=refinements)
    np.testing.assert_allclose(float(result), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("refinements", [0, 4])
def test_gradient_through_normalization_and_iterations(refinements):
    generator = torch.Generator().manual_seed(41)
    A = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, generator=generator, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda matrix: exp.residual_penalty(matrix, y, refinements=refinements),
        (A,), eps=1e-7, atol=2e-6, rtol=2e-5)


def test_fixed_span_counterexample_and_refinement():
    t = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    A = torch.stack((torch.tensor([1., 0., 0.], dtype=torch.float64),
                     torch.stack((t*0, t, t*0))), dim=1)
    y = torch.tensor([0., 1., 0.], dtype=torch.float64)
    # y is exactly represented with readout (0, 1/t) for every positive t.
    literal = exp.residual_penalty(A, y)
    derivative, = torch.autograd.grad(literal, t, retain_graph=True)
    refined = exp.residual_penalty(A, y, refinements=4)
    assert literal > .05
    assert abs(derivative) > 1e-3
    assert refined < literal*1e-8


def test_penalty_does_not_change_readout_partial_gradient():
    generator = torch.Generator().manual_seed(3)
    A = torch.randn(9, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.randn(9, generator=generator, dtype=torch.float64)
    v = torch.randn(4, generator=generator, dtype=torch.float64, requires_grad=True)
    L = .5*(A@v-y).square().sum()
    Fhat = exp.residual_penalty(A, y, refinements=4)
    gradient_v, = torch.autograd.grad(L+100*Fhat, v)
    torch.testing.assert_close(gradient_v, A.T@(A@v-y))


@pytest.mark.parametrize("mu,refinements", [(0, 0), (100, 0), (100, 4)])
@pytest.mark.parametrize("scheduled", [False, True])
def test_multiple_combined_adam_steps_against_literal_reference(monkeypatch, mu, refinements, scheduled):
    cfg = exp.config() | dict(n_train=41, steps=3, diagnostic_snapshots=4)
    if scheduled:
        cfg["lr_schedule"] = dict(kind="cosine", start_step=0, min_factor=.001)
    initial = dict(a=np.array([1., 2., 3.]), b=np.array([-.8, .2, .5]),
                   v=np.array([.3, -.2, .1, .4]))
    monkeypatch.setattr(exp.previous.profile, "initial_state",
                        lambda arm, cfg: {k: v.copy() for k, v in initial.items()})
    x = torch.tensor(exp.previous.profile.previous.midpoint_grid(cfg["n_train"]))
    y = torch.tensor(exp.previous.profile.previous.matched.target_values(cfg["target"], x.numpy(), cfg))
    p = {k: torch.tensor(v, requires_grad=True) for k, v in initial.items()}
    m, variance = [{k: torch.zeros_like(v) for k, v in p.items()} for _ in range(2)]
    reference = {k: [v.detach().numpy().copy()] for k, v in p.items()}
    beta1, beta2 = cfg["adam_betas"]
    for step in range(1, cfg["steps"]+1):
        h = torch.tanh(x[:, None]*p["a"]+p["b"])
        prediction = h@p["v"][:-1]+p["v"][-1]
        objective = .5*(prediction-y).square().mean()
        if mu:
            A = torch.cat((h, torch.ones_like(x[:, None])), dim=1)/len(x)**.5
            X = A/torch.sqrt(torch.sum(A*A))
            for _ in range(5):
                gram = X.T@X
                X = 3.4445*X - 4.7750*(X@gram) + 2.0315*((X@gram)@gram)
            for _ in range(refinements):
                X = .5*X@(3*torch.eye(X.shape[1], dtype=X.dtype)-X.T@X)
            residual = y/len(x)**.5-X@(X.T@(y/len(x)**.5))
            objective = objective + mu*.5*residual.square().sum()
        gradients = torch.autograd.grad(objective, tuple(p.values()))
        with torch.no_grad():
            for (key, value), grad in zip(p.items(), gradients):
                m[key] = beta1*m[key]+(1-beta1)*grad
                variance[key] = beta2*variance[key]+(1-beta2)*grad.square()
                direction = (m[key]/(1-beta1**step))/(torch.sqrt(variance[key]/(1-beta2**step))+cfg["adam_epsilon"])
                eta = cfg["learning_rate"]
                if scheduled:
                    eta *= .001 + .999*.5*(1+np.cos(np.pi*(step-1)/cfg["steps"]))
                value.sub_(eta*direction)
                reference[key].append(value.detach().numpy().copy())
    actual = exp.train(cfg, mu, refinements)
    if scheduled:
        assert actual["learning_rate"][0] == cfg["learning_rate"]
        assert actual["learning_rate"][-1] == pytest.approx(.001*cfg["learning_rate"])
    for key in p:
        np.testing.assert_allclose(actual[key], reference[key], rtol=1e-11, atol=1e-12)
