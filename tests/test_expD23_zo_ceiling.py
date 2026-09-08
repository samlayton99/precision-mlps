"""expD23 sanity: the batched loss oracle IS the QIMlp loss, the upstream
ports reproduce expD22's CD-RGE stream, the finite-difference oracle arms are
accurate, and the estimator covariance identity holds.

Loads run.py / zo.py by explicit path (module-name collision bug, see
docs/ORIENTATION.md section 7b).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def d22():
    return _load("expD22_run_for23", REPO_ROOT / "experiments" / "expD22_cdrge" / "run.py")


@pytest.fixture(scope="module")
def zo22():
    return _load("expD22_cdrge_for23", REPO_ROOT / "experiments" / "expD22_cdrge" / "cdrge.py")


@pytest.fixture(scope="module")
def zo():
    return _load("expD23_zo", REPO_ROOT / "experiments" / "expD23_zo_ceiling" / "zo.py")


@pytest.fixture(scope="module")
def cell(d22, zo):
    model = d22.build_model("qi", 64, seed=0)
    X_tr, y_tr, X_ev, y_ev, y_norm = d22.data_bundle("sine")
    W = model.readout.weight.shape[1]
    oracle = zo.FlatLoss(X_tr, y_tr, W)
    return model, (X_tr, y_tr, X_ev, y_ev, y_norm), oracle, d22.get_flat(model)


def test_flat_oracle_is_the_model_loss(d22, cell):
    """FlatLoss.losses on a batch of parameter vectors == QIMlp's MSE for each
    vector, to summation-order rounding; and the batch path == the single path."""
    model, (X_tr, y_tr, X_ev, y_ev, y_norm), oracle, x0 = cell
    gen = torch.Generator().manual_seed(1)
    thetas = x0[None, :] + 1e-2 * torch.randn(5, x0.numel(), generator=gen)
    ref = torch.tensor([d22.make_loss_fn(model, X_tr, y_tr)(t) for t in thetas])
    got = oracle.losses(thetas)
    assert torch.allclose(got, ref, rtol=1e-13, atol=0)
    assert abs(oracle.loss(thetas[2]) - float(got[2])) <= 1e-15 * float(got[2])
    d22.set_flat(model, x0)
    assert abs(oracle.rel_l2(x0, X_ev, y_ev, y_norm)
               - d22.eval_rel_l2(model, X_ev, y_ev, y_norm)) < 1e-13


def test_cdrge_port_reproduces_expD22_stream(zo, zo22, cell):
    """Same seed, same probes, same update: the expD23 CD-RGE (batched oracle)
    and expD22's cdrge.py agree to rounding over the first steps, both for the
    plain lr = eps rule and for the Adam-style rule that was expD22's headline."""
    model, (X_tr, y_tr, *_), oracle, x0 = cell
    loss22 = lambda x: oracle.loss(x)            # same oracle, so only the update code differs
    for kw22, kw23 in [
        (dict(schedule="constant", eps0=1e-3), dict(eps=1e-3)),
        (dict(schedule="constant", eps0=1e-3, beta1=0.9, beta2=0.999, adam_lr=0.01),
         dict(eps=1e-3, beta1=0.9, beta2=0.999, adam_lr=0.01)),
    ]:
        x22, _ = zo22.cdrge_minimize(x0, loss22, max_steps=3, n_perturb=20, seed=5, **kw22)
        x23, _ = zo.cdrge_minimize(x0, oracle, steps=3, n_perturb=20, seed=5, **kw23)
        assert torch.allclose(x22, x23, rtol=1e-10, atol=1e-14), (x22 - x23).abs().max()


def test_spsa15_curvature_is_the_directional_second_derivative(zo):
    """On a quadratic, the 1.5-SPSA normaliser |f+ - 2f0 + f-|/eps^2 equals
    z^T H z exactly (up to rounding), and z^T H z concentrates at tr(H) with
    std sqrt(2)(|H|_F^2 - sum H_ii^2)^{1/2}: it is a scalar, not a preconditioner."""
    m = 40
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(80, m, generator=gen)
    H = 2 * A.T @ A / 80

    class Q:
        n_evals = 0

        def losses(self, T):
            T = T.reshape(-1, m)
            return ((T @ A.T) ** 2).mean(dim=1)

        def loss(self, t):
            return float(self.losses(t[None])[0])

    q = Q()
    theta = torch.randn(m, generator=gen)
    Z = zo.rademacher(m, gen, 2000)
    eps = 1e-2
    f0 = q.loss(theta)
    fp = q.losses(theta[None] + eps * Z)
    fm = q.losses(theta[None] - eps * Z)
    curv = (fp - 2 * f0 + fm).abs() / eps ** 2
    ztHz = torch.einsum("jm,mk,jk->j", Z, H, Z)
    assert torch.allclose(curv, ztHz, rtol=1e-6)
    trH = float(H.trace())
    pred_std = float(torch.sqrt(2 * ((H ** 2).sum() - (H.diagonal() ** 2).sum())))
    assert abs(float(curv.mean()) - trH) < 0.1 * pred_std
    assert abs(float(curv.std()) - pred_std) < 0.15 * pred_std


def test_fd_gradient_matches_autograd(zo, cell):
    """Coordinate central differences of the fp64 loss oracle reproduce the
    autograd gradient of the same loss to ~1e-8 relative at the qi init."""
    model, (X_tr, y_tr, *_), oracle, x0 = cell
    theta = x0.clone().requires_grad_(True)
    loss = ((oracle.predict(theta[None])[0] - oracle.y) ** 2).mean()
    g_true = torch.autograd.grad(loss, theta)[0]
    g_fd = zo.fd_gradient(x0, oracle, 1e-5)
    rel = float((g_fd - g_true).norm() / g_true.norm())
    print(f"FD gradient rel err = {rel:.2e}")
    assert rel < 1e-7


def test_fd_hessian_matches_autograd_on_readout_block(zo, cell):
    """On the readout block the loss is exactly quadratic; the coordinate FD
    Hessian equals 2 Phi_aug^T Phi_aug / n to absolute ~eps_mach * diag(H)
    (the polarization floor), NOT to relative eps_mach in every eigenvalue."""
    model, (X_tr, y_tr, *_), oracle, x0 = cell
    Phi = oracle.features(x0)
    ro = zo.ReadoutLoss(Phi, y_tr)
    v0 = torch.cat([x0[2 * oracle.W:3 * oracle.W], x0[3 * oracle.W:]])
    k = 30                                             # a sub-block keeps the test fast
    sub = ro.__class__(Phi[:, :k], y_tr)
    Aug = torch.cat([Phi[:, :k], torch.ones(Phi.shape[0], 1)], 1)
    H_true = 2 * Aug.T @ Aug / Phi.shape[0]
    H_fd = zo.fd_hessian(torch.cat([v0[:k], v0[-1:]]), sub, 1e-1)
    err = (H_fd - H_true).abs().max()
    floor = zo.EPS_MACH * H_true.diagonal().max()
    print(f"FD Hessian max abs err = {float(err):.2e}, eps_mach*max H_ii = {float(floor):.2e}")
    assert float(err) < 100 * float(floor)
    assert float(err) > 0.01 * float(floor)


def test_zo_estimator_covariance_identity(zo):
    """Empirical covariance of ghat = (1/n) sum_j (z_j . g) z_j over Rademacher
    probes matches (|g|^2 I + g g^T - 2 diag(g^2))/n."""
    m, n = 12, 8
    gen = torch.Generator().manual_seed(3)
    g = torch.randn(m, generator=gen)
    draws = []
    for _ in range(20000):
        Z = zo.rademacher(m, gen, n)
        draws.append((Z @ g) @ Z / n)
    G = torch.stack(draws)
    assert torch.allclose(G.mean(0), g, atol=0.05 * g.norm())
    C_emp = torch.cov(G.T)
    C_th = zo.zo_estimator_covariance(g) / n
    assert float((C_emp - C_th).abs().max()) < 0.08 * float(C_th.abs().max())


def test_truncated_newton_on_quadratic_reaches_machine_precision(zo):
    """With the exact Hessian, one truncated-Newton step solves a well-conditioned
    quadratic to machine precision; with the FD Hessian it does too."""
    m = 15
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(60, m, generator=gen)
    xs = torch.randn(m, generator=gen)
    y = A @ xs

    class Q:
        n_evals = 0

        def losses(self, T):
            T = T.reshape(-1, m)
            return ((T @ A.T - y) ** 2).mean(dim=1)

        def loss(self, t):
            return float(self.losses(t[None])[0])

    q = Q()
    x0 = torch.zeros(m)
    g = zo.fd_gradient(x0, q, 1e-2)
    H = zo.fd_hessian(x0, q, 1e-1)
    x1, kept = zo.truncated_newton_step(x0, g, H, 1e-12)
    assert kept == m
    assert float((x1 - xs).norm() / xs.norm()) < 1e-10
