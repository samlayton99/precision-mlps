"""expD22 sanity: the CD-RGE port is faithful and its estimator is a gradient.

Loads run.py / cdrge.py by explicit path (module-name collision bug, see
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
    return _load("expD22_run",
                 REPO_ROOT / "experiments" / "expD22_cdrge" / "run.py")


@pytest.fixture(scope="module")
def zo():
    return _load("expD22_cdrge_mod",
                 REPO_ROOT / "experiments" / "expD22_cdrge" / "cdrge.py")


def test_one_step_matches_reference_formula(zo):
    """One cdrge_minimize step == x0 - (lr/eps) * (1/n) sum_j [(f+ - f-)/2] z_j
    computed by hand with the same generator draws (the upstream update rule)."""
    m, n_pert, eps = 7, 5, 1e-3
    A = torch.randn(12, m, generator=torch.Generator().manual_seed(3))
    y = torch.randn(12, generator=torch.Generator().manual_seed(4))
    x0 = torch.randn(m, generator=torch.Generator().manual_seed(5))

    def loss_fn(x):
        return float(((A @ x - y) ** 2).mean())

    x1, _ = zo.cdrge_minimize(x0, loss_fn, max_steps=1, n_perturb=n_pert,
                              eps0=eps, schedule="constant", seed=42)

    gen = torch.Generator().manual_seed(42)
    buf = torch.zeros(m)
    for _ in range(n_pert):
        z = (torch.randint(0, 2, (m,), generator=gen, dtype=torch.int8)
             .to(torch.float64) * 2 - 1)
        coeff = -(loss_fn(x0 + eps * z) - loss_fn(x0 - eps * z)) / (2 * n_pert)
        buf += coeff * z
    x1_ref = x0 + buf                       # lr_over_eps = 1
    assert torch.equal(x1, x1_ref)


def test_estimate_aligns_with_true_gradient(d22, zo):
    """On the real model/loss, the CD-RGE step direction at small eps and large
    n_perturb must align with -grad (cosine > 0.5; the zz^T average is I plus
    O(sqrt(m/n)) noise, so alignment is far above chance but below 1)."""
    model = d22.build_model("qi", 64, seed=0)
    X_tr, y_tr, X_ev, y_ev, y_norm = d22.data_bundle("sine")
    loss = ((model(X_tr) - y_tr) ** 2).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()))
    g_true = torch.cat([g.reshape(-1) for g in grads])

    x0 = d22.get_flat(model)
    loss_fn = d22.make_loss_fn(model, X_tr, y_tr)
    captured = {}

    def cb(step, x, mean_loss, eps):
        captured["step_vec"] = x - x0

    zo.cdrge_minimize(x0, loss_fn, max_steps=1, n_perturb=2000, eps0=1e-6,
                      schedule="constant", seed=0, step_callback=cb)
    step_vec = captured["step_vec"]
    cos = float(torch.dot(step_vec, -g_true)
                / (step_vec.norm() * g_true.norm()))
    print(f"cosine(step, -grad) = {cos:.3f}")
    assert cos > 0.5


def test_cdrge_solves_small_quadratic_to_machine_precision(zo):
    """CD-RGE with lr = eps (constant) on a well-conditioned least-squares
    problem must reach machine precision. Central differences are exact in eps
    on quadratics, so constant eps is the correct schedule here; the halve-
    every-step recipe leaves a geometric travel budget and provably stalls
    (measured: 3e-2 at the same budget)."""
    m = 10
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(50, m, generator=gen)
    x_star = torch.randn(m, generator=gen)
    y = A @ x_star

    def loss_fn(x):
        return float(((A @ x - y) ** 2).mean())

    x0 = torch.zeros(m)
    x_fin, info = zo.cdrge_minimize(x0, loss_fn, max_steps=1000, n_perturb=100,
                                    eps0=0.2, schedule="constant", seed=0)
    final = loss_fn(x_fin)
    print(f"final loss = {final:.3e} after {info['steps_run']} steps")
    assert final < 1e-25


def test_qi_init_geometry_reaches_floor(d22):
    """The copied expD16 builder still produces a floor-quality qi geometry."""
    model = d22.build_model("qi", 128, seed=0)
    X_tr, y_tr, X_ev, y_ev, y_norm = d22.data_bundle("sine")
    with torch.no_grad():
        Phi_tr = model.features(X_tr).numpy()
        Phi_ev = model.features(X_ev).numpy()
    Aug = np.hstack([Phi_tr, np.ones((Phi_tr.shape[0], 1))])
    U, s, Vt = np.linalg.svd(Aug, full_matrices=False)
    s_inv = np.where(s > 1e-13 * s[0], 1.0 / np.where(s > 0, s, 1.0), 0.0)
    sol = Vt.T @ (s_inv * (U.T @ y_tr.numpy().ravel()))
    pred = Phi_ev @ sol[:-1] + sol[-1]
    err = float(np.linalg.norm(pred - y_ev.numpy().ravel()) / y_norm)
    print(f"QI init + lstsq readout, sine N=128: rel L2 = {err:.3e}")
    assert err < 1e-12
