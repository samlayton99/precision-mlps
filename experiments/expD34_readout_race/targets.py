"""Paired upstream initialization and targets in the empirical training metric."""
from __future__ import annotations

import hashlib
import numpy as np
from numpy.polynomial.legendre import legvander

TARGETS = ("sine", "runge", "moment3", "moment5", "moment9")
RATIOS = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
WIDTHS = ((64, 12), (128, 24), (256, 48))


def grid(m):
    return -1 + 2 * (np.arange(m, dtype=np.float64) + .5) / m


def polynomial_map(x):
    """Coefficients in the Legendre basis, fitted once on training samples."""
    _, r = np.linalg.qr(legvander(x, 9) / np.sqrt(len(x)))
    return np.linalg.solve(r, np.diag(np.sign(np.diag(r))))


def values(name, x, mapping):
    if name == "sine":
        return np.sin(2 * np.pi * x)
    if name == "runge":
        return 1 / (1 + 25 * x*x)
    if name not in TARGETS:
        raise ValueError(name)
    coefficients = np.zeros(10)
    coefficients[[0, 1, int(name[6:])]] = [.3, .4, np.sqrt(.75)]
    return legvander(x, 9) @ (mapping @ coefficients)


def data(m, name):
    x = grid(m)
    mapping = polynomial_map(x)
    y = values(name, x, mapping)
    powers = x[:, None] ** np.arange(11)
    return dict(x=x, y=y, powers=powers, Q=powers.T @ powers/m,
                ym=powers.T @ y/m, sy=np.mean(y*y), sigma=np.sqrt(np.mean(x*x)),
                mapping=mapping)


def initial(n, halo, seed):
    # Import lazily: compiled GPU training itself does not require PyTorch.
    from experiments.expD28_loss_gradient_decomposition.run import initial_state
    state = initial_state("xavier", dict(resolution=n, halo=halo, seed=seed, lambda_star=.25))
    return np.stack((state["a"], state["b"], state["v"][:-1])), float(state["v"][-1])


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
