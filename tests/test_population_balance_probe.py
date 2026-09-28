import numpy as np
import pytest

from experiments.expD34_readout_race.population_balance_probe import balance


@pytest.mark.parametrize("scale", [.05, .5, 3.])
def test_exact_balance_identity_with_biases_and_coarse_compensation(scale):
    rng = np.random.default_rng(121)
    p = rng.normal(size=22)*scale
    x = np.linspace(-1, 1, 129)
    y = .3+.4*x+np.sin(5*x)
    row = balance(p, x, y)
    reference = max(1., abs(row["full_drift"]))
    assert row["direct_identity_error"] < 2e-12*reference
    assert row["effective_identity_error"] < 2e-12*reference
    assert row["full_identity_error"] < 2e-12*reference
    assert row["force_split_error"] < 2e-12*reference
    # Compare to an independently evaluated directional derivative of the loss.
    a, b, c = p[:-1].reshape(3, -1)
    v = np.r_[a, b, -c, 0.]
    def loss(q):
        aa, bb, cc = q[:-1].reshape(3, -1)
        r = np.tanh(x[:, None]*aa+bb)@cc+q[-1]-y
        return np.mean(r*r)/2
    eps = 1e-5/max(1., np.linalg.norm(v))
    directional = (loss(p+eps*v)-loss(p-eps*v))/(2*eps)
    np.testing.assert_allclose(-directional, row["full_drift"], rtol=3e-7, atol=1e-9)
    # The quadratic observable has an exact GD increment, with no trajectory tube.
    phi = np.tanh(x[:, None]*a+b)
    error = phi@c+p[-1]-y
    local = error[:, None]*(1-phi**2)*c
    gradient = np.r_[np.mean(local*x[:, None], axis=0), np.mean(local, axis=0),
                     np.mean(error[:, None]*phi, axis=0), np.mean(error)]
    eta = .007
    aa, bb, cc = (p-eta*gradient)[:-1].reshape(3, -1)
    next_imbalance = (aa@aa+bb@bb-cc@cc)/2
    np.testing.assert_allclose(next_imbalance-row["imbalance"],
                               eta*row["full_drift"]+eta**2*row["gd_quadratic_coefficient"],
                               rtol=2e-11, atol=2e-14)
