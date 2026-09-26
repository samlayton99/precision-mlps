"""Eight-target definitions and consistency with single-target measurements."""
import numpy as np
import pytest
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
import yaml

from experiments.expC09_bandwidth_figures import convergence as c


@pytest.fixture
def config():
    return yaml.safe_load((c.HERE / "convergence_config.yaml").read_text())


def test_requested_width_range_and_eight_targets(config):
    widths = c.widths(config)
    assert len(widths) == 41
    assert widths[0] == 64 and widths[-1] == 2048
    assert len(set(widths)) == len(widths) and all(w % 2 == 0 for w in widths)
    assert len(config["targets"]) == 8
    assert config["targets"] == list(c.LABELS)


def test_selected_targets_and_bump_boundaries(config):
    x = np.array([-1., -.5, -.25, 0., .125, .25, .5, 1.])
    with np.errstate(divide="raise", invalid="raise"):
        y = c.values(x, config)
    np.testing.assert_allclose(y[:, 0], np.sin(4*np.pi*x))
    np.testing.assert_allclose(y[:, 1], np.sin(24*np.pi*x))
    np.testing.assert_allclose(y[:, 2]-np.sin(2*np.pi*x), 1e-3*np.sin(40*np.pi*x), atol=1e-16)
    np.testing.assert_allclose(y[:, 3], np.sin(8*np.pi*(x+1)**2))
    np.testing.assert_allclose(y[:, 4], 1/(1+25*x*x))
    np.testing.assert_allclose(y[:, 5], 1/(1+100*x*x))
    np.testing.assert_allclose(y[:, 6], c.earlier_target(x, "gaussian_envelope"))
    assert y[x == 0, 7].item() == pytest.approx(np.exp(-1))
    assert np.all(y[np.abs(x) >= .5, 7] == 0)


def test_readouts_match_single_target_reference(config):
    w, nt, ne = 96, 601, 1201
    with threadpool_limits(limits=1):
        row, coefficients = c.measure(w, config, train_points=nt, eval_points=ne)
        _, h, centers = c.geometry(w, config["halo_per_side"])
        gamma = config["lambda"]/h
        x = np.linspace(-1, 1, nt)
        a = np.column_stack((np.tanh(gamma*(x[:, None]-centers)), np.ones(nt)))
        y = c.values(x, config)
        xe = np.linspace(-1, 1, ne)
        truth = c.values(xe, config)
        features = np.tanh(gamma*(xe[:, None]-centers))
        for j in range(len(config["targets"])):
            readout = lstsq(a, y[:, j], cond=2**-52, lapack_driver="gelsd")[0]
            direct = np.linalg.norm(features@readout[:-1]+readout[-1]-truth[:, j])/np.linalg.norm(truth[:, j])
            assert row["relative_l2"][j] == pytest.approx(direct, rel=1e-7, abs=1e-12)
        assert coefficients.shape == (w+1, 8)
        assert row["N"] == w-49
