"""Check target accuracy and consistency with the prior convergence experiment."""
import json

import mpmath as mp
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from experiments.expC09_bandwidth_figures import smooth_absolute as smooth, convergence


def test_target_agrees_with_high_precision_formula():
    x = np.array([-1., -.2, -.01, 0., .01, .2, 1.])
    with mp.workdps(60):
        expected = [float(mp.log(mp.cosh(10*mp.mpf(float(t))))/10) for t in x]
    np.testing.assert_allclose(smooth.target(x), expected, rtol=1e-13, atol=2e-17)
    assert smooth.target(np.array([0.]))[0] == 0


def test_solver_matches_previous_convergence(monkeypatch):
    config = json.loads((smooth.OUT.parent / "eight_target_convergence/data/config.json").read_text())["config"]
    config = dict(config, targets=["sine24"])
    monkeypatch.setattr(smooth, "target", lambda x: np.sin(24*np.pi*x))
    with threadpool_limits(limits=1):
        expected, expected_coef = convergence.measure(256, config)
        actual, coef = smooth.measure(256, config)
    np.testing.assert_array_equal(coef, expected_coef[:, 0])
    assert actual["relative_l2"] == pytest.approx(expected["relative_l2"][0], rel=1e-12)
