"""Verify the replacement uses the existing comparison's numerical conventions."""
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from experiments.expC09_bandwidth_figures import analytic_packet as packet, convergence


def test_measurement_matches_existing_convergence_solver(monkeypatch):
    config = json.loads((packet.OUT.parent / "eight_target_convergence/data/config.json").read_text())["config"]
    config = dict(config, targets=["sine24"])
    monkeypatch.setattr(packet, "target", lambda x: np.sin(24*np.pi*x))
    with threadpool_limits(limits=1):
        expected, expected_coef = convergence.measure(256, config)
        actual, coef = packet.measure(256, config)
    np.testing.assert_array_equal(coef, expected_coef[:, 0])
    assert actual["relative_l2"] == pytest.approx(expected["relative_l2"][0], rel=1e-12)


def test_packet_energy_matches_analytic_gaussian_integral():
    # Whole-line integral: sin^2 = (1-cos(16*pi*x))/2. Use a wider
    # integration domain here so Gaussian tails are negligible.
    x = np.linspace(-4, 4, 128001)
    expected = .5*np.sqrt(np.pi/16)*(1-np.exp(-(16*np.pi)**2/64)*np.cos(16*np.pi*.2))
    assert np.trapezoid(packet.target(x)**2, x) == pytest.approx(expected, rel=1e-12)
