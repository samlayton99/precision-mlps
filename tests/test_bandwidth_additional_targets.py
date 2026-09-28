"""Keep the follow-up panels identical to the approved convergence targets."""
import numpy as np
import pytest
import yaml

from experiments.expC09_bandwidth_figures import additional_targets as targets
from experiments.expC09_bandwidth_figures import convergence, run


def test_targets_match_convergence_including_bump_boundary():
    x = np.unique(np.r_[np.linspace(-1, 1, 1001), -.5, .5, 0.])
    config = {"targets": list(targets.EXTRA_NAMES), "bump_half_width": .5}
    expected = convergence.values(x, config)
    for j, target in enumerate(targets.EXTRA_NAMES):
        np.testing.assert_array_equal(targets.target_values(x, target), expected[:, j])
    bump = targets.target_values(x, "smooth_bump")
    assert np.all(bump[np.abs(x) >= .5] == 0)
    assert bump[x == 0][0] == pytest.approx(np.exp(-1))


def test_frequency_estimates_are_stable_and_identified():
    assert targets.frequency_scale("runge100") == 10
    x = np.linspace(-1, 1, 1001)
    assert targets.frequency_scale("chirp") == pytest.approx(np.trapezoid(16*np.pi*(x+1), x)/2)
    coarse = targets.bump_frequency_scale(length=64, points=65536)
    assert targets.frequency_scale("smooth_bump") == pytest.approx(coarse, rel=3e-5)


def test_new_suite_uses_requested_width_and_precision_conventions():
    config = yaml.safe_load((run.HERE / "additional_config.yaml").read_text())
    jobs, predictions = run.make_jobs(config)
    assert {p["width"] for p in predictions} == {96, 128, 192, 256}
    assert {p["p"] for p in predictions} == {24, 32, 40, 53}
    assert all(p[rule]["status"] == "threshold" for p in predictions for rule in ("basic", "refined"))
    for name in targets.EXTRA_NAMES:
        assert {w for t, w, h, lam, p in jobs
                if t == name and h == 24 and lam == .25 and p == 53} == set(range(64, 257, 2))
