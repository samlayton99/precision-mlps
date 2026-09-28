"""Check the conventions that could silently change the plotted experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figure = load("experiments/expC09_bandwidth_figures/run.py", "bandwidth_figure")


@pytest.mark.parametrize("width", [64, 96, 128, 192, 256])
def test_total_width_counts_halo_neurons(width):
    n, h, centers = figure.geometry(width, 24)
    assert len(centers) == width
    assert np.count_nonzero(centers < -1) == 24
    assert np.count_nonzero(centers > 1) == 24
    np.testing.assert_allclose(centers, -centers[::-1], atol=2e-15)
    np.testing.assert_allclose(np.diff(centers), h, rtol=1e-12)
    _, h2, larger = figure.geometry(n + 2*128 + 1, 128)
    assert h == h2
    np.testing.assert_array_equal(larger[104:-104], centers)


def test_precision_rounding_matches_float32_and_ties_to_even():
    rng = np.random.default_rng(184)
    x = rng.normal(size=1000) * np.exp2(rng.uniform(-80, 80, 1000))
    np.testing.assert_array_equal(figure.round_bits(x, 24), x.astype(np.float32).astype(float))
    np.testing.assert_array_equal(figure.round_bits([1.125, 1.375, -1.125], 3), [1., 1.5, -1.])
    np.testing.assert_array_equal(figure.round_bits(x, 53), x)


@pytest.mark.parametrize("bits", [24, 32, 40, 53])
def test_rule_threshold_uses_spacing_and_requested_precision(bits):
    pred = figure.predictions(512, 32, bits)
    assert pred["N"] == 447
    for name, theta in [("basic", None), ("refined", 10/447)]:
        pr = pred[name]
        assert pr["status"] == "threshold"
        log_score = figure.selector.log_alias_score("tanh", pr["lambda"], theta)
        assert log_score == pytest.approx(np.log(2.0**(1-bits)), abs=1e-12)
        assert pr["gamma"] == pytest.approx(pr["lambda"] * 447/2)


@pytest.mark.parametrize("bits", [24, 40, 53])
def test_measurement_agrees_with_previous_bandwidth_experiment(bits):
    old = load("experiments/expC08_anchor_rule/anchor_rule.py", "previous_bandwidth")
    with threadpool_limits(limits=1):
        result = figure.measure(128, 32, .3, bits, 2003, 4001)
        expected = old.solve({"act": "tanh", "lam": .3, "N": 63, "fn": "runge25", "p": bits})
    assert result["relative_l2"] == pytest.approx(expected["rel_l2"], rel=1e-12, abs=1e-18)


def test_invalid_width_cannot_silently_drop_halo():
    with pytest.raises(ValueError):
        figure.geometry(64, 32)


def test_targets_match_existing_four_target_suite():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.targets import get_target
    from experiments.expC09_bandwidth_figures.targets import target_values, SIGMA
    import yaml
    config = yaml.safe_load((ROOT / "experiments/expD24_gd_residual_spectrum/readout_comparison.yaml").read_text())
    x = np.linspace(-1, 1, 257)
    for new, old in [("runge", "runge"), ("mixed_sine", "sine_mixture")]:
        np.testing.assert_allclose(target_values(x, new), get_target(old).fn_numpy(x), atol=1e-15)
    assert SIGMA == config["envelope_sigma"]
    expected = np.exp(-.5*(x/SIGMA)**2) * get_target("sine_mixture").fn_numpy(x)
    np.testing.assert_allclose(target_values(x, "gaussian_envelope"), expected, atol=1e-15)


def test_regular_sine_has_four_cycles_on_the_interval():
    from experiments.expC09_bandwidth_figures.targets import target_values, frequency_scale
    x = np.array([0., 1/8, 1/4, 3/8, 1/2])
    np.testing.assert_allclose(target_values(x, "sine"), [0., 1., 0., -1., 0.], atol=1e-15)
    assert frequency_scale("sine") == pytest.approx(4*np.pi)


def test_gaussian_frequency_summary_against_spectral_quadrature():
    from scipy.integrate import quad
    from experiments.expC09_bandwidth_figures.targets import MODES, AMPLITUDES, SIGMA, frequency_scale
    def spectrum(w):
        return sum(a*(np.exp(-.5*(SIGMA*(w-k*np.pi))**2)
                      - np.exp(-.5*(SIGMA*(w+k*np.pi))**2))
                   for a,k in zip(AMPLITUDES, MODES))
    mass = quad(spectrum, 0, 100, epsabs=1e-12)[0]
    first = quad(lambda w: w*spectrum(w), 0, 100, epsabs=1e-12)[0]
    assert frequency_scale("gaussian_envelope") == pytest.approx(first/mass, rel=1e-12)


def test_jobs_use_requested_widths_and_all_targets():
    import yaml
    config = yaml.safe_load((figure.HERE / "config.yaml").read_text())
    jobs, prediction = figure.make_jobs(config)
    main = [j for j in jobs if j[2] == 24]
    refinement = [j for j in main if j[3] == config["refinement_lambda"] and j[4] == 53]
    for target in config["targets"]:
        assert {j[1] for j in refinement if j[0] == target} == set(range(64, 257, 2))
    assert {pr["width"] for pr in prediction} == {96, 128, 192, 256}
    assert not any(j[1] == 64 and j[3] != config["refinement_lambda"] for j in main)
    assert {j[0] for j in main} == {"runge", "mixed_sine", "sine", "gaussian_envelope"}
    assert {j[-1] for j in main} == {24, 32, 40, 53}
    assert all(p[r]["status"] == "threshold" for p in prediction for r in ("basic", "refined"))
