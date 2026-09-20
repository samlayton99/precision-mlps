"""Initialization, GD, dense Fourier evaluation, and animation checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expd24_gd_residual_spectrum", ROOT / "experiments/expD24_gd_residual_spectrum/run.py")
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


@pytest.mark.parametrize("resolution", [64, 128, 256])
def test_c05_scaling_preserves_centers_signs_and_readout(resolution):
    xav, meta = exp.initial_arrays(resolution, 0, "xavier")
    scaled, scaled_meta = exp.initial_arrays(resolution, 0, "scaled_xavier")
    qi, _ = exp.initial_arrays(resolution, 0, "qi_zero")
    assert meta["halo"] == 24
    assert meta["width"] == resolution + 49
    expected_a, expected_b = exp.c05.xavier_draw(meta["width"], 0, resolution)
    np.testing.assert_array_equal(xav["a"], expected_a)
    np.testing.assert_array_equal(xav["b"], expected_b)
    factor = (0.25 / meta["h"]) / np.mean(np.abs(xav["a"]))
    assert scaled_meta["scale_factor"] == factor
    np.testing.assert_array_equal(scaled["a"], xav["a"] * factor)
    np.testing.assert_array_equal(scaled["b"], xav["b"] * factor)
    np.testing.assert_allclose(-scaled["b"] / scaled["a"], -xav["b"] / xav["a"], rtol=1e-14)
    np.testing.assert_array_equal(np.sign(scaled["a"]), np.sign(xav["a"]))
    np.testing.assert_array_equal(scaled["c"], xav["c"])
    assert np.mean(np.abs(scaled["a"])) * meta["h"] == pytest.approx(0.25, rel=1e-14)
    np.testing.assert_allclose(qi["a"] * meta["h"], 0.25, rtol=1e-14)
    centers = -qi["b"] / qi["a"]
    np.testing.assert_allclose(np.diff(centers), meta["h"], atol=1e-14)
    assert centers[meta["halo"]] == -1.0
    assert centers[-meta["halo"] - 1] == 1.0
    assert np.count_nonzero(qi["c"]) == np.count_nonzero(qi["d"]) == 0


def test_all_parameter_gd_update_matches_analytic_half_mse_gradient():
    # Unequal, signed slopes and nonzero biases/readouts exercise every block.
    arrays = {"a": np.array([-0.8, 2.0, 4.5]), "b": np.array([0.3, -0.5, 0.9]),
              "c": np.array([0.4, -0.7, 0.2]), "d": np.array([0.13])}
    x = np.array([-0.91, -0.4, 0.1, 0.65, 0.94])
    y = 1 / (1 + 25 * x**2)
    features = np.tanh(x[:, None] * arrays["a"] + arrays["b"])
    residual = features @ arrays["c"] + arrays["d"][0] - y
    inner_signal = residual[:, None] * arrays["c"] * (1 - features**2)
    expected = {"a": np.mean(inner_signal * x[:, None], axis=0),
                "b": np.mean(inner_signal, axis=0),
                "c": np.mean(residual[:, None] * features, axis=0),
                "d": np.array([np.mean(residual)])}
    model = exp.model_from_arrays(arrays)
    lr = 0.002
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = 0.5 * (model(torch.from_numpy(x)[:, None]).ravel() - torch.from_numpy(y)).square().mean()
    loss.backward()
    for key, parameter in exp.named_arrays(model).items():
        np.testing.assert_allclose(parameter.grad.numpy().ravel(), expected[key], rtol=2e-13, atol=1e-15)
    optimizer.step()
    for key, value in exp.snapshot(model).items():
        np.testing.assert_allclose(value, arrays[key] - lr * expected[key], rtol=1e-14, atol=1e-15)


def test_saved_frames_reconstruct_the_residual_spectrum(tmp_path):
    spectrum = load_spectrum_module()
    config = {"seed": 0, "lambda_star": 0.25, "halo": 24, "domain": [-1.0, 1.0],
              "resolutions": [8], "n_train": 61, "n_eval": 122,
              "learning_rate": 0.002, "steps": 3,
              "targets": ["runge"], "arms": ["qi_zero"]}
    x = exp.midpoint_grid(122, config["domain"]).numpy().ravel()
    transform = spectrum.FiniteIntervalTransform(x, max_mode=16, points=129)
    result = exp.train_case(config, "runge", "qi_zero", transform, [0, 1, 2, 3])
    assert len(result["loss"]) == 4
    assert result["loss"][-1] < result["loss"][0]
    model = exp.model_from_arrays(result["final_parameters"])
    with torch.no_grad():
        y = exp.get_target("runge").fn(torch.from_numpy(x)[:, None]).numpy().ravel()
        residual = model(torch.from_numpy(x)[:, None]).numpy().ravel() - y
    np.testing.assert_allclose(result["spectra"][-1], transform(residual), atol=1e-14)
    np.testing.assert_allclose(result["relative_l2"][-1], np.linalg.norm(residual) / np.linalg.norm(y))
    path = tmp_path / "data.npz"
    exp.save_data([result], config, transform.modes, [0, 1, 2, 3], path)
    cases, saved_config, modes, steps = exp.load_data(path)
    assert saved_config == config and steps == [0, 1, 2, 3]
    np.testing.assert_array_equal(modes, transform.modes)
    np.testing.assert_array_equal(cases[0]["loss"], result["loss"])
    for index in range(4):
        arrays = {key: values[index] for key, values in cases[0]["frame_parameters"].items()}
        model = exp.model_from_arrays(arrays)
        with torch.no_grad():
            residual = model(torch.from_numpy(x)[:, None]).numpy().ravel() - y
        np.testing.assert_allclose(cases[0]["spectra"][index], transform(residual), atol=1e-14)
        np.testing.assert_allclose(cases[0]["relative_l2"][index], np.linalg.norm(residual) / np.linalg.norm(y))
    assert list(tmp_path.iterdir()) == [path]


def load_spectrum_module():
    spec = importlib.util.spec_from_file_location(
        "expd24_spectrum_test", ROOT / "experiments/expD24_gd_residual_spectrum/spectrum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_smoothing_preserves_interior_and_retains_the_new_tails():
    from experiments.expD24_gd_residual_spectrum.end_smoothing import smooth_ends

    dx = 0.0002
    x = -1.2 + (np.arange(12000) + 0.5) * dx
    raw = (abs(x) < 1).astype(float)
    actual = smooth_ends(x, raw, half_width=1)
    np.testing.assert_array_equal(actual[abs(x) <= 0.88], raw[abs(x) <= 0.88])
    assert actual[np.argmin(abs(x - 1.02))] > 0.15
    # The old unit jump across x=1 is now a smooth transition over sigma=0.02.
    boundary = np.searchsorted(x, 1)
    assert abs(actual[boundary] - actual[boundary - 1]) < 0.005
    assert max(abs(actual[abs(x) > 1.18])) < 1e-13


def test_whole_line_formula_matches_direct_integration_and_parseval():
    from experiments.expD24_gd_residual_spectrum import whole_line as whole
    from scipy.integrate import quad_vec
    import yaml

    config = yaml.safe_load((whole.HERE / "whole_line.yaml").read_text())
    state = {"a": np.array([0.9, 1.7, 3.2]), "b": np.array([0.72, -0.17, -2.24]),
             "v": np.array([0.4, -0.2, 0.8])}
    x, weights = whole.quadrature(config, refinement=2)
    r = whole.predict(state, x) - whole.target(x, config)
    omega = np.pi * np.array([-6.25, -2.1, 0, 0.01, 1.2, 2, 6, 14, 31.375])
    exact = whole.model_spectrum(state, omega) - whole.target_spectrum(omega, config)
    # Loss quadrature need not resolve exp(-i*omega*x) in the long tail panels.
    # Adapt independently to that oscillatory factor for the transform check.
    direct, _ = quad_vec(lambda t: (whole.predict(state, t) - whole.target(t, config))
                         * np.exp(-1j * omega * t), -24, 24,
                         points=[-2, 0, 2], epsabs=1e-12, epsrel=1e-12)
    np.testing.assert_allclose(exact, direct, atol=2e-12)
    k = np.linspace(0, 64, 4097)
    spectrum = whole.model_spectrum(state, np.pi * k) - whole.target_spectrum(np.pi * k, config)
    np.testing.assert_allclose(np.trapezoid(abs(spectrum) ** 2, k), weights @ (r * r), rtol=1e-10)
    y = whole.target(x, config)
    np.testing.assert_allclose(weights @ (y * y), whole.target_energy(config), rtol=1e-11)
    np.testing.assert_allclose(whole.predict(state, np.array([-1000, 1000])), 0, atol=0)


def test_whole_line_zero_sum_readout_gradient_matches_spatial_pairings():
    from experiments.expD24_gd_residual_spectrum import whole_line as whole
    import yaml

    config = yaml.safe_load((whole.HERE / "whole_line.yaml").read_text())
    values = {"a": np.array([0.9, 1.7, 3.2]), "b": np.array([0.72, -0.17, -2.24]),
              "v": np.array([0.4, -0.2, 0.8])}
    params = {key: torch.tensor(value, requires_grad=True) for key, value in values.items()}
    x, weights = whole.quadrature(config)
    tx, tw = torch.tensor(x), torch.tensor(weights)
    c = params["v"] - params["v"].mean()
    features = torch.tanh(tx[:, None] * params["a"] + params["b"])
    basis = features - torch.tanh(tx)[:, None]
    r = basis @ c - torch.tensor(whole.target(x, config))
    (0.5 * torch.sum(tw * r**2)).backward()
    signal = (weights * r.detach().numpy())[:, None]
    tangent = c.detach().numpy() * (1 - features.detach().numpy() ** 2)
    expected_v = np.sum(signal * basis.detach().numpy(), axis=0)
    expected = {"a": np.sum(signal * x[:, None] * tangent, axis=0),
                "b": np.sum(signal * tangent, axis=0), "v": expected_v - expected_v.mean()}
    for key in params:
        np.testing.assert_allclose(params[key].grad.numpy(), expected[key], atol=5e-14)


def test_dense_fourier_matches_direct_complex_quadrature():
    spectrum = load_spectrum_module()
    x = exp.midpoint_grid(128, [-1.0, 1.0]).numpy().ravel()
    residual = np.random.default_rng(24).normal(size=128)
    transform = spectrum.FiniteIntervalTransform(x, max_mode=32, points=257)
    expected = transform.dx * np.exp(-1j * transform.omega[:, None] * x) @ residual
    np.testing.assert_allclose(transform(residual), expected, atol=2e-13)
    assert np.any(transform.modes % 1 != 0)


def test_ten_viridis_snapshot_times_include_endpoints_and_cluster_early():
    spectrum = load_spectrum_module()
    selected = spectrum.snapshot_steps(2000, available=spectrum.frame_steps(2000))
    assert len(selected) == 10 and selected[0] == 0 and selected[-1] == 2000
    assert selected == sorted(set(selected))
    assert np.count_nonzero(np.asarray(selected) < 200) >= 6


def test_whole_line_signed_xavier_features_and_gradients_match_spatial_loss():
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    from experiments.expD24_gd_residual_spectrum import whole_line as whole
    import yaml

    config = yaml.safe_load((comparison.HERE / "readout_comparison.yaml").read_text())
    problem = comparison.Problem("gaussian_envelope", config)
    arrays = {"a": np.array([0.9, -1.7, 3.2]), "b": np.array([0.72, 0.17, -2.24]),
              "v": np.array([0.4, -0.2, 0.8])}
    p = {key: torch.tensor(value, requires_grad=True) for key, value in arrays.items()}
    loss = problem.objective(problem.features(p["a"], p["b"]), p["v"])
    loss.backward()
    spatial_config = yaml.safe_load((whole.HERE / "whole_line.yaml").read_text())
    x, weights = whole.quadrature(spatial_config, refinement=2)
    q = {key: torch.tensor(value, requires_grad=True) for key, value in arrays.items()}
    tx = torch.tensor(x)
    signs = torch.sign(q["a"])
    canonical = torch.tanh(tx[:, None] * abs(q["a"]) + signs * q["b"]) - torch.tanh(tx)[:, None]
    basis = (canonical - canonical.mean(dim=1, keepdim=True)) * signs
    r = basis @ q["v"] - torch.tensor(whole.target(x, config))
    spatial_loss = 0.5 * torch.sum(torch.tensor(weights) * r.square())
    spatial_loss.backward()
    np.testing.assert_allclose(loss.detach(), spatial_loss.detach(), rtol=1e-11)
    for key in p:
        np.testing.assert_allclose(p[key].grad, q[key].grad, rtol=1e-10, atol=1e-12)
    # Zero readout produces extremely tiny/zero complex residuals in the tails;
    # a real^2+imag^2 objective must give finite, zero geometry gradients.
    initial = problem.initial("qi_zero")
    p = {key: torch.tensor(value, requires_grad=True) for key, value in initial.items()}
    problem.objective(problem.features(p["a"], p["b"]), p["v"]).backward()
    for key in p:
        assert torch.isfinite(p[key].grad).all()
    np.testing.assert_array_equal(p["a"].grad, 0)
    np.testing.assert_array_equal(p["b"].grad, 0)


def test_varpro_envelope_gradient_matches_resolved_finite_differences():
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    import yaml

    config = yaml.safe_load((comparison.HERE / "readout_comparison.yaml").read_text())
    config["n_train"] = 128
    problem = comparison.Problem("runge", config)
    arrays = {"a": np.array([0.9, 1.7, 3.2]), "b": np.array([0.72, -0.17, -2.24])}
    p = {key: torch.tensor(value, requires_grad=True) for key, value in arrays.items()}
    features = problem.features(p["a"], p["b"])
    v, info = comparison.solve_readout(features.detach().numpy(), problem.y, problem.weights)
    assert info["rank"] == 4 and info["projection_stationarity"] < 1e-13
    problem.objective(features, torch.tensor(v), info["optimal_residual"]).backward()
    for key in arrays:
        for j in range(3):
            values = []
            for direction in (-1, 1):
                perturbed = {name: value.copy() for name, value in arrays.items()}
                perturbed[key][j] += direction * 1e-5
                A = problem.features(torch.tensor(perturbed["a"]), torch.tensor(perturbed["b"])).numpy()
                _, fit = comparison.solve_readout(A, problem.y, problem.weights)
                values.append(fit["loss"])
            expected = (values[1] - values[0]) / 2e-5
            np.testing.assert_allclose(p[key].grad[j], expected, rtol=2e-5, atol=1e-10)


def test_whole_line_least_squares_is_resolved_away_from_its_training_quadrature():
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    from experiments.expD24_gd_residual_spectrum import whole_line as whole
    import yaml

    config = yaml.safe_load((comparison.HERE / "readout_comparison.yaml").read_text())
    problem = comparison.Problem("gaussian_envelope", config)
    params = problem.initial("qi_zero")
    A = problem.features(torch.tensor(params["a"]), torch.tensor(params["b"])).numpy()
    v, info = comparison.solve_readout(A, problem.y, problem.weights)
    # A least-squares solve can exploit underresolved quadrature even when GD
    # passes Parseval. Check this fitted readout by independent spatial integration.
    spatial_config = yaml.safe_load((whole.HERE / "whole_line.yaml").read_text())
    x, weights = whole.quadrature(spatial_config, refinement=2)
    r = comparison.whole_spatial_features(params["a"], params["b"], x) @ v - whole.target(x, config)
    spatial_energy = weights @ r**2
    np.testing.assert_allclose(spatial_energy, 2 * info["actual_loss"], rtol=1e-5)


def test_diagnostic_readout_refit_cannot_modify_the_saved_gd_trajectory():
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    import yaml

    config = yaml.safe_load((comparison.HERE / "readout_comparison.yaml").read_text())
    config.update(resolution=8, halo=2, n_train=64, n_eval=128)
    problem = comparison.Problem("runge", config)
    case = comparison.train(problem, "qi_zero", "gd", steps=2)
    original_parameters = {key: value.copy() for key, value in case["parameters"].items()}
    original_loss = case["loss"].copy()
    views = comparison.diagnose(case, problem)
    for key in original_parameters:
        np.testing.assert_array_equal(case["parameters"][key], original_parameters[key])
    np.testing.assert_array_equal(case["loss"], original_loss)
    assert np.all(views["gd_refit"]["relative_l2"] < views["gd"]["relative_l2"])


@pytest.mark.parametrize("method", ["gd", "varpro"])
def test_recording_animation_states_replays_the_same_training(method):
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    from experiments.expD24_gd_residual_spectrum.animate_comparison import check_replay
    import yaml

    config = yaml.safe_load((comparison.HERE / "readout_comparison.yaml").read_text())
    config.update(resolution=8, halo=2, n_train=64, steps=30)
    problem = comparison.Problem("runge", config)
    sparse = comparison.train(problem, "qi_zero", method)
    frames = comparison.frame_steps(config["steps"])
    dense = comparison.train(problem, "qi_zero", method, displayed_steps=frames)
    assert len(dense["snapshot_steps"]) > len(sparse["snapshot_steps"])
    check_replay(sparse, dense)


def test_gamma_sweep_changes_only_slopes_not_centers_width_or_readout():
    from experiments.expD24_gd_residual_spectrum.gamma_comparison import GammaProblem, HERE
    import yaml

    config = yaml.safe_load((HERE / "gamma_comparison.yaml").read_text())
    expected_centers = -1 + np.arange(-24, 129 + 24) / 64
    for target in config["targets"]:
        for gamma in config["initial_gammas"]:
            initial = GammaProblem(target, config, gamma).initial()
            np.testing.assert_array_equal(initial["a"], np.full(177, gamma))
            np.testing.assert_array_equal(-initial["b"] / initial["a"], expected_centers)
            np.testing.assert_array_equal(initial["v"], 0)
            assert len(initial["v"]) == (177 if target == "gaussian_envelope" else 178)


def test_gamma_motion_reports_sign_cancellation_without_losing_physical_scale():
    from experiments.expD24_gd_residual_spectrum.gamma_comparison import motion_statistics

    # Opposite changes cancel in the mean, but both neurons have moved by 1.
    # A sign flip of a tanh slope does not make its physical scale negative.
    case = {"target": "sine", "method": "gd", "parameters": {
        "a": np.array([[2., 2.], [3., -1.]]), "b": np.zeros((2, 2))}}
    stats = motion_statistics(case)
    assert stats["mean_signed_gamma_change"] == 0
    assert stats["mean_absolute_gamma_change"] == 1
    assert stats["slopes_changing_sign"] == 1


def test_gamma_one_varpro_first_step_is_resolved_by_refined_whole_line_rule():
    from experiments.expD24_gd_residual_spectrum.gamma_comparison import GammaProblem, HERE, independent_spatial_energy
    from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
    import yaml

    config = yaml.safe_load((HERE / "gamma_comparison.yaml").read_text())
    problem = GammaProblem("gaussian_envelope", config, 1)
    initial = problem.initial()
    params = {key: torch.tensor(value, requires_grad=True) for key, value in initial.items()}
    A = problem.features(params["a"], params["b"])
    v, info = comparison.solve_readout(A.detach().numpy(), problem.y, problem.weights)
    problem.objective(A, torch.tensor(v), info["optimal_residual"]).backward()
    moved = {key: params[key].detach() - config["learning_rate"] * params[key].grad for key in ("a", "b")}
    assert float(moved["a"].abs().max()) > 100  # Exercise the formerly unresolved regime.
    with torch.no_grad():
        A = problem.features(moved["a"], moved["b"])
        v, info = comparison.solve_readout(A.numpy(), problem.y, problem.weights)
        coarse = 2 * float(problem.objective(A, torch.tensor(v)))
        fine = GammaProblem("gaussian_envelope", config, 1, refinement=2)
        refined = 2 * float(fine.objective(fine.features(moved["a"], moved["b"]), torch.tensor(v)))
    assert abs(refined - coarse) / problem.target_energy < 1e-5
    spatial = independent_spatial_energy(moved["a"].numpy(), moved["b"].numpy(), v, config, order=24)
    assert abs(spatial - refined) / problem.target_energy < 1e-5


def test_dense_fourier_sine_cosine_phase_and_finite_window_lobes():
    spectrum = load_spectrum_module()
    x = exp.midpoint_grid(8192, [-1.0, 1.0]).numpy().ravel()
    residual = 3 + np.cos(3 * np.pi * x) + 2 * np.sin(5 * np.pi * x)
    transform = spectrum.FiniteIntervalTransform(x, max_mode=32, points=2049)
    k = transform.modes
    expected = (6 * np.sinc(k) + np.sinc(k - 3) + np.sinc(k + 3)
                - 2j * (np.sinc(k - 5) - np.sinc(k + 5)))
    np.testing.assert_allclose(transform(residual), expected, atol=2e-6)


@pytest.mark.parametrize("half_width", [1.0, 0.8])
def test_cropped_spectrum_preserves_physical_frequency_and_integral_scale(half_width):
    from experiments.expD24_gd_residual_spectrum.analyze import window_spectrum

    # A zero predictor gives e(x) = -sin(2*pi*x), with an exact window transform.
    case = {"target": "sine", "frame_parameters": {
        "a": np.ones((1, 1)), "b": np.zeros((1, 1)),
        "c": np.zeros((1, 1)), "d": np.zeros((1, 1))}}
    result = window_spectrum(case, half_width, max_k=32, points=513)
    k = np.linspace(0, 32, 513)
    expected = 1j * half_width * (np.sinc(half_width * (k - 2))
                                - np.sinc(half_width * (k + 2)))
    np.testing.assert_allclose(result["k"], k, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(result["spectrum"], expected, atol=5e-8)
    assert result["relative_l2"] == pytest.approx(1)


@pytest.mark.parametrize("half_width", [1.0, 0.5])
def test_flat_residual_boundary_lobes_match_closed_form_and_converge(half_width):
    spectrum = load_spectrum_module()
    errors = []
    for n in (2048, 8192):
        x = exp.midpoint_grid(n, [-1.0, 1.0]).numpy().ravel()
        transform = spectrum.FiniteIntervalTransform(x, max_mode=32, points=2049)
        # Integral from -a to a of exp(-i*omega*x) dx = 2a*sinc(a*omega/pi).
        # Both boundaries coincide with quadrature-cell edges at these grids.
        exact = 2 * half_width * np.sinc(half_width * transform.modes)
        actual = transform((abs(x) < half_width).astype(float))
        errors.append(np.max(abs(actual - exact)))
        zero_indices = (np.arange(1, int(32 * half_width) + 1) / half_width * 64).astype(int)
        np.testing.assert_allclose(actual[zero_indices], 0, atol=1e-13)
    assert errors[-1] < 6e-7
    assert errors[-1] < errors[0] / 15  # Four times as many midpoint samples: second-order convergence.


def test_smooth_gaussian_transform_has_no_numerical_sidelobes():
    spectrum = load_spectrum_module()
    x = exp.midpoint_grid(8192, [-1.0, 1.0]).numpy().ravel()
    transform = spectrum.FiniteIntervalTransform(x, max_mode=16, points=1025)
    sigma = 0.12
    actual = transform(np.exp(-x * x / (2 * sigma * sigma)))
    exact = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (sigma * transform.omega) ** 2)
    # The omitted whole-line Gaussian tails are below 3e-17 in L1.
    np.testing.assert_allclose(actual, exact, atol=1e-13)
    assert np.all(np.diff(abs(actual)) < 0)


def test_frame_schedule_matches_archived_expD08_source():
    import ast
    spectrum = load_spectrum_module()
    source = ROOT / "experiments/expD06_derivative_readout_init/run.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "frame_steps")
    namespace = {"STEPS": 1000}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    for total in (20, 500, 1000, 3000):
        assert spectrum.frame_steps(total) == namespace["frame_steps"](total)
    steps = spectrum.frame_steps(1000)
    assert len(steps) == 67 and steps[0] == 0 and steps[-1] == 1000


def test_gif_encoding_and_slowed_loop_duration(tmp_path):
    spectrum = load_spectrum_module()
    steps = spectrum.frame_steps(2000)
    durations = spectrum.frame_durations(len(steps), 16.0)
    assert len(steps) == 77 and steps[-1] == 2000
    assert sum(durations) == 16000 and min(durations) >= 200
    from PIL import Image
    palette = None
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    image[:, :6, 0] = 255
    image[:, 6:12, 1] = 255
    image[:, 12:, 2] = 255
    path = tmp_path / "stream.gif"
    with path.open("wb") as stream:
        for index, shift in enumerate((0, 6, 12)):
            palette = spectrum.write_gif_frame(stream, Image.fromarray(np.roll(image, shift, axis=1)), palette, int(durations[index]))
        stream.write(b";")
    with Image.open(path) as gif:
        assert gif.n_frames == 3
        assert gif.info["loop"] == 0
        for index, shift in enumerate((0, 6, 12)):
            gif.seek(index)
            assert gif.info["duration"] == durations[index]
            np.testing.assert_array_equal(np.asarray(gif.convert("RGB")), np.roll(image, shift, axis=1))
