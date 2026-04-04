"""Tests for QI construction and readout solve.

Verifies:
1. Toeplitz solve produces cardinal coefficients (residual ~ machine eps)
2. QI construction achieves machine-precision on analytic targets
3. Precision improves with width (exponential convergence)
4. Feature matrix Phi has correct shape and conditioning
5. Readout solvers agree on well-conditioned problems
6. Full pipeline: construct -> initialize model -> evaluate matches construction
"""

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.construction.qi_mpmath import construct_qi, evaluate_qi, default_halo, QIResult
from src.construction.readout import build_phi, solve_readout, solve_readout_with_bias
from src.construction.initialize import initialize_from_construction, initialize_with_readout_solve
from src.data.targets import get_target
from src.models.mlp import QIMlp
from src.config.schema import ModelConfig


# Construction parameters that achieve machine precision.
# lambda=0.3 gives intrinsic alias ~ exp(-pi^2/0.3) ~ 5e-15.
# Kc=120: cardinal coefficients decay to ~1e-20 at the stencil edge.
# halo=50: boundary truncation error ~1e-13.
# mp_dps=120: mpmath precision well beyond fp64.
QI_PARAMS = dict(lambda_star=0.3, Kc=120, halo=50, mp_dps=120)


def _rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def _linf(pred, true):
    return float(np.max(np.abs(pred - true)))


# ---------------------------------------------------------------------------
# 1. QIResult shape and type checks
# ---------------------------------------------------------------------------

class TestQIResultBasics:

    @pytest.fixture(scope="class")
    def qi_sine(self):
        target = get_target("sine")
        return construct_qi(target.fn_numpy, target.deriv_numpy, N=32, **QI_PARAMS)

    def test_types(self, qi_sine):
        r = qi_sine
        assert isinstance(r, QIResult)
        assert isinstance(r.N, int)
        assert isinstance(r.h, float)
        assert isinstance(r.gamma, float)
        assert isinstance(r.c0, float)
        assert isinstance(r.centers, np.ndarray)
        assert isinstance(r.a_coeffs, np.ndarray)

    def test_shapes(self, qi_sine):
        r = qi_sine
        n_full = r.N + 2 * r.halo + 1
        assert r.centers.shape == (n_full,)
        assert r.a_coeffs.shape == (n_full,)
        assert r.interior_centers.shape == (r.N + 1,)
        assert r.interior_a_coeffs.shape == (r.N + 1,)

    def test_gamma_scaling(self, qi_sine):
        r = qi_sine
        expected_gamma = r.lambda_val / r.h
        assert abs(r.gamma - expected_gamma) < 1e-12

    def test_grid_spacing(self, qi_sine):
        r = qi_sine
        assert abs(r.h - 2.0 / r.N) < 1e-15

    def test_centers_uniform(self, qi_sine):
        r = qi_sine
        diffs = np.diff(r.centers)
        assert np.allclose(diffs, r.h, atol=1e-14)

    def test_toeplitz_residual_small(self, qi_sine):
        r = qi_sine
        assert r.toeplitz_residual < 1e-12, f"Toeplitz residual: {r.toeplitz_residual}"


# ---------------------------------------------------------------------------
# 2. Machine-precision on sine target
# ---------------------------------------------------------------------------

class TestQIPrecisionSine:

    @pytest.fixture(scope="class")
    def result(self):
        target = get_target("sine")
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64, **QI_PARAMS)
        x_eval = np.linspace(-1, 1, 4001)
        y_true = target.fn_numpy(x_eval)
        y_pred = evaluate_qi(qi, x_eval)
        return qi, x_eval, y_true, y_pred

    def test_linf_below_1e11(self, result):
        _, _, y_true, y_pred = result
        err = _linf(y_pred, y_true)
        assert err < 1e-11, f"L_inf = {err:.3e}"

    def test_rel_l2_below_1e11(self, result):
        _, _, y_true, y_pred = result
        err = _rel_l2(y_pred, y_true)
        assert err < 1e-11, f"Rel L2 = {err:.3e}"

    def test_boundary_interpolation(self, result):
        qi, _, _, _ = result
        target = get_target("sine")
        q_left = evaluate_qi(qi, np.array([-1.0]))[0]
        g_left = target.fn_numpy(np.array([-1.0]))[0]
        assert abs(q_left - g_left) < 1e-14


# ---------------------------------------------------------------------------
# 3. Exponential convergence with width
# ---------------------------------------------------------------------------

class TestExponentialConvergence:

    @pytest.fixture(scope="class")
    def convergence_data(self):
        target = get_target("sine")
        x_eval = np.linspace(-1, 1, 2001)
        y_true = target.fn_numpy(x_eval)

        widths = [16, 32, 64, 128]
        errors = []
        for N in widths:
            qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
            y_pred = evaluate_qi(qi, x_eval)
            errors.append(_linf(y_pred, y_true))
        return widths, errors

    def test_error_decreases_monotonically(self, convergence_data):
        _, errors = convergence_data
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"Error not decreasing: {errors[i]:.3e} >= {errors[i-1]:.3e}"
            )

    def test_reaches_near_machine_precision(self, convergence_data):
        """Largest width should reach ~1e-12 (limited by fp64 evaluation)."""
        _, errors = convergence_data
        assert errors[-1] < 1e-11, f"N=128 error: {errors[-1]:.3e}"

    def test_exponential_rate(self, convergence_data):
        """Error should improve by at least 1.5 OOM from N=16 to N=128."""
        widths, errors = convergence_data
        log_errors = [math.log10(max(e, 1e-16)) for e in errors]
        improvement = log_errors[0] - log_errors[-1]
        assert improvement > 1.5, f"Expected >1.5 OOM improvement, got {improvement:.1f}"


# ---------------------------------------------------------------------------
# 4. Multiple target functions
# ---------------------------------------------------------------------------

class TestMultipleTargets:

    @pytest.mark.parametrize("target_name,threshold", [
        ("sine", 1e-11),
        ("cosine", 1e-11),
        ("exp", 1e-11),
        ("poly5", 1e-10),
        ("runge", 1e-7),  # narrow analyticity strip -> slower convergence
    ])
    def test_precision_at_N64(self, target_name, threshold):
        target = get_target(target_name)
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64, **QI_PARAMS)
        x_eval = np.linspace(-1, 1, 2001)
        y_true = target.fn_numpy(x_eval)
        y_pred = evaluate_qi(qi, x_eval)
        err = _linf(y_pred, y_true)
        assert err < threshold, f"Target '{target_name}' L_inf = {err:.3e}"

    @pytest.mark.parametrize("target_name", ["sine", "exp"])
    def test_near_machine_precision_at_N128(self, target_name):
        target = get_target(target_name)
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=128, **QI_PARAMS)
        x_eval = np.linspace(-1, 1, 4001)
        y_true = target.fn_numpy(x_eval)
        y_pred = evaluate_qi(qi, x_eval)
        err = _linf(y_pred, y_true)
        assert err < 1e-11, f"Target '{target_name}' L_inf = {err:.3e} at N=128"


# ---------------------------------------------------------------------------
# 5. Feature matrix and readout solve
# ---------------------------------------------------------------------------

class TestReadout:

    def test_phi_shape(self):
        x = np.linspace(-1, 1, 100)
        gamma = np.full(32, 24.0)
        centers = np.linspace(-1, 1, 32)
        Phi = build_phi(x, gamma, centers)
        assert Phi.shape == (100, 32)

    def test_phi_values_bounded(self):
        x = np.linspace(-1, 1, 100)
        gamma = np.full(32, 24.0)
        centers = np.linspace(-1, 1, 32)
        Phi = build_phi(x, gamma, centers)
        assert np.all(np.abs(Phi) <= 1.0)

    def test_solve_lstsq_recovers_known_weights(self):
        np.random.seed(42)
        x = np.linspace(-1, 1, 200)
        gamma = np.full(20, 15.0)
        centers = np.linspace(-1, 1, 20)
        Phi = build_phi(x, gamma, centers)
        v_true = np.random.randn(20)
        y = Phi @ v_true
        v_solved, info = solve_readout(Phi, y, method="lstsq")
        assert np.allclose(v_solved, v_true, atol=1e-10)

    def test_solvers_agree(self):
        np.random.seed(42)
        x = np.linspace(-1, 1, 200)
        gamma = np.full(20, 15.0)
        centers = np.linspace(-1, 1, 20)
        Phi = build_phi(x, gamma, centers)
        v_true = np.random.randn(20)
        y = Phi @ v_true

        v_lstsq, _ = solve_readout(Phi, y, method="lstsq")
        v_qr, _ = solve_readout(Phi, y, method="qr")
        v_svd, _ = solve_readout(Phi, y, method="svd")

        assert np.allclose(v_lstsq, v_qr, atol=1e-10)
        assert np.allclose(v_lstsq, v_svd, atol=1e-10)

    def test_solve_with_bias(self):
        np.random.seed(42)
        x = np.linspace(-1, 1, 200)
        gamma = np.full(20, 15.0)
        centers = np.linspace(-1, 1, 20)
        Phi = build_phi(x, gamma, centers)
        v_true = np.random.randn(20)
        b_true = 0.73
        y = Phi @ v_true + b_true

        v, b, info = solve_readout_with_bias(Phi, y, method="lstsq")
        assert abs(b - b_true) < 1e-10
        assert np.allclose(v, v_true, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Full pipeline: construct -> model -> evaluate
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_model_matches_construction(self):
        """Model output matches evaluate_qi after initialization."""
        target = get_target("sine")
        N = 32
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
        width = len(qi.centers)  # N + 2*halo + 1

        config = ModelConfig(width=width, layer_type="gamma_linear")
        model = QIMlp(config)
        initialize_from_construction(model, qi)

        x_eval = torch.linspace(-1, 1, 501, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            y_model = model(x_eval).squeeze(1).numpy()
        y_qi = evaluate_qi(qi, x_eval.squeeze(1).numpy())

        err = _linf(y_model, y_qi)
        assert err < 1e-13, f"Model vs construction mismatch: {err:.3e}"

    def test_model_achieves_precision(self):
        """Initialized model achieves same precision as raw construction."""
        target = get_target("sine")
        N = 64
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
        width = len(qi.centers)

        config = ModelConfig(width=width, layer_type="gamma_linear")
        model = QIMlp(config)
        initialize_from_construction(model, qi)

        x_eval = torch.linspace(-1, 1, 4001, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            y_model = model(x_eval).squeeze(1).numpy()
        y_true = target.fn_numpy(x_eval.squeeze(1).numpy())

        err = _linf(y_model, y_true)
        assert err < 1e-11, f"Model L_inf = {err:.3e}"

    def test_readout_solve_matches_construction(self):
        """Solving readout from QI geometry gets close to construction precision."""
        target = get_target("sine")
        N = 64
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
        width = len(qi.centers)

        config = ModelConfig(width=width, layer_type="gamma_linear")
        model = QIMlp(config)
        initialize_from_construction(model, qi, construct_readout=False)

        x_train = torch.linspace(-1, 1, 2048, dtype=torch.float64).unsqueeze(1)
        y_train = torch.tensor(
            target.fn_numpy(x_train.squeeze(1).numpy()), dtype=torch.float64,
        ).unsqueeze(1)
        initialize_with_readout_solve(model, x_train, y_train)

        x_eval = torch.linspace(-1, 1, 4001, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            y_model = model(x_eval).squeeze(1).numpy()
        y_true = target.fn_numpy(x_eval.squeeze(1).numpy())

        err = _linf(y_model, y_true)
        assert err < 1e-10, f"Readout solve L_inf = {err:.3e}"


# ---------------------------------------------------------------------------
# 7. Lambda sensitivity
# ---------------------------------------------------------------------------

class TestLambdaSensitivity:

    def test_optimal_lambda_beats_extremes(self):
        """lambda=0.3 should beat lambda=0.1 and lambda=2.0."""
        target = get_target("sine")
        x_eval = np.linspace(-1, 1, 2001)
        y_true = target.fn_numpy(x_eval)

        errors = {}
        for lam in [0.1, 0.3, 2.0]:
            qi = construct_qi(
                target.fn_numpy, target.deriv_numpy, N=64,
                lambda_star=lam, Kc=50, halo=55, mp_dps=80,
            )
            y_pred = evaluate_qi(qi, x_eval)
            errors[lam] = _linf(y_pred, y_true)

        assert errors[0.3] < errors[0.1], (
            f"lambda=0.3 ({errors[0.3]:.3e}) not better than lambda=0.1 ({errors[0.1]:.3e})"
        )
        assert errors[0.3] < errors[2.0], (
            f"lambda=0.3 ({errors[0.3]:.3e}) not better than lambda=2.0 ({errors[2.0]:.3e})"
        )


# ---------------------------------------------------------------------------
# 8. Diagnostic: gamma and weight scaling
# ---------------------------------------------------------------------------

class TestScaling:

    def test_gamma_grows_with_N(self):
        """gamma = lambda_star * N / 2, so gamma ~ O(N)."""
        target = get_target("sine")
        for N in [32, 64, 128]:
            qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
            expected = 0.3 * N / 2.0
            assert abs(qi.gamma - expected) < 1e-10

    def test_outer_weights_bounded(self):
        """Outer weights a_n should stay bounded as N grows."""
        target = get_target("sine")
        for N in [32, 64, 128]:
            qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=N, **QI_PARAMS)
            max_weight = np.max(np.abs(qi.a_coeffs))
            assert max_weight < 100, f"N={N}: max |a_n| = {max_weight:.3e}"


# ---------------------------------------------------------------------------
# 9. Dual precision paths (fp64 vs mpmath)
# ---------------------------------------------------------------------------

class TestPrecisionFlag:
    """Test that precision="fp64" and precision="mpmath" both work."""

    def test_default_halo_scaling(self):
        """default_halo scales with both N and 1/lambda."""
        # Scales with 1/lambda: smaller lambda -> larger min halo.
        assert default_halo(64, lambda_star=0.30) < default_halo(64, lambda_star=0.20)
        # Scales with N for large N.
        assert default_halo(256, lambda_star=0.30) > default_halo(64, lambda_star=0.30)
        # For N=64, lambda=0.30: halo = max(ceil(35/0.6), 25) = max(59, 25) = 59
        assert default_halo(64, lambda_star=0.30) == 59

    def test_fp64_path_precision(self):
        """fp64 path reaches ~1e-11 on sine at N=64 with default params."""
        target = get_target("sine")
        qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64,
                          precision="fp64", use_cache=False)
        assert qi.lambda_val == 0.30  # default for fp64
        x_eval = np.linspace(-1, 1, 2001)
        y_true = target.fn_numpy(x_eval)
        err = _linf(evaluate_qi(qi, x_eval), y_true)
        assert err < 1e-11, f"fp64 L_inf = {err:.3e}"

    @pytest.mark.slow
    def test_mpmath_path_reaches_machine_eps(self):
        """mpmath path reaches machine eps on sine at N=64."""
        target = get_target("sine")
        with tempfile.TemporaryDirectory() as tmp:
            qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64,
                              precision="mpmath", cache_dir=Path(tmp))
        assert qi.lambda_val == 0.25  # default for mpmath
        x_eval = np.linspace(-1, 1, 2001)
        y_true = target.fn_numpy(x_eval)
        err = _linf(evaluate_qi(qi, x_eval), y_true)
        assert err < 1e-14, f"mpmath L_inf = {err:.3e}, expected machine-eps"


class TestCache:
    """Test disk cache for cardinal coefficients."""

    def test_fp64_cache_roundtrip(self):
        """fp64 cache: cold -> warm returns identical coefficients."""
        target = get_target("sine")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            qi1 = construct_qi(target.fn_numpy, target.deriv_numpy, N=32,
                               precision="fp64", cache_dir=tmp_path)
            # Cache file exists
            cache_files = list(tmp_path.glob("*.npz"))
            assert len(cache_files) == 1

            qi2 = construct_qi(target.fn_numpy, target.deriv_numpy, N=32,
                               precision="fp64", cache_dir=tmp_path)
            # Coefficients identical
            assert np.allclose(qi1.a_coeffs, qi2.a_coeffs, atol=0)
            assert qi1.c0 == qi2.c0

    def test_cache_is_target_independent(self):
        """Same (lambda, Kc, N) cache entry serves different targets."""
        t1 = get_target("sine")
        t2 = get_target("exp")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            construct_qi(t1.fn_numpy, t1.deriv_numpy, N=32,
                         precision="fp64", cache_dir=tmp_path)
            # Second construction with different target should reuse cache.
            construct_qi(t2.fn_numpy, t2.deriv_numpy, N=32,
                         precision="fp64", cache_dir=tmp_path)
            # Still only one cache file
            assert len(list(tmp_path.glob("*.npz"))) == 1

    def test_cache_distinguishes_params(self):
        """Different (lambda, Kc, N) produce different cache entries."""
        target = get_target("sine")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            construct_qi(target.fn_numpy, target.deriv_numpy, N=32,
                         precision="fp64", cache_dir=tmp_path)
            construct_qi(target.fn_numpy, target.deriv_numpy, N=64,
                         precision="fp64", cache_dir=tmp_path)
            construct_qi(target.fn_numpy, target.deriv_numpy, N=32,
                         precision="fp64", lambda_star=0.25, cache_dir=tmp_path)
            assert len(list(tmp_path.glob("*.npz"))) == 3

    def test_use_cache_false_skips_cache(self):
        """use_cache=False neither reads nor writes the cache."""
        target = get_target("sine")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            construct_qi(target.fn_numpy, target.deriv_numpy, N=32,
                         precision="fp64", cache_dir=tmp_path, use_cache=False)
            assert len(list(tmp_path.glob("*.npz"))) == 0
