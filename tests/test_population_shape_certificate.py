"""Force concentration closure checked against the exact evolving field."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel
from experiments.expD34_readout_race import population_shape_certificate as certificate
from experiments.expD34_readout_race.population_shape_certificate import (
    PREFIX, evaluate, force_polynomial, travel_integral,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('width,scale', [(5, .12), (9, .4), (7, .8)])
def test_concentration_bounds_against_independent_effective_flow_jvp(width, scale):
    p = jnp.asarray(np.random.default_rng(821+width).normal(size=3*width+1)*scale)
    x = jnp.linspace(-1., 1., 51)
    y = jnp.sin(4*x)+.15*x*x
    state = kernel.decomposition(p, x, y)
    F = state['F']
    f = float(jnp.linalg.norm(F))
    r2 = jnp.sum(p[:-1].reshape(3, -1)**2, axis=0)
    weights = jnp.sum(F[:-1].reshape(3, -1)**2, axis=0)/(F@F)
    I = float(width*jnp.sum(weights**2))
    omega = float(width*jnp.sum(weights*r2))
    Y = float(jnp.sqrt(jnp.mean(state['eH']**2)))
    sigma = float(jnp.sqrt(jnp.linalg.eigvalsh(state['gram'])[0]))/2

    def population_norm(point):
        radius2 = jnp.sum(point[:-1].reshape(3, -1)**2, axis=0)
        return (width*jnp.sum(radius2**2))**.25

    q, qdot = jax.jvp(population_norm, (p,), (-F,))
    _, fdot = jax.jvp(lambda point: jnp.linalg.norm(kernel.effective(point, x, y)), (p,), (-F,))
    q = float(q)
    assert abs(float(qdot)) <= I**.25*f+1e-12
    assert f <= 3*Y*I**.25*q**3/width+1e-12
    assert omega <= np.sqrt(I)*q*q+1e-12
    coefficients = force_polynomial(q, width*f, Y, sigma, width, I)
    assert float(fdot) <= coefficients[1]*f/width+1e-12


def test_polynomial_and_regularized_quadrature_against_direct_integral():
    q0, g0, Y, sigma, width, I, travel = 1.3, .01, .7, .4, 512, 32., .03
    coefficients = force_polynomial(q0, g0, Y, sigma, width, I)
    s = I**.25
    q = q0+s*travel
    expected = (g0+2*np.sqrt(2)*Y*s*(q**3-q0**3)
                +6*Y*(q**5-q0**5)/(5*sigma**2*s)
                +2*np.sqrt(2)*Y*(q**6-q0**6)/(sigma**2*np.sqrt(width)))
    assert np.polynomial.polynomial.polyval(travel, coefficients) == pytest.approx(expected, rel=1e-12)
    direct = quad(lambda A: 1/np.polynomial.polynomial.polyval(A, coefficients),
                  0., travel, epsabs=1e-13, epsrel=1e-13)[0]
    for tolerance in (1e-9, 1e-11):
        actual, _ = travel_integral(coefficients, travel, tolerance)
        assert actual == pytest.approx(direct, rel=1e-9)


@pytest.mark.parametrize('g0', [1e-20, 1e-100, 1e-250])
def test_tiny_initial_force_against_independent_log_coordinate_quadrature(g0):
    travel = .03
    coefficients = force_polynomial(1.3, g0, .7, .4, 512, 32.)
    beta0 = coefficients[1]
    # A=(g0/beta0)*(exp(u)-1) resolves the small-A layer without
    # subtracting the analytical linear integral from the numerical result.
    upper = np.logaddexp(0., np.log(beta0)+np.log(travel)-np.log(g0))

    def transformed(u):
        jacobian = np.exp(np.log(g0)-np.log(beta0)+u)
        A = jacobian*(-np.expm1(-u))
        return jacobian/np.polynomial.polynomial.polyval(A, coefficients)

    reference, reference_error = quad(transformed, 0., upper, epsabs=1e-11,
                                      epsrel=1e-11, limit=200)
    actual, error = travel_integral(coefficients, travel, 1e-11)
    assert reference_error/reference < 1e-8
    assert error/actual < 1e-8
    assert actual == pytest.approx(reference, rel=1e-9)


def fixture_row():
    return dict(width=512, M=1., M4=2., fine_norm=1., target_norm=1.,
                target_fine_norm=1., eta=.002, reinforcement_F_norm=.001,
                reinforcement_hidden_force_energy_concentration=8.,
                reinforcement_coarse_sigma=.5, reinforcement_resolved=True)


def test_initial_only_evaluation_no_sixth_moment_or_kurtosis_required():
    row = fixture_row()
    result = evaluate(row)
    assert result[PREFIX+'status'] == 'conditional_FP64_effective_flow'
    assert result[PREFIX+'assumptions_certified'] is False
    assert result[PREFIX+'flow_time'] > 0
    assert result[PREFIX+'explicit_flow_time'] > 0
    assert result[PREFIX+'nominal_time_over_eta_not_GD_certificate'] == pytest.approx(result[PREFIX+'flow_time']/.002)
    A = result[PREFIX+'path']
    assert (np.sqrt(2)+4*np.sqrt(row['M']))*A+2*A*A == pytest.approx(row['reinforcement_coarse_sigma']/2)
    assert result[PREFIX+'time_relative_tolerance_difference'] < 1e-6
    assert evaluate(row, I_star=7.)[PREFIX+'status'] == 'initial_concentration_outside_allowance'
    assert evaluate(dict(row, M4=float('nan')))[PREFIX+'status'] == 'invalid_initial_data'
    assert evaluate(dict(row, reinforcement_resolved=False))[PREFIX+'status'] == 'unresolved_coarse_solve'
    assert evaluate(dict(row, M6=float('nan'), K_F=1e100, future_I=100000, actual_error=0.)) == result


def test_explicit_physical_time_scales_with_width():
    row = fixture_row()
    baseline = evaluate(row)
    wider = evaluate(dict(row, width=2*row['width'], reinforcement_F_norm=row['reinforcement_F_norm']/2))
    assert wider[PREFIX+'explicit_flow_time']/baseline[PREFIX+'explicit_flow_time'] == pytest.approx(2.)


def test_agreeing_quadratures_with_large_error_are_not_accepted(monkeypatch):
    monkeypatch.setattr(certificate, 'travel_integral', lambda *args: (1., .001))
    result = evaluate(fixture_row())
    assert result[PREFIX+'time_relative_tolerance_difference'] == 0.
    assert result[PREFIX+'status'] == 'quadrature_error_too_large'


def test_quadrature_warning_is_not_accepted(monkeypatch):
    def warned_quad(*args, **kwargs):
        certificate.warnings.warn('Failed integration', certificate.IntegrationWarning)
        return 1., 0.

    monkeypatch.setattr(certificate, 'quad', warned_quad)
    assert evaluate(fixture_row())[PREFIX+'status'] == 'quadrature_integration_warning'
