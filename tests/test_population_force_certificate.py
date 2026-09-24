"""Initial-force comparison and the exact constrained-gradient norm identity."""
import csv
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race import adam_forces as af
from experiments.expD34_readout_race.population_force_certificate import (
    FRACTIONS, evaluate, force_time_integral, write_audit,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('ratio,zeta,q', [(.01, .3, 1.2), (.2, 3., 1.9), (2., .01, 1.5)])
def test_time_integral_matches_direct_reference(ratio, zeta, q):
    value, _ = force_time_integral(ratio, zeta, q)
    reference, _ = quad(lambda u: 1/(ratio+2*math.sqrt(2)*(u**3-1)+zeta*(u**4-1)),
                        1., q, epsabs=1e-11, epsrel=1e-11)
    assert value == pytest.approx(reference, rel=1e-10)


def test_tiny_force_gives_logarithmic_delay():
    first, _ = force_time_integral(1e-20, .7, 1.9)
    second, _ = force_time_integral(1e-40, .7, 1.9)
    assert second-first == pytest.approx(20*math.log(10)/(6*math.sqrt(2)+4*.7), rel=1e-10)


def state():
    return dict(B=.05, fine_norm=.2, F_norm=.03*.2*.05**3, M=1., Es=.5,
                coarse_kappa=.1, input_variance=1/3, target_norm=.5, resolved=True)


def test_selection_is_initial_only_and_keeps_all_candidates():
    row = state()
    summary, candidates = evaluate(row)
    other, _ = evaluate(dict(row, future_error=0., future_force=1e99, checkpoint_after=123456))
    assert summary == other
    assert len(candidates) == len(FRACTIONS)
    eligible = [c for c in candidates if c['force_certificate_status'] == 'eligible_half_retention']
    assert summary['force_certificate_flow_time'] == max(c['force_certificate_flow_time'] for c in eligible)
    assert summary['force_certificate_relative_fine_floor'] >= .5
    assert sum(c['force_certificate_selected'] for c in candidates) == 1


@pytest.mark.parametrize('field,value', [('F_norm', np.nan), ('B', np.inf), ('fine_norm', -.1)])
def test_nonfinite_or_negative_initial_data_retained_as_failure(field, value):
    row = state(); row[field] = value
    summary, candidates = evaluate(row)
    assert summary['force_certificate_status'] == 'invalid_initial_data'
    assert len(candidates) == len(FRACTIONS)


def test_zero_force_stationary_and_unresolved_is_not_stationary():
    row = state(); row['F_norm'] = 0.
    summary, _ = evaluate(row)
    assert summary['force_certificate_status'] == 'stationary_effective_flow'
    row['resolved'] = 'False'
    assert evaluate(row)[0]['force_certificate_status'] == 'unresolved_coarse_solve'


def test_csv_preserves_input_and_records_failed_candidates(tmp_path):
    source = tmp_path/'input.csv'
    rows = [dict(state(), arbitrary_original_field='preserve me'), dict(state(), F_norm='nan')]
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = tmp_path/'audit'
    manifest = write_audit(source, output)
    with (output/'states.csv').open() as stream:
        result = list(csv.DictReader(stream))
    assert result[0]['arbitrary_original_field'] == 'preserve me'
    assert manifest['rows'] == 2
    assert manifest['candidates'] == 2*len(FRACTIONS)
    assert result[1]['force_certificate_status'] == 'invalid_initial_data'


def test_projected_force_energy_identity_against_jvp():
    # Small exact network, nonzero biases and mixed signs; no surrogate model.
    x = jnp.linspace(-1., 1., 41)
    y = jnp.sin(3*x)+.2*x*x
    theta = jnp.array(np.random.default_rng(45).normal(size=19)*.3)
    q = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)

    def residual(p):
        a, b, c = p[:-1].reshape(3, -1)
        return jnp.tanh(x[:, None]*a+b)@c+p[-1]-y

    def coarse(p):
        return q.T@residual(p)/len(x)

    def fine(p):
        return residual(p)-q@coarse(p)

    def force(p):
        g, _, jc, ec = af.field(p, x, y)
        return af.split(g, jc, ec)[0][0]

    F = force(theta)
    _, derivative = jax.jvp(force, (theta,), (-F,))
    _, fine_velocity = jax.jvp(fine, (theta,), (F,))
    _, fine_curvature = jax.jvp(lambda p: jax.jvp(fine, (p,), (F,))[1], (theta,), (F,))
    _, coarse_curvature = jax.jvp(lambda p: jax.jvp(coarse, (p,), (F,))[1], (theta,), (F,))
    g, _, jc, ec = af.field(theta, x, y)
    ell = jnp.linalg.solve(jc@jc.T, jc@(g-jc.T@ec))
    expected = -jnp.mean(fine_velocity**2)-jnp.mean(fine(theta)*fine_curvature)+ell@coarse_curvature
    np.testing.assert_allclose(jnp.dot(F, derivative), expected, rtol=2e-10, atol=1e-14)
