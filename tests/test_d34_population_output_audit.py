"""Exact output inequalities and moving-population directional identities."""
import jax
import numpy as np

from experiments.expD34_readout_race import population_output_audit as audit


def test_output_capacity_and_tail_bounds_are_global():
    rng = np.random.default_rng(872)
    x = np.linspace(-1, 1, 101)
    y = np.sin(5*x)+.2*x*x
    for scale in (.05, .5, 5., 50.):
        p = rng.normal(size=22)*scale
        row = {k: np.asarray(v).item() for k, v in audit.observables(p, x, y).items()}
        assert row['output_fine_norm'] <= row['Q']+1e-12
        assert row['Q_error_floor'] <= row['relative_l2']+1e-12
        tails = audit.tail_bounds(p, x, y)
        for degree in (2, 3, 5, 9):
            assert tails[f'tail{degree}_output'] <= tails[f'tail{degree}_capacity']+1e-11
            assert tails[f'tail{degree}_relative_floor'] <= row['relative_l2']+1e-12


def test_moment_derivatives_match_finite_differences():
    rng = np.random.default_rng(84)
    p = rng.normal(size=19)*.2
    direction = rng.normal(size=p.shape)*.1
    _, exact = jax.jvp(audit.moment_values, (p,), (direction,))
    eps = 1e-6
    numerical = (audit.moment_values(p+eps*direction)-audit.moment_values(p-eps*direction))/(2*eps)
    np.testing.assert_allclose(exact, numerical, rtol=2e-7, atol=1e-9)


def test_projected_energy_and_population_decomposition():
    rng = np.random.default_rng(33)
    p = rng.normal(size=25)*.05
    x = np.linspace(-1, 1, 101); y = np.sin(3*x)+.1*x*x
    row = {k: np.asarray(v).item() for k, v in audit.observables(p, x, y).items()}
    assert row['resolved']
    assert row['energy_identity'] < 1e-13
    assert row['force_identity'] < 1e-13
    assert row['residual_sensitivity'] <= row['sensitivity_bound']+1e-13
    assert row['residual_sensitivity'] <= row['fine_jacobian_hs_norm']**2+1e-13
    assert row['fine_jacobian_hs_norm']**2 <= row['sensitivity_bound']+1e-13
    for name in ('M', 'M6', 'C6', 'Es', 'Q'):
        np.testing.assert_allclose(row[name+'_dot_full'], row[name+'_dot_effective']+row[name+'_dot_tracking'], atol=1e-12)
        np.testing.assert_allclose(row[name+'_dot_effective'], row[name+'_dot_direct']+row[name+'_dot_compensation'], atol=1e-12)
        np.testing.assert_allclose(row[name+'_dot_effective'], row[name+'_dot_generated']+row[name+'_dot_target'], atol=1e-12)
    cert = audit.initial_certificate(dict(row, input_variance=float(np.mean(x*x))))
    assert cert['certificate_flow_time'] > 0
    assert cert['certificate_rank_margin'] > 0
    assert 0 < cert['certificate_relative_fine_floor'] <= 1
    for order in (8, 12):
        assert cert[f'p{order}_flow_time'] > 0
        assert cert[f'p{order}_rank_margin'] > 0
