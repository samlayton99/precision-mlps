"""Independent differential-bound and envelope behavior checks."""
import numpy as np

from experiments.expD34_readout_race import population_gd_certificate as audit
from experiments.expD34_readout_race.adam_output_audit import field


def state_data(p, x, y):
    residual, J = field(p, x, y)
    q, _ = np.linalg.qr(np.c_[np.ones_like(x), x])
    jc = q.T @ J
    eh = residual-q@(q.T@residual)
    gh = J.T@eh
    gram = jc@jc.T
    ell = np.linalg.solve(gram, jc@gh)
    a, b, c = p[:-1].reshape(3, -1)
    row = dict(B=float(np.sum((a*a+b*b+c*c)**3)**(1/6)),
               M=float(np.sum(a*a+b*b+c*c)), Es=float(np.sum(a*a+c*c)),
               input_variance=float(np.mean(x*x)), target_norm=float(np.sqrt(np.mean(y*y))),
               relative_l2=float(np.linalg.norm(residual)/np.sqrt(np.mean(y*y))),
               fine_norm=float(np.linalg.norm(eh)),
               tracking_z_norm=float(np.linalg.norm(q.T@residual+ell)),
               coarse_kappa=float(np.linalg.eigvalsh(gram)[0]))
    return row, ell


def test_compensation_derivative_bound_by_independent_differences():
    rng = np.random.default_rng(973)
    p = rng.normal(size=61)*.03
    x = np.linspace(-1, 1, 101)
    y = .3+.4*x+np.sin(5*x)
    row, _ = state_data(p, x, y)
    constants = audit.constants(row, 1.001, .002)
    assert constants['status'] == 'eligible'
    for _ in range(4):
        direction = rng.normal(size=p.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-6
        _, plus = state_data(p+h*direction, x, y)
        _, minus = state_data(p-h*direction, x, y)
        derivative = np.linalg.norm((plus-minus)/(2*h))
        assert derivative <= constants['D_ell']*(1+1e-7)


def test_zero_residual_zero_tracking_is_stationary():
    row = dict(B=.05, M=1., Es=.8, input_variance=1/3,
               target_norm=1., relative_l2=0., fine_norm=0., tracking_z_norm=0., coarse_kappa=.2)
    answer = audit.evaluate(row, 1.25, max_steps=20)
    assert answer['accepted_updates'] == 20
    assert answer['travel_upper'] == 0
    assert answer['z_upper'] == 0
    assert answer['error_floor_final'] == 0


def test_more_initial_tracking_cannot_improve_enclosure():
    row = dict(B=.05, M=1., Es=.8, input_variance=1/3,
               target_norm=1., relative_l2=.7, fine_norm=.6, tracking_z_norm=1e-7, coarse_kappa=.2)
    low = audit.evaluate(row, 1.5, max_steps=1000)
    high = audit.evaluate(dict(row, tracking_z_norm=.01), 1.5, max_steps=1000)
    assert high['accepted_updates'] <= low['accepted_updates']
    assert high['half_initial_fine_error_through_update'] <= low['half_initial_fine_error_through_update']
    assert low['accepted_updates'] > 0


def test_step_failure_is_not_an_ode_certificate():
    row = dict(B=.05, M=1., Es=.8, input_variance=1/3,
               target_norm=1., relative_l2=.7, fine_norm=.6, tracking_z_norm=0., coarse_kappa=.2)
    answer = audit.evaluate(row, 1.25, eta=2., max_steps=20)
    assert answer['status'] == 'descent_step_condition_fails'
    assert answer['accepted_updates'] == 0
