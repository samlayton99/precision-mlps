"""Direct small-matrix references for structured target-filter calculations."""
import numpy as np
import pytest
from scipy.special import expit

from experiments.expD36_frozen_gamma_probe.structured_resolvent import (
    signed_lowrank_solver, rational_filter_mass, filter_cdf_bounds,
)


def test_signed_woodbury_in_complex_known_basis():
    rng = np.random.default_rng(391)
    q, _ = np.linalg.qr(rng.normal(size=(9, 9))+1j*rng.normal(size=(9, 9)))
    d = np.linspace(.02, 2., 9)
    u = rng.normal(size=(9, 3))+1j*rng.normal(size=(9, 3))
    values = np.array([.2, -.03, 0.])
    h = q@(np.diag(d)+(u*values)@u.conj().T)@q.conj().T
    solve = signed_lowrank_solver(d, u, values,
        to_basis=lambda x:q.conj().T@x, from_basis=lambda x:q@x)
    rhs = rng.normal(size=(9, 2))
    for shift in [.1+.2j, -.003]:
        np.testing.assert_allclose(solve(rhs, shift), np.linalg.solve(h-shift*np.eye(9), rhs),
                                   rtol=3e-12, atol=2e-12)
    basis_solve = signed_lowrank_solver(d, u, values)
    basis_h = np.diag(d)+(u*values)@u.conj().T
    np.testing.assert_allclose(basis_solve(rhs, -.003),
                               np.linalg.solve(basis_h+.003*np.eye(9), rhs), atol=2e-12)


@pytest.mark.parametrize('order', [2, 8, 64])
def test_filter_matches_sample_spectrum_including_nullspace(order):
    rng = np.random.default_rng(30)
    j = rng.normal(size=(11, 7))*.1
    y = rng.normal(size=(11, 3))
    h, forcing = j.T@j, j.T@y
    eig, u = np.linalg.eigh(j@j.T)
    weights = np.abs(u.T@y)**2/np.sum(y*y, axis=0)
    scale = .07
    expected = np.sum(weights*expit(-order*np.log(np.maximum(eig, 1e-300)/scale))[:, None], axis=0)
    result = rational_filter_mass(forcing, np.sum(y*y, axis=0), scale, order,
        lambda rhs,z:np.linalg.solve(h-z*np.eye(7), rhs), lambda v:h@v)
    np.testing.assert_allclose(result['mass'], expected, atol=2e-14)
    assert np.all(result['lower'] <= expected+2e-14)
    assert np.all(result['upper'] >= expected-2e-14)


def test_residual_correction_covers_deliberately_inaccurate_solve():
    rng = np.random.default_rng(41)
    j = rng.normal(size=(10, 6))*.1
    y = rng.normal(size=(10, 2))
    h, g = j.T@j, j.T@y
    perturbation = rng.normal(size=g.shape)*.07
    direct = lambda rhs,z:np.linalg.solve(h-z*np.eye(6), rhs)
    exact = rational_filter_mass(g, np.sum(y*y, axis=0), .1, 8, direct, lambda v:h@v)
    inaccurate = rational_filter_mass(g, np.sum(y*y, axis=0), .1, 8,
        lambda rhs,z:direct(rhs,z)+perturbation*(1+2j), lambda v:h@v)
    assert np.all(np.abs(inaccurate['mass']-exact['mass']) <= inaccurate['solve_radius']+1e-14)
    assert np.all(inaccurate['solve_radius'] > 0)


def test_construction_error_and_cdf_transfer():
    rng = np.random.default_rng(4)
    j = rng.normal(size=(12, 6))*.1
    approximate = j+rng.normal(size=j.shape)*1e-5
    y = rng.normal(size=(12, 2))
    h, forcing = approximate.T@approximate, approximate.T@y
    delta = np.linalg.norm(j@j.T-approximate@approximate.T, 2)
    action = lambda v:h@v
    solve = lambda rhs,z:np.linalg.solve(h-z*np.eye(6), rhs)
    cutoff, margin, order = .08, 1.4, 16
    lo = rational_filter_mass(forcing, np.sum(y*y, axis=0), cutoff/margin, order,
                             solve, action, kernel_error=delta)
    hi = rational_filter_mass(forcing, np.sum(y*y, axis=0), cutoff*margin, order,
                             solve, action, kernel_error=delta)
    np.testing.assert_allclose(lo['construction_radius'], delta*order/(2*lo['scale']))
    bounds = filter_cdf_bounds(lo, hi, cutoff)
    eig, u = np.linalg.eigh(j@j.T)
    mass = np.sum(np.abs(u[:, eig <= cutoff].T@y)**2, axis=0)/np.sum(y*y, axis=0)
    assert np.all(bounds['lower'] <= mass)
    assert np.all(bounds['upper'] >= mass)


def test_arithmetic_guard_is_separate_and_widens_bounds():
    j = np.diag([.1, .2, .3])
    y = np.ones((3, 1)); g = j.T@y; h = j.T@j
    args = (g, np.sum(y*y, axis=0), .05, 8,
            lambda rhs,z:np.linalg.solve(h-z*np.eye(3), rhs), lambda v:h@v)
    plain = rational_filter_mass(*args)
    guarded = rational_filter_mass(*args, residual_guard=lambda v,z,r:np.linalg.norm(v, axis=0)*1e-10)
    np.testing.assert_array_equal(plain['mass'], guarded['mass'])
    np.testing.assert_array_equal(plain['solve_radius'], guarded['solve_radius'])
    assert np.all(guarded['arithmetic_radius'] > plain['arithmetic_radius'])
    assert np.all(guarded['lower'] <= plain['lower'])
    assert np.all(guarded['upper'] >= plain['upper'])
