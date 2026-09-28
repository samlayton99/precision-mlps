"""Independent finite features, lattice sums, capacity, and GD checks."""
import numpy as np
import pytest

from experiments.expD36_frozen_gamma_probe.structured_gamma import finite_geometry
from experiments.expD36_frozen_gamma_probe.structured_symbol_audit import (
    center_normalize, direct, necessary_time, symbol_integral,
)


def test_nonzero_mean_is_rejected_before_infinite_lattice_extension():
    with pytest.raises(ValueError, match='mean-zero'):
        symbol_integral(np.ones(13), 12, 1, 3.)
    with pytest.raises(ValueError, match='mean-zero'):
        direct(np.linspace(-1, 1, 13)+.01, 12, 1, 3., 1)
    with pytest.raises(ValueError, match='constant'):
        center_normalize(np.ones(13))


@pytest.mark.parametrize('n,q,gamma,cap', [(12,1,2.,4.), (15,3,2.,3.), (16,4,4.,6.)])
def test_symbol_and_cap_against_independent_finite_features(n, q, gamma, cap):
    rng = np.random.default_rng(391+n+q)
    v = center_normalize(rng.normal(size=(q*n+1, 3)))
    finite, extended, tail, _ = direct(v, n, q, gamma, halo=2)
    j = finite_geometry(n, q, 2, gamma)['j']
    np.testing.assert_allclose(finite, np.sum((j.T@v)**2, axis=0), rtol=2e-13, atol=2e-14)
    exact, scalar, general, _ = symbol_integral(v, n, q, gamma, points=4096, aliases=12)
    np.testing.assert_allclose(exact, extended, rtol=3e-12, atol=2e-13)
    assert np.max(tail) <= 1e-26
    assert np.all(finite <= general+2e-13)
    _, scalar_cap, general_cap, _ = symbol_integral(v, n, q, cap, points=4096, aliases=12)
    assert np.all(finite <= general_cap+2e-13)
    if q == 1:
        assert np.all(exact <= scalar+2e-13)
        assert np.all(finite <= scalar_cap+2e-13)


def test_alias_allowance_and_nonunit_scaling():
    n,q,gamma=12,3,12.
    v=center_normalize(np.random.default_rng(103).normal(size=(q*n+1,2)))
    coarse,_,coarse_cap,allowance=symbol_integral(v,n,q,gamma,4096,aliases=0)
    fine,_,fine_cap,_=symbol_integral(v,n,q,gamma,4096,aliases=18)
    assert np.all(np.abs(np.sqrt(fine)-np.sqrt(coarse)) <= allowance)
    assert np.all(np.sqrt(fine_cap)-np.sqrt(coarse_cap) <= allowance)
    scaled,_,scaled_cap,scaled_allowance=symbol_integral(3*v,n,q,gamma,4096,aliases=0)
    np.testing.assert_allclose(scaled,9*coarse,rtol=3e-14)
    np.testing.assert_allclose(scaled_cap,9*coarse_cap,rtol=3e-14)
    np.testing.assert_allclose(scaled_allowance,3*allowance,rtol=3e-14)


@pytest.mark.parametrize('gamma', [2.,4.])
def test_q1_capacity_via_cauchy_columns(gamma):
    g=finite_geometry(6,1,1,gamma)
    x,c,j=g['x'],g['centers'],g['j']
    m=len(x)
    a=np.exp(2*gamma*x)
    b=np.exp(2*gamma*c[:m])
    cauchy=1/(a[:,None]+b[None,:])
    np.testing.assert_allclose((j[:,1:m+1]-j[:,[0]])*np.sqrt(m),
                               -2*cauchy*b[None,:],atol=6e-16)
    assert np.linalg.matrix_rank(cauchy) == m
    assert np.linalg.matrix_rank(j) == m
    target=np.cos(4*x)
    coefficients=np.linalg.lstsq(j,target,rcond=None)[0]
    np.testing.assert_allclose(j@coefficients,target,atol=2e-11)


def test_cap_jensen_bound_against_executed_raw_gd():
    n,q,gamma,cap,halo=12,1,3.,4.,2
    g=finite_geometry(n,q,halo,gamma)
    x=g['x']-1
    target=center_normalize(np.exp(-.5*(x/.3)**2)*np.sin(5*np.pi*x))[:,0]
    j=g['j']; eta=.5/j.shape[1]
    _,cap_form,_,_=symbol_integral(target,n,q,cap,4096,aliases=12)
    rate=eta*float(cap_form[0])
    coefficients=np.zeros(j.shape[1])
    for n_update in range(201):
        residual=j@coefficients-target
        assert np.linalg.norm(residual)+3e-13 >= (1-rate)**n_update
        coefficients-=eta*j.T@residual
    threshold=.9
    necessary=necessary_time(rate,threshold)
    # No optimization threshold can have been crossed before the bound's integer.
    if necessary <= 200:
        coefficients=np.zeros(j.shape[1])
        for _ in range(necessary-1):
            coefficients-=eta*j.T@(j@coefficients-target)
        assert np.linalg.norm(j@coefficients-target)>threshold
