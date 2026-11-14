"""Numerical checks of the independent quadrature and interval machinery."""
from pathlib import Path
import sys
import gmpy2 as gp
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import posterior as p


def test_enclosed_gauss_rule_integrates_all_polynomials_through_degree_49():
    nodes,weights,intervals=p.quadrature()
    with gp.context(gp.get_context(),precision=p.PREC):
        powers=[p.point(1) for _ in nodes]
        for degree in range(50):
            total=p.point(0)
            for power,w in zip(powers,weights):total=p.add(total,p.mul(w,power))
            exact=gp.mpq(0) if degree%2 else gp.mpq(1,degree+1)
            assert total[0]<=exact<=total[1],degree
            powers=[p.mul(power,z) for power,z in zip(powers,intervals)]


def test_tanh_ellipse_and_local_target_bounds():
    with gp.context(gp.get_context(),precision=p.PREC):
        for N in [128,256,512,1024,2048]:
            B=p.local_bounds(N)
            h=gp.mpfr(2)/N;b=h*(p.RHO-gp.mpfr(1)/p.RHO)/4
            assert gp.mpfr(N)/8*b < gp.const_pi()/4
            # Check representative complex ellipse points against local bounds.
            rho=p.RHO
            for midpoint in [-1+1/N,0,1-1/N]:
                for theta in np.linspace(0,2*np.pi,33):
                    z=midpoint+float(h)/4*(rho*np.exp(1j*theta)+np.exp(-1j*theta)/rho)
                    vals=[np.exp(z),np.sin(4*np.pi*z),np.sin(24*np.pi*z),
                          np.sin(8*np.pi*(z+1)**2),np.sin(2*np.pi*z)+.001*np.sin(40*np.pi*z),
                          np.exp(-8*(z-.2)**2)*np.sin(8*np.pi*z)]
                    assert all(abs(v)<=float(bound)*(1+1e-13) for v,bound in zip(vals,B))
