import numpy as np
import pytest
from experiments.expD36_frozen_gamma_probe.adam_spectral_analysis import sustained_acquisition, projected_energy

@pytest.mark.parametrize('trace,expected', [
    ([1.,.2,.09,.01],(2,1,2,2)),
    ([1.,.09,.2,.08,.07],(1,2,3,2)),
    ([1.,.2,.1],(2,1,2,1)),
    ([.09,.08,.01],(0,-1,0,3)),
    ([1.,.09,.2],(1,2,-1,0)),
    ([1.,.2],(-1,1,-1,0)),
])
def test_sustained(trace,expected):
    result=sustained_acquisition(trace,.1)
    assert tuple(result[k] for k in ['first','last_above','sustained','confirmation_updates']) == expected


def test_projected_energy_with_unresolved_tail():
    rng=np.random.default_rng(182)
    j=rng.normal(size=(13,7)); y=rng.normal(size=(13,3)); theta=rng.normal(size=(7,3))
    u,s,vh=np.linalg.svd(j,full_matrices=False); keep=4
    norm=np.sum(y*y,axis=0)
    actual=j@theta-y
    energy=projected_energy(s[:keep],vh[:keep],u[:,:keep].T@y,theta,norm)
    np.testing.assert_allclose(energy,(u[:,:keep].T@actual)**2/norm,atol=2e-14)
    unresolved=np.sum(actual**2,axis=0)/norm-energy.sum(axis=0)
    perpendicular=actual-u[:,:keep]@(u[:,:keep].T@actual)
    np.testing.assert_allclose(unresolved,np.sum(perpendicular**2,axis=0)/norm,atol=2e-14)
    assert np.all(unresolved>0)
