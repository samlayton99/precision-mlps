"""An exact constant-feature witness and the necessary descent condition."""
import numpy as np
import pytest
from flint import arb

from experiments.expD34_readout_race.population_frozen_witness_certificate import certify


def parameters():
    return dict(a=np.array([0.]), b=np.array([0.]), c=np.array([0.]), d=0.,
                x=np.array([-1., 1.]), y=np.array([1., -1.]), v=np.array([1., -1.]), eta=.25)


@pytest.mark.parametrize('N', [0, 100000])
def test_constant_features_preserve_target_error(N):
    result = certify(**parameters(), N=N)
    floor = arb(result['intervals']['relative_error_floor_lower'])
    assert floor > arb(1)-arb(2)**-100
    assert all(result['certified_above'].values())
    assert arb(result['intervals']['adjoint_witness_norm_upper']) == 0


def test_rejects_analytic_descent_condition_failure():
    case = parameters(); case['eta'] = 1.
    with pytest.raises(ValueError, match='eta.*W'):
        certify(**case, N=100)
