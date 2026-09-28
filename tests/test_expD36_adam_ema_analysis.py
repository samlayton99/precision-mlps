import numpy as np
import pytest
from experiments.expD36_frozen_gamma_probe.adam_ema_analysis import ema_squared, crossing_summary


def test_ema_matches_independent_explicit_recurrence():
    rng=np.random.default_rng(516)
    errors=rng.uniform(0,2,(100,2,3));half_life=30;beta=2**(-1/half_life)
    expected=[errors[0]**2]
    for error in errors[1:]:expected.append(beta*expected[-1]+(1-beta)*error**2)
    np.testing.assert_allclose(ema_squared(errors,half_life),expected,rtol=2e-15,atol=0)


def test_constant_and_initialization():
    errors=np.full((200,2),.25)
    np.testing.assert_allclose(ema_squared(errors,100),errors**2,atol=2e-16)
    np.testing.assert_array_equal(ema_squared(np.array([.3]),10),np.array([.09]))
    np.testing.assert_allclose(ema_squared(np.array([1.,0.,0.]),1),[1.,.5,.25])


@pytest.mark.parametrize('values,first,recross',[
    ([1.,.0001,.0002,.00005],1,1),
    ([.00001,.00002],0,0),
    ([1.,.1],-1,0),
    ([1.,.0001],1,0),
])
def test_first_censor_and_recross(values,first,recross):
    got=crossing_summary(np.array(values))
    assert int(got[0])==first and int(got[1])==recross
