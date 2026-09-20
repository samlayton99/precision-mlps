"""Independent checks of the frozen-dictionary GD reference calculation."""
import numpy as np
import pytest

from experiments.expD26_freeze_and_readout_spectrum import spectrum as exp


def test_modal_prediction_matches_actual_gd_with_null_residual_and_bias():
    x = np.linspace(-1, 1, 31)
    centers = np.array([-.6, .2, .7])
    A = exp.dictionary(x, centers, 2.)/np.sqrt(len(x))
    np.testing.assert_array_equal(A[:, -1], np.full(len(x), 1/np.sqrt(len(x))))
    y = np.column_stack((np.sin(4*x), np.cos(3*x)))/np.sqrt(len(x))
    U,s,Vh,keep,alpha,floor = exp.decompose(A,y,1e-13)
    assert np.all(floor > 1e-8)
    error,*_ = exp.validate_explicit_gd(A,y,U,s,keep,alpha,floor,1/s[0]**2,steps=120)
    assert error < 1e-12


def test_step_count_is_first_satisfying_integer_and_respects_floor():
    s = np.array([2., .4]); alpha=np.array([.8, .6]); norm=1.
    logs = exp.contraction_logs(s,.1)
    for tolerance in (.8,.4,.1,.01):
        count=exp.steps_to_error(logs,alpha,1e-6,norm,tolerance)
        assert np.isfinite(count)
        assert exp.predicted_relative_error(count,logs,alpha,1e-6,norm) <= tolerance
        assert exp.predicted_relative_error(count-1,logs,alpha,1e-6,norm) > tolerance
    assert np.isinf(exp.steps_to_error(logs,alpha,.01,norm,.05))
    assert exp.steps_to_error(logs,alpha,0.,norm,1.) == 0


def test_tiny_resolved_contractions_are_not_rounded_to_zero():
    logs=exp.contraction_logs(np.array([1.,1e-12]),.1)
    assert logs[-1] < 0
    np.testing.assert_allclose(logs[-1],-1e-25,rtol=1e-14)
    assert exp.predicted_relative_error(1e25,logs,np.array([0.,1.]),0.,1.) < .4


def test_numerical_cutoff_is_a_retained_span_floor_not_true_null_space():
    A=np.diag([1.,1e-15,0.]);y=np.eye(3)
    U,s,Vh,keep,alpha,floor=exp.decompose(A,y,1e-13)
    np.testing.assert_array_equal(keep,[True,False,False])
    np.testing.assert_allclose(floor,[0.,1.,1.])
    assert A[1,1] != 0  # The excluded second direction is still mathematically learnable.


def test_monotone_rate_extinguishes_top_mode_in_one_step():
    logs=exp.contraction_logs(np.array([2.,1.]),.25)
    assert np.isneginf(logs[0])
    assert exp.predicted_relative_error(0,logs,np.array([1.,0.]),0.,1.) == 1.
    assert exp.predicted_relative_error(1,logs,np.array([1.,0.]),0.,1.) == 0.
    with pytest.raises(ValueError): exp.contraction_logs(np.array([2.]),.3)
