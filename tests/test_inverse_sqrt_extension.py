import numpy as np
from experiments.expK02_inverse_sqrt_extension.construction import Extension, build, metrics


def test_extension_preserves_target_and_is_finite_outside_domain():
    e=Extension();x=np.geomspace(e.eps,1,1001)
    np.testing.assert_allclose(e.value(x),x**-.5,rtol=3e-16,atol=0)
    np.testing.assert_allclose(e.derivative(x),-.5*x**-1.5,rtol=3e-16,atol=0)
    outside=np.array([-100.,-1.,0.,e.eps/2,e.eps,1.,4.,100.])
    assert np.isfinite(e.value(outside)).all() and np.all(e.value(outside)>0)
    assert np.isfinite(e.derivative(outside)).all()
    assert e.derivative(0.)==0 and e.derivative(100.)==0
    np.testing.assert_array_equal(e.physical(np.array([-1.,1.])),[e.eps,1.])


def test_dyadic_partition_is_exact_on_requested_interval():
    e=Extension();x=np.geomspace(e.eps,1,4001);total=np.zeros_like(x)
    for s in e.scales:
        total+=s**-1.5*e.prototype_derivative((x/s-2)/2)/2
    np.testing.assert_allclose(total,-.5*x**-1.5,rtol=2e-15,atol=0)


def test_export_is_one_accurate_centered_tanh_network(tmp_path):
    e=Extension();net=build(e,128,tmp_path,halo=32)
    assert net.width==15*(128+2*32+1)
    x=np.geomspace(e.eps,1,2001)
    assert np.max(np.abs((net.predict(x)-x**-.5)*np.sqrt(x)))<3e-12
    assert metrics(net,e.eps,128)<1e-12


def test_l2_integrates_physical_measure():
    class RelativePerturbation:
        def predict(self,x):return 1.01*np.asarray(x)**-.5
    np.testing.assert_allclose(metrics(RelativePerturbation(),1e-4),.01,atol=1e-16,rtol=0)


def test_non_power_of_two_width_keeps_stable_export(tmp_path):
    e=Extension();net=build(e,96,tmp_path,halo=32)
    expected=np.concatenate([12.*(e.eps/(2*s))-.25*np.arange(-32,129) for s in e.scales])
    np.testing.assert_array_equal(net.hidden_bias-net.weight,expected)
    assert metrics(net,e.eps,128)<1e-8
