import numpy as np
from experiments.expD36_frozen_gamma_probe.section34_joint_metric import jacobians
from experiments.expD36_frozen_gamma_probe.section34_feature_access import metric_spectrum


def test_joint_jacobian_and_actual_diagonal_kernel():
    rng=np.random.default_rng(71)
    parameters=rng.normal(size=13)
    x=np.linspace(-1,1,19)
    readout,full,prediction=jacobians(parameters,x)
    for i in range(len(parameters)):
        delta=np.zeros_like(parameters);delta[i]=1e-6
        finite=(jacobians(parameters+delta,x)[2]-jacobians(parameters-delta,x)[2])/2e-6
        np.testing.assert_allclose(full[:,i],finite,rtol=2e-7,atol=5e-10)
    np.testing.assert_array_equal(readout,full[:,8:])
    y=np.sin(x);residual=prediction-y
    v=rng.uniform(.001,.1,len(parameters));count=37;epsilon=1e-8
    diagonal=1/(np.sqrt(v/(1-.999**count))+epsilon)
    mu,rho,w,rw,checks=metric_spectrum(full,y,residual,diagonal)
    kernel=(full*diagonal)@full.T/len(x)
    np.testing.assert_allclose(checks['residual']['direct_quadratic_form'],residual@kernel@residual/(y@y),rtol=1e-12)
    np.testing.assert_allclose(np.sum(rw)+checks['residual']['complement_energy'],residual@residual/(y@y),rtol=1e-12)
