"""Numerical checks before relying on the 200k-update sweep."""
import numpy as np
import torch

from experiments.expD56_scale_acquisition.model import (
    TARGETS, diagnostics, gd_chunk, initial_parameters, midpoint_grid, prediction, target,
)
from experiments.expD56_scale_acquisition.run import all_cases, load_config
from experiments.expD56_scale_acquisition.plot import cohort_summary, neuron_slope_summary


def test_matrix_and_initialization():
    config=load_config()
    assert len(all_cases(config))==144
    w=config["width"]
    assert w==196
    torch.manual_seed(config["seed"])
    first=torch.nn.Linear(1,w,dtype=torch.float64)
    last=torch.nn.Linear(w,1,dtype=torch.float64)
    expected=torch.cat((first.weight.detach().flatten(),first.bias.detach(),last.weight.detach().flatten(),last.bias.detach()))
    base=initial_parameters(w,config["seed"],1.)
    torch.testing.assert_close(base,expected,rtol=0,atol=0)
    for factor in (2.,4.):
        scaled=initial_parameters(w,config["seed"],factor)
        torch.testing.assert_close(scaled[:2*w],base[:2*w]*factor,rtol=0,atol=0)
        torch.testing.assert_close(scaled[2*w:],base[2*w:],rtol=0,atol=0)
        torch.testing.assert_close(-scaled[w:2*w]/scaled[:w],-base[w:2*w]/base[:w])


def test_gd_matches_autograd_and_chunk_resume():
    torch.set_num_threads(1)
    x=torch.from_numpy(midpoint_grid(64))
    y=torch.from_numpy(target("mixed_sine",x.numpy()))
    w=17
    start=initial_parameters(w,30,2.)
    actual,n,ok=gd_chunk(start,x,y,.01,37)
    assert n==37 and ok
    expected=start.clone().requires_grad_(True)
    optimizer=torch.optim.SGD([expected],lr=.01)
    for _ in range(37):
        a,b,c=expected[:w],expected[w:2*w],expected[2*w:3*w]
        prediction_=(torch.tanh(x[:,None]*a+b)*c).sum(1)+expected[-1]
        loss=.5*(prediction_-y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-13)
    first,_,_=gd_chunk(start,x,y,.01,11)
    resumed,_,_=gd_chunk(first.clone(),x,y,.01,26)
    torch.testing.assert_close(actual,resumed,rtol=0,atol=0)


def test_targets_and_population_diagnostics():
    x,xe=midpoint_grid(1024),midpoint_grid(4096)
    assert not np.intersect1d(x,xe).size
    for name in TARGETS:
        y=target(name,x)
        assert np.all(np.isfinite(y)) and np.sqrt(np.mean(y*y))>0
    w=196
    theta=initial_parameters(w,30,1.).numpy()
    y=target("legendre_5",x)
    norm=np.sqrt(np.mean(y*y))
    report=diagnostics(theta,x,y/norm,xe,target("legendre_5",xe)/norm,2/147)
    np.testing.assert_allclose(report["gamma_rms"],np.linalg.norm(theta[:w])/np.sqrt(w))
    for k in (4,6,10,14): assert report[f"chi_{k}"]>=1
    np.testing.assert_allclose(report["gradient_norm"]**2,
                               report["fine_norm"]**2+report["tracking_norm"]**2,rtol=1e-10)
    assert report["coarse_kernel_min_eigenvalue"]>0
    assert report["tau_3"]<1e-4 and report["tau_5"]>.99
    # Generic nonlinear-output allowance holds on the observed state.
    pred=prediction(theta,x)
    basis=np.column_stack((np.ones_like(x),x/np.sqrt(np.mean(x*x))))
    nonaffine=pred-basis@(basis.T@pred/len(x))
    assert np.sqrt(np.mean(nonaffine**2))<=report["q_generic_current"]


def test_cohort_quantiles_are_across_functions():
    values=np.arange(16.)[:,None]*np.ones((1,7))
    median,low,high=cohort_summary(values)
    np.testing.assert_allclose(median,7.5)
    np.testing.assert_allclose(low,.375)
    np.testing.assert_allclose(high,14.625)


def test_neuron_distribution_uses_absolute_input_weights_only():
    snapshots=np.full((2,13),10000.)
    snapshots[0,:4]=[-3.,-1.,1.,7.]
    snapshots[1,:4]=2*snapshots[0,:4]
    stats=neuron_slope_summary(snapshots,4)
    np.testing.assert_allclose(stats["gamma_neuron_median"],[2.,4.])
    np.testing.assert_allclose(stats["gamma_neuron_p01"],[1.,2.])
    np.testing.assert_allclose(stats["gamma_neuron_p99"],[6.88,13.76])
