import numpy as np
import torch
from experiments.expK06_hard_analytic_suite.common import NS,WIDTHS,training_grid,halo,quadrature
from experiments.expK06_hard_analytic_suite.targets import MAIN,EXTRA,NAMES,target,kepler
from experiments.expK06_hard_analytic_suite.models import build
from docs.qi_practical_guide.reference import fit_qi

def test_target_suite_is_finite_and_disjoint():
    assert len(MAIN)==4 and len(EXTRA)==12 and len(set(NAMES))==16
    for name in NAMES:assert np.isfinite(target(name,np.linspace(-2,2,4097))).all()

def test_implicit_target_has_unique_correct_root():
    x=np.linspace(-2,2,4097);y=kepler(x)
    np.testing.assert_allclose(y-.9*np.sin(y),np.pi*x,rtol=1e-14,atol=1e-14)
    assert np.min(1-.9*np.cos(y))>=.1-1e-15

def test_same_sampled_quill_geometry():
    n=64;x=training_grid(n)
    m,info=fit_qi(x,np.sin(x),directions=np.ones((1,1)),intervals=n,lam=.25,rcond=1e-14)
    assert len(m.weight)==WIDTHS[0]==n+2*halo(n)+1
    np.testing.assert_allclose(m(np.linspace(-1,1,1001)),np.sin(np.linspace(-1,1,1001)),atol=1e-12)

def test_models_and_gradients():
    x=torch.tensor([[-.95],[-.1],[.2],[.9]],dtype=torch.float64)
    for method,variants in [('Adam',['pytorch','xavier']),('NPU',['native','positive','composed'])]:
        for variant in variants:
            m=build(method,64,variant,0);m(x).square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            assert m.head.weight.shape==(1,64)

def test_no_target_specific_tanh_initialization():
    m=build('Adam',64,'pytorch',0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);a=torch.nn.Linear(1,64,dtype=torch.float64);b=torch.nn.Linear(64,1,dtype=torch.float64)
    for p,q in zip(m.parameters(),list(a.parameters())+list(b.parameters())):assert torch.equal(p,q)

def test_physical_measure():
    x,w=quadrature(64)
    assert abs(w.sum()-2)<1e-14
    for name in NAMES:
        y=target(name,x);err=np.sqrt(np.dot(w,(.01*y)**2)/np.dot(w,y*y))
        assert abs(err-.01)<1e-15

def test_log_panel_integration_resolves_integrable_spike():
    from experiments.expK06_hard_analytic_suite.integration import zero_refined_grid
    x,w=zero_refined_grid([0.],32);eps=np.finfo(float).eps
    integral=np.dot(w,(abs(x)+eps)**(-.8))
    exact=2*((1+eps)**.2-eps**.2)/.2
    np.testing.assert_allclose(integral,exact,rtol=1e-13)
    assert abs(w.sum()-2)<1e-14
