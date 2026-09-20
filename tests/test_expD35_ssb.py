import os
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('optimistix')
import jax.numpy as jnp
from experiments.expD35_optimization_exploration import core,run,ssb
from experiments.expD06_fixed_center_scales import higher_order as higher


def test_metric_reset_retains_readout_block_and_removes_cross_terms():
    a=jnp.arange(36,dtype=float).reshape(6,6)/36
    matrix=a@a.T+jnp.eye(6)
    reset=np.asarray(ssb.reset_matrix(matrix,'geometry',3))
    np.testing.assert_array_equal(reset[:3,:3],matrix[:3,:3])
    np.testing.assert_array_equal(reset[3:,3:],np.eye(3))
    assert not np.any(reset[:3,3:]) and not np.any(reset[3:,:3])
    assert np.min(np.linalg.eigvalsh(reset))>0


def test_metric_restart_and_resume_preserve_exact_state(tmp_path):
    source=os.environ.get('SSB_SOURCE')
    if not source or not Path(source).exists():pytest.skip('Pinned external source required')
    c=run.case(n=64,optimizer='ssbroyden')
    g,loss,phys=ssb.problem(c)
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    state=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
    def step(length):return ssb.kernel(64,'individual','sine',source,1e-30,1e-15,'geometry',2,length)
    prefix,_=step(2)(state)
    ssb.save_solver(tmp_path/'state.npz',prefix)
    restored=ssb.load_solver(tmp_path/'state.npz',state)
    split,_=step(3)(restored);whole,trace=step(5)(state)
    np.testing.assert_array_equal(split['z'],whole['z'])
    np.testing.assert_array_equal(trace[:,14],[0,0,1,0,1])
    fresh=ssb.restart(prefix,solver,loss,jnp.eye(len(prefix['z'])))
    np.testing.assert_array_equal(fresh['z'],prefix['z'])
    assert bool(fresh['solver'].first_step)


def test_covariant_initial_metric_matches_first_physical_step():
    source=os.environ.get('SSB_SOURCE')
    if not source or not Path(source).exists():pytest.skip('Pinned external source required')
    c=run.case(n=64,optimizer='ssbroyden',coordinates='individual')
    g,loss,phys=ssb.problem(c)
    cp=dict(c,coordinates='physical');_,loss_p,phys_p=ssb.problem(cp)
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    a=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
    b=higher.ssb_initial(solver,loss_p,core.initialize(cp)['z'])
    diagonal=jnp.r_[jnp.asarray(g.alpha),jnp.full(g.width,1/g.h)]
    b=ssb.restart(b,solver,loss_p,jnp.diag(diagonal**2))
    a,_=higher.ssb_step(solver,loss,phys,1e-30)(a)
    b,_=higher.ssb_step(solver,loss_p,phys_p,1e-30)(b)
    np.testing.assert_allclose(np.r_[*phys(a['z'])],b['z'],rtol=2e-12,atol=2e-13)


def test_non_descent_reset_is_explicit_and_restores_descent():
    source=os.environ.get('SSB_SOURCE')
    if not source or not Path(source).exists():pytest.skip('Pinned external source required')
    import equinox as eqx
    c=run.case(n=64,optimizer='ssbroyden');g,loss,phys=ssb.problem(c)
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    s=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
    s,_=higher.ssb_step(solver,loss,phys,1e-30)(s)
    s['solver']=eqx.tree_at(lambda st:st.f_info.hessian_inv.pytree,s['solver'],-jnp.eye(len(s['z'])))
    s['solver']=eqx.tree_at(lambda st:st.descent_state.newton,s['solver'],-s['solver'].f_info.grad)
    out,trace=ssb.kernel(64,'individual','sine',source,1e-30,1e-15,'non_descent',1000,1)(s)
    assert float(trace[0,14])==1. and float(trace[0,15])>0
    assert int(out['status'])==0 and int(out['count'])==2
