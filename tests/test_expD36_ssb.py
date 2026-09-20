import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
pytest.importorskip('optimistix')
from experiments.expD35_optimization_exploration import core,ssb,run as old
from experiments.expD06_fixed_center_scales import higher_order as higher
from experiments.expD36_ssb_geometry_switching import run,accessibility as access,diagnostics,reinitialization


def test_controller_persistence_cooldown_and_secondary_veto():
    state=dict(streak=0,last_event=-100)
    row=dict(beta_candidate=.4)
    beta,state=run.controller(row,state,0,'adaptive');assert beta is None
    beta,state=run.controller(row,state,1,'adaptive');assert beta==.4
    beta,state=run.controller(row,state,5,'adaptive');assert beta is None
    beta,state=run.controller(row,state,100,'adaptive');assert beta is None
    beta,state=run.controller(row,state,200,'adaptive');assert beta==.4
    assert run.next_diagnostic(200,20000)==500
    assert run.next_diagnostic(500,20000)==1000


def test_network_accessibility_derivatives_and_physical_initialization():
    z=[];physical=[]
    for coordinates in ('individual','neighbor'):
        c=old.case(n=64,optimizer='ssbroyden',coordinates=coordinates)
        g=core.old.geometry(64);initial=core.initialize(c)['z'];z.append(initial)
        physical.append(np.r_[*core.physical(initial,g,coordinates)])
        row,arrays=diagnostics.diagnostic(initial,jnp.eye(len(initial)),c,[10.,50.],audit=True)
        assert row['resolved']
        np.testing.assert_allclose(row['gain'],row['gain_svd'],atol=2e-10)
        np.testing.assert_allclose(arrays['finite_difference'][:,:,1],row['exposure'],rtol=3e-3,atol=2e-6)
    np.testing.assert_allclose(*physical,rtol=1e-12,atol=1e-13)


def test_metric_change_restart_and_resume(tmp_path):
    source=os.environ.get('SSB_SOURCE')
    if not source or not Path(source).exists():pytest.skip('Pinned source required')
    c=old.case(n=64,optimizer='ssbroyden',coordinates='individual')
    g,loss,physical=ssb.problem(c)
    solver=higher.ssb_solver(source,1e-30,1e-15,'accepted_step')
    state=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
    step=ssb.kernel(64,'individual','sine',source,1e-30,1e-15,'non_descent',1000,5)
    state,_=step(state)
    previous=state['z'];h=state['solver'].f_info.hessian_inv.pytree
    mixed,_=access.mix_metric(h,jax.grad(loss)(state['z']),.2)
    state=ssb.restart(state,solver,loss,mixed)
    np.testing.assert_array_equal(state['z'],previous)
    ssb.save_solver(tmp_path/'solver.npz',state)
    restored=ssb.load_solver(tmp_path/'solver.npz',state)
    a,_=step(state);b,_=step(restored)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


def test_parameter_reset_keeps_centers_and_separates_zero_weight_signal():
    c=old.case(n=64,optimizer='ssbroyden',coordinates='individual',reset_mask=[10],reinitialization='zero_physical')
    initial=core.initialize(c)['z'];g,loss,physical=ssb.problem(c)
    reset=jnp.asarray(reinitialization.replace(initial,c))
    assert float(physical(reset)[0][11])==0.
    assert float(jax.grad(loss)(reset)[g.width+1+10])==0.
    nonzero=jnp.asarray(reinitialization.replace(initial,dict(c,reinitialization='scaled_physical')))
    assert float(physical(nonzero)[0][11])!=0.
    assert float(physical(nonzero)[1][10])==float(physical(reset)[1][10])
    np.testing.assert_array_equal(reinitialization.replace(initial,dict(c,reinitialization='state_only')),initial)
    unchanged=np.ones(len(initial),dtype=bool);unchanged[[11,g.width+1+10]]=False
    np.testing.assert_array_equal(nonzero[unchanged],initial[unchanged])
