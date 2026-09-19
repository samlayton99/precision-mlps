"""Numerical contracts for full Newton and scale-controlled handoffs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD06_fixed_center_scales import core, higher_order as ho, full_newton as nt


@pytest.mark.parametrize('coordinate',['physical','parameter_scale','parameter_differences'])
def test_full_hessian_and_physical_roundtrip(coordinate):
    g,r,j,loss=ho.problem(64,coordinate,2)
    z=ho.initial_parameters(g,3,coordinate)
    c,gamma=ho.physical(z,g,coordinate)
    np.testing.assert_allclose(ho.encode_physical(c,gamma,g,coordinate),z,rtol=3e-15,atol=1e-15)
    _,gradient,hessian,gn=nt.derivatives(g,coordinate,j,2)(z)
    np.testing.assert_allclose(gradient,jax.grad(loss)(z),rtol=2e-10,atol=3e-12)
    np.testing.assert_allclose(hessian,jax.hessian(loss)(z),rtol=2e-10,atol=3e-10)
    assert np.linalg.norm(hessian-gn)>1e-3
    if coordinate=='physical':
        _,_,js,_=ho.problem(64,'parameter_scale',2)
        zs=ho.encode_physical(c,gamma,g,'parameter_scale')
        _,gs,hs,_=nt.derivatives(g,'parameter_scale',js,2)(zs)
        scales=np.r_[g.alpha,np.full(g.width,1/g.h)]
        np.testing.assert_allclose(gs,scales*gradient,rtol=2e-10,atol=3e-12)
        np.testing.assert_allclose(hs,scales[:,None]*hessian*scales[None,:],rtol=2e-10,atol=3e-10)


@pytest.mark.parametrize('values,gradient,radius',[
    ([1.,3.,7.],[.1,.2,.3],1.),
    ([1e-20,1.,10.],[1.,2.,3.],.25),
    ([-2.,1.,3.],[1.,2.,3.],1.),
    ([-2.,1.,3.],[0.,.1,.1],1.),
    ([0.,1.,3.],[0.,.1,.1],1.),
    ([0.,1.,3.],[1.,.1,.1],1.),
    ([-2.,-2.,3.],[0.,0.,0.],1.),
    ([0.,0.,0.],[0.,0.,0.],1.),
])
def test_trust_region_global_optimality(values,gradient,radius):
    values=jnp.array(values);gradient=jnp.array(gradient)
    p,shift,hard=nt.eigen_step(values,jnp.eye(3),gradient,radius)
    p=np.asarray(p);shift=float(shift)
    assert np.linalg.norm(p)<=radius*(1+2e-14)
    assert np.min(np.asarray(values)+shift)>=-1e-14
    np.testing.assert_allclose((values+shift)*p+gradient,0,atol=2e-13)
    assert abs(shift*(np.linalg.norm(p)-radius))<2e-12
    assert gradient@p+.5*p@(values*p)<=1e-14


def test_newton_exact_quadratic_and_checkpoint_resume(tmp_path):
    a=jnp.diag(jnp.array([1.,2.,3.]));z=jnp.array([.1,.1,.1])
    residual=lambda z:a@z
    evaluate=lambda z:(a@z,a.T@a@z,a.T@a,a.T@a)
    step=nt.step(residual,evaluate,lambda z:(z,z))
    result,ev=step(nt.initial(z))
    assert int(result['count'])==1 and float(ev['damping'])==0
    np.testing.assert_allclose(result['z'],0,atol=1e-15)
    g,r,j,loss=ho.problem(64,'parameter_scale',1)
    z=ho.initial_parameters(g,0,'parameter_scale')
    step=nt.step(r,nt.derivatives(g,'parameter_scale',j,1),lambda z:ho.physical(z,g,'parameter_scale'))
    state,ev=step(nt.initial(z))
    assert int(state['status'])==0
    ho.save_state(tmp_path/'state.pkl',state,1)
    restored,count=ho.load_state(tmp_path/'state.pkl',nt.initial(z))
    direct,_=step(state);resumed,_=step(restored)
    assert count==1 and int(direct['count'])==2
    for a,b in zip(jax.tree.leaves(direct),jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a,b)


def test_ssb_transformed_metric_equivalence():
    eqx=pytest.importorskip('equinox')
    from pathlib import Path
    import os
    source=Path(os.environ.get('SSBROYDEN_SOURCE','/tmp/precision-ssbroyden-4c87785'))
    if not (source/'ssbrodyen_family.py').exists():pytest.skip('Pinned SSB source unavailable')
    s=jnp.array([.5,2.,4.]);p=jnp.array([.7,-.5,.9]);z=p/s
    a=jnp.array([[2.,.2,0.],[.2,3.,.1],[0.,.1,1.]])
    loss=lambda p:.5*p@a@p
    solver=ho.ssb_solver(source,1e-30,integration='accepted_step')
    sp=ho.ssb_initial(solver,loss,p);sz=ho.ssb_initial(solver,lambda z:loss(s*z),z)
    sp=eqx.tree_at(lambda st:st['solver'].f_info.hessian_inv.pytree,sp,jnp.diag(s*s))
    fp=ho.ssb_step(solver,loss,lambda p:(p,p),1e-30)
    fz=ho.ssb_step(solver,lambda z:loss(s*z),lambda z:(s*z,s*z),1e-30)
    for _ in range(4):
        sp,ep=fp(sp);sz,ez=fz(sz)
        assert int(sp['status'])==int(sz['status'])==0
        np.testing.assert_allclose(sp['z'],s*sz['z'],rtol=1e-10,atol=1e-12)
        hp=sp['solver'].f_info.hessian_inv.pytree;hz=sz['solver'].f_info.hessian_inv.pytree
        np.testing.assert_allclose(hp,s[:,None]*hz*s[None,:],rtol=1e-10,atol=1e-12)


def test_handoff_hash_and_fresh_newton_resume(tmp_path):
    import hashlib
    import json
    import time
    from experiments.expD06_fixed_center_scales import joint_conditioning as jc, difference_training as dt, run
    origin=tmp_path/'old'/'adam';origin.mkdir(parents=True)
    root=tmp_path/'new';root.mkdir()
    g=core.geometry(64);c,gamma=core.initial_physical(g,0,'xavier_a_reference')
    run.save_arrays(origin/'checkpoint_000000123.npz',c=c,gamma=gamma)
    run.write_json(origin/'case.json',dict(n=64,seed=0,target='sine',samples_per_cell=1,optimizer='adam'))
    digest=hashlib.sha256((origin/'checkpoint_000000123.npz').read_bytes()).hexdigest()
    config=jc.case('newton','parameter_scale',n=64,diagnostics=True,
                   warm_start=dict(root='../old',source_key='adam',step=123,sha256=digest))
    jc.advance_higher(root,config,1,time.monotonic()+120,None,samples=1)
    folder=root/dt.case_key(config)
    assert json.loads((folder/'latest.json').read_text())['completed_updates']==1
    assert json.loads((folder/'case.json').read_text())['initialization']=='checkpoint_handoff'
    with np.load(folder/'checkpoint_000000000.npz') as a:
        np.testing.assert_allclose(a['c'],c,rtol=3e-15,atol=0)
        np.testing.assert_array_equal(a['gamma'],gamma)
    jc.advance_higher(root,config,2,time.monotonic()+120,None,samples=1)
    assert json.loads((folder/'latest.json').read_text())['completed_updates']==2
    assert len(jc.read_trace(folder,2,'newton'))==2
    config['warm_start']['sha256']='wrong'
    with pytest.raises(ValueError,match='hash changed'):jc.warm_parameters(root,config,g,1)
