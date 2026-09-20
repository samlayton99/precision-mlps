import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from experiments.expD06_fixed_center_scales import core, difference_training as dt, run


def test_parameter_difference_coordinates_and_jacobian():
    g = core.geometry(128)
    c, gamma = core.initial_physical(g, 0, "xavier_a_reference")
    z = dt.encode(c, g, "parameter_differences")
    np.testing.assert_allclose(dt.decode(z, g, "parameter_differences"), c, atol=3e-17)
    x = np.linspace(-1, 1, 129)
    phi = np.tanh((x[:, None]-g.centers)*gamma)
    q = z[1:]*np.cumsum(g.alpha[1:])
    alternate = g.alpha[0]*z[0]+(phi[:, :-1]-phi[:, 1:])@q[:-1]+phi[:, -1]*q[-1]
    np.testing.assert_allclose(alternate, c[0]+phi@c[1:], atol=2e-16)
    gc = jnp.asarray(np.random.default_rng(7).normal(size=c.size))
    auto = jax.grad(lambda a: dt.decode(a,g,"parameter_differences")@gc)(jnp.asarray(z))
    np.testing.assert_allclose(auto, dt.pullback(gc,g,"parameter_differences"), atol=2e-14)
    lam = jnp.asarray(g.h*gamma)
    y = core.target(jnp.asarray(x), "sine")
    loss = lambda a: dt.physical_loss(dt.decode(a,g,"parameter_differences"),lam,jnp.asarray(x),y,g)
    direction = jnp.asarray(np.random.default_rng(2).normal(size=z.size))
    direction /= jnp.linalg.norm(direction)
    eps = 1e-6
    np.testing.assert_allclose((loss(z+eps*direction)-loss(z-eps*direction))/(2*eps),
                               jax.grad(loss)(z)@direction, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("optimizer", ["gd", "adam"])
@pytest.mark.parametrize("coordinate", ["parameter_scale", "parameter_differences"])
def test_joint_first_order_updates_and_resume(optimizer,coordinate,tmp_path):
    g = core.geometry(128); eta = 1e-5; epsilon = 1e-12
    state = dt.initial(g,0,coordinate,optimizer)
    c = np.asarray(dt.decode(state["z"],g,coordinate)); lam = np.asarray(state["lam"])
    basis = np.eye(g.width+1)
    transform = np.stack([dt.decode(jnp.asarray(v),g,coordinate) for v in basis],axis=1)
    x=jnp.linspace(-1,1,129); y=core.target(x,"sine")
    gc,gl=jax.grad(dt.physical_loss,argnums=(0,1))(jnp.asarray(c),jnp.asarray(lam),x,y,g)
    gz=transform.T@np.asarray(gc)
    dz=-eta*gz if optimizer=="gd" else -eta*gz/(np.abs(gz)+epsilon)
    dl=-eta*np.asarray(gl) if optimizer=="gd" else -eta*np.asarray(gl)/(np.abs(gl)+epsilon)
    fn=dt.chunk(128,coordinate,1,True,1,False,optimizer,epsilon)
    next_state,(_,dense)=fn(state,eta,0)
    np.testing.assert_allclose(dense["delta_c"][0],transform@dz,atol=3e-16,rtol=1e-10)
    np.testing.assert_allclose(dense["delta_lambda"][0],dl,atol=3e-16,rtol=1e-10)
    whole,_=dt.chunk(128,coordinate,4,False,1,False,optimizer,epsilon)(state,eta,0)
    run.save_state(tmp_path/'state.pkl',next_state,1)
    resumed,start=run.load_state(tmp_path/'state.pkl')
    resumed,_=dt.chunk(128,coordinate,3,False,1,False,optimizer,epsilon)(resumed,eta,start)
    for a,b in zip(jax.tree.leaves(whole),jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a,b)


def test_explicit_epsilon_required_for_difference_adam():
    with pytest.raises(ValueError,match="explicit native epsilon"):
        dt.chunk(128,"parameter_differences",1,optimizer="adam")


@pytest.mark.parametrize("coordinate", ["parameter_scale", "parameter_differences"])
def test_analytic_joint_jacobian(coordinate):
    from experiments.expD06_fixed_center_scales import higher_order as ho
    g,residual,jacobian,loss=ho.problem(128,coordinate,1)
    z=ho.initial_parameters(g,0,coordinate)
    r,j=jacobian(z)
    np.testing.assert_allclose(j,jax.jacfwd(residual)(z),atol=2e-14,rtol=3e-12)
    np.testing.assert_allclose(j.T@r,jax.grad(loss)(z),atol=2e-14,rtol=3e-12)


@pytest.mark.parametrize('coordinate',['parameter_scale','parameter_differences'])
def test_native_hessian_includes_residual_curvature(coordinate):
    from experiments.expD06_fixed_center_scales import higher_order as ho
    from experiments.expD06_fixed_center_scales.joint_mechanism_probes import native_hessians
    g,residual,jacobian,loss=ho.problem(128,coordinate,1)
    z=ho.initial_parameters(g,0,coordinate)
    z=z.at[g.width+1:].multiply(20)
    c,gamma=map(np.asarray,ho.physical(z,g,coordinate))
    hessian,gn,gradient=native_hessians(g,c,gamma,coordinate,1)
    np.testing.assert_allclose(hessian,jax.hessian(loss)(z),atol=3e-10,rtol=2e-10)
    np.testing.assert_allclose(gradient,jax.grad(loss)(z),atol=3e-12,rtol=2e-10)
    assert np.linalg.norm(hessian-gn)>1


@pytest.mark.parametrize('coordinate',['parameter_scale','parameter_differences'])
def test_multiprecision_full_gradient(coordinate):
    from experiments.expD06_fixed_center_scales import higher_order as ho
    from experiments.expD06_fixed_center_scales.ssb_gradient_audit import gradient_block,pullback
    g=core.geometry(128);z=ho.initial_parameters(g,0,coordinate);c,gamma=map(np.asarray,ho.physical(z,g,coordinate))
    x=np.linspace(-1,1,9)
    gradients,losses=gradient_block((x,g.centers,gamma,[c],g.h))
    native=np.array(pullback(gradients[0],g.alpha,coordinate),float)/len(x)
    def loss(p):
        cc,gg=ho.physical(p,g,coordinate)
        residual=cc[0]+core.tanh((jnp.asarray(x)[:,None]-g.centers)*gg)@cc[1:]-core.target(jnp.asarray(x),'sine')
        return .5*jnp.mean(residual**2)
    np.testing.assert_allclose(native,jax.grad(loss)(z),atol=3e-14,rtol=3e-12)
    np.testing.assert_allclose(float(losses[0])/len(x),2*loss(z),atol=3e-15)


def test_gn_invariance_rank_deficiency_and_damping_metric():
    from experiments.expD06_fixed_center_scales import higher_order as ho
    j=np.array([[1.,2.],[2.,-1.],[.3,.7]]);r=np.array([.2,-.7,.9]);t=np.array([[2.,.7],[0.,.4]])
    dp=np.linalg.lstsq(j,-r,rcond=None)[0]
    dz=np.linalg.lstsq(j@t,-r,rcond=None)[0]
    np.testing.assert_allclose(t@dz,dp,atol=2e-15)
    mu=.03
    step=ho.augmented_qr(jnp.asarray(j@t),jnp.asarray(r),mu)
    expected=np.linalg.solve(j.T@j+mu*np.linalg.inv(t).T@np.linalg.inv(t),-j.T@r)
    np.testing.assert_allclose(t@step,expected,atol=2e-15)
    deficient=j[:1]
    d1=np.linalg.lstsq(deficient,-r[:1],rcond=None)[0]
    d2=t@np.linalg.lstsq(deficient@t,-r[:1],rcond=None)[0]
    assert np.linalg.norm(d1-d2)>1e-3
    np.testing.assert_allclose(deficient@d1,deficient@d2,atol=1e-15)


def test_gn_progress_resume_and_failure(tmp_path):
    from experiments.expD06_fixed_center_scales import higher_order as ho
    a=jnp.array([[1.,2.],[2.,-1.],[.3,.7]]);target=jnp.array([.2,-.7,.9])
    residual=lambda z:a@z-target
    jacobian=lambda z:(residual(z),a)
    advance=ho.gn_step(residual,jacobian,lambda p:(p,p))
    initial=ho.gn_initial(jnp.array([1.,-2.]))
    middle,evidence=advance(initial)
    assert int(middle['count'])==1 and int(middle['status'])==0
    assert evidence['actual']>0 and evidence['linear_residual']<1e-12
    run.save_state(tmp_path/'gn.pkl',middle,1)
    loaded,_=run.load_state(tmp_path/'gn.pkl')
    for left,right in zip(jax.tree.leaves(advance(middle)[0]),jax.tree.leaves(advance(loaded)[0])):
        np.testing.assert_array_equal(left,right)
    bad=ho.gn_step(lambda z:jnp.full((3,),jnp.nan),jacobian,lambda p:(p,p))(initial)[0]
    assert int(bad['status'])==1 and int(bad['count'])==0
    np.testing.assert_array_equal(bad['z'],initial['z'])
    uphill=ho.gn_step(lambda z:residual(initial['z'])+1e6,jacobian,lambda p:(p,p),max_trials=2)(initial)[0]
    assert int(uphill['status'])==2 and int(uphill['count'])==0


def ssb_source():
    import os
    from pathlib import Path
    pytest.importorskip('optimistix')
    path=os.environ.get('SSBROYDEN_SOURCE','/tmp/precision-ssbroyden-4c87785')
    if not (Path(path)/'ssbrodyen_family.py').exists():pytest.skip('Pinned external SSBroyden source unavailable')
    return path


def test_ssb_guard_and_default_equivalence(tmp_path):
    import importlib.util,sys
    from pathlib import Path
    from experiments.expD06_fixed_center_scales import higher_order as ho
    source=ssb_source()
    eps=np.finfo(float).eps
    guarded=ho.ssb_solver(source,eps)
    spec=importlib.util.spec_from_file_location('precision_ssbroyden_upstream',Path(source)/'ssbrodyen_family.py')
    upstream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)
    original=upstream.SSBroyden(rtol=0.,atol=0.,search=guarded.search)
    loss=lambda z:.5*jnp.sum(jnp.array([1.,3.])*(z-jnp.array([.1,.3]))**2)
    z=jnp.array([1.,2.]);states=[]
    for solver in (guarded,original):
        state=ho.ssb_initial(solver,loss,z);advance=ho.ssb_step(solver,loss,lambda p:(p,p),eps)
        state,ev=advance(state)
        assert int(state['count'])==1 and int(state['status'])==0 and int(ev['attempts'])>=2
        states.append(state)
    for a,b in zip(jax.tree.leaves(states[0]),jax.tree.leaves(states[1])):
        np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-15)
    ho.save_state(tmp_path/'ssb.pkl',states[0],1)
    resumed,_=ho.load_state(tmp_path/'ssb.pkl',ho.ssb_initial(guarded,loss,z))
    advance=ho.ssb_step(guarded,loss,lambda p:(p,p),eps)
    for a,b in zip(jax.tree.leaves(advance(states[0])[0]),jax.tree.leaves(advance(resumed)[0])):
        np.testing.assert_array_equal(a,b)
    module=ho.ssb_module(source,1e-24)
    grad=jnp.array([-1e-10,-2e-10]);s=jnp.array([1e-10,1e-10]);new_grad=grad+jnp.array([2.,3.])*s
    matrices=[]
    for threshold in (eps,1e-24,1e-30):
        solver=ho.ssb_solver(source,threshold)
        info,update=solver.init_hessian(jnp.zeros(2),jnp.array(1.),grad)
        result,_=solver.update_hessian(jnp.zeros(2),s,info,module.FunctionInfo.EvalGrad(jnp.array(.9),new_grad),update,jnp.array(1.))
        matrices.append(np.asarray(result.hessian_inv.pytree))
    np.testing.assert_array_equal(matrices[0],np.eye(2))
    assert np.linalg.norm(matrices[1]-np.eye(2))>.1
    np.testing.assert_array_equal(matrices[1],matrices[2])
    assert np.linalg.eigvalsh(matrices[1]).min()>0


@pytest.mark.parametrize('optimizer',['gn','ssbroyden'])
def test_higher_runner_resume_and_trace(tmp_path,optimizer):
    import time,json
    from experiments.expD06_fixed_center_scales import joint_conditioning as jc
    source=ssb_source() if optimizer=='ssbroyden' else None
    config=jc.case(optimizer,'parameter_differences',n=128)
    jc.advance_higher(tmp_path,config,2,time.monotonic()+120,source,samples=1)
    jc.advance_higher(tmp_path,config,4,time.monotonic()+120,source,samples=1)
    folder=tmp_path/dt.case_key(config)
    status=json.loads((folder/'latest.json').read_text())
    assert status['completed_updates']==4 and status['status']=='continuing'
    trace=jc.read_trace(folder,4,optimizer)
    assert trace.shape==(4,len(jc.HIGHER_COLUMNS)) and np.all(np.isfinite(trace))
    with np.load(folder/'dense_latest.npz') as a:
        np.testing.assert_array_equal(a['step'],np.arange(4))
    assert (folder/'state_000000004.pkl').exists()


def test_ssb_nonfinite_and_unchanged_states_do_not_count():
    from experiments.expD06_fixed_center_scales import higher_order as ho
    solver=ho.ssb_solver(ssb_source())
    z=jnp.zeros(2)
    for loss,expected in ((lambda p:jnp.sum(p*jnp.nan),1),(lambda p:jnp.sum(p*p),3)):
        state=ho.ssb_initial(solver,loss,z)
        result,ev=ho.ssb_step(solver,loss,lambda p:(p,p))(state)
        assert int(result['status'])==expected and int(result['count'])==0
        np.testing.assert_array_equal(result['z'],z)
        assert not bool(ev['accepted'])


def test_detached_joint_decomposition():
    from experiments.expD06_fixed_center_scales import joint_analysis as analysis
    from experiments.expD06_fixed_center_scales import higher_order as ho
    from experiments.expD06_fixed_center_scales import diagnostics, ratio_analysis
    g=core.geometry(128);c,gamma=core.initial_physical(g,0,'xavier_a_reference')
    x,y,a,r,j=analysis.linearize(g,c,gamma,1)
    rng=np.random.default_rng(91);dc=rng.normal(size=c.size)*1e-5;dl=rng.normal(size=gamma.size)*1e-7
    for coord in ('parameter_scale','parameter_differences'):
        np.testing.assert_allclose(analysis.native_features(a,g,coord),a@ho.readout_map(g,coord),atol=3e-15)
    bounds,rb,_=diagnostics.band_residuals(r)
    assert (bounds==[64,65]).all(axis=1).any()
    np.testing.assert_allclose(np.sum(rb**2),r@r,atol=1e-14)
    np.testing.assert_allclose((rb@j).sum(axis=0),j.T@r,atol=2e-14)
    _,pieces,budget=ratio_analysis.update_budget(x,y,g.centers,g.h,c,gamma,dc,dl)
    np.testing.assert_allclose(budget['mse_change'][-1],budget['measured_mse_change'],atol=1e-14)
    assert budget['closure_max']<1e-14
    eps=1e-3
    predicted=(a@dc+j@dl)*np.sqrt(len(x))
    f=lambda e:diagnostics.prediction(x,g.centers,c+e*dc,gamma+e*dl/g.h)
    np.testing.assert_allclose((f(eps)-f(-eps))/(2*eps),predicted,atol=2e-12)


def test_ssb_accepted_step_matches_secant_scaling_formula():
    from experiments.expD06_fixed_center_scales import higher_order as ho
    source=ssb_source();z=jnp.array([1.,2.,3.]);target=jnp.full(3,.1);diag=jnp.geomspace(1.,30.,3)
    loss=lambda p:.5*jnp.sum(diag*(p-target)**2)
    solver=ho.ssb_solver(source,integration='accepted_step')
    result,evidence=ho.ssb_step(solver,loss,lambda p:(p,p))(ho.ssb_initial(solver,loss,z))
    assert int(result['count'])==1 and 0<float(evidence['step_size'])<.1
    s=np.asarray(result['z']-z);y=np.asarray(diag)*s;grad=np.asarray(diag*(z-target))
    rho=1/(s@y);h=(y@y)*rho;b=(s@s)*rho;a=b*h-1
    np.testing.assert_allclose(b,-float(evidence['step_size'])*(s@grad)*rho,rtol=1e-12)
    ck=np.sqrt(abs(a/(1+a)));rm=min(1.,h*(1-ck))
    theta=max((rm-1)/a,min(1/rm,(1-b)/b));sigma=1+theta*a
    rp=min(1.,1/b);sigma_power=abs(sigma)**(1/(1-len(z)))
    tau=min(rp*sigma_power,sigma) if theta<=0 else rp*min(sigma_power,1/theta)
    phi=(1-theta)/(1+a*theta);v=s*rho-y/(y@y)
    expected=(np.eye(len(z))-np.outer(y,y)/(y@y)+phi*(y@y)*np.outer(v,v))/tau+rho*np.outer(s,s)
    np.testing.assert_allclose(result['solver'].f_info.hessian_inv.pytree,expected,rtol=2e-12,atol=2e-14)
    assert np.linalg.eigvalsh(expected).min()>0
    original=ho.ssb_solver(source,integration='pinned')
    original_result,_=ho.ssb_step(original,loss,lambda p:(p,p))(ho.ssb_initial(original,loss,z))
    assert np.linalg.norm(np.asarray(original_result['solver'].f_info.hessian_inv.pytree)-expected)>1e-3


def test_metric_direction_audit_detects_non_descent():
    from experiments.expD06_fixed_center_scales.joint_analysis import inverse_hessian_direction
    g=np.array([1.,.5])
    for matrix,expected in ((np.diag([2.,3.]),-2.75),(np.diag([-2.,3.]),1.25)):
        result=inverse_hessian_direction(matrix,g)
        assert result['stored_metric_directional_derivative_fp64']==expected
        assert result['stored_metric_directional_derivative_mp80']==expected


def test_fixed_horizon_analysis_preserves_case_identity(tmp_path,monkeypatch):
    from experiments.expD06_fixed_center_scales import joint_analysis as analysis, joint_conditioning as jc
    config=jc.case('gn','parameter_scale',n=128);key=dt.case_key(config)
    folder=tmp_path/key;folder.mkdir();g=core.geometry(128)
    c,gamma=core.initial_physical(g,0,'xavier_a_reference')
    run.write_json(folder/'latest.json',dict(completed_updates=20001,step=20001,status='continuing',train_mse=1.,validation_mse=1.))
    run.save_arrays(folder/'checkpoint_000020000.npz',c=c,gamma=gamma,train_mse=.5,validation_mse=.5)
    trace=np.zeros((20000,len(jc.HIGHER_COLUMNS)));trace[:,0]=.25
    trace[-1,jc.HIGHER_COLUMNS.index('function_evaluations')]=40000
    monkeypatch.setattr(jc,'read_trace',lambda *args:trace)
    monkeypatch.setattr(analysis,'probe',lambda *args:({},{}))
    monkeypatch.setattr(analysis,'dense_audit',lambda *args:{})
    monkeypatch.setattr(analysis,'precision_check',lambda *args:{})
    result=analysis.analyze_case((tmp_path,tmp_path/'analysis',config,True))
    assert result['key']==key and result['end']==20000
    assert result['status']['function_evaluations']==40000
    assert result['source_latest']['completed_updates']==20001
    assert result['late_mean_mse']==.5


def test_frozen_decay_matches_explicit_gradient_descent():
    from experiments.expD06_fixed_center_scales.joint_mechanism_probes import frozen_decay
    rng=np.random.default_rng(14);u,_=np.linalg.qr(rng.normal(size=(7,4)))
    s=np.array([1.,.2,.01,.0001]);residual=rng.normal(size=7)
    for retained in (2,4):
        basis=u[:,:retained];b=basis*s[:retained];modal=basis.T@residual
        perpendicular=residual-basis@modal;steps=np.arange(501);eta=.7
        predicted=frozen_decay(s[:retained],modal,perpendicular,steps,eta)
        live=residual.copy();measured=[]
        for _ in steps:
            measured.append(live@live);live-=eta*b@(b.T@live)
        np.testing.assert_allclose(predicted,measured,rtol=2e-13,atol=2e-14)


def test_paired_gn_restart_preserves_origin_and_damping(tmp_path):
    import json,time
    from experiments.expD06_fixed_center_scales import joint_conditioning as jc
    original=jc.case('gn','parameter_scale',n=128)
    jc.advance_higher(tmp_path,original,2,time.monotonic()+120,None,samples=1)
    source=tmp_path/dt.case_key(original)
    origin_leaves,_=run.load_state(source/'state_000000002.pkl')
    branches=[]
    for coord in jc.MAPS:
        config=jc.case('gn',coord,n=128,restart=dict(source_key=source.name,step=2))
        jc.advance_higher(tmp_path,config,2,time.monotonic()+120,None,samples=1)
        folder=tmp_path/dt.case_key(config);branches.append(folder)
        with np.load(folder/'checkpoint_000000000.npz') as branch,np.load(source/'checkpoint_000000002.npz') as start:
            np.testing.assert_allclose(branch['c'],start['c'],atol=3e-16,rtol=1e-14)
            np.testing.assert_array_equal(branch['gamma'],start['gamma'])
        leaves,_=run.load_state(folder/'state_000000000.pkl')
        # Dictionary leaves are ordered count, damping, evaluation counters, z.
        np.testing.assert_array_equal(leaves[1],origin_leaves[1])
        assert json.loads((folder/'latest.json').read_text())['completed_updates']==2
        assert json.loads((folder/'case.json').read_text())['initialization']=='checkpoint_restart'
    jc.advance_higher(tmp_path,original,4,time.monotonic()+120,None,samples=1)
    with np.load(source/'checkpoint_000000004.npz') as uninterrupted,np.load(branches[0]/'checkpoint_000000002.npz') as restarted:
        np.testing.assert_array_equal(restarted['native_parameters'],uninterrupted['native_parameters'])
