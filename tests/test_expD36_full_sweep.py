import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
import time
from scipy.linalg import svd, lstsq

from experiments.expD36_frozen_gamma_probe import core, full_core as f, full_kernels as k


def test_full_maps_roundtrip_gradient_and_anchor():
    g = core.geometry(64)
    x = np.linspace(-1, 1, 129)
    a = core.design(x, g.centers, 4)
    rng = np.random.default_rng(10)
    c = rng.normal(size=(g.width+1, 2))
    y = rng.normal(size=(len(x), 2))
    for name in f.config()['maps']+f.config()['coordinate_controls']:
        scale, neighbor = f.map_spec(g, name)
        r = f.map_matrix(g, name)
        theta = f.encode(c, scale, neighbor)
        j = f.design_from_physical(a, scale, neighbor)
        np.testing.assert_allclose(f.decode(theta, scale, neighbor), c, atol=2e-14)
        np.testing.assert_allclose(j, a@r, atol=2e-15)
        np.testing.assert_allclose(j@theta, a@c, atol=2e-13)
        loss = lambda z: .5*jnp.sum((jnp.asarray(a)@f.decode(z,jnp.asarray(scale),neighbor,jnp)-y)**2)
        np.testing.assert_allclose(jax.grad(loss)(jnp.asarray(theta)), j.T@(j@theta-y), rtol=1e-10, atol=1e-12)
        qh=core.transform(core.polynomial_transform(x,8),j)[9:]
        assert np.linalg.norm(qh,2)**2 <= np.exp(f.envelopes(4,np.array([8]),r)['used'][0])


def test_full_batched_adam_and_resume():
    rng=np.random.default_rng(81)
    j=rng.normal(size=(2,31,6))/np.sqrt(31)
    y=rng.normal(size=(2,31,3))/np.sqrt(31)
    scale=rng.uniform(.5,2,size=(2,6))
    theta=rng.normal(size=(2,6,3))*.1
    rates=np.array([[.001,.01,.02],[.002,.005,.01]])
    eps=np.array([[1e-8,1e-12,1e-8],[1e-12,1e-8,1e-12]])
    state=k.initialize(theta,2)
    kernel=k.make_chunk('adam',True,steps=30)
    out,trace=kernel(state,j,y,scale,rates,eps,np.array([.1,.01]))
    for b in range(2):
        for t in range(3):
            p=jnp.asarray(theta[b,:,t]); tx=optax.adam(rates[b,t],eps=eps[b,t]); s=tx.init(p)
            for _ in range(30):
                grad=j[b].T@(j[b]@p-y[b,:,t]); delta,s=tx.update(grad,s,p); p=optax.apply_updates(p,delta)
            np.testing.assert_allclose(out['theta'][b,:,t],p,rtol=1e-11,atol=1e-13)
    twice,_=kernel(out,j,y,scale,rates,eps,np.array([.1,.01]))
    whole,_=k.make_chunk('adam',True,60)(state,j,y,scale,rates,eps,np.array([.1,.01]))
    np.testing.assert_array_equal(twice['theta'],whole['theta'])
    np.testing.assert_allclose(trace[0,:,:,0],np.linalg.norm(j@theta-y,axis=1)/np.linalg.norm(y,axis=1))
    np.testing.assert_allclose(k.multiplier(jnp.array([1,20000,50000,200000])),[1,1,.001,.001])


def test_general_gd_shift_bounds_and_zero_target():
    rng=np.random.default_rng(51)
    j=rng.normal(size=(1,25,5))/5; y=rng.normal(size=(1,25,2))/5
    theta=rng.normal(size=(1,5,2)); eta=.1
    original,_=k.make_chunk('gd',False,40)(k.initialize(theta,1),j,y,np.ones((1,5)),
        np.full((1,2),eta),np.zeros((1,2)),np.array([.01]))
    shifted,_=k.make_chunk('gd',False,40)(k.initialize(np.zeros_like(theta),1),j,y-j@theta,np.ones((1,5)),
        np.full((1,2),eta),np.zeros((1,2)),np.array([.01]))
    np.testing.assert_allclose(original['theta'],shifted['theta']+theta,atol=2e-14)
    for chi in [.25,.5,.9]:
        b=f.bound(np.array([1.]),np.log(np.array([.04])),.01,.04,chi)
        assert b['bound']==np.ceil(np.log(.01)/np.log1p(-chi))
    assert f.bound(np.array([1.]),np.array([0.]),1.,1.)['bound']==0
    zero, trace=k.make_chunk('gd',False,2)(k.initialize(np.zeros((1,5,1)),1),j,np.zeros((1,25,1)),
        np.ones((1,5)),np.ones((1,1))*.1,np.zeros((1,1)),np.array([.01]))
    assert np.isfinite(trace).all() and zero['hits'][0,0,0]==0


def test_damping_and_initialization_pairing():
    rng=np.random.default_rng(5); j=rng.normal(size=(20,7)); q=rng.normal(size=20); q/=np.linalg.norm(q)
    u,s,vh=svd(j,full_matrices=False)
    for penalty in [.1,1,100]:
        step,res,lower=f.damping(j,u,s,vh,q,penalty)
        other=lstsq(np.vstack([j,np.sqrt(penalty)*np.eye(7)]),np.r_[-q,np.zeros(7)])[0]
        np.testing.assert_allclose(step,other,atol=1e-13)
        assert np.linalg.norm(res)>=lower-1e-13
    g=core.geometry(64)
    a=f.initial_physical(g,'sqrt_alpha_xavier',0); b=f.initial_physical(g,'alpha_xavier',0)
    np.testing.assert_allclose(b[1:],a[1:]*np.sqrt(g.alpha[1:]))


def test_joint_matches_optax_and_finite_difference():
    # A nonzero mean avoids an exactly zero bias gradient whose roundoff is
    # strongly amplified by Adam's epsilon during this implementation check.
    x=jnp.linspace(-1,1,33); y=.2+jnp.sin(2*jnp.pi*x)
    initial=k.joint_initial(7,[0,1]); state,values=k.make_joint_chunk(10)(initial,x,y)
    tx=optax.adam(.001,eps=1e-8); p=initial['params']; s=tx.init(p)
    objective=lambda p: .5*jnp.sum(jax.vmap(lambda q: jnp.mean((k.joint_predict(q,x)-y)**2))(p))
    delta=jnp.zeros_like(p['hidden']).at[0,0,2].set(1e-5)
    numerical=(objective(dict(p,hidden=p['hidden']+delta))-objective(dict(p,hidden=p['hidden']-delta)))/2e-5
    np.testing.assert_allclose(numerical,jax.grad(objective)(p)['hidden'][0,0,2],rtol=1e-7,atol=1e-10)
    for _ in range(10):
        delta,s=tx.update(jax.grad(objective)(p),s,p); p=optax.apply_updates(p,delta)
    for actual,expected in zip(jax.tree.leaves(state['params']),jax.tree.leaves(p)):
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
    assert np.isfinite(values).all()


def test_full_screen_selection_resume_and_controls(tmp_path):
    from experiments.expD36_frozen_gamma_probe import full_screen as screen, full_train as train
    cfg=dict(f.config(),n=64,gammas=[4],robust_gammas=[4],widths=[64],maps=['raw'],
        coordinate_controls=[],k_max=8,k_max_extension=8,n_eval=128,n_validation=65,
        chunk_size=5,training_steps=20,adam_pilot_steps=10,checkpoints=[0,5,10,20],
        validation_steps=[5,10],adam_rates=[.001,.01],seeds=[0],polynomial_degrees=[2,4])
    screen.run(tmp_path,cfg,60)
    deadline=time.monotonic()+60
    train.primary(tmp_path,cfg,64,'raw',[4],cfg['targets'],deadline)
    selection=json.loads((tmp_path/'training/N64_raw_selection.json').read_text())
    pilot=dict(np.load(tmp_path/'training/N64_raw_adam_pilot/state.npz'))
    continued=dict(np.load(tmp_path/'training/N64_raw_adam_continue/checkpoint_000010.npz'))
    indices=np.array(selection['indices'])
    for key in ['theta','mu','nu']:
        np.testing.assert_array_equal(continued[key],np.take_along_axis(pilot[key],indices[:,None,:],axis=2))
    assert continued['count']==10
    arrays,banks=train.load_banks(tmp_path,64,'raw',[4]); columns=train.plain_columns(cfg,cfg['targets'])
    initial=np.random.default_rng(77).normal(size=(1,banks[0]['J'].shape[1],5))*.01
    rates=np.full((1,5),.5/banks[0]['meta']['L']); eps=np.zeros_like(rates)
    train.run_batch(tmp_path,cfg,64,'raw',[4],'resumed','gd',columns,rates,eps,10,deadline,initial=initial)
    resumed,_=train.run_batch(tmp_path,cfg,64,'raw',[4],'resumed','gd',columns,rates,eps,20,deadline,initial=initial)
    whole,_=train.run_batch(tmp_path,cfg,64,'raw',[4],'whole','gd',columns,rates,eps,20,deadline,initial=initial)
    np.testing.assert_array_equal(resumed['theta'],whole['theta'])
    for stage in ['rate','polynomial']:
        train.controls(tmp_path,cfg,'raw',stage,deadline)
    probe=f.polynomial_probe(arrays['x_train'],4,len(arrays['x_train']))
    np.testing.assert_allclose(probe,core.discrete_polynomials(arrays['x_train'],4)[:,4],atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(probe),1,atol=1e-14)
    from experiments.expD36_frozen_gamma_probe import full_diagnostics as diagnostics
    rows=diagnostics.case_certificates(tmp_path,tmp_path/'training/N64_raw_gd',cfg,deadline)
    assert len(rows)==len(cfg['targets'])*len(cfg['tolerances'])*8
    np.testing.assert_allclose([row['epsilon_residual'] for row in rows],
                               [row['epsilon_target'] for row in rows],rtol=1e-14)
    diagnostics.hitting_audit(tmp_path,cfg,deadline)
    audit=np.load(tmp_path/'training/N64_raw_adam_continue/hitting_audit.npz')
    assert audit['first'].shape==np.load(tmp_path/'training/N64_raw_adam_continue/state.npz')['hits'].shape


def test_residual_access_and_effective_generator():
    from experiments.expD36_frozen_gamma_probe.full_diagnostics import tail_measurements
    rng=np.random.default_rng(8); j=rng.normal(size=(31,7)); residual=rng.normal(size=31)
    u,s,_=svd(j,full_matrices=False); eta=.5/s[0]**2
    delta,mu,effective,gradient=tail_measurements(j,residual,u,s,eta,8)
    for degree in range(9):
        q=residual.copy(); q[:degree+1]=0; q/=np.linalg.norm(q)
        np.testing.assert_allclose(gradient[degree],j.T@q,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(mu[degree],np.linalg.norm(j.T@q)**2,rtol=1e-13)
        np.testing.assert_allclose(effective[degree],np.sum((u.T@q)**2*(-np.log1p(-eta*s*s))),rtol=1e-13)
        assert effective[degree]<=np.log(2)*mu[degree]/s[0]**2+1e-14
