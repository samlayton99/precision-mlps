"""Independent checks of the discrete dynamics and moment hierarchy."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD34_readout_race import core, references, targets


def example(name="sine"):
    z, d = targets.initial(64, 12, 3)
    return jnp.asarray(z), jnp.asarray(d), {k: jnp.asarray(v) for k,v in targets.data(64,name).items()}


@pytest.mark.parametrize("degree", [0,1,3,5,7])
def test_gradients_against_sample_autodiff(degree):
    z,d,data=example()
    pred=lambda z,d: (jnp.tanh(data['x'][:,None]*z[0]+z[1]) @ z[2]+d if degree==0
                      else core.sample_polynomial(z,d,data['x'],degree))
    f=lambda z,d:.5*jnp.mean((pred(z,d)-data['y'])**2)
    expected=jax.grad(f,argnums=(0,1))(z,d)
    loss,_,g,gd,_=core.field(z,d,data,degree)
    np.testing.assert_allclose(loss,f(z,d),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(g,expected[0],rtol=1e-11,atol=1e-14)
    np.testing.assert_allclose(gd,expected[1],rtol=1e-11,atol=1e-14)


@pytest.mark.parametrize("kappa", [1e-4,1.,100.])
def test_exact_affine_discrete_closure(kappa):
    z,d,data=example()
    z0=np.asarray(z);G=z0@z0.T;U=np.eye(3);dd=float(d)
    beta=np.asarray(data['ym'][:2])/np.array([1.,float(data['sigma'])])
    step=jax.jit(lambda z,d:core.update(z,d,data,1,.002,kappa))
    for _ in range(100):
        G,dd,U=references.affine_step(G,dd,U,beta,float(data['sigma']),.002,kappa)
        z,d=step(z,d)
    np.testing.assert_allclose(z,U@z0,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(G,np.asarray(z)@np.asarray(z).T,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(d,dd,atol=1e-14)


def test_matched_targets_and_visibility_orders():
    data=[targets.data(1024,f'moment{p}') for p in (3,5,9)]
    for d in data:
        np.testing.assert_allclose(d['sy'],1.,atol=2e-15)
        np.testing.assert_allclose(d['ym'][:2]/[1,d['sigma']],[.3,.4],atol=2e-15)
        phi=np.polynomial.legendre.legvander(d['x'],9)@d['mapping']
        np.testing.assert_allclose(phi.T@phi/1024,np.eye(10),atol=4e-15)
    np.testing.assert_allclose(data[1]['ym'][:4],data[2]['ym'][:4],atol=1e-15)
    assert abs(data[0]['ym'][3]-data[2]['ym'][3])>.01
    z,b,_=example()
    for degree,pair in ((1,(0,2)),(3,(1,2))):
        fields=[references.polynomial(z,b,degree,jnp.asarray(data[i]['Q']),
                    jnp.asarray(data[i]['ym']),data[i]['sy']) for i in pair]
        np.testing.assert_allclose(fields[0][2],fields[1][2],atol=1e-15)


def test_upstream_initialization_and_simultaneous_torch_steps():
    from experiments.expD28_loss_gradient_decomposition import run as previous
    import torch
    torch.set_default_dtype(torch.float64)
    cfg=previous.config()|dict(resolution=64,halo=12,seed=3,n_train=64,steps=25,diagnostic_snapshots=30)
    old=previous.train('sine','xavier',cfg)
    z,d,data=example()
    step=jax.jit(lambda z,d:core.update(z,d,data,0,.002,1.))
    for i in range(26):
        np.testing.assert_allclose(z,np.stack((old['a'][i],old['b'][i],old['v'][i,:-1])),atol=2e-14,rtol=1e-12)
        np.testing.assert_allclose(d,old['v'][i,-1],atol=2e-14)
        z,d=step(z,d)


def test_pointwise_moment_bound_with_nonlinear_bias():
    z,d,data=example()
    z=z.at[1].add(.8)
    loss,moments,g,_,_=core.field(z,d,data,0)
    R=float(jnp.sqrt(2*loss))
    for order in (2,4,6,8):
        approx,rho=references.slope_interval(z,np.asarray(moments),R,order)
        a,b,c=np.asarray(z)
        bound=(1/np.cos(rho)**2/rho**order * np.sqrt(np.mean(np.asarray(data['x'])**(2*order+2)))
               *np.linalg.norm(c*a**order))
        assert np.linalg.norm(np.asarray(g[0])/R-approx)<=bound+2e-15


@pytest.mark.parametrize('degree',[0,3,7])
def test_batched_chunk_resume_and_signed_accounting(degree):
    from experiments.expD34_readout_race import run
    z,d,data=example()
    state=run.initialize(z,float(d),2)
    kappa=jnp.array([.01,100.])
    args=[jnp.stack([data[key]]*2) for key in ('y','ym','sy')]
    moments=core.field(z,d,data,degree)[1]
    m0=jnp.full(2,jnp.linalg.norm(moments[:2]/jnp.array([1.,data['sigma']])))
    fn=run.chunk(64,64,degree,blocks=2,stride=20)
    end,(trace,snap)=fn(state,*args,kappa,m0,.002,0)
    short=run.chunk(64,64,degree,blocks=1,stride=20)
    half,_=short(state,*args,kappa,m0,.002,0)
    # Round-trip all state, including event and failure metadata.
    restored=jax.tree.map(lambda x:jnp.asarray(np.array(x)),half)
    resumed,_=short(restored,*args,kappa,m0,.002,20)
    for key in end:
        np.testing.assert_array_equal(end[key],resumed[key])
    for i,k in enumerate(kappa):
        zz,dd=z,d
        advance=jax.jit(lambda zz,dd:core.update(zz,dd,data,degree,.002,k))
        for _ in range(40):
            zz,dd=advance(zz,dd)
        np.testing.assert_allclose(end['z'][i],zz,atol=2e-14,rtol=1e-11)
        np.testing.assert_allclose(end['d'][i],dd,atol=2e-14)
    flat=np.asarray(trace).reshape(2,40,len(run.TRACE))
    np.testing.assert_allclose(flat[:,:,18],.002*(flat[:,:,16]+flat[:,:,17])+flat[:,:,19],atol=1e-19)
    np.testing.assert_allclose(flat[:,:,18].sum(axis=1),np.mean(abs(np.asarray(end['z'])[:,0]),axis=1)-np.mean(abs(np.asarray(z[0]))),atol=1e-16)
    assert np.any(np.asarray(end['z'])[:,1]!=np.asarray(z)[1])


def test_campaign_manifest_and_failed_member_isolation():
    from pathlib import Path
    from experiments.expD34_readout_race import run
    groups=run.bundles('core',Path('/tmp/not-created'))
    assert len(groups)==15
    assert sum(len(c['targets'])*len(c['ratios']) for _,c in groups)==525
    z,d,data=example();state=run.initialize(z,float(d),2)
    args=[jnp.stack([data[key]]*2) for key in ('y','ym','sy')]
    end,(trace,_)=run.chunk(64,64,0,blocks=1,stride=20)(state,*args,jnp.array([1.,1e308]),jnp.ones(2),.002,0)
    assert int(end['failed'][0])==0 and int(end['failed'][1])>0
    assert np.all(np.isfinite(np.asarray(trace[0])))


def test_disk_resume_and_manifest_identity(tmp_path):
    import time
    from experiments.expD34_readout_race import run
    cfg=run.specification(64,12,3,64,.002,('sine',),(1.,))
    assert run.advance(tmp_path,cfg,0,40,time.monotonic()+60)
    assert run.advance(tmp_path,cfg,0,80,time.monotonic()+60)
    with np.load(tmp_path/'p0/state.npz') as saved:
        assert saved['step']==80
        z,d,data=example()
        step=jax.jit(lambda z,d:core.update(z,d,data,0,.002,1.))
        for _ in range(80): z,d=step(z,d)
        np.testing.assert_allclose(saved['z'][0],z,atol=1e-14)
    with pytest.raises(ValueError,match='Incompatible resume'):
        run.prepare(tmp_path,cfg|dict(eta=.003))
    from experiments.expD34_readout_race import analyze
    output=tmp_path/'analysis';output.mkdir()
    rows,_,probes,_=analyze.analyze_bundle(tmp_path,output,80)
    assert len(rows)==1 and rows[0]['complete'] and rows[0]['completed_updates']==80
    assert rows[0]['heldout_mse']>0
    assert len(probes)>20
    sparse_steps,sparse=analyze.load_snapshots(tmp_path/'p0',80,sparse=True)
    assert np.all(sparse_steps<80) and sparse['z'].shape[1]==len(sparse_steps)
    for row in probes:
        for order in (2,4,6,8):
            assert row[f'pointwise_p{order}_error']<=row[f'pointwise_p{order}_bound']+1e-14


def test_coarse_velocity_matches_finite_difference():
    from experiments.expD34_readout_race import analyze
    z,d,data=example('moment5');loss,mom,g,gd,_=core.field(z,d,data,0)
    kappa=37.
    probe=analyze.sample_probe(np.asarray(z),float(d),np.asarray(g),float(gd),np.asarray(data['x']),np.asarray(data['y']),kappa,0)
    eta=1e-7
    zn,dn=core.update(z,d,data,0,eta,kappa)
    next_mom=core.field(zn,dn,data,0)[1]
    change=np.asarray((next_mom[:2]-mom[:2])/jnp.array([1.,data['sigma']])/eta)
    np.testing.assert_allclose(change,probe['Vv']+probe['Vq'],rtol=2e-6,atol=1e-8)


@pytest.mark.parametrize('degree',[0,1,3,5,7])
def test_direct_sample_probe_gradient(degree):
    from experiments.expD34_readout_race import analyze
    z,d,data=example('moment3');loss,mom,g,gd,_=core.field(z,d,data,degree)
    probe=analyze.sample_probe(np.asarray(z),float(d),np.asarray(g),float(gd),np.asarray(data['x']),np.asarray(data['y']),100.,degree)
    np.testing.assert_allclose(probe['gradient'],g,atol=5e-16)
    np.testing.assert_allclose(probe['sample_half_mse'],loss,atol=5e-16)


def test_offline_hessian_includes_residual_curvature():
    from experiments.expD34_readout_race.curvature_check import hessian
    z,d,data=example('moment3');z=z[:,:6];theta=jnp.r_[z.reshape(-1),d]
    def loss(theta):
        a,b,c=theta[:-1].reshape(3,6)
        e=jnp.tanh(data['x'][:,None]*a+b)@c+theta[-1]-data['y']
        return .5*jnp.mean(e**2)
    expected=jax.hessian(loss)(theta)
    actual=hessian(np.asarray(z),float(d),np.asarray(data['x']),np.asarray(data['y']))
    np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-13)
