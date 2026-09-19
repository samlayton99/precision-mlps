import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.expD06_fixed_center_scales import core, difference_training as dt, parameter_scale as campaign, run


def close_trees(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=3e-12, atol=3e-14)


@pytest.mark.parametrize("optimizer", campaign.OPTIMIZERS)
def test_paired_physical_initialization(optimizer):
    g=core.geometry(128)
    a,b=[dt.initial(g,0,c,optimizer) for c in campaign.MAPS]
    np.testing.assert_allclose(dt.decode(a["z"],g,"scaled"),dt.decode(b["z"],g,"parameter_scale"),atol=1e-16)
    np.testing.assert_array_equal(a["lam"],b["lam"])


@pytest.mark.parametrize("optimizer",campaign.OPTIMIZERS)
@pytest.mark.parametrize("coord",campaign.MAPS)
def test_physical_multistep_updates_and_resume(optimizer,coord,tmp_path):
    g=core.geometry(128);eta=1e-5;steps=6
    initial=dt.initial(g,0,coord,optimizer)
    end,(trace,dense)=dt.chunk(128,coord,steps,True,1,False,optimizer)(initial,eta,0)
    scale=g.d if coord=="scaled" else g.alpha
    c=np.asarray(dt.decode(initial["z"],g,coord));lam=np.asarray(initial["lam"])
    x=jnp.linspace(-1,1,129);y=core.target(x,"sine")
    derivative=jax.grad(dt.physical_loss,argnums=(0,1))
    mc=np.zeros_like(c);vc=mc.copy();ml=np.zeros_like(lam);vl=ml.copy()
    for k in range(steps):
        gc,gl=[np.asarray(v) for v in derivative(jnp.asarray(c),jnp.asarray(lam),x,y,g)]
        if optimizer=="gd":
            dc=-eta*scale**2*gc;dl=-eta*gl
        else:
            mc=.9*mc+.1*gc;vc=.999*vc+.001*gc**2
            ml=.9*ml+.1*gl;vl=.999*vl+.001*gl**2
            dc=-eta*scale*(mc/(1-.9**(k+1)))/(np.sqrt(vc/(1-.999**(k+1)))+1e-8/g.d)
            dl=-eta*(ml/(1-.9**(k+1)))/(np.sqrt(vl/(1-.999**(k+1)))+1e-8)
        np.testing.assert_allclose(dense["delta_c"][k],dc,atol=2e-16,rtol=1e-10)
        np.testing.assert_allclose(dense["delta_lambda"][k],dl,atol=2e-16,rtol=1e-10)
        c=c+dc;lam=lam+dl
    np.testing.assert_allclose(dt.decode(end["z"],g,coord),c,atol=2e-15)
    np.testing.assert_allclose(end["lam"],lam,atol=2e-15)
    middle,_=dt.chunk(128,coord,3,False,1,False,optimizer)(initial,eta,0)
    run.save_state(tmp_path/"resume.pkl",middle,3)
    loaded,start=run.load_state(tmp_path/"resume.pkl")
    resumed,_=dt.chunk(128,coord,3,False,1,False,optimizer)(loaded,eta,start)
    close_trees(resumed,end)


@pytest.mark.parametrize("optimizer",campaign.OPTIMIZERS)
def test_failure_isolation(optimizer):
    initial=dt.initial(core.geometry(128),0,"parameter_scale",optimizer)
    single,_=dt.chunk(128,"parameter_scale",4,False,1,False,optimizer)(initial,1e-5,0)
    batch,_=dt.chunk(128,"parameter_scale",4,False,1,True,optimizer)(run.stack_states([initial]*2),jnp.array([1e-5,1e308]),0)
    close_trees(single,run.unstack_state(batch,0))
    assert int(batch["failed"][1])>0


def test_parameter_scale_finite_difference_and_manifest():
    g=core.geometry(128);state=dt.initial(g,0,"parameter_scale")
    x=jnp.linspace(-1,1,129);y=core.target(x,"sine")
    loss=lambda z,lam:dt.physical_loss(dt.decode(z,g,"parameter_scale"),lam,x,y,g)
    grads=jax.grad(loss,argnums=(0,1))(state["z"],state["lam"])
    for block,key in enumerate(("z","lam")):
        direction=jnp.asarray(np.random.default_rng(4+block).normal(size=state[key].shape))
        direction/=jnp.linalg.norm(direction);eps=1e-6
        args=[state["z"],state["lam"]]
        args[block]=state[key]+eps*direction;plus=loss(*args)
        args[block]=state[key]-eps*direction;minus=loss(*args)
        np.testing.assert_allclose((plus-minus)/(2*eps),grads[block]@direction,rtol=1e-5,atol=1e-8)
    assert len(campaign.pilot())==40
    selected=[campaign.case(o,c,1e-3 if c=="scaled" else 3e-3) for o in campaign.OPTIMIZERS for c in campaign.MAPS]
    confirmation,continuation=campaign.followups(selected)
    assert len(confirmation)==24 and len(continuation)==16
    assert len({dt.case_key(c) for c in campaign.pilot()})==40


def test_saved_campaign_records_early_dense_and_adam_moments(tmp_path):
    c=campaign.case("adam","parameter_scale",1e-5,n=128)
    dt.advance_group(tmp_path,[c],4,samples_per_cell=1)
    dt.advance_group(tmp_path,[c],8,samples_per_cell=1)
    folder=tmp_path/dt.case_key(c)
    state,step=run.load_state(folder/"state_000000008.pkl")
    assert step==8 and int(state["opt"][0].count)==8
    with np.load(folder/"checkpoint_000000008.npz") as cp:
        assert "adam_readout_sqrt_v_over_epsilon" in cp
    indices=np.concatenate([np.load(p)["step"] for p in sorted(folder.glob("dense_*.npz"))])
    np.testing.assert_array_equal(indices,np.arange(8))


@pytest.mark.parametrize("optimizer",campaign.OPTIMIZERS)
def test_common_projection_and_actual_next_update(optimizer):
    from experiments.expD06_fixed_center_scales import parameter_scale_analysis as analysis
    g=core.geometry(128);records=[]
    for coord in campaign.MAPS:
        case=campaign.case(optimizer,coord,1e-5,n=128)
        state=dt.initial(g,0,coord,optimizer)
        cp={k:np.asarray(v[0]) for k,v in dt.evaluator(128,coord,1,optimizer)(run.stack_states([state])).items()}
        _,(_,dense)=dt.chunk(128,coord,1,True,1,False,optimizer)(state,case["eta"],0)
        record,arrays,_=analysis.probe(g,cp,case,samples=1)
        np.testing.assert_allclose(arrays["next_delta_c"],dense["delta_c"][0],atol=2e-15)
        np.testing.assert_allclose(arrays["next_delta_lambda"],dense["delta_lambda"][0],atol=2e-15)
        assert record["fourier_closure"]<1e-12
        assert record["update_budget"]["closure_max"]<1e-12
        np.testing.assert_allclose(arrays["band_gradient_lambda"].sum(axis=0),arrays["gradient_lambda"],atol=1e-12)
        records.append(record)
    assert records[0]["reference_retained_rank"]==records[1]["reference_retained_rank"]
    np.testing.assert_allclose([r["mse"] for r in records[0]["refits"]],[r["mse"] for r in records[1]["refits"]],atol=1e-12)
