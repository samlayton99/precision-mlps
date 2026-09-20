import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD35_optimization_exploration import core
from experiments.expD06_fixed_center_scales import core as old, difference_training as previous


def test_stable_neighbor_evaluation_and_saturated_derivative():
    from experiments.expD35_optimization_exploration import stable,run
    c=run.case(n=64,coordinates='neighbor',slope_initialization='lambda_xavier')
    state=core.initialize(c);g=old.geometry(64);x=jnp.linspace(-1,1,257)
    w,gamma=core.physical(state['z'],g,'neighbor')
    expected=w[0]+old.tanh((x[:,None]-g.centers)*gamma)@w[1:]
    np.testing.assert_allclose(stable.predict(state['z'],g,x),expected,atol=3e-15,rtol=3e-14)
    y=core.target(x,'mixed')
    derivative=jax.grad(lambda z:.5*jnp.mean((stable.predict(z,g,x)-y)**2))(state['z'])
    np.testing.assert_allclose(derivative,core.field(state['z'],x,y,g,'neighbor')[1],atol=3e-14,rtol=3e-12)
    assert float(stable.tanh_difference(jnp.array(25.),jnp.array(24.)))>0
    assert float(jnp.tanh(25.)-jnp.tanh(24.))==0
    assert float(jax.grad(lambda a:stable.tanh_difference(a,jnp.array(0.)))(0.))==1.


def test_paired_warm_recipes_resolve_parent_hyperparameters(tmp_path):
    from experiments.expD35_optimization_exploration import design,run
    rows=[]
    for seed in (0,1):
        parent=run.case(seed=seed);folder=tmp_path/run.key(parent);folder.mkdir()
        run.write_json(folder/'case.json',parent)
        child=run.case(seed=seed,origin=dict(checkpoint=str(folder/'checkpoint_000100000.npz'),sha256=str(seed),carry_optimizer=True))
        rows.append(dict(config=child,score=1.+seed,step=20000))
    paired=design.paired_recipes(rows,root=tmp_path)
    assert len(paired)==1 and paired[0]['score']==1.5
    parent['eta']*=3;run.write_json(folder/'case.json',parent)
    assert design.paired_recipes(rows,root=tmp_path)==[]


def test_archive_sequence_survives_verified_offload(tmp_path):
    from experiments.expD35_optimization_exploration import history,run
    run.write_json(tmp_path/'offloaded_archives.json',{'history_0007.zip':'retained elsewhere'})
    run.save(tmp_path/'snapshot_000020000.npz',step=20000,z=np.zeros(2))
    history.consolidate(tmp_path)
    assert (tmp_path/'history_0008.zip').exists()
    assert not (tmp_path/'history_0000.zip').exists()


def test_dense_samples_cover_cycle_phases():
    from experiments.expD35_optimization_exploration import run
    indices=run.sample_indices(2048,0)
    np.testing.assert_array_equal(indices//16,np.arange(128))
    np.testing.assert_array_equal(np.bincount(indices%16),np.full(16,8))
    assert run.sample_indices(3,0)[-1]<3


def test_schedule_handoff_preserves_filter_and_optimizer_state(tmp_path):
    from experiments.expD35_optimization_exploration import run
    import hashlib
    parent=run.case(n=64,ema_strength=2.);folder,state,_=run.prepare(tmp_path,parent)
    state.update(ema=jnp.ones_like(state['ema'])*3,post_ema=jnp.ones_like(state['post_ema'])*4,
                 age=jnp.ones_like(state['age'])*100)
    p=folder/'checkpoint_000100000.npz';run.save(p,**state,step=100000)
    child=dict(parent,schedule='decay',origin=dict(checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reset_filter=False))
    _,actual,_=run.prepare(tmp_path,child)
    for name in state:np.testing.assert_array_equal(actual[name],state[name])


def case(**extra):
    return dict(n=64, seed=0, coordinates='individual', optimizer='gd', eta=1e-4, **extra)


@pytest.mark.parametrize('coordinates', core.COORDINATES)
def test_physical_pairing_and_gradient(coordinates):
    c = case(); c['coordinates'] = coordinates
    g = old.geometry(c['n']); state = core.initialize(c)
    w, gamma = old.initial_physical(g, 0, 'xavier_a_reference')
    wr, gr = core.physical(state['z'], g, coordinates)
    np.testing.assert_allclose(wr, w, atol=2e-15, rtol=2e-14)
    np.testing.assert_array_equal(gr, gamma)
    x = jnp.linspace(-1, 1, 129); y = core.target(x, 'mixed')
    analytic = core.field(state['z'], x, y, g, coordinates)[1]
    def loss(z):
        w, ga = core.physical(z, g, coordinates)
        return .5*jnp.mean((w[0]+old.tanh((x[:,None]-g.centers)*ga)@w[1:]-y)**2)
    np.testing.assert_allclose(analytic, jax.grad(loss)(state['z']), rtol=3e-13, atol=2e-14)


@pytest.mark.parametrize('optimizer', ['gd','adam'])
def test_unmodified_matches_existing_training(optimizer):
    c = case(); c['optimizer'] = optimizer
    state = core.initialize(c); hp = core.hyperparameters(c)
    batch = lambda v: jax.tree.map(lambda a:a[None], v)
    actual, _ = core.chunk(64, 'individual', optimizer, 'sine', 20)(batch(state), batch(hp), 0)
    prior = previous.initial(old.geometry(64), 0, 'parameter_scale', optimizer)
    expected, _ = previous.chunk(64, 'parameter_scale', 20, batched=False,
                                optimizer=optimizer, native_epsilon=1e-15)(prior, 1e-4, 0)
    z = np.r_[expected['z'], expected['lam']]
    np.testing.assert_allclose(actual['z'][0], z, rtol=2e-12, atol=2e-14)


def test_ema_active_formula_and_gain_control():
    c=case(ema_strength=2., ema_alpha=.98)
    st=core.initialize(c);hp=core.hyperparameters(c);g=jnp.arange(len(st['z']))*.001
    direction, hist, filtered, _=core.optimizer_direction(st,g,hp,'gd')
    np.testing.assert_allclose(filtered,3*g)
    st.update(hist)
    _,_,f,_=core.optimizer_direction(st,-g,hp,'gd')
    np.testing.assert_allclose(f,-g+2*(.98*g+.02*(-g)))
    hp['ema_alpha']=jnp.array(0.)
    hp['ema_normalized']=jnp.array(True)
    d,_,_,_=core.optimizer_direction(st,g,hp,'gd')
    np.testing.assert_allclose(d,-g,atol=1e-16)


def test_tiny_gradient_cosine_and_zero():
    a=jnp.array([1e-250,2e-250,-1e-250])
    assert float(core.cosine(a,a))==pytest.approx(1.)
    assert float(core.cosine(a,-a))==pytest.approx(-1.)
    assert np.isnan(core.cosine(a,a*0))


def test_polynomial_target_does_not_change_with_grid():
    x=np.linspace(-1,1,137)
    for name in ('moment3','moment5','moment9'):
        expected=core.moments.values(name,x,core.polynomial_mapping())
        np.testing.assert_allclose(core.target(x,name),expected,atol=5e-13,rtol=2e-12)


def test_runner_resume_preserves_adam_and_ema(tmp_path):
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64, ema_strength=2., eta=1e-5)
    run.advance(tmp_path/'split',[c],10)
    run.advance(tmp_path/'split',[c],20)
    run.advance(tmp_path/'whole',[c],20)
    a,_=run.restore(tmp_path/'split'/run.key(c)/'state.npz')
    b,_=run.restore(tmp_path/'whole'/run.key(c)/'state.npz')
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])


def test_agreement_uses_one_state_and_separate_batches():
    st=core.initialize(case())
    stats,gradients,full=core.agreement(64,'individual','sine',128)(st['z'][None],st['key'][None])
    assert gradients.shape==(1,8,len(st['z']))
    assert not np.array_equal(gradients[0,0],gradients[0,1])
    expected=np.mean([float(core.cosine(gradients[0,i],gradients[0,j])) for i in range(8) for j in range(i)])
    assert float(stats[0,0,0])==pytest.approx(expected,abs=1e-14)
    x=jnp.linspace(-1,1,1025)
    np.testing.assert_allclose(full[0],core.field(st['z'],x,core.target(x,'sine'),old.geometry(64),'individual')[1],rtol=2e-13,atol=2e-14)


def test_neighbor_reset_clears_suffix_and_preserves_other_physical_neurons():
    from experiments.expD35_optimization_exploration import recycling
    c=case();c['coordinates']='neighbor'
    st=core.initialize(c);g=old.geometry(64);hp=core.hyperparameters(c)
    for name in ('m','v','ema','post_ema','age'): st[name]=jnp.ones_like(st[name])
    mask=jnp.arange(g.width)==7
    out,actual,jump=recycling.apply(st,jnp.ones(g.width),g,'neighbor',hp,'replay_state',mask,1,jnp.linspace(-1,1,129))
    np.testing.assert_array_equal(out['z'],st['z'])
    expected=np.r_[False,np.arange(g.width)>=7,np.arange(g.width)==7]
    np.testing.assert_array_equal(out['age'],~expected)
    assert float(jump)==0
    hp['maturity']=jnp.array(0);hp['replacement_rate']=jnp.array(1/g.width)
    u=jnp.ones(g.width).at[7].set(0.)
    out,mask,jump=recycling.apply(st,u,g,'neighbor',hp,'utility',mask,1,jnp.linspace(-1,1,129))
    before,ga0=core.physical(st['z'],g,'neighbor');after,ga1=core.physical(out['z'],g,'neighbor')
    expected=np.array(before);expected[8]=0
    np.testing.assert_allclose(after,expected,atol=2e-15)
    np.testing.assert_allclose(np.asarray(ga1)[~np.asarray(mask)],np.asarray(ga0)[~np.asarray(mask)])
    assert int(jnp.sum(mask))==1 and float(jump)>0


def test_no_replacement_does_not_roundtrip_neighbor_coordinates():
    from experiments.expD35_optimization_exploration import recycling
    c=case();c['coordinates']='neighbor'
    st=core.initialize(c);g=old.geometry(64)
    out,mask,jump=recycling.apply(st,jnp.ones(g.width),g,'neighbor',core.hyperparameters(c),
                                'utility',jnp.zeros(g.width,dtype=bool),0,jnp.linspace(-1,1,129))
    np.testing.assert_array_equal(st['z'],out['z'])
    assert not np.any(mask)


def test_fourier_accounting_resolves_dc_and_high_frequency_bands():
    from experiments.expD35_optimization_exploration.diagnose import band_energy,band_product
    x=np.arange(2048)/2048
    r=2+np.cos(2*np.pi*90*x);update=-.1+1e-3*np.sin(2*np.pi*7*x)
    bands=band_energy(r)
    assert bands[0]==pytest.approx(4.)
    assert bands[7]==pytest.approx(.5)
    assert np.sum(bands)==pytest.approx(np.mean(r*r))
    np.testing.assert_allclose(band_energy(r+update),bands+2*band_product(r,update)+band_energy(update),atol=1e-15)


def test_dense_capture_does_not_change_updates():
    st=core.initialize(case());hp=core.hyperparameters(case())
    batch=lambda v:jax.tree.map(lambda a:a[None],v)
    a,_=core.chunk(64,'individual','gd','sine',3)(batch(st),batch(hp),0)
    b,(trace,z,gradient,filtered)=core.chunk(64,'individual','gd','sine',3,capture=True)(batch(st),batch(hp),0)
    x=jnp.linspace(-1,1,1025)
    expected=core.field(st['z'],x,core.target(x,'sine'),old.geometry(64),'individual')[1]
    np.testing.assert_allclose(gradient[0,0],expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_array_equal(a['z'],b['z'])
    np.testing.assert_array_equal(z[:,-1],b['z'])


def test_adam_normalized_ema_requires_scaled_epsilon():
    c=case(ema_strength=2.,epsilon=1e-3)
    a=core.initialize(c);b=core.initialize(c)
    ha=core.hyperparameters(c);hb=core.hyperparameters(dict(c,ema_normalized=True,epsilon=c['epsilon']/3))
    for i in range(4):
        grad=jnp.sin(jnp.arange(len(a['z']))+i)*1e-3
        da,ua,_,_=core.optimizer_direction(a,grad,ha,'adam')
        db,ub,_,_=core.optimizer_direction(b,grad,hb,'adam')
        np.testing.assert_allclose(da,db,rtol=1e-13,atol=1e-15)
        a.update(ua);b.update(ub)


def test_selection_can_compare_same_horizon_after_promotion(tmp_path):
    from experiments.expD35_optimization_exploration import run,design
    folder=tmp_path/'trial';folder.mkdir()
    run.write_json(folder/'case.json',run.case())
    run.write_json(folder/'latest.json',dict(step=100000,failed_update=0))
    for step,value in ((16000,.2),(20000,.1),(100000,1e-8)):
        run.write_json(folder/f'evaluation_{step:09d}.json',dict(step=step,validation_relative_mse=value))
    assert design.rank(tmp_path,horizon=20000)[0]['score']==pytest.approx(.15)
    assert design.rank(tmp_path)[0]['score']==pytest.approx(1e-8)


def test_selection_does_not_reweight_policies_that_save_extra_evaluations(tmp_path):
    from experiments.expD35_optimization_exploration import run,design
    for policy in ('constant','decay'):
        folder=tmp_path/policy;folder.mkdir()
        run.write_json(folder/'case.json',run.case(schedule=policy))
        run.write_json(folder/'latest.json',dict(step=100000,failed_update=0))
        for step in range(80000,100001,5000):
            run.write_json(folder/f'evaluation_{step:09d}.json',dict(step=step,validation_relative_mse=.1))
        if policy=='decay':
            run.write_json(folder/'evaluation_000081000.json',dict(step=81000,validation_relative_mse=100.))
    assert [r['score'] for r in design.rank(tmp_path,horizon=100000)]==pytest.approx([.1,.1])


def test_replay_cannot_silently_run_without_source_events(tmp_path):
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64,reset='replay_state',replay_directory=str(tmp_path/'missing'))
    with pytest.raises(ValueError,match='recycling run must reach'):
        run.advance(tmp_path/'out',[c],20000)


def test_history_consolidation_preserves_bytes_and_selection(tmp_path):
    import zipfile,json,hashlib
    from experiments.expD35_optimization_exploration import run,design,history
    folder=tmp_path/'trial';folder.mkdir()
    run.write_json(folder/'case.json',run.case())
    run.write_json(folder/'latest.json',dict(step=20000,failed_update=0))
    for step,value in ((16000,.2),(20000,.1)):
        run.write_json(folder/f'evaluation_{step:09d}.json',dict(step=step,validation_relative_mse=value))
        run.save(folder/f'snapshot_{step:09d}.npz',z=np.arange(4.)+step,step=step)
    expected={p.name:p.read_bytes() for p in folder.glob('snapshot_*.npz')}
    before=design.rank(tmp_path)[0]['score']
    assert history.consolidate(folder)==3
    assert design.rank(tmp_path)[0]['score']==before
    with zipfile.ZipFile(next(folder.glob('history_*.zip'))) as archive:
        hashes=json.loads(archive.read('sha256.json'))
        for name,data in expected.items():
            assert archive.read(name)==data
            assert hashlib.sha256(data).hexdigest()==hashes[name]
    assert (folder/'snapshot_000020000.npz').exists()
    assert not (folder/'snapshot_000016000.npz').exists()
    recovered=list(history.arrays(folder,'snapshot_*.npz'))
    assert [int(data['step']) for _,data in recovered]==[16000,20000]


@pytest.mark.parametrize('kind',['grid','jitter','random'])
def test_affine_all_hidden_gradients_match_autodiff(kind):
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64,architecture='affine',center_initialization=kind)
    st=core.initialize(c);g=old.geometry(64);x=jnp.linspace(-1,1,257);y=core.target(x,'mixed')
    def loss(z):
        w,ga=core.physical(z,g,'individual')
        f=w[0]+old.tanh((x[:,None]-g.centers)*ga+core.offsets(z,g))@w[1:]
        return .5*jnp.mean((f-y)**2)
    actual=core.field(st['z'],x,y,g,'individual')[1]
    np.testing.assert_allclose(actual,jax.grad(loss)(st['z']),rtol=3e-13,atol=2e-14)
    assert len(actual)==3*g.width+1 and np.any(np.asarray(actual[-g.width:])!=0)


def test_affine_release_keeps_function_and_existing_optimizer_state(tmp_path):
    import hashlib
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64,eta=1e-4);folder,st,_=run.prepare(tmp_path,c)
    batch=lambda v:jax.tree.map(lambda a:a[None],v)
    st,_=core.chunk(64,'individual','adam','sine',3)(batch(st),batch(core.hyperparameters(c)),0)
    st=jax.tree.map(lambda a:a[0],st);checkpoint=folder/'checkpoint_000000003.npz'
    run.save(checkpoint,**st,step=3)
    child=dict(c,architecture='affine',origin=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),carry_optimizer=True))
    _,released,_=run.prepare(tmp_path,child)
    np.testing.assert_array_equal(released['z'][:len(st['z'])],st['z'])
    for name in ('m','v','age'):np.testing.assert_array_equal(released[name][:len(st['z'])],st[name])
    assert np.all(np.asarray(core.offsets(released['z'],old.geometry(64)))==0)


def test_adam_to_gd_handoff_keeps_parameters_and_takes_plain_gd_step(tmp_path):
    import hashlib
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64,coordinates='neighbor',eta=1e-5,ema_strength=2.)
    folder,st,_=run.prepare(tmp_path,c)
    batch=lambda v:jax.tree.map(lambda a:a[None],v)
    st,_=core.chunk(64,'neighbor','adam','sine',3)(batch(st),batch(core.hyperparameters(c)),0)
    st=jax.tree.map(lambda a:a[0],st);checkpoint=folder/'checkpoint_000000003.npz'
    run.save(checkpoint,**st,step=3)
    child=dict(c,optimizer='gd',eta=1e-4,ema_strength=0.,origin=dict(
        checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),carry_optimizer=False))
    _,released,_=run.prepare(tmp_path,child)
    np.testing.assert_array_equal(released['z'],st['z'])
    for name in ('m','v','age'):assert np.all(np.asarray(released[name])==0)
    x=jnp.linspace(-1,1,1025);g=old.geometry(64)
    grad=core.field(st['z'],x,core.target(x,'sine'),g,'neighbor')[1]
    after,_=core.chunk(64,'neighbor','gd','sine',1)(batch(released),batch(core.hyperparameters(child)),0)
    np.testing.assert_allclose(after['z'][0],st['z']-child['eta']*grad,rtol=2e-14,atol=2e-14)
