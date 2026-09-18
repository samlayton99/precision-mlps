import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.expD06_fixed_center_scales import core, difference_training as dt, run


def tree_close(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=3e-12, atol=3e-14)


@pytest.mark.parametrize("coord", dt.COORDINATES)
def test_invertible_map_and_pullback(coord):
    g = core.geometry(128)
    c, gamma = core.initial_physical(g, 0, "xavier_a_reference")
    z = jnp.asarray(dt.encode(c, g, coord))
    np.testing.assert_allclose(dt.decode(z, g, coord), c, atol=2e-16)
    x = jnp.linspace(-1, 1, 257)
    y = core.target(x, "sine")
    lam = jnp.asarray(gamma * g.h) * jnp.where(jnp.arange(g.width) % 2, -1, 1)
    gc = jax.grad(dt.physical_loss)(jnp.asarray(c), lam, x, y, g)
    actual = jax.grad(lambda a: dt.physical_loss(dt.decode(a, g, coord), lam, x, y, g))(z)
    np.testing.assert_allclose(dt.pullback(gc, g, coord), actual, atol=2e-14)


@pytest.mark.parametrize("coord", dt.COORDINATES)
def test_shared_rate_physical_step_and_dense_motion(coord):
    g = core.geometry(128)
    state = dt.initial(g, 0, coord)
    eta = 1e-5
    end, (trace, dense) = dt.chunk(128, coord, 6, True, 1, False)(state, eta, 0)
    c = np.asarray(dt.decode(state["z"], g, coord))
    gc = np.asarray(dense["gradient_c"][0])
    if coord == "scaled":
        expected_dc = -eta * g.alpha * gc
    else:
        s2 = np.cumsum(g.alpha[1:])
        dq = -eta * s2 * (gc[1:] - np.r_[gc[2:], 0.])
        expected_dc = np.r_[-eta*g.alpha[0]*gc[0], dq - np.r_[0., dq[:-1]]]
    np.testing.assert_allclose(dense["delta_c"][0], expected_dc, atol=1e-16)
    np.testing.assert_allclose(dense["delta_lambda"][0], -eta*dense["gradient_lambda"][0], atol=1e-17)
    final_c = dt.decode(end["z"], g, coord)
    np.testing.assert_allclose(np.diff(np.r_[dense["c"], final_c[None]], axis=0), dense["delta_c"], atol=3e-16)
    np.testing.assert_allclose(np.diff(np.r_[dense["gamma"]*g.h, end["lam"][None]], axis=0), dense["delta_lambda"], atol=1e-17)
    np.testing.assert_array_equal(trace[:, -1], np.full(6, eta))


def test_batched_failure_isolation_and_paired_initialization():
    g = core.geometry(128)
    states = [dt.initial(g, 0, coord) for coord in dt.COORDINATES]
    np.testing.assert_allclose(dt.decode(states[0]["z"], g, dt.COORDINATES[0]),
                               dt.decode(states[1]["z"], g, dt.COORDINATES[1]), atol=2e-16)
    np.testing.assert_array_equal(states[0]["lam"], states[1]["lam"])
    single, _ = dt.chunk(128, "scaled", 8, False, 1, False)(states[0], 1e-5, 0)
    batch, _ = dt.chunk(128, "scaled", 8, False, 1)(run.stack_states([states[0]]*2), jnp.array([1e-5, 1e308]), 0)
    tree_close(single, run.unstack_state(batch, 0))
    assert int(batch["failed"][1]) > 0
    assert int(batch["failed"][0]) == 0


def test_saved_resume_and_fixed_centers(tmp_path):
    case = dict(n=128, seed=0, coordinates="scaled_differences", eta=1e-5)
    dt.advance_group(tmp_path, [case], 4, samples_per_cell=1)
    dt.advance_group(tmp_path, [case], 8, samples_per_cell=1)
    path = tmp_path/dt.case_key(case)
    actual, step = run.load_state(path/"state_000000008.pkl")
    g = core.geometry(128)
    expected, _ = dt.chunk(128, case["coordinates"], 8, False, 1, False)(dt.initial(g, 0, case["coordinates"]), 1e-5, 0)
    tree_close(actual, expected)
    assert step == 8
    assert json.loads((path/"latest.json").read_text())["completed_updates"] == 8
    with np.load(path/"reference.npz") as a:
        np.testing.assert_array_equal(a["centers"], g.centers)
    with np.load(path/"checkpoint_000000008.npz") as a:
        assert a["alternate_eval_max"] < 1e-14
    assert sum(np.load(p)["trace"].shape[0] for p in path.glob("trace_*.npz")) == 8


@pytest.mark.parametrize("coord", dt.COORDINATES)
def test_detached_spectral_normalization_and_frozen_step(coord):
    from experiments.expD06_fixed_center_scales import diagnostics, difference_analysis as analysis
    g = core.geometry(128)
    rng = np.random.default_rng(22)
    c = rng.normal(size=g.width+1)*.02
    gamma = rng.uniform(.15,.5,size=g.width)/g.h
    eta = 1e-4
    record, arrays, _ = analysis.spectral_probe(g, c, gamma, coord, eta, samples=2)
    assert record["modal_prediction_max_error"] < 1e-12
    assert record["fourier_parseval_error"] < 1e-12
    x = np.linspace(-1,1,257)
    a = diagnostics.features(x,g.centers,gamma)/np.sqrt(len(x))
    b = analysis.mapped_features(a,g,coord)
    z = dt.encode(c,g,coord)
    np.testing.assert_allclose(a@c,b@z,atol=1e-14)
    np.testing.assert_allclose(arrays["band_mse"].sum(),record["train_mse"],atol=1e-14)
    assert record["readout_stability_limit"] == 2/record["sigma_max"]**2
    assert all(r["train_mse"] < record["train_mse"] for r in record["refits"])


def test_projected_force_removes_large_in_span_roundoff():
    from experiments.expD06_fixed_center_scales.difference_analysis import projected_forces
    rng=np.random.default_rng(391)
    u=np.linalg.qr(rng.normal(size=(80,9)))[0]
    j=u@rng.normal(size=(9,13))
    small=rng.normal(size=80)*1e-10
    small-=u@(u.T@small)
    r=u@rng.normal(size=9)+small
    gp,gn,leak,closure=projected_forces(j,r,u)
    assert np.linalg.norm(gn)<1e-23
    assert leak>1000*np.linalg.norm(gn)
    np.testing.assert_allclose(gp+gn,j.T@r,atol=3e-14)
    assert closure<3e-14
    complete=np.linalg.qr(rng.normal(size=(80,80)))[0]
    u,v=complete[:,:9],complete[:,9:14]
    outside=rng.normal(size=(5,13));beta=rng.normal(size=5)
    j=u@rng.normal(size=(9,13))+1e-4*v@outside
    r=u@rng.normal(size=9)+1e-8*v@beta
    _,gn,_,_=projected_forces(j,r,u)
    np.testing.assert_allclose(gn,1e-12*outside.T@beta,rtol=1e-6,atol=2e-19)


@pytest.mark.parametrize("coord",dt.COORDINATES)
def test_halo_pair_bound_and_cancellation(coord):
    from experiments.expD06_fixed_center_scales import diagnostics,difference_analysis as analysis
    g=core.geometry(128);lam=.125
    a=diagnostics.features(np.linspace(-1,1,16*g.n+1),g.centers,np.full(g.width,lam/g.h))/np.sqrt(16*g.n+1)
    b=analysis.mapped_features(a,g,coord)
    scale=g.d[-1] if coord=="scaled" else np.sqrt(g.alpha[1:].sum())
    v=np.zeros(g.width+1);v[0]=scale/g.d[0];v[-1]=1;v/=np.linalg.norm(v)
    result=analysis.halo_cancellation_bound(g,lam,coord)
    np.testing.assert_allclose(np.linalg.norm(b@v),result["normalized_pair_image_norm"],rtol=1e-12)
    assert result["normalized_pair_image_norm"]<=result["sigma_min_upper_bound"]
    saturated=analysis.halo_cancellation_bound(core.geometry(512),1.,coord)
    assert saturated["naive_tanh_plus_one_max"]==0.
    assert saturated["normalized_pair_image_norm"]>0.
