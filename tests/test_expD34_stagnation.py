"""Independent loss gradients, coarse projection, and exact GD motion checks."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from experiments.expD34_readout_race import stagnation as st, targets, transport as tr


def example():
    rng = np.random.default_rng(304)
    z = rng.normal(size=(3, 7))*.3
    x = targets.grid(64)
    return z, .12, x, targets.data(64, 'moment9')['y']


def independent_gradient(z, d, x, y, arm):
    zz = torch.tensor(z, requires_grad=True)
    dd = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    xx, yy = torch.tensor(x), torch.tensor(y)
    r = torch.tanh(xx[:, None]*zz[0]+zz[1]) @ zz[2]+dd-yy
    if arm != 'full':
        q = torch.tensor(st.mode_matrix(x, arm))
        projected = q @ (q.T @ r/len(x))
        r = r-projected if arm == 'remove_lower' else projected
    (.5*torch.mean(r*r)).backward()
    return zz.grad.numpy(), dd.grad.item()


@pytest.mark.parametrize('arm', st.ARMS)
def test_active_loss_gradient_and_coarse_decomposition(arm):
    z, d, x, y = example()
    expected, ed = independent_gradient(z, d, x, y, arm)
    g, gd, *_ = tr.field(jnp.array(z), d, jnp.array(x), jnp.array(y),
        jnp.ones(7)/7, 7, jnp.array(st.mode_matrix(x, arm)), remove=arm == 'remove_lower')
    np.testing.assert_allclose(g, expected, rtol=2e-13, atol=2e-15)
    assert float(gd) == pytest.approx(ed, abs=2e-15)
    row, arr = st.diagnostics(z, d, x, y, arm, degree=17)
    np.testing.assert_allclose(arr['gradient'], expected, rtol=2e-13, atol=2e-15)
    assert row['actual_outward'] == pytest.approx(-np.mean(np.sign(z[0])*expected[0]), abs=1e-15)
    assert row['actual_outward'] == pytest.approx(sum(row[k+'_outward'] for k in
        ('generated', 'hard', 'higher', 'tracking', 'omitted')), abs=1e-15)
    h = np.tanh(x[:, None]*z[0]+z[1]); s = 1-h*h
    jc = np.asarray(st.coarse_jacobian(z, x, h, s, tr.basis(x, 9)[:, :2]))
    flat = np.r_[expected.ravel(), ed]
    qg = jc.T @ np.linalg.solve(jc @ jc.T, jc @ flat)
    np.testing.assert_allclose(jc @ (flat-qg), 0, atol=3e-15)
    np.testing.assert_allclose(arr['exact_tracking_a'], qg[:7], atol=2e-15)
    if arm == 'remove_lower':
        np.testing.assert_array_equal(arr['generated_a'], np.zeros(7))


@pytest.mark.parametrize('arm', st.ARMS)
def test_simultaneous_updates_and_stepwise_motion_identities(arm):
    z, d, x, y = example(); eta = .002
    state = st.initial(z[None], np.array([d]))
    result = st.advance_factory(len(x), 7, arm, eta)(state, jnp.array(y[None]), 12)
    zz, dd = z.copy(), d
    positive = np.zeros(7); negative = np.zeros(7); tracking = np.zeros(7)
    for _ in range(12):
        g, gd = independent_gradient(zz, dd, x, y, arm)
        h = np.tanh(x[:, None]*zz[0]+zz[1]); s = 1-h*h
        jc = np.asarray(st.coarse_jacobian(zz, x, h, s, tr.basis(x, 9)[:, :2]))
        qg = jc.T @ np.linalg.solve(jc @ jc.T, jc @ np.r_[g.ravel(), gd])
        tracking -= eta*np.sign(zz[0])*qg[:7]
        zn = zz-eta*g; delta = abs(zn[0])-abs(zz[0])
        positive += np.maximum(delta, 0); negative += np.maximum(-delta, 0)
        zz, dd = zn, dd-eta*gd
    np.testing.assert_allclose(result['z'][0], zz, atol=2e-15, rtol=2e-14)
    assert float(result['d'][0]) == pytest.approx(dd, abs=2e-15)
    for key, value in [('positive', positive), ('negative', negative), ('tracking_travel', tracking)]:
        np.testing.assert_allclose(result[key][0], value, atol=2e-15, rtol=2e-13)
    balance = lambda v: np.sum(v[0]**2+v[1]**2-v[2]**2)
    assert balance(zz)-balance(z) == pytest.approx(float(result['balance_flow'][0]+result['balance_discrete'][0]), abs=3e-15)
    assert .5*np.sum(zz[0]**2-z[0]**2) == pytest.approx(float(result['slope_energy_flow'][0]+result['slope_energy_discrete'][0]), abs=2e-15)


def test_probe_accounts_for_changed_raw_coarse_force():
    z, d, x, y = example()
    probes = st.force_probes(z, d, x, y, degree=17)
    zero = next(p for p in probes if p['probe'] == 'lower_penalty' and p['value'] == 0)
    row, _ = st.diagnostics(z, d, x, y, 'remove_lower', degree=17)
    for name in ('effective_outward', 'tracking_outward', 'actual_outward'):
        assert zero[name] == pytest.approx(row[name], abs=2e-15)
    q9 = tr.basis(x, 9)[:, 9]; y9 = q9 @ y/len(x)
    for probe in (p for p in probes if p['probe'] == 'hard_target'):
        row, _ = st.diagnostics(z, d, x, y+(probe['value']-y9)*q9, degree=17)
        for name in ('effective_outward', 'tracking_outward', 'actual_outward'):
            assert probe[name] == pytest.approx(row[name], abs=2e-13, rel=2e-12)


def test_positive_travel_does_not_disappear_in_mean_cancellation():
    z, d, x, y = example()
    out = st.advance_factory(len(x), 7, 'full', .002)(st.initial(z[None], np.array([d])), jnp.array(y[None]), 12)
    assert float(out['positive'].mean()) > 0
    assert float(out['negative'].mean()) > 0
    np.testing.assert_allclose(out['positive'][0]-out['negative'][0], abs(out['z'][0, 0])-abs(z[0]), atol=2e-15)
