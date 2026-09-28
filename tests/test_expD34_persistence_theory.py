"""Independent Hessians, finite GD spectra, and conservative neighborhood bounds."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import plateau
from experiments.expD34_readout_race import persistence_theory as pt


def example():
    rng = np.random.default_rng(71)
    return rng, rng.normal(0, .2, 22), np.linspace(-.99, .99, 64)


def test_loss_hessian_matches_autodiff():
    _, p, x = example(); y = np.sin(7*x)+.2
    h = jax.hessian(lambda z: jnp.mean((plateau.prediction(z, x)-y)**2)/2)(jnp.array(p))
    np.testing.assert_allclose(pt.tensors(p, x, y)['hessian'], h, rtol=2e-11, atol=3e-14)


def test_ball_bounds_cover_perturbed_hessians():
    rng, p, x = example(); y = np.sin(7*x)
    state = pt.tensors(p, x, y); radius = .03; bound = pt.ball_constants(p, state, radius)
    for _ in range(20):
        direction = rng.normal(size=len(p)); direction *= radius*rng.uniform()/np.linalg.norm(direction)
        new = pt.tensors(p+direction, x, y)
        ev = np.linalg.eigvalsh(new['hessian'])
        assert ev.min() >= -bound['negative']
        assert ev.max() <= bound['upper']
        assert np.linalg.norm(new['hessian']-state['hessian'], 2) <= bound['third']*np.linalg.norm(direction)


def test_frozen_spectrum_and_path_against_explicit_updates():
    _, p, x = example(); y = np.sin(7*x); spectrum = pt.frozen_spectrum(p, x, y)
    g = spectrum['initial']['g'].copy(); pn = p.copy(); path = 0.; eta = .002
    for _ in range(257):
        pn -= eta*g; path += eta*np.linalg.norm(g[:7]); g = g-eta*spectrum['gram'] @ g
    got, gg, _, bound = pt.frozen_at(spectrum, 257)
    np.testing.assert_allclose(got, pn, atol=3e-14)
    np.testing.assert_allclose(gg, g, atol=3e-14)
    assert path <= bound
    np.testing.assert_allclose(pt.geometric(1., 17), 17)
    np.testing.assert_allclose(pt.geometric(1.01, 17), sum(1.01**i for i in range(17)))


def test_enclosure_contains_actual_nonlinear_gd():
    _, p, x = example(); y = np.sin(7*x)*.01
    rows = pt.frozen_enclosure(p, x, y, horizon=200, block=10)
    assert all(r['closed'] for r in rows)
    spectrum = pt.frozen_spectrum(p, x, y); pn = p.copy()
    for n in range(200):
        pn -= .002*pt.tensors(pn, x, y)['g']
        row = rows[n//10]; predicted = pt.frozen_at(spectrum, n+1)[0]
        assert np.linalg.norm(pn-predicted) <= row['end_error']+1e-14
    a = np.array([.1, -.2, 1.2])
    np.testing.assert_allclose(pt.acquisition_distance(a, 2/3, 1.), .8)
