"""Check the hard-mode floor and finite-step descent accounting independently."""
import numpy as np
from experiments.expD34_readout_race import persistence_energy as en, persistence_theory as pt, transport


def test_hard_cap_covers_perturbed_network_coefficients():
    rng = np.random.default_rng(32); x = (np.arange(256)+.5)/128-1
    q = transport.basis(x, 9)[:, 9]; p = rng.normal(0, .15, 22); radius = .08
    cap = en.hard_cap(p, radius)
    for _ in range(100):
        direction = rng.normal(size=len(p)); direction *= radius*rng.random()/np.linalg.norm(direction)
        z = p+direction; a,b,c = z[:-1].reshape(3,7)
        coefficient = q @ (np.tanh(x[:,None]*a+b) @ c+z[-1])/len(x)
        assert abs(coefficient) <= cap


def test_energy_ball_contains_nonlinear_gd_and_each_path_prefix():
    rng = np.random.default_rng(91); x = np.linspace(-.99, .99, 64)
    p = rng.normal(0, .12, 22); q = transport.basis(x, 9)
    initial = pt.tensors(p,x,np.zeros_like(x))['r']
    y = q[:,:2] @ (q[:,:2].T @ initial/len(x))+q[:,9]
    row = max(en.candidates(p,x,y), key=lambda v:v['updates'])
    assert row['updates'] > 1000
    pn = p.copy(); path = 0.
    for n in range(1000):
        gradient = pt.tensors(pn,x,y)['g']; pn -= .002*gradient
        path += .002*np.linalg.norm(gradient)
        bound = np.sqrt(.002*(n+1)*row['available_loss']/row['descent_factor'])
        assert path <= bound and np.linalg.norm(pn-p) <= row['inner_radius']
