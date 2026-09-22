"""Check the physical two-residual solution against its discrete updates."""
import numpy as np
from experiments.expD34_readout_race import persistence_reduction as pr, transport


def test_reduced_solution_matches_coupled_updates_and_path_bound():
    rng = np.random.default_rng(82); x = np.linspace(-1, 1, 128)
    p = rng.normal(0, .4, 22); y = np.sin(7*x)+x
    state = pr.reduced_state(p, x, y, transport.basis(x, 9))
    for driven in (False, True):
        pn = p.copy(); residual = state['initial'].copy(); path = 0.
        for _ in range(319):
            force = state['generated'] @ residual
            if driven: force += state['hard']
            pn -= .002*force; path += .002*np.linalg.norm(force[:7])
            residual -= .002*state['generated'].T @ force
        got = pr.reduced_at(state, 319, driven=driven)
        np.testing.assert_allclose(got['p'], pn, atol=3e-14)
        np.testing.assert_allclose(got['residual'], residual, atol=3e-14)
        assert path <= got['slope_path_bound']+1e-14
    np.testing.assert_allclose(state['generated'].T @ (state['hard']+state['generated'] @ state['equilibrium']), 0., atol=1e-16)
