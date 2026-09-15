"""Protocol checks on tiny synthetic cases; not additional research arms."""
import numpy as np
import torch

from experiments.expD27_readout_information import run as exp


def problem():
    rng = np.random.default_rng(73)
    x = np.linspace(-1, 1, 29)
    state = dict(a=np.array([1., 2., 3., 4.]), b=rng.normal(size=4), v=rng.normal(size=5))
    return state, x, np.sin(5*x)


def test_solve_at_zero_and_each_step_never_readout_gd():
    torch.set_num_threads(2)
    state, x, y = problem()
    trajectory, info = exp.train(state, x, y, 3, .002, 'solve')
    assert info['solve_count'] == 4
    for step in range(4):
        current = exp.state_at(trajectory, step)
        expected, _ = exp.previous.solve_readout(current, x, y, 1e-13)
        np.testing.assert_allclose(current['v'], expected, rtol=2e-13, atol=2e-13)
        if step < 3:
            gradients = exp.previous.numpy_gradients(current, x, y)
            for key in ('a', 'b'):
                np.testing.assert_allclose(trajectory[key][step+1], current[key]-.002*gradients[key], rtol=1e-13, atol=1e-14)


def test_branch_keeps_solved_marker_and_freezes_all_coefficients():
    state, x, y = problem()
    reference, _ = exp.train(state, x, y, 5, .002, 'solve')
    cfg = dict(frozen_steps=4, learning_rate=.002)
    branch, _ = exp.branch_from(reference, 2, x, y, cfg)
    for key in exp.FIELDS:
        np.testing.assert_array_equal(branch[key][:3], reference[key][:3])
    np.testing.assert_array_equal(branch['v'][2:], np.broadcast_to(reference['v'][2], branch['v'][2:].shape))
    assert branch['steps'][-1] == 6
    assert np.all(branch['solve_rank'][3:] == -1)
    for key in ('a', 'b'):
        np.testing.assert_array_equal(branch[key][3], reference[key][3])


def test_zero_readout_has_no_first_geometry_update():
    state, x, y = problem()
    state['v'][:] = 0
    trajectory, _ = exp.train(state, x, y, 2, .002, 'gd')
    for key in ('a', 'b'):
        np.testing.assert_array_equal(trajectory[key][0], trajectory[key][1])
    assert np.any(trajectory['v'][1] != 0)
    assert np.any(trajectory['a'][2] != trajectory['a'][1])


def test_transfer_and_scale_noise_preserve_their_requested_invariants():
    cfg = exp.config()
    cfg.update(resolution=8, halo=2, n_train=48, qi_gamma=2., transfer_reset='gamma_1',
               noise_kind='multiplicative_scale', noise_level=.1)
    teacher, info = exp.solved_qi_reference('sine', cfg)
    transfer = exp.transfer_state(teacher, cfg)
    np.testing.assert_array_equal(transfer['v'], teacher['v'])
    np.testing.assert_allclose(-transfer['b']/transfer['a'], -teacher['b']/teacher['a'])
    np.testing.assert_array_equal(transfer['a'], np.ones_like(transfer['a']))
    noisy, factors = exp.noisy_qi_state(cfg)
    np.testing.assert_allclose(-noisy['b']/noisy['a'], -teacher['b']/teacher['a'])
    np.testing.assert_allclose(noisy['a'], cfg['qi_gamma']*factors)
    assert np.all(noisy['v'] == 0)


def test_offline_diagnostics_cannot_mutate_training_states():
    cfg = exp.config()
    cfg.update(resolution=8, halo=2, n_train=48, n_eval=96, qi_gamma=2.,
               freeze_steps=[2, 4], frozen_steps=5)
    cases, metadata = exp.run_cases('qi_zero', 'sine', cfg)
    before = {name: {key: value[key].copy() for key in exp.FIELDS} for name, value in cases.items()}
    exp.diagnose(cases, 'sine', cfg, metadata)
    for name in cases:
        for key in exp.FIELDS:
            np.testing.assert_array_equal(cases[name][key], before[name][key])
    for step in cfg['freeze_steps']:
        np.testing.assert_array_equal(cases[f'freeze_{step}']['relative_l2'][:step+1], cases['reference']['relative_l2'][:step+1])
