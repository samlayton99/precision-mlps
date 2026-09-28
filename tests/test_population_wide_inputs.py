"""Reuse archived initial states without loading the original PyTorch initializer."""
import json

import numpy as np
import pytest

from experiments.expD34_readout_race import population_wide_inputs as inputs


def reference(tmp_path, mismatch=False):
    pp = np.arange(4*2116, dtype=float).reshape(4, 2116)/10000
    pp[1] = pp[0]; pp[3] = pp[2]
    if mismatch:
        pp[1, 0] += .001
    cases = [dict(seed=seed, start=0, width=705, nref=512) for seed in (30, 30, 31, 31)]
    path = tmp_path/'reference.npz'
    np.savez(path, p=pp, cases=np.array(json.dumps(cases)))
    return path, pp


def test_reuse_exact_initial_states_without_initializer(tmp_path, monkeypatch):
    path, pp = reference(tmp_path)
    def forbidden(*args):
        raise AssertionError('Archived initialization must bypass PyTorch')
    monkeypatch.setattr(inputs.targets, 'initial', forbidden)
    monkeypatch.setattr(inputs.widths, 'data', lambda name, *args:
                        (np.array([-.5, .5]), np.array([1., -1.]), None, 1.))
    output = tmp_path/'prepared.npz'
    inputs.prepare(output, path)
    with np.load(output) as data:
        cases = json.loads(str(data['cases']))
        assert len(cases) == 34
        assert str(data['initialization_reference_sha256']) == inputs.ef.digest(path)
        for p, case in zip(data['p'], cases):
            np.testing.assert_array_equal(p, pp[0 if case['seed'] == 30 else 2])


def test_reject_differing_same_seed_states(tmp_path):
    path, _ = reference(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match='Same-seed initial parameters differ'):
        inputs.archived_initializations(path)
