import json

import numpy as np
import pytest

from experiments.expD34_readout_race import plateau_followup as follow


def test_extract_preserves_case_order_and_every_state_field(tmp_path):
    cases = [dict(target=t, seed=s) for s in (1, 0) for t in ('moment9', 'mixed_sine')]
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(cases=cases)))
    (tmp_path / 'status.json').write_text(json.dumps(dict(complete=True, failed=[0]*4, unresolved=[0]*4)))
    arrays = dict(p=np.arange(24.).reshape(4, 2, 3), m=np.arange(24.).reshape(4, 2, 3)+100,
                  count=np.arange(8).reshape(4, 2), failed=np.zeros((4, 2), bool))
    np.savez(tmp_path / 'snapshots.npz', steps=[600000, 6000000], **arrays)
    selected, state = follow.extract(tmp_path, 0, ('mixed_sine', 'moment9'))
    assert selected == cases[2:]
    assert set(state) == set(arrays)
    for name in arrays:
        np.testing.assert_array_equal(state[name], arrays[name][2:, 1])
    with pytest.raises(ValueError, match='Missing checkpoint'):
        follow.extract(tmp_path, 0, ('moment9', 'mixed_sine'), step=6000001)
    with pytest.raises(ValueError, match='Missing or duplicate'):
        follow.extract(tmp_path, 2, ('moment9', 'mixed_sine'))
    (tmp_path / 'status.json').write_text(json.dumps(dict(complete=False, failed=[0]*4)))
    with pytest.raises(ValueError, match='Incomplete'):
        follow.extract(tmp_path, 0, ('moment9', 'mixed_sine'))


def test_probe_rejects_modified_forecast_and_duplicate_owner(tmp_path, monkeypatch):
    (tmp_path / 'pilot').mkdir()
    (tmp_path / 'pilot/passed.json').write_text('{"passed": true}')
    forecast = tmp_path / 'forecast.npz'
    forecast.write_bytes(b'issued before updates')
    (tmp_path / 'prepared.json').write_text(json.dumps(dict(artifacts={'forecast.npz': follow.sha(forecast)})))
    calls = []
    monkeypatch.setattr(follow.pp, 'run', lambda args: calls.append(args))
    follow.probe(tmp_path, 0, 100)
    assert calls[0].start == 6000000 and calls[0].horizon == 500000
    with pytest.raises(FileExistsError):
        follow.probe(tmp_path, 0, 100)
    forecast.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed prepared input or forecast'):
        follow.probe(tmp_path, 1, 100)
