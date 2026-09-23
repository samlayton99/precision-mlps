"""Source-integrity checks for original and combined forecast archives."""
import json

import numpy as np
import pytest

from experiments.expD34_readout_race.cross_function_audit import (
    COHORTS, HORIZONS, THRESHOLDS, digest, load_cohort,
)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def archive(tmp_path):
    stem, relative = COHORTS['fresh']
    pack = tmp_path/'inputs'/f'{stem}.npz'
    prediction = tmp_path/'predictions'/stem/'manifest.json'
    run = tmp_path/relative
    for directory in (pack.parent, prediction.parent, run/'snapshots'):
        directory.mkdir(parents=True)
    cases = [dict(target='sine', seed=20, start=100000)]
    p = np.array([[.1, .2, .3, .4]])
    np.savez(pack, p=p, x=np.array([-.5, .5]), y=np.array([[1., -1.]]),
             cases=np.array(json.dumps(cases)))
    # Combined manifests keep step/basis on records, not at the top level.
    write_json(prediction, dict(input_sha256=digest(pack), records=[
        dict(case=cases[0], eta=.002, degree=65, file='case_000.npz')]))
    write_json(run/'manifest.json', dict(cases=cases, arm='joint', eta=.002,
        degree=65, input_sha256=digest(pack), thresholds=THRESHOLDS))
    write_json(run/'status.json', dict(complete=True, valid=True))
    for n in HORIZONS:
        np.savez(run/'snapshots'/f'{n:09d}.npz', p=p, offset=n,
                 count=np.array([n]), failed=np.array([False]))
    return tmp_path, pack, prediction, run


def test_combined_manifest_without_top_level_step_or_degree(archive):
    root, _, prediction, _ = archive
    assert 'eta' not in json.loads(prediction.read_text())
    pack, cases, snapshots, _, records, _ = load_cohort(root, 'fresh')
    assert pack['p'].shape == (1, 4)
    assert tuple(snapshots) == HORIZONS
    assert list(records) == [('sine', 20, 100000)]
    assert cases[0]['seed'] == 20


def test_changed_input_pack_is_rejected(archive):
    root, pack, _, _ = archive
    pack.write_bytes(pack.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Changed input pack'):
        load_cohort(root, 'fresh')


@pytest.mark.parametrize('field', ['count', 'offset', 'failed'])
def test_changed_snapshot_update_accounting_is_rejected(archive, field):
    root, _, _, run = archive
    path = run/'snapshots'/'000010000.npz'
    with np.load(path) as source:
        data = {key: source[key].copy() for key in source.files}
    data[field] = np.array([True]) if field == 'failed' else data[field]+1
    np.savez(path, **data)
    with pytest.raises(ValueError, match='Invalid update count or failed state'):
        load_cohort(root, 'fresh')


@pytest.mark.parametrize('field,value,message', [
    ('thresholds', [3.2, 1., 16.], 'threshold order'),
    ('input_sha256', 'different', 'Run and analysis inputs differ'),
    ('cases', [dict(target='chirp', seed=20, start=100000)], 'cases differ'),
])
def test_changed_run_provenance_is_rejected(archive, field, value, message):
    root, _, _, run = archive
    path = run/'manifest.json'
    data = json.loads(path.read_text())
    data[field] = value
    write_json(path, data)
    with pytest.raises(ValueError, match=message):
        load_cohort(root, 'fresh')


@pytest.mark.parametrize('field,value', [('eta', .001), ('degree', 129)])
def test_combined_record_protocol_is_checked(archive, field, value):
    root, _, path, _ = archive
    data = json.loads(path.read_text())
    data['records'][0][field] = value
    write_json(path, data)
    with pytest.raises(ValueError, match='Forecast uses a different step or basis'):
        load_cohort(root, 'fresh')
