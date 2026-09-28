"""Artifact-level smoke checks, independent of scientific GPU execution."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.expD34_readout_race import population_dilation_summary as dilation
from experiments.expD34_readout_race import population_output_summary as output


def artifacts(tmp_path):
    p = np.arange(14, dtype=float).reshape(2, 7)/100
    source = tmp_path/'prepared.npz'
    np.savez(source, p=p)
    cases = [dict(target='fixture', seed=30, start=20000, cohort='wide', scale=scale,
                  reference='primary', arm=arm, input_index=i, eta=.002,
                  repair=dict(valid=True, reason='accepted'))
             for i, (arm, scale) in enumerate((('repaired', 1.), ('s10_primary', 10.)))]
    runs = []
    for name, eta, steps in (('ordinary', .002, 10), ('half', .001, 20)):
        directory = tmp_path/name; directory.mkdir(); runs.append(directory)
        manifest = dict(input=str(source), input_sha256=dilation.digest(source), cases=cases,
                        skipped_invalid=[dict(cases[1], arm='s100_primary', repair=dict(valid=False))],
                        eta=eta, steps=steps, channels=['direct_fine', 'compensation', 'tracking', 'unresolved'],
                        error_thresholds=[.1, .01])
        (directory/'manifest.json').write_text(json.dumps(manifest))
        for count in (0, steps):
            state = {key: np.full(2, .1) for key in dilation.SCALARS}
            state.update(p=p+count*eta, count=np.full(2, count), failed=np.zeros(2, bool),
                         resolved=np.array([True, count == 0]), diagnostic_unresolved_steps=np.array([0, int(count > 0)]),
                         signed=np.zeros((2, 4, 2)), norm_integral=np.ones((2, 4)),
                         error_first_hit=np.array([[-1, -1], [7 if count else -1, -1]]),
                         first_hit=np.array([[0, -1], [-1, -1]]))
            np.savez(directory/f'{count:09d}.npz', **state)
    return runs


def test_dilation_manifest_first_hits_unresolved_and_matched_time(tmp_path):
    runs = artifacts(tmp_path)
    destination = tmp_path/'summary'
    dilation.run(SimpleNamespace(runs=runs, output=destination, input_root=None, no_plots=True))
    summary = json.loads((destination/'summary.json').read_text())
    assert summary['attempts'] == 6 and summary['skipped'] == 2
    assert summary['endpoint_count'] == 4 and summary['matched_step_controls'] == 2
    with (destination/'endpoints.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    primary = next(r for r in rows if r['arm'] == 's10_primary')
    assert primary['status'] == 'unresolved_diagnostic'
    assert primary['first_hit_relative_l2_0.1'] == '7'  # online counter, not endpoint update
    assert primary['tracking_integrated_norm_to_effective'] == ''
    assert primary['unresolved_norm_integral'] == '1.0'
    assert primary['baseline_available'] == 'True'
    with (destination/'step_controls.csv').open() as stream:
        controls = list(csv.DictReader(stream))
    assert all(r['equal_completed_flow_time'] == 'True' for r in controls)


def test_dilation_rejects_prepared_start_mismatch(tmp_path):
    runs = artifacts(tmp_path)
    path = runs[0]/'000000000.npz'
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    state['p'] = state['p']+.01
    np.savez(path, **state)
    with pytest.raises(ValueError, match='Start-state mismatch'):
        dilation.run(SimpleNamespace(runs=runs, output=tmp_path/'summary', input_root=None, no_plots=True))


def test_output_summary_csv_flags_and_role_classification(tmp_path):
    source = tmp_path/'states.csv'
    source.write_text('role,arm,resolved,status,Q,Q_dot_generated,Q_dot_target\n'
                      'static,,True,finite,0.1,-2,3\n'
                      'trajectory,natural,False,finite,0.2,,\n'
                      'trajectory,s10_primary,True,finite,0.3,1,-1\n')
    rows = output.load(source)
    assert [output.role(r) for r in rows] == ['static', 'natural', 'dilation']
    assert output.usable(rows[1]) and not output.usable(rows[1], resolved=True)
    assert rows[0]['Q_dot_generated'] == -2 and rows[1]['Q_dot_target'] is None
