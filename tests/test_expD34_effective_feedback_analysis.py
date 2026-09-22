"""Hand-computable archive comparisons without training or report generation."""
import hashlib
import json

import numpy as np
import pytest

from experiments.expD34_readout_race import effective_feedback_analysis as analysis


def fixtures(tmp_path):
    source, predictions = tmp_path/'runs', tmp_path/'predictions'
    predictions.mkdir()
    case = dict(target='sine', seed=0, start=400000)
    p0 = np.array([.5, 0., .25, 0.])
    changes = dict(joint=.1, freeze_map=.2, clamp_residual=-.05)
    arrays = dict(steps=np.array([0, 2]), p0=p0)
    for arm, change in changes.items():
        p = p0.copy(); p[0] += change
        estimate = p.copy()
        if arm == 'freeze_map':
            estimate[0] += .02
        arrays[arm+'_p'] = np.stack([p0, estimate])
        arrays[arm+'_effective_a'] = np.array([[.1], [.1]])
        arrays[arm+'_supported'] = np.array([True, arm != 'clamp_residual'])
        directory = source/arm/'snapshots'; directory.mkdir(parents=True)
        manifest = dict(cases=[case], arm=arm, eta=.002, degree=65, reference_eta=.002)
        (directory.parent/'manifest.json').write_text(json.dumps(manifest))
        for offset, point in [(0, p0), (2, p)]:
            movement = float(point[0]-p0[0])
            np.savez(directory/f'{offset:09d}.npz', offset=offset, count=np.array([offset]),
                p=point[None], failed=np.array([0]), positive=np.array([[max(movement, 0)]]),
                negative=np.array([[max(-movement, 0)]]), effective=np.array([[movement]]),
                tracking=np.zeros((1, 1)), omitted=np.zeros((1, 1)), crossing=np.zeros((1, 1)),
                first_hit=np.full((1, 3, 1), -1), metric_effective_a=np.array([[.1]]))
    archive = predictions/'case_000.npz'
    np.savez(archive, **arrays)
    record = dict(case=case, degree=65, eta=.002, file=archive.name,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (predictions/'manifest.json').write_text(json.dumps(dict(records=[record])))
    return source, predictions


def test_forecast_errors_and_branch_contrasts_match_hand_values(tmp_path):
    source, predictions = fixtures(tmp_path)
    rows, contrasts = analysis.collect(source, [predictions], eval_samples=32)
    row = next(r for r in rows if r['arm'] == 'freeze_map' and r['offset'] == 2)
    assert row['forecast_supported'] and row['prediction_matched']
    assert row['actual_mean_gamma'] == pytest.approx(.7)
    assert row['predicted_mean_gamma'] == pytest.approx(.72)
    assert row['slope_motion_relative_error'] == pytest.approx(.1)
    assert row['effective_force_relative_error'] == 0
    assert row['motion_identity'] == 0 and row['signed_identity'] == 0
    contrast = next(r for r in contrasts if r['arm'] == 'freeze_map' and r['offset'] == 2)
    assert contrast['actual_mean_gamma_contrast'] == pytest.approx(.1)
    assert contrast['predicted_mean_gamma_contrast'] == pytest.approx(.12)
    assert contrast['contrast_relative_error'] == pytest.approx(.2)
    assert contrast['contrast_alignment'] == pytest.approx(1.)
    assert contrast['mean_gamma_contrast_sign_agrees']
    unsupported = next(r for r in rows if r['arm'] == 'clamp_residual' and r['offset'] == 2)
    assert not unsupported['forecast_supported']
    assert 'predicted_mean_gamma' not in unsupported
    initial = next(r for r in rows if r['arm'] == 'joint' and r['offset'] == 0 and r['model'] == 'affine')
    assert np.isnan(initial['slope_motion_relative_error'])
    assert all('predicted_positive_travel' not in r for r in rows)


def test_unmatched_initial_state_cannot_use_same_label_prediction(tmp_path):
    source, predictions = fixtures(tmp_path)
    path = source/'freeze_map'/'snapshots'/'000000000.npz'
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays['p'][0, 2] += .01
    np.savez(path, **arrays)
    rows, contrasts = analysis.collect(source, [predictions], eval_samples=32)
    assert all(not r['prediction_matched'] for r in rows if r['arm'] == 'freeze_map')
    assert all(r['arm'] != 'freeze_map' for r in contrasts)


def test_changed_prediction_archive_is_rejected(tmp_path):
    _, predictions = fixtures(tmp_path)
    with (predictions/'case_000.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed prediction archive'):
        analysis.prediction_index([predictions])


def test_evaluation_uses_independent_grid_and_original_normalization():
    p = np.array([.7, -.2, .3, .1])
    m = 8192
    x = -1+2*(np.arange(m)+.5)/m
    y = np.sin(2*np.pi*x)
    expected = np.mean((.3*np.tanh(.7*x-.2)+.1-y)**2)/np.mean(y*y)
    assert analysis.evaluation_mse(p, 'sine') == pytest.approx(expected, rel=2e-15)


def test_initial_tail_and_new_visits_are_separate():
    p0 = np.r_[np.array([3.5, .2]), np.zeros(5)]
    p = p0.copy(); p[0] = .1; p[1] = 1.2
    change = abs(p[:2])-abs(p0[:2])
    data = dict(p=p[None], positive=np.maximum(change, 0)[None], negative=np.maximum(-change, 0)[None],
                effective=change[None], tracking=np.zeros((1, 2)), omitted=np.zeros((1, 2)), crossing=np.zeros((1, 2)),
                first_hit=np.array([[[0, 2], [0, -1], [-1, -1]]]))
    row = analysis.actual_metrics(data, 0, p0)
    assert row['initial_fraction_1'] == .5
    assert row['new_ever_fraction_1'] == .5
    assert row['ever_fraction_1'] == 1.
    assert row['current_fraction_1'] == .5
