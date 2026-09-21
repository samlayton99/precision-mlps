import json
import numpy as np

from experiments.expD36_frozen_gamma_probe import cap_analyze as a, cap_campaign as c


def test_audit_checks_strict_tolerance_and_cap_nesting_on_held_out_case(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    case = tmp_path/'cases'/'held_out'
    write(case/'meta.json', dict(id='held_out', n=128, cap=4, family='uniform',
        seed=100, eta=.1, matrix_hash='j', target_hash='y', source_commit='test', screen_hits={}))
    hits = [[-1]*len(c.EPSILONS) for _ in c.TARGETS]
    hits[0][:2] = [20, 80]
    write(case/'training.json', dict(steps=200000, hits=hits))
    # A cap-8 proof applies to this cap-4 case. Only the stricter hit violates it.
    bounds = {str(e):dict(bound=10 if i == 0 else 100) for i, e in enumerate(c.EPSILONS)}
    write(tmp_path/'certificates'/'larger_cap'/'result.json', dict(n=128, cap=8,
        target=c.TARGETS[0], rank=4, max_intervals=8, source_commit='test',
        grid_hash='x', centers_hash='centers', target_hash='y', bounds=bounds, certificates=[]))
    write(tmp_path/'confirmation_cases.json', ['held_out', 'missing'])
    result = a.collect(tmp_path)
    assert len(result['bound_violations']) == 1
    violation = result['bound_violations'][0]
    assert violation['epsilon'] == 1e-4 and violation['held_out']
    assert violation['certificate'] == 'larger_cap'
    assert result['coverage']['confirmation']['incomplete'] == ['missing']
    write(tmp_path/'development_cases.json', ['held_out'])
    reused = a.collect(tmp_path)
    assert not any(row['held_out'] for row in reused['certificate_checks'])
    assert reused['coverage']['confirmation']['new_dictionaries_vs_development'] == 0


def test_cdf_envelope_reuses_larger_caps_without_interpolating_mass():
    certificates = [dict(n=128, cap=4, target='y', cdf=dict(thresholds=[0., .2, 1.], mass=[0., .4, 1.])),
                    dict(n=128, cap=8, target='y', cdf=dict(thresholds=[0., .1, .5, 1.], mass=[0., .1, .6, 1.]))]
    thresholds, mass = a.cdf_envelope(certificates, 128, 4, 'y')
    np.testing.assert_array_equal(thresholds, [0., .1, .2, .5, 1.])
    np.testing.assert_array_equal(mass, [0., .1, .4, .6, 1.])
    _, larger = a.cdf_envelope(certificates, 128, 8, 'y')
    np.testing.assert_array_equal(larger, [0., .1, .6, 1.])
