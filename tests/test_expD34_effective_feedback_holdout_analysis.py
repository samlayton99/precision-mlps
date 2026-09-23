"""Scientific baseline and aggregation semantics for held-out assessment."""
import json

import numpy as np

from experiments.expD34_readout_race import effective_feedback_holdout_analysis as audit


def test_motion_skill_uses_change_not_large_residual_level():
    actual_change = np.array([1e-7, -2e-7])
    assert audit.score(np.zeros(2), actual_change)['skill'] == 0
    assert audit.score(actual_change, actual_change)['skill'] == 1
    assert audit.score(-actual_change, actual_change)['skill'] == -3
    unresolved = audit.score(np.ones(2), np.zeros(2))
    assert unresolved['zero_actual_change']
    assert np.isnan(unresolved['relative_error'])
    assert np.isnan(unresolved['skill'])


def test_family_macro_does_not_weight_function_by_number_of_repeated_states():
    rows = [dict(target='a', family='F', start=k, offset=1000,
                 arm='joint', model='affine', metric=1.) for k in range(5)]
    rows += [dict(target='b', family='F', start=0, offset=1000,
                  arm='joint', model='affine', metric=3.)]
    result = audit.summaries(rows, ['metric'])
    macro = result['family_macro'][0]
    assert macro['per_function_statistic'] == 'mean'
    assert macro['case_count'] == 2
    assert macro['metrics']['metric']['mean'] == 2.
    assert macro['metrics']['metric']['min'] == 1.
    assert macro['metrics']['metric']['max'] == 3.


def test_unresolved_values_are_counted_and_json_has_no_nan():
    result = audit.stats([1., np.nan, np.inf, -np.inf])
    assert result['finite_count'] == 1
    assert result['unresolved_or_nonfinite_count'] == 3
    assert result['mean'] == 1.
    empty = audit.stats([np.nan])
    assert empty['mean'] is None
    json.dumps(empty, allow_nan=False)
