"""Separate intervention achievement, learned crossings, and endpoint accuracy."""
from experiments.expD34_readout_race.population_dilation_findings import group_facts


def case(hit, error, **extra):
    return dict(target='fixture', seed=30, failed=False, completed_steps=20000, requested_steps=20000,
                relative_l2=error, initial_relative_l2=.2, relative_eval_l2=error*1.1,
                initial_relative_eval_l2=.21, initially_above_lambda025=2,
                learned_new_hits_lambda025=0, **{'first_hit_relative_l2_0.01': hit}, **extra)


def test_first_hits_remain_separate_from_endpoints_and_injection():
    first = case(0, .02)
    first['initial_relative_l2'] = .009
    first['initial_relative_eval_l2'] = .0099
    second = case(123, .009, relative_l2_minus_repaired=-.1)
    third = case(-1, .5); third['failed'] = True
    result = group_facts([first, second, third])
    assert result['attempted'] == 3 and result['complete'] == 2
    event = result['thresholds']['0.01']
    assert event['already_met_after_repair'] == 1
    assert event['newly_met_during_training'] == 1
    assert event['training_endpoint_meets'] == 1
    assert result['initially_above_lambda025_labels'] == 4
    assert result['learned_new_lambda025_labels'] == 0
