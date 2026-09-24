"""Grouping and signed-rate interpretation must preserve scope and signs."""
import pytest

from experiments.expD34_readout_race.population_reinforcement_summary import (
    derive, static_groups, trajectories, trajectory_endpoints,
)


def row(role='static', age=20000, horizon=0):
    value = dict(role=role, panel='test', target='sine', seed='0', index='0', width=705,
                 start=age, horizon=horizon, arm='natural', reinforcement_F_norm=2.,
                 reinforcement_structural_curvature_rate_bound=100.,
                 reinforcement_directional_curvature_rate_bound=10.,
                 reinforcement_generated_rate=-8., reinforcement_target_rate=10.,
                 reinforcement_curvature_rate=2., reinforcement_log_force_rate=-1.)
    return derive(value)


def test_cancellation_and_positive_only_ratios():
    value = row()
    assert value['structural_to_directional'] == 10.
    assert value['generated_target_cancellation'] == pytest.approx(1-2/18)
    assert value['directional_to_positive_curvature'] == 5.
    assert value['structural_to_positive_net_growth'] != value['structural_to_positive_net_growth']


def test_static_group_does_not_mix_ages_or_trajectory_checkpoints():
    groups = static_groups([row(), row(age=600000), row(role='trajectory')])
    assert len(groups['W705_age20000']) == 1


def test_natural_endpoints_use_first_retained_not_assumed_zero():
    first, last = row(role='trajectory', horizon=10), row(role='trajectory', horizon=20000)
    last['F_norm'] = 6.
    paths = trajectories([last, first])
    endpoints = trajectory_endpoints(paths)
    assert endpoints[0]['first_retained_horizon'] == 10
    assert endpoints[0]['force_ratio_last_to_first'] == 3.
