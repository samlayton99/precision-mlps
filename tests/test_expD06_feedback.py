from experiments.expD06_fixed_center_scales.feedback import decide


def test_policy_requires_two_stalled_windows_and_both_span_cutoffs():
    assert decide([1, .9, .89], [.999, .999], 0, 220000, 160000) == "hold_improving"
    assert decide([1, .99, .98], [.94, .999], 0, 220000, 160000) == "hold_out_of_span"
    assert decide([1, .99, .98], [.96, .999], 0, 220000, 160000) == "slow_geometry"
    assert decide([1, .99], [.99, .99], 0, 200000, 160000) == "hold_insufficient_history"


def test_readout_intervention_needs_cooldown_and_counterfactual():
    args = ([1, 1, 1], [.99, .99], 1)
    assert decide(*args, 260000, 240000) == "hold_cooldown"
    assert decide(*args, 280000, 240000) == "test_readout"
    assert decide(*args, 280000, 240000, {"mean_delta_mse": -1e-8, "decrease_fraction": .96}) == "increase_readout"
    assert decide(*args, 280000, 240000, {"mean_delta_mse": -1e-8, "decrease_fraction": .94}) == "reject_readout"
    assert decide(*args, 280000, 240000, {"mean_delta_mse": 1e-8, "decrease_fraction": .96}) == "reject_readout"
    assert decide([1, 1, 1], [.99, .99], 2, 400000, 300000) == "hold_limit"
