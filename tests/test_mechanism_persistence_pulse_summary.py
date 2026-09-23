"""Independent finite-amplitude contamination bookkeeping examples."""
import numpy as np

from experiments.expD34_readout_race import mechanism_persistence_pulse_summary as summary


def test_even_tracking_can_exceed_matched_force_while_odd_part_cancels():
    amplitudes = (0., .01, -.01, .005, -.005, .0025, -.0025)
    cases, values, pulses = [], [], []
    points = np.ones((7, 4))
    for amplitude in amplitudes:
        cases.append(dict(target='weak', seed=30, start=20000, source_index=0,
                          reference_baseline_index=0, amplitude=amplitude))
        contamination = 1.4*amplitude**2
        values.append(dict(F=np.array([2e-5, 0., 0., 0.]),
            R=np.array([1e-9+contamination, 0., 0., 0.]),
            ga=np.array([2e-5+1e-9+contamination]), z=np.array([1e-9+contamination, 0.]),
            coarse=np.array([amplitude**2, 0.])))
        pulses.append(dict(amplitude=amplitude, coarse_resolved=True, finite=True,
            intrinsic_k_change=2*amplitude,
            diagnostics=dict(k_pure=2*amplitude, k=2*amplitude-amplitude**2)))
    diagnostics = [dict(source_index=0, pulses=pulses,
                        direction_diagnostics=dict(intrinsic_k_derivative=2.))]
    rows, pairs = summary.matching_rows(points, cases, values, diagnostics)
    assert rows[1]['tracking_over_current_force'] > 7
    np.testing.assert_allclose(rows[1]['induced_tracking_over_baseline_force'], 7)
    assert rows[1]['force_squared_change_over_baseline_squared'] == 0
    assert rows[1]['slopes_unchanged']
    assert pairs[0]['tracking_odd_norm'] == 0
    np.testing.assert_allclose(pairs[0]['tracking_even_over_baseline'], 7)
    np.testing.assert_allclose(pairs[1]['tracking_even_over_baseline'], 7/4)
    np.testing.assert_allclose(pairs[0]['k_pure_central_derivative'], 2)


def test_zero_denominator_does_not_invent_a_relative_scale():
    assert summary.ratio(1., 0.) is None
    assert summary.ratio(0., 0.) is None
    assert summary.ratio(0., 1.) == 0
