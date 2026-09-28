from experiments.expD34_readout_race.population_frozen_summary import improvement_fraction
import pytest


def test_improvement_fraction_respects_denominator_and_sign():
    fraction, status = improvement_fraction(1., .2, .3)
    assert fraction == pytest.approx(.875) and status == 'positive_resolved_full_improvement'
    assert improvement_fraction(1., 1.+1e-4, .3)[1] == 'full_error_increased'
    assert improvement_fraction(1., 1.-1e-14, .3)[0] is None
    fraction, status = improvement_fraction(1., .5, 1.1)
    assert fraction < 0 and status == 'positive_resolved_full_improvement'
