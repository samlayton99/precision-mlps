"""Selection must recover known decay rates without fitting a numerical floor."""
import numpy as np
import pytest

from experiments.expC09_bandwidth_figures.refinement_rates import select_fit


def test_exponential_middle_between_two_plateaus():
    w = np.r_[np.arange(50, 385, 2), np.arange(512, 1025, 64)]
    log_error = np.clip(12-.15*w, -13., 0.)
    fit = select_fit(w, 10**log_error, "semilog")
    assert fit["status"] == "fitted"
    assert fit["slope"] == pytest.approx(-.15, rel=.03)
    assert fit["width_start"] >= 76
    assert fit["width_stop"] <= 160
    assert fit["points"] >= 20
    assert fit["r2"] >= .995


def test_power_law_middle_between_two_plateaus():
    w = np.r_[np.arange(50, 385, 2), np.arange(512, 1025, 64)]
    log_error = np.clip(-20*np.log10(w/80), -13., 0.)
    fit = select_fit(w, 10**log_error, "loglog")
    assert fit["status"] == "fitted"
    assert fit["slope"] == pytest.approx(-20, rel=.03)
    assert fit["width_start"] >= 76
    assert fit["width_stop"] < 320


def test_flat_curve_does_not_get_a_decay_fit():
    w = np.r_[np.arange(50, 385, 2), np.arange(512, 1025, 64)]
    assert select_fit(w, np.full(w.size, 1e-13), "loglog")["status"] == "no_qualified_segment"


def test_selected_segment_does_not_silently_drop_outliers():
    w = np.r_[np.arange(50, 385, 2), np.arange(512, 1025, 64)]
    y = np.clip(12-.15*w, -13., 0.)
    y[w == 120] += 2
    fit = select_fit(w, 10**y, "semilog")
    assert fit["status"] == "fitted"
    assert not fit["width_start"] <= 120 <= fit["width_stop"]
    assert fit["selected_widths"] == list(range(fit["width_start"], fit["width_stop"]+1, 2))
