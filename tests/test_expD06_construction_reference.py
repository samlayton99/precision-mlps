import numpy as np

from experiments.expD06_fixed_center_scales.construction_reference import sine_coefficients


def test_corrected_sine_reference_and_precision_refinement():
    low = sine_coefficients(dps=50, quadrature_degree=7)
    high = sine_coefficients(dps=80, quadrature_degree=9)
    np.testing.assert_allclose(low["c"], high["c"], rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(high["c"][1:], high["c"][:0:-1], atol=1e-15)
    assert abs(high["c"][0]) < 1e-15
    x = -1 + (np.arange(4096)+.5)*2/4096
    a = np.column_stack([np.ones(len(x)), np.tanh((x[:, None]-high["centers"])*high["gamma"])])
    y = np.sqrt(2)*np.sin(2*np.pi*x)
    assert np.max(np.abs(a@high["c"]-y)) < 2e-13
    assert np.max(np.abs(a@high["baseline_c"]-y)) > 1e-8
    # Interior density in A.2 is explicit for this sine; correction touches only
    # the twelve outer slots on each side, not the other 535 coefficients.
    h = high["h"]
    d = np.pi/(2*high["gamma"][0])
    expected = h*np.sqrt(2)*np.cos(2*np.pi*high["centers"])*np.sinh(2*np.pi*d)/(2*d)
    np.testing.assert_allclose(high["c"][13:-12], expected[12:-12], atol=1e-16)
