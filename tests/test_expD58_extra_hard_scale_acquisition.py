import numpy as np
from scipy import special
from experiments.expD56_scale_acquisition.model import midpoint_grid
from experiments.expD58_extra_hard_scale_acquisition.targets import TARGETS,target,special_probe_points
from experiments.expD58_extra_hard_scale_acquisition.run import load_config,cases


def test_suite_is_finite_and_normalized_on_required_grids():
    for name in TARGETS:
        train=target(name,midpoint_grid(1024))
        scale=np.sqrt(np.mean(train*train))
        assert np.isfinite(scale) and scale>0
        np.testing.assert_allclose(np.mean((train/scale)**2),1)
        for grid in (np.linspace(-1,1,65537),midpoint_grid(4096),special_probe_points()):
            assert np.all(np.isfinite(target(name,grid))),name
    assert len(cases(load_config()))==72


def test_singular_points_and_piecewise_definitions():
    np.testing.assert_array_equal(target("smooth_bump",np.array([-.35,.35,-1.,1.])),0)
    np.testing.assert_allclose(target("smooth_bump",np.array([0.])),np.exp(-1))
    assert target("oscillatory_cusp",np.array([0.]))[0]==0
    assert target("signed_sqrt_cusp",np.array([.137]))[0]==0
    np.testing.assert_array_equal(target("shifted_step",np.array([.137-1e-8,.137,.137+1e-8])),[0,1,1])
    np.testing.assert_allclose(target("gamma_shifted",np.array([-1.,1.])),special.gamma([.05,5.]))
    assert np.all(np.isin(target("square_wave",midpoint_grid(4096)),[-1.,0.,1.]))


def test_disjoint_grids_and_off_grid_jump_probes():
    train=midpoint_grid(1024);evaluation=midpoint_grid(4096)
    assert not np.intersect1d(train,evaluation).size
    assert not np.any(train==.137)
    probes=special_probe_points()
    assert np.any(probes==.137-1e-8) and np.any(probes==.137+1e-8)
