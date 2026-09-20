import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "expA02_qi_vs_lstsq"
    / "plot_coefficients.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("expA02_plot_coefficients", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_panel_limit_clips_extreme_halo_coefficients_at_three_times_interior():
    module = _load_script()
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    qi = np.array([100.0, 1.0, -2.0, 1.5, -80.0])
    lstsq = np.array([50.0, -1.0, 1.0, 0.5, -40.0])

    assert module.panel_limit(centers, qi, lstsq) == 6.0


def test_panel_limit_includes_all_coefficients_when_below_interior_cap():
    module = _load_script()
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    qi = np.array([4.0, 1.0, -2.0, 1.5, -3.0])
    lstsq = np.array([2.0, -1.0, 1.0, 0.5, -2.0])

    assert np.isclose(module.panel_limit(centers, qi, lstsq), 4.2)
