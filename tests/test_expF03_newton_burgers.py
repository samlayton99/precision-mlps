import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF02_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF03_newton_burgers"))


def test_burgers_manufactured_fd_verified():
    import problems as bp
    bp.verify_all(nu=0.1)
    bp.verify_all(nu=0.01)
