import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver


def test_no_holdout_sine_reaches_fp64_floor():
    """Dense full-grid fit of sine at lambda=0.25, N=128 hits the fp64 floor,
    matching expG01's 3.6e-14 sanity number. N sets the geometry (center
    spacing + halo); the number of training samples is an independent knob and
    must exceed the geometry's effective DOF (~N) to reach the floor."""
    N, lam = 128, 0.25
    centers, gamma_vec = solver.geometry(N, lam)
    x = np.linspace(-1.0, 1.0, 400)
    f = lambda t: np.sin(2 * np.pi * t)
    v, bias, info = solver.fit(x, f(x), centers, gamma_vec)
    x_test = np.linspace(-1.0, 1.0, 257)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    assert solver.rel_l2(u_hat, f(x_test)) < 1e-12
