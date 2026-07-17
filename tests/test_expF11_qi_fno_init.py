import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for d in ("expF08_darcy_sweep", "expF10_qi_operator", "expF11_qi_fno_init"):
    sys.path.append(str(REPO_ROOT / "experiments" / d))

import qi_solve as qs
import data as dd


def test_u_qi_is_a_sane_solution():
    a, u = dd.load_darcy("test", n=2, res=32)
    uq = qs.u_qi(a[0], res=32)
    assert uq.shape == (32, 32) and np.isfinite(uq).all()
    assert np.max(np.abs(uq[0, :])) < 1e-2         # ~Dirichlet (cell-centered)
    rel = np.linalg.norm(uq - u[0]) / np.linalg.norm(u[0])
    assert rel < 5e-2                              # a real (approx) solution
