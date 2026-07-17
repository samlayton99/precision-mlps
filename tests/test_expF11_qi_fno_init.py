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


import torch
import fno2d
import qi_codec as qc
import init_methods as im


def test_qi_spectral_init_changes_weights_and_runs():
    codec = qc.QICodec(W=256, lam=0.25)
    net = fno2d.FNO2d(width=16, modes=12, n_layers=3)
    before = net.specs[0].w1.detach().clone()
    im.qi_spectral_init(net, codec, res=64)
    assert not torch.allclose(before, net.specs[0].w1)     # init changed weights
    for res in (64, 32):                                   # still runs, incl low-res
        y = net(torch.randn(2, 1, res, res))
        assert y.shape == (2, 1, res, res)
