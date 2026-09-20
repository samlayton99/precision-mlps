"""QI physics-solve teacher for expF11: u_QI(a) via the expF08 Darcy collocation
solver. Disk-cached so target generation is one-time."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))

import core
import darcy_problems as dp

CACHE = HERE / "_uqi_cache"
W_DEFAULT, LAM, SIGMA = 576, 0.25, 4.0


def _eval_grid(n):
    g = dp.grid_1d(n, cell_centered=True)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def _zero_bc():
    Pb = core.boundary_points_square(360)
    return [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]


def u_qi(a_grid, res, W=W_DEFAULT, lam=LAM, sigma=SIGMA):
    """QI Darcy solution for one coefficient field [res,res] -> [res,res]."""
    coeff = dp.DarcyCoefficient(np.asarray(a_grid, float), sigma_px=sigma,
                                cell_centered=True)
    model = core.solve_square(coeff.terms(), dp.DARCY_FORCING, _zero_bc(),
                              W, lam, seed=42)
    return core.eval_model(model, _eval_grid(res)).reshape(res, res)


def batch_u_qi(a_batch, res, tag, **kw):
    """[n,res,res] -> [n,res,res], cached to _uqi_cache/{tag}.npy."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}_res{res}_n{len(a_batch)}.npy"
    if path.exists():
        return np.load(path)
    out = np.stack([u_qi(a, res, **kw) for a in a_batch])
    np.save(path, out)
    return out
