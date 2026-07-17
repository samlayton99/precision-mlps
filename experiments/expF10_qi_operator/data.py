"""Darcy dataset loading + area-average downsampling for expF10."""
from __future__ import annotations

import numpy as np

BASE = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax"


def _avg_downsample(field, res):
    """[n,H,W] -> [n,res,res] by block-mean over an index binning that tolerates
    non-integer ratios."""
    n, H, W = field.shape
    if H == res:
        return field
    yi = (np.linspace(0, H, res + 1)).astype(int)
    xi = (np.linspace(0, W, res + 1)).astype(int)
    out = np.empty((n, res, res), dtype=np.float64)
    for i in range(res):
        for j in range(res):
            out[:, i, j] = field[:, yi[i]:yi[i + 1], xi[j]:xi[j + 1]].mean((1, 2))
    return out


def load_darcy(split, n, res, source_res=421):
    """Returns (a, u) as [n,res,res] float64, downsampled from source_res."""
    d = np.load(f"{BASE}/darcy_{split}_{source_res}_jax.npz")
    a = np.asarray(d["x"][:n], dtype=np.float64)
    u = np.asarray(d["y"][:n], dtype=np.float64)
    return _avg_downsample(a, res), _avg_downsample(u, res)
