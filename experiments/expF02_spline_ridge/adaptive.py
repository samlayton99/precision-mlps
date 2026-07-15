"""Residual-guided knot insertion for the spline ridge basis.

insert_knots: project |residual| mass onto each ridge direction (Radon
binning), allocate new offsets across directions proportionally to mass,
place them at the highest-mass bin centers.
local_gammas: per-neuron gamma from local offset spacing within a direction,
gamma = lam / (0.875 * local_gap) — matches the global expF01 gamma
(lam / (2.8/sqrt(W))) on the uniform init grid.
"""
from __future__ import annotations

import numpy as np

from ridge_core import COLLAR_SQUARE

GAP_TO_HREF = 0.875  # (2.8/sqrt(W)) / (3.2/sqrt(W)) on the uniform grid


def _theta_groups(dirs):
    thetas = np.round(np.arctan2(dirs[:, 1], dirs[:, 0]), 9)
    return thetas, np.unique(thetas)


def local_gammas(dirs, offs, lam):
    thetas, uniq = _theta_groups(dirs)
    gammas = np.empty(len(offs))
    for th in uniq:
        idx = np.where(thetas == th)[0]
        order = idx[np.argsort(offs[idx])]
        t = offs[order]
        gaps = np.diff(t)
        g = np.empty(len(t))
        if len(t) == 1:
            g[:] = 2 * COLLAR_SQUARE
        else:
            g[0], g[-1] = gaps[0], gaps[-1]
            g[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
        gammas[order] = lam / (GAP_TO_HREF * np.maximum(g, 1e-6))
    return gammas


def insert_knots(dirs, offs, P_res, r_abs, n_new, collar=COLLAR_SQUARE, n_bins=48):
    """Return (new_dirs [k,2], new_offs [k]) with k == n_new."""
    thetas, uniq = _theta_groups(dirs)
    infos, masses = [], []
    for th in uniq:
        w = np.array([np.cos(th), np.sin(th)])
        s = P_res @ w
        hist, edges = np.histogram(s, bins=n_bins, range=(-collar, collar), weights=r_abs)
        infos.append((w, hist, edges))
        masses.append(hist.sum())
    masses = np.asarray(masses)
    alloc = np.floor(n_new * masses / masses.sum()).astype(int)
    short = n_new - alloc.sum()
    if short > 0:
        alloc[np.argsort(-masses)[:short]] += 1
    new_dirs, new_offs = [], []
    for (w, hist, edges), k in zip(infos, alloc):
        if k == 0:
            continue
        k = int(min(k, len(hist)))
        top = np.argsort(-hist)[:k]
        centers = 0.5 * (edges[top] + edges[top + 1])
        for c in centers:
            new_dirs.append(w)
            new_offs.append(c)
    return np.asarray(new_dirs), np.asarray(new_offs)
