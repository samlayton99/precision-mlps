"""Geometry presets. Generated ones are built at the current neuron count; the JSON ones
are exact arrays from repo runs or formulas (see each file's 'source' field;
presets/_make_presets.py regenerated them from the repo on 2026-09-26)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine import (Params, canonical, uniform_geometry, xavier_geometry,
                    xavier_rescaled_geometry)

HERE = Path(__file__).resolve().parent / "presets"

# (key, group, label, kind) -- kind 'gen' is built at the current count, 'json' is fixed
CATALOG = [
    ("uniform_sqrt", "Construction", "Uniform, lambda*, halo max(10, ceil(sqrt N))  [current count]", "gen"),
    ("uniform_none", "Construction", "Uniform, lambda*, no halo  [current count]", "gen"),
    ("qi_uniform", "Construction", "QI standard: uniform + default halo (N=64, W=205)", "json"),
    ("xavier_standard", "Xavier and trained", "Xavier (Glorot uniform), c=-b/w, g=|w|  [current count]", "gen"),
    ("xavier_rescaled", "Xavier and trained", "Xavier slopes rescaled to median lambda* on uniform centers (expD05)  [current count]", "gen"),
    ("xavier_d06_start", "Xavier and trained", "expD06 Xavier start: QI centers, Xavier slopes (N=512, W=559)", "json"),
    ("xavier_adam_gn_best", "Xavier and trained", "Xavier -> Adam 5.3M -> Gauss-Newton: best trained sine, L2RE 3e-11 (N=512)", "json"),
    ("xavier_ssbroyden_trained", "Xavier and trained", "Xavier -> SSBroyden 8k updates: L2RE 5e-10 (N=512)", "json"),
    ("xavier_newton_collapsed", "Xavier and trained", "Xavier -> Newton 100k: collapsed slopes, MSE 6e-10 (N=512)", "json"),
    ("graded_halfgauss", "Graded meshes (expH02)", "Half-Gaussian graded, g=lambda*/h_k: reaches floor", "json"),
    ("graded_bimodal", "Graded meshes (expH02)", "Bimodal clusters, 10x spacing range: floor at 4x width", "json"),
    ("graded_beta", "Graded meshes (expH02)", "Beta(2,5) graded: end-gap jump, stalls (failure case)", "json"),
    ("chebyshev_local_lambda", "Graded meshes (expH02)", "Chebyshev-Lobatto centers, g=lambda*/h_k: predicted stall", "json"),
    ("cascade_3band", "Irregular", "Three-band cascade lambda=0.25/0.10/0.05 (expG04)", "json"),
    ("random_centers", "Irregular", "Random i.i.d. centers, constant g (expC04): plateaus", "json"),
    ("clustered_centers", "Irregular", "Clustered centers, constant g (expC04): plateaus", "json"),
    ("spike_monitor_mesh", "Irregular", "Slope-monitor mesh for a spike at x=0.3 (expH04 machinery)", "json"),
]


def catalog():
    out = []
    for key, group, label, kind in CATALOG:
        desc, target = "", None
        if kind == "json":
            d = _load(key)
            desc, target = d.get("description", ""), d.get("target")
        else:
            desc = {
                "uniform_sqrt": "x_n = a + n h, h=(b-a)/N, n=-R..N+R, R=max(10, ceil(sqrt N)); g = lambda*/h.",
                "uniform_none": "N+1 uniform centers over the domain, g = lambda*/h.",
                "xavier_standard": "w,b ~ U(+-sqrt(6/(1+W))) (expD16/expD02 build_model); c=-b/w heavy-tailed, lambda ~ 1e-3.",
                "xavier_rescaled": "expD05 scale_center_spread_xavier: uniform centers, |w| Xavier, scaled so median g = lambda*/h.",
            }[key]
        out.append({"key": key, "group": group, "label": label, "kind": kind,
                    "description": desc, "target": target})
    return out


def _load(key):
    src = "xavier_adam_gn_best" if key == "xavier_d06_start" else key
    return json.loads((HERE / f"{src}.json").read_text())


TARGET_EXPR = {"sqrt(2) sin(2 pi x)": "sqrt(2)*sin(2*pi*x)",
               "exp(-((x-0.3)/0.03)^2)": "exp(-((x-0.3)/0.03)**2)"}


def build(key, W, domain, lambda_star, seed=0):
    """Returns (Params canonicalized so g >= 0, has_readout, target_expr_or_None)."""
    if key == "uniform_sqrt":
        return uniform_geometry(W, domain, "sqrt", lambda_star), False, None
    if key == "uniform_none":
        return uniform_geometry(W, domain, "none", lambda_star), False, None
    if key == "xavier_standard":
        return canonical(xavier_geometry(W, seed)), False, None
    if key == "xavier_rescaled":
        return xavier_rescaled_geometry(W, domain, "sqrt", lambda_star, seed), False, None
    d = _load(key)
    c = np.asarray(d["centers"], dtype=np.float64)
    target = d.get("target")
    if key == "xavier_d06_start":
        g = np.asarray(d["init_gammas"], dtype=np.float64)
        return canonical(Params(c, g, np.zeros(c.size), 0.0)), False, TARGET_EXPR.get(target, target)
    g = np.asarray(d["gammas"], dtype=np.float64)
    if "readout_weights" in d:
        P = Params(c, g, np.asarray(d["readout_weights"], dtype=np.float64), float(d["readout_bias"]))
        return canonical(P), True, TARGET_EXPR.get(target, target)
    if key == "spike_monitor_mesh":
        target = "exp(-((x-0.3)/0.03)^2)"
    return canonical(Params(c, g, np.zeros(c.size), 0.0)), False, TARGET_EXPR.get(target, target)
