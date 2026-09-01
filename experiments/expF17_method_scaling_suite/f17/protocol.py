"""expF17 protocol: the ladder, the knob table, tuning, and the work queue.

Tuning (SPEC 17.1/17.5): per (task, method, width), sweep the method's single
knob over its declared 3-point grid at seed 0 in the ORACLE regime, select
argmin rel L2 on the scored metric (Sam: "this is pinns... it is fine to tune
on everything"), FREEZE, then run the seeds. Equal declared effort is the sole
fairness mechanism; the chosen configuration is recorded on every cell.
The dynamic regime and the poly variant INHERIT the frozen config (same
dictionary both regimes; tuning them separately would multiply effort).

J/M is BAKED at the exact-identity split J = n_ax, M = n_ax + 2 (SPEC 17.6,
adopted on Sam's build go-ahead): both QI arms carry exactly one knob (halo).
lambda is BAKED at 0.25 (SPEC 17.3). ELM's R is NOT a lambda and stays swept.
"""
from __future__ import annotations

import numpy as np

LADDER = [128, 256, 512, 1024, 2048]
EXTRA_LADDER = {"convection_c40": [4096], "convection_c80": [4096],
                "wave": [4096, 8192], "darcy_man": [4096]}
SEEDS = [0, 1, 2]

SCALING_METHODS = ["qi_radon", "qi_tensor", "elm"]
COMPARATORS = ["spectral", "bwler", "rbf_imq", "rbf_phs"]
ALL_METHODS = SCALING_METHODS + COMPARATORS

TASK_ORDER = ["convection_c40", "convection_c80", "reaction", "wave",
              "burgers", "poisson_man", "darcy_man", "darcy_orig"]


def default_halo(n_ax):
    return max(1, min(int(round(n_ax / 8)), 8))


def halo_candidates(n_ax, geometry):
    """{h0-2, h0, h0+2}, clamped to >=1 and to a valid interior (16.6 clamp rule).
    radon interior = n_ax + 2 - 2h; tensor interior = n_ax - 2h; both need >= 3."""
    h0 = default_halo(n_ax)
    h_max = ((n_ax + 2 - 3) // 2) if geometry == "qi_radon" else ((n_ax - 3) // 2)
    cand = sorted({max(1, min(h, h_max)) for h in (h0 - 2, h0, h0 + 2)})
    return cand


def knob_grid(method, n_ax):
    """-> (knob_name, [config dicts]) for the declared 3-point check (17.5)."""
    if method in ("qi_radon", "qi_tensor"):
        return "halo", [{"halo": h} for h in halo_candidates(n_ax, method)]
    if method == "elm":
        return "R", [{"R": r} for r in (1.0, 2.0, 4.0)]
    if method == "rbf_imq":
        return "epsh", [{"epsh": e} for e in (0.5, 1.0, 2.0)]
    return None, [{}]


def cell_key(task, method, C, seed, regime, variant="base"):
    return f"{task}|{method}|{C}|{seed}|{regime}|{variant}"


# --- the dysts (1-D) arm ----------------------------------------------------

LADDER_1D = [48, 96, 128, 192, 256, 384]
LADDER_1D_BY_SYSTEM = {"MacArthur": [48, 96, 128, 192, 256],
                       "InteriorSquirmer": [48, 96, 128, 192, 256, 384, 512, 768]}
SCALING_1D = ["qi_grid", "elm"]
COMPARATORS_1D = ["spectral", "bwler", "rbf_imq", "rbf_phs"]
METHODS_1D = SCALING_1D + COMPARATORS_1D


def ladder_1d(system_name):
    return LADDER_1D_BY_SYSTEM.get(system_name, LADDER_1D)


def knob_grid_1d(method, W):
    if method == "qi_grid":
        n_c = W - 1
        h0 = max(1, min(int(round(n_c / 8)), 8))
        h_max = (n_c - 3) // 2
        cand = sorted({max(1, min(h, h_max)) for h in (h0 - 2, h0, h0 + 2)})
        return "halo", [{"halo": h} for h in cand]
    if method == "elm":
        return "R", [{"R": r} for r in (1.0, 2.0, 4.0)]
    if method == "rbf_imq":
        return "epsh", [{"epsh": e} for e in (0.5, 1.0, 2.0)]
    return None, [{}]


def build_queue(include_extras=False):
    """Plot-resolution order (Sam, 2026-09-01): plot 1a fully resolves first
    (oracle regime, scaling methods, all 16 tasks), then 1b (dynamic, scaling),
    then plot 2 (comparators, oracle then dynamic), then plot 3 (poly).
    Within a phase: task (outer) -> method -> width -> seed (inner), so a
    single line fills to completion, then the next method, then the next task."""
    from .tasks import TASKS
    from .tasks1d import SYSTEMS_1D
    q = []

    def add_2d(regime, variant, methods, ladder=None):
        for task in TASK_ORDER:
            t = TASKS[task]
            if t.get("blocked") or (regime == "dynamic" and not t["dynamic"]):
                continue
            for method in methods:
                for C in (ladder or LADDER):
                    for seed in SEEDS:
                        q.append(dict(task=task, method=method, C=C, seed=seed,
                                      regime=regime, variant=variant))

    def add_1d(regime, methods):
        for name in SYSTEMS_1D:
            for method in methods:
                for W in ladder_1d(name):
                    for seed in SEEDS:
                        q.append(dict(task=f"dysts_{name}", method=method, C=W,
                                      seed=seed, regime=regime, variant="base"))

    # phase A -- plot 1a: oracle, scaling methods
    add_2d("oracle", "base", SCALING_METHODS)
    add_1d("oracle", SCALING_1D)
    # phase B -- plot 1b: dynamic, scaling methods
    add_2d("dynamic", "base", SCALING_METHODS)
    add_1d("dynamic", SCALING_1D)
    # phase C -- plot 2: comparators, oracle then dynamic
    add_2d("oracle", "base", COMPARATORS)
    add_1d("oracle", COMPARATORS_1D)
    add_2d("dynamic", "base", COMPARATORS)
    add_1d("dynamic", COMPARATORS_1D)
    # phase D -- plot 3: the polynomial ablation (2-D QI arms)
    add_2d("oracle", "poly", ["qi_radon", "qi_tensor"])
    add_2d("dynamic", "poly", ["qi_radon", "qi_tensor"])
    if include_extras:
        for task, ladder in EXTRA_LADDER.items():
            for regime in ("oracle", "dynamic"):
                for method in SCALING_METHODS:
                    for C in ladder:
                        for seed in SEEDS:
                            q.append(dict(task=task, method=method, C=C,
                                          seed=seed, regime=regime,
                                          variant="base"))
    return q


def geomean_gsd(vals):
    v = np.asarray([max(x, 1e-300) for x in vals], dtype=np.float64)
    lg = np.log(v)
    g = float(np.exp(lg.mean()))
    sg = float(np.exp(lg.std())) if len(v) > 1 else 1.0
    return g, sg
