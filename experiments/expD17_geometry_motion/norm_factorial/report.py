"""expD17/norm_factorial -- the tables the writeup is built from.

Reads the grid's JSONL and prints, in order:
  1. movement:   relative and absolute first-layer drift per arm, per class
  2. performance: final error per arm, per class, plus a rank summary
  3. probes:     pre/post lstsq probe -> geometry score, readout score
  4. pinn:       recovered PDE parameter, absolute error, correct decimals
  5. tabular:    all seven tasks, variance-normalized test error
  6. colnorm:    the column-scale pathology and what each arm does to it
  7. dead:       dead-neuron fractions split by preactivation sign class

Usage:
    uv run --with scikit-learn --with pandas --extra dev python \\
        experiments/expD17_geometry_motion/norm_factorial/report.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
_HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NF = _load("expD17n_run", _HERE / "run.py")
ARMS, CLASSES, WIDTHS, PROBLEMS = NF.ARMS, NF.CLASSES, NF.WIDTHS, NF.PROBLEMS
LBL = NF.ARM_LABEL


def all_rows():
    rows = []
    for p in sorted(NF.DATA_DIR.glob("*_w*_*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def gmean(v):
    v = [x for x in v if x is not None and x > 0 and math.isfinite(x)]
    return float(np.exp(np.mean(np.log(v)))) if v else float("nan")


def section(t):
    print(f"\n{'='*96}\n{t}\n{'='*96}")


def q1_movement(rows):
    section("Q1  PARAMETER MOVEMENT -- first-layer geometry drift at the end of training")
    print("geometric mean over the class's problems and both widths\n")
    print(f"{'arm':28s} " + " ".join(f"{NF.CLASS_LABEL[c]:>17s}" for c in CLASSES))
    for tag, key in (("relative  ||g-g0||/||g0||", "rel_drift_end"),
                     ("absolute  ||g-g0||", "abs_drift_end")):
        print(f"\n-- {tag}")
        for arm in ARMS:
            cells = [gmean([r[key] for r in sel(rows, arm=arm, **{"class": c})])
                     for c in CLASSES]
            print(f"{LBL[arm]:28s} " + " ".join(f"{v:17.3e}" for v in cells))
    print("\n-- QI / standard drift ratio (same activation+norm, same cells)")
    print(f"{'family':28s} " + " ".join(f"{NF.CLASS_LABEL[c]:>17s}" for c in CLASSES))
    for fam in ("tanh_none", "gelu_none", "gelu_rmsc"):
        out = []
        for c in CLASSES:
            q = gmean([r["rel_drift_end"] for r in sel(rows, arm=f"{fam}_qi", **{"class": c})])
            s = gmean([r["rel_drift_end"] for r in sel(rows, arm=f"{fam}_std", **{"class": c})])
            out.append(q / s if s and math.isfinite(q / s) else float("nan"))
        print(f"{fam+' QI/std':28s} " + " ".join(f"{v:17.3f}" for v in out))


def q2_performance(rows):
    section("Q2  PERFORMANCE -- final eval error (geometric mean over problems x widths)")
    print("tabular = variance-normalized test error (mean-predictor = 1.0); "
          "PINN = field rel L2\n")
    print(f"{'arm':28s} " + " ".join(f"{NF.CLASS_LABEL[c]:>17s}" for c in CLASSES))
    for arm in ARMS:
        cells = [gmean([r["final_err"] for r in sel(rows, arm=arm, **{"class": c})])
                 for c in CLASSES]
        print(f"{LBL[arm]:28s} " + " ".join(f"{v:17.3e}" for v in cells))

    print("\n-- rank per cell (1 = best of the six arms), mean and worst")
    ranks = {a: [] for a in ARMS}
    for cls in CLASSES:
        for prob in PROBLEMS[cls]:
            for w in WIDTHS[cls]:
                cell = {r["arm"]: r["final_err"]
                        for r in sel(rows, **{"class": cls}, problem=prob, width=w)}
                if len(cell) < len(ARMS):
                    continue
                order = sorted(cell, key=lambda a: cell[a])
                for k, a in enumerate(order):
                    ranks[a].append(k + 1)
    print(f"{'arm':28s} {'mean rank':>10s} {'worst':>7s} {'#best':>7s} {'n cells':>8s}")
    for a in ARMS:
        v = ranks[a]
        if not v:
            continue
        print(f"{LBL[a]:28s} {np.mean(v):10.2f} {max(v):7d} "
              f"{sum(1 for x in v if x == 1):7d} {len(v):8d}")

    print("\n-- head-to-head: does rms_center change the GELU ranking? "
          "(ratio gelu_none / gelu_rmsc, >1 means rms_center better)")
    print(f"{'cell':44s} {'std init':>12s} {'QI init':>12s}")
    for cls in CLASSES:
        for prob in PROBLEMS[cls]:
            for w in WIDTHS[cls]:
                got = {r["arm"]: r["final_err"]
                       for r in sel(rows, **{"class": cls}, problem=prob, width=w)}
                if len(got) < len(ARMS):
                    continue
                rs = got["gelu_none_std"] / got["gelu_rmsc_std"]
                rq = got["gelu_none_qi"] / got["gelu_rmsc_qi"]
                print(f"{cls+'/'+prob+'/w'+str(w):44s} {rs:12.3f} {rq:12.3f}")


def q3_probes(rows):
    section("Q3  PROBES -- geometry score (post/pre) and readout score (final/post)")
    print("post/pre  < 1 : training improved the geometry;  > 1 : damaged it")
    print("final/post> 1 : Adam left readout accuracy on the table\n")
    print(f"{'arm':28s} " + " ".join(f"{NF.CLASS_LABEL[c]:>17s}" for c in CLASSES))
    for tag, fn in (("geometry post/pre", lambda r: r["post_probe"] / r["pre_probe"]),
                    ("readout  final/post", lambda r: r["final_err"] / r["post_probe"])):
        print(f"\n-- {tag}")
        for arm in ARMS:
            cells = [gmean([fn(r) for r in sel(rows, arm=arm, **{"class": c})])
                     for c in CLASSES]
            print(f"{LBL[arm]:28s} " + " ".join(f"{v:17.3f}" for v in cells))


def q4_pinn(rows):
    section("Q4  INVERSE PINN -- recovered PDE parameter (absolute accuracy)")
    print("decimals = -log10(|p_hat - p| / |p|); plain Adam, machine precision "
          "not expected\n")
    print(f"{'problem':14s} {'W':>5s} {'arm':28s} {'true':>8s} {'recovered':>13s} "
          f"{'abs err':>11s} {'dec':>6s} {'field err':>11s}")
    for prob in PROBLEMS["pinn_inverse"]:
        for w in WIDTHS["pinn_inverse"]:
            for arm in ARMS:
                r = next(iter(sel(rows, **{"class": "pinn_inverse"},
                                 problem=prob, width=w, arm=arm)), None)
                if r is None or "param_final" not in r:
                    continue
                pt, pf = r["param_true"], r["param_final"]
                ae = abs(pf - pt)
                dec = -math.log10(ae / abs(pt)) if ae > 0 else 99.0
                print(f"{prob:14s} {w:5d} {LBL[arm]:28s} {pt:8.4g} {pf:13.8g} "
                      f"{ae:11.2e} {dec:6.2f} {r['final_err']:11.2e}")


def q5_tabular(rows):
    section("Q5  TABULAR (2 hidden layers, layer-1 geometry) -- variance-normalized test error")
    print("1.0 = a mean-predictor. naval is the precision target, the rest are "
          "parity tasks.\n")
    print(f"{'task':16s} {'W':>5s} " + " ".join(f"{a.replace('_none','').replace('_rmsc','.rc'):>14s}"
                                               for a in ARMS))
    for prob in PROBLEMS["tabular"]:
        for w in WIDTHS["tabular"]:
            got = {r["arm"]: r["final_err"]
                   for r in sel(rows, **{"class": "tabular"}, problem=prob, width=w)}
            if not got:
                continue
            best = min(got.values())
            cells = " ".join(f"{got.get(a, float('nan')):14.4f}"
                             + ("*" if got.get(a) == best else " ")
                             for a in ARMS)
            print(f"{prob:16s} {w:5d} {cells}")
    print("\n-- QI vs standard init, same activation+norm (ratio std/QI, >1 = QI better)")
    print(f"{'task':16s} " + " ".join(f"{f:>14s}" for f in
                                      ("tanh_none", "gelu_none", "gelu_rmsc")))
    for prob in PROBLEMS["tabular"]:
        out = []
        for fam in ("tanh_none", "gelu_none", "gelu_rmsc"):
            q = gmean([r["final_err"] for r in sel(rows, **{"class": "tabular"},
                                                   problem=prob, arm=f"{fam}_qi")])
            s = gmean([r["final_err"] for r in sel(rows, **{"class": "tabular"},
                                                   problem=prob, arm=f"{fam}_std")])
            out.append(s / q if q else float("nan"))
        print(f"{prob:16s} " + " ".join(f"{v:14.3f}" for v in out))


def q6_colnorm(rows):
    section("Q6  COLUMN-SCALE PATHOLOGY -- max/min column RMS of the QI layer (live columns)")
    print(f"{'arm':28s} " + " ".join(f"{NF.CLASS_LABEL[c]:>17s}" for c in CLASSES))
    for arm in ARMS:
        cells = [gmean([r["colnorm_init"]["max_over_min_live"]
                        for r in sel(rows, arm=arm, **{"class": c})]) for c in CLASSES]
        print(f"{LBL[arm]:28s} " + " ".join(f"{v:17.3e}" for v in cells))


def q7_dead(rows):
    section("Q7  DEAD NEURONS -- fraction receiving no Adam update, by preactivation sign")
    print("'neg' = preactivation always negative (tanh saturates, gelu -> 0); "
          "'pos' = always positive (tanh saturates, gelu -> identity)\n")
    print(f"{'arm':28s} {'class':16s} {'dead%':>7s} {'dead|neg%':>10s} "
          f"{'dead|pos%':>10s} {'zero-col%':>10s}")
    for arm in ARMS:
        for cls in CLASSES:
            rs = sel(rows, arm=arm, **{"class": cls})
            if not rs:
                continue
            d = [NF.WS.dead_stats(r) for r in rs]
            d = [x for x in d if x]
            if not d:
                continue
            f = lambda k: 100 * float(np.nanmean([x.get(k, np.nan) for x in d]))
            print(f"{LBL[arm]:28s} {NF.CLASS_LABEL[cls]:16s} {f('dead_run_frac'):7.1f} "
                  f"{f('dead_of_neg'):10.1f} {f('dead_of_pos'):10.1f} "
                  f"{100*np.mean([r['colnorm_init']['frac_zero'] for r in rs]):10.1f}")


if __name__ == "__main__":
    rows = all_rows()
    print(f"loaded {len(rows)} runs from {NF.DATA_DIR}")
    have = {(r["class"], r["problem"], r["width"]) for r in rows}
    want = {(c, p, w) for c in CLASSES for p in PROBLEMS[c] for w in WIDTHS[c]}
    if want - have:
        print(f"MISSING {len(want-have)} cells: {sorted(want-have)[:8]}")
    q1_movement(rows)
    q2_performance(rows)
    q3_probes(rows)
    q4_pinn(rows)
    q5_tabular(rows)
    q6_colnorm(rows)
    q7_dead(rows)
