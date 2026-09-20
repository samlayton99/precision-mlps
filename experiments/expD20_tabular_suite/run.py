"""expD20 -- run the headroom sweep.

    uv run --with scikit-learn --extra dev python experiments/expD20_tabular_suite/run.py --incumbents
    uv run --with scikit-learn --extra dev python experiments/expD20_tabular_suite/run.py --candidates
    uv run --with scikit-learn --extra dev python experiments/expD20_tabular_suite/run.py --plot

Results append to results/checkpoint_D_optimizers/expD20_tabular_suite/data/*.jsonl.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, path):
    """Explicit-path import (the repo's `import run` collision rule). The module
    MUST be registered in sys.modules before exec: @dataclass resolves field
    types through sys.modules[cls.__module__] and fails otherwise."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EV = _load("expD20_eval", HERE / "evaluate.py")
DS = _load("expD20_data", HERE / "datasets.py")

OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD20_tabular_suite"
DATA = OUT / "data"
FIGS = OUT / "figures"
for d in (DATA, FIGS):
    d.mkdir(parents=True, exist_ok=True)


def run_incumbents(quick=False):
    res = []
    for name in DS.INCUMBENTS:
        try:
            Xtr, ytr, Xte, yte = DS.load_incumbent(name)
            r = EV.evaluate_task(name, Xtr, ytr, Xte, yte, quick=quick, note="incumbent")
            EV.print_row(r)
            res.append(r)
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    EV.save(res, DATA / "incumbents.jsonl")
    return res


def run_candidates(only=None, quick=False):
    res, failed = [], []
    for name, fn in DS.CANDIDATES.items():
        if only and name not in only:
            continue
        try:
            Xtr, ytr, Xte, yte = fn()
        except Exception as e:
            print(f"DOWNLOAD FAILED {name}: {type(e).__name__}: {str(e)[:120]}")
            failed.append((name, f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        try:
            note = "candidate" + (" near-noiseless" if name in DS.NEAR_NOISELESS else "")
            r = EV.evaluate_task(name, Xtr, ytr, Xte, yte, quick=quick, note=note)
            EV.print_row(r)
            res.append(r)
        except Exception as e:
            print(f"EVAL FAILED {name}: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            failed.append((name, f"eval: {type(e).__name__}"))
    # merge with anything already saved
    path = DATA / "candidates.jsonl"
    if path.exists() and only:
        old = [json.loads(l) for l in open(path)]
        done = {r.task for r in res}
        keep = [o for o in old if o["task"] not in done]
        with open(path, "w") as f:
            for o in keep:
                f.write(json.dumps(o) + "\n")
            for r in res:
                d = r.__dict__.copy()
                d.update(best_linear=r.best_linear, best_linear_or_poly=r.best_linear_or_poly,
                         best_nonlinear=r.best_nonlinear,
                         headroom_vs_linear=r.headroom_vs_linear,
                         headroom_vs_poly=r.headroom_vs_poly)
                f.write(json.dumps(d) + "\n")
    else:
        EV.save(res, path)
    if failed:
        (DATA / "failures.json").write_text(json.dumps(failed, indent=2))
    return res, failed


def plot():
    """Two panels, because the headline is a CONTRAST, not a ranking.

    Left: how much nonlinear headroom each task has (what a strong deep model
    or GBDT buys over the best linear/poly-2 model). Right: how much of that a
    ONE-HIDDEN-LAYER net -- the architecture the QI theory covers -- can
    actually reach, and whether width helps it at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for f, kind in ((DATA / "incumbents.jsonl", "incumbent"),
                    (DATA / "candidates.jsonl", "candidate")):
        if f.exists():
            for l in open(f):
                d = json.loads(l)
                d["kind"] = kind
                rows.append(d)
    one_layer = {}
    for f in (DATA / "one_layer_scaling.jsonl", DATA / "one_layer_candidates.jsonl"):
        if f.exists():
            for l in open(f):
                d = json.loads(l)
                e = {int(k): v for k, v in d["one_layer"].items()}
                one_layer[d["task"]] = dict(best=min(e.values()),
                                            width_gain=e[256] / e[4096])
    rows.sort(key=lambda d: d["headroom_vs_poly"])
    names = [d["task"] for d in rows]
    yy = np.arange(len(rows))
    colors = ["#d62728" if d["kind"] == "incumbent" else "#1f77b4" for d in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, max(5, 0.40 * len(rows))),
                                   sharey=True)

    axL.barh(yy, [d["headroom_vs_poly"] for d in rows], color=colors, alpha=0.85)
    axL.plot([d["headroom_vs_linear"] for d in rows], yy, "k.", ms=7)
    for i, d in enumerate(rows):
        if d["task"] in DS.NEAR_NOISELESS:
            axL.text(1.03, yy[i], "~noiseless", va="center", fontsize=6.5, color="#444")
    axL.axvline(1.0, color="k", lw=0.8)
    axL.axvline(3.0, color="green", ls="--", lw=1.2)
    axL.set_xscale("log")
    axL.set_xlabel("headroom available\n(best lin/poly-2) / (best deep or GBDT)")
    axL.set_yticks(yy)
    axL.set_yticklabels(names, fontsize=8)
    axL.grid(axis="x", alpha=0.3, which="both")

    hr1, wg = [], []
    for d in rows:
        ol = one_layer.get(d["task"])
        hr1.append(d["best_linear_or_poly"] / ol["best"] if ol else np.nan)
        wg.append(ol["width_gain"] if ol else np.nan)
    axR.barh(yy, hr1, color=colors, alpha=0.85)
    axR.plot(wg, yy, "kv", ms=5)
    axR.axvline(1.0, color="k", lw=0.8)
    axR.axvline(3.0, color="green", ls="--", lw=1.2)
    axR.set_xscale("log")
    axR.set_xlabel("headroom a 1-HIDDEN-LAYER net reaches\n"
                   "bars: (best lin/poly-2)/(best 1-layer);  triangles: width gain 256->4096")
    axR.grid(axis="x", alpha=0.3, which="both")

    handles = [plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=.85),
               plt.Rectangle((0, 0), 1, 1, color="#1f77b4", alpha=.85),
               plt.Line2D([], [], color="k", marker=".", ls="", ms=7),
               plt.Line2D([], [], color="k", marker="v", ls="", ms=5),
               plt.Line2D([], [], color="green", ls="--")]
    labels = ["incumbent (expF04 suite)", "candidate", "vs best linear only",
              "1-layer width gain 256->4096", "adoption bar (3x)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=8)
    fig.suptitle("expD20: nonlinear headroom, and how much of it one hidden layer can reach",
                 y=0.945, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGS / "expD20_headroom.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if "--incumbents" in sys.argv:
        run_incumbents(quick=quick)
    if "--candidates" in sys.argv:
        only = None
        for a in sys.argv:
            if a.startswith("--only="):
                only = set(a.split("=", 1)[1].split(","))
        run_candidates(only=only, quick=quick)
    if "--plot" in sys.argv:
        plot()
