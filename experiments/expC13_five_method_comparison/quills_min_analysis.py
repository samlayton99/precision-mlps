"""Fit the theory's min model to the QUILLS grid of quills_min_test.py and plot it.

Model: log10 E(W, p) = max(a_W, c + k log10 W - p log10 2): a width floor a_W (the approximation error of
the width, constant in p) and a precision limit C W^k 2^-p (u poly(W), as in the construction and
evaluation bound). Fitted per target and lambda choice ('rule', each fixed lambda, and 'best': the
smallest error over the four lambda choices at each (W, p)). Residuals are reported over p >= 12 and
E < 0.1 (outside the low-precision breakdown).

    .venv/bin/python experiments/expC13_five_method_comparison/quills_min_analysis.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC13_five_method_comparison import common as C  # noqa: E402

L2 = np.log10(2.0)
DATA = C.OUT / "data"


def grid(rows, target, lam):
    E = {}
    for r in rows:
        if r["target"] != target or r.get("status") != "ok" or not np.isfinite(r["rel_l2"]):
            continue
        if lam == "best" or r["lambda"] == lam:
            k = (r["width"], r["p"])
            E[k] = min(E.get(k, np.inf), r["rel_l2"])
    return {k: np.log10(v) for k, v in E.items()}


def fit(logE):
    Ws = sorted({w for w, _ in logE})
    pts = [(w, p, v) for (w, p), v in logE.items() if p >= 12 and v < -1]
    a = {w: min(v for ww, p, v in pts if ww == w) for w in Ws if any(ww == w for ww, _, _ in pts)}
    c, k = 3.0, 0.0
    for _ in range(6):
        S = [(w, p, v) for w, p, v in pts if v > a[w] + 0.5]          # clearly above the width floor
        X = np.array([[1.0, np.log10(w)] for w, p, v in S])
        y = np.array([v + p * L2 for w, p, v in S])
        c, k = np.linalg.lstsq(X, y, rcond=None)[0]
        for w in a:
            flo = [v for ww, p, v in pts if ww == w and c + k * np.log10(w) - p * L2 < v - 0.5]
            a[w] = float(np.median(flo)) if flo else min(v for ww, p, v in pts if ww == w)
    res = np.array([v - max(a[w], c + k * np.log10(w) - p * L2) for w, p, v in pts])
    return {"c": float(c), "k": float(k), "floor": a, "residuals": res}


def main():
    rows = [json.loads(s) for s in (DATA / "quills_min_test.jsonl").read_text().splitlines()]
    targets = [t for t in ("chirp", "runge", "exp") if any(r["target"] == t for r in rows)]
    summary = []
    for target in targets:
        for lam in ["rule", 0.25, 0.35, 0.5, "best"]:
            logE = grid(rows, target, lam)
            if not logE:
                continue
            f = fit(logE)
            r = np.abs(f["residuals"])
            summary.append({"target": target, "lambda": lam, "C": 10 ** f["c"], "k": f["k"], "points": int(r.size),
                            "median": float(np.median(r)), "p90": float(np.quantile(r, .9)), "max": float(r.max()),
                            "within_half_decade": float(np.mean(r <= 0.5))})
            s = summary[-1]
            print(f"{target:6s} lambda={str(lam):5s} C={s['C']:.3g} k={s['k']:+.2f}  |residual| decades: "
                  f"median {s['median']:.2f}, 90% {s['p90']:.2f}, max {s['max']:.2f}, within 0.5: {s['within_half_decade']:.0%}")
    (DATA / "quills_min_fit.json").write_text(json.dumps(summary, indent=1) + "\n")
    # figure: error against p, one curve per width, fitted min model dashed; rule and best lambda
    fig, axes = plt.subplots(2, len(targets), figsize=(5.2 * len(targets), 8.4), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for j, target in enumerate(targets):
        for i, lam in enumerate(["rule", "best"]):
            ax = axes[i, j]
            logE = grid(rows, target, lam)
            Ws = sorted({w for w, _ in logE})
            for n, w in enumerate(Ws):
                ps = sorted(p for ww, p in logE if ww == w)
                col = cmap(n / max(1, len(Ws) - 1))
                ax.plot(ps, [10 ** logE[(w, p)] for p in ps], color=col, lw=1.4, marker="o", ms=2.5,
                        label=f"W = {w}")
            ax.set_yscale("log")
            ax.set_ylim(1e-15, 10)
            ax.grid(True, color="0.92", lw=0.6)
            ax.set_title(f"{C.TITLES[target]}, " + ("bandwidth rule" if lam == "rule" else "best of 4 bandwidths"),
                         fontsize=11)
            if i == 1:
                ax.set_xlabel("bits of precision $p$")
            if j == 0:
                ax.set_ylabel("relative error (QUILLS)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = C.OUT / "figures" / "quills_min_structure.png"
    fig.savefig(out, dpi=150)
    print("figure:", out)


if __name__ == "__main__":
    main()
