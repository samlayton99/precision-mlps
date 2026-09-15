"""(ex-expC10, consolidated into expC07/lambda_rule on 2026-09-08; see hardened_rule.md.) NOTE (skeptic review P16): this
scorer was not covered by the pre-registration hashes; only rule.py, targets.py and the predictions were.
Pre-registered test of the two-wall rule on held-out targets, fp64 and fp32.

Read PREREGISTRATION_c10.md first. This script measures the lambda curves (expC09 solver, halo 32, gelsd in the
stated dtype), scores every cell against the locked predictions.json, and draws the figures. It changes no
prediction. Run:  uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/run_c10_prereg.py   (--smoke, --score-only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from rule import HALO  # noqa: E402
from targets import evaluate, make_targets  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
DATA = OUT_DIR / "data"
LAM_GRID = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 60)]
ACTS = ["tanh", "gelu", "swish"]
WIDTHS = [64, 128, 256, 512]
PRECISIONS = ["fp64", "fp32"]
DTYPE = {"fp64": np.float64, "fp32": np.float32}
RESOLVED_MIN = {"fp64": 1e-6, "fp32": 1e-3}


def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)


def solve_cell(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, lam, N, prec = job["act"], job["lam"], job["N"], job["precision"]
    t = job["target"]
    dt = DTYPE[prec]
    h = 2.0 / N
    centers = (-1.0 + np.arange(-HALO, N + HALO + 1, dtype=np.float64) * h).astype(dt)
    gamma = dt(lam / h)
    x = np.linspace(-1, 1, max(2003, 2 * centers.size + 3)).astype(dt)
    A = np.hstack([apply_act(gamma * (x[:, None] - centers[None, :]), act), np.ones((x.size, 1), dtype=dt)]).astype(dt)
    b = evaluate(t, x.astype(np.float64)).astype(dt)
    sol, _, rank, sv = sp_lstsq(A, b, lapack_driver="gelsd")
    xe = np.linspace(-1, 1, 4001).astype(dt)
    fit = (apply_act(gamma * (xe[:, None] - centers[None, :]), act) @ sol[:-1].astype(dt) + sol[-1]).astype(np.float64)
    fe = evaluate(t, xe.astype(np.float64))
    resid = fit - fe
    return {"target": t["name"], "act": act, "lam": lam, "N": N, "precision": prec,
            "rel_l2": float(np.linalg.norm(resid) / np.linalg.norm(fe)), "linf": float(np.max(np.abs(resid))),
            "rank": int(rank), "s_max": float(sv[0])}


def smooth_log(v, k=5):
    lv = np.log(np.asarray(v, dtype=float))
    r = k // 2
    return np.exp(np.array([np.median(lv[max(0, i - r): i + r + 1]) for i in range(len(lv))]))


def curve(rows, target, act, N, prec):
    pts = sorted([r for r in rows if r["target"] == target and r["act"] == act and r["N"] == N and r["precision"] == prec],
                 key=lambda r: r["lam"])
    return np.array([r["lam"] for r in pts]), np.array([r["rel_l2"] for r in pts])


def score(rows, preds):
    """Regret per cell for the rule (exact, x3, div3) and the constant; the pre-registered split; verdicts."""
    cells = []
    for c in preds["cells"]:
        lam, rel = curve(rows, c["target"], c["act"], c["N"], c["precision"])
        if not lam.size:
            continue
        sm = smooth_log(rel)
        emin = float(sm.min())
        q = c["exact"]["q"]
        thr = RESOLVED_MIN[c["precision"]]
        status = "unresolved" if emin > thr else ("resolved" if q <= 0.25 else "under-resolved")
        rec = {**{k: c[k] for k in ("target", "act", "N", "precision")}, "q": q, "E_min": emin, "argmin": float(lam[int(sm.argmin())]),
               "status": status}
        for tag in ("exact", "x3", "div3"):
            ls = c[tag]["lambda_star"]
            if ls is None:
                rec[tag] = None
                continue
            e = float(np.exp(np.interp(np.log(ls), np.log(lam), np.log(sm))))
            rec[tag] = {"lambda_star": ls, "E_star": c[tag]["E_star"], "E_at": e, "regret": e / emin}
        lc = c["constant"]
        e = float(np.exp(np.interp(np.log(lc), np.log(lam), np.log(sm))))
        rec["constant"] = {"lambda": lc, "E_at": e, "regret": e / emin}
        cells.append(rec)

    verdict = {}
    for prec in PRECISIONS:
        res = [x["exact"]["regret"] for x in cells if x["precision"] == prec and x["status"] == "resolved" and x["exact"]]
        und = [x["exact"]["regret"] for x in cells if x["precision"] == prec and x["status"] == "under-resolved" and x["exact"]]
        unr = [x for x in cells if x["precision"] == prec and x["status"] == "unresolved"]
        v = {"n_resolved": len(res), "n_under": len(und), "n_unresolved": len(unr)}
        if res:
            v["resolved_median"] = float(np.median(res)); v["resolved_p90"] = float(np.percentile(res, 90)); v["resolved_max"] = float(max(res))
        if und:
            v["under_p90"] = float(np.percentile(und, 90)); v["under_median"] = float(np.median(und)); v["under_max"] = float(max(und))
        ok_res = bool(res) and v["resolved_median"] <= 2 and v["resolved_p90"] <= 10
        ok_und = (not und) or v["under_p90"] <= 10
        v["H1_pass"] = bool(ok_res and ok_und)
        # secondary
        sc = [x for x in cells if x["precision"] == prec and x["status"] != "unresolved" and x["exact"]]
        v["S1_frac_Estar_within_10x"] = float(np.mean([abs(np.log10(x["exact"]["E_star"] / x["E_min"])) <= 1 for x in sc])) if sc else None
        for tag in ("x3", "div3", "constant"):
            rr = [x[tag]["regret"] for x in sc if x.get(tag)]
            if rr:
                v[f"S_{tag}_median"] = float(np.median(rr)); v[f"S_{tag}_p90"] = float(np.percentile(rr, 90)); v[f"S_{tag}_max"] = float(max(rr))
        verdict[prec] = v
    return cells, verdict


def make_figures(rows, cells, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tnames = [t["name"] for t in preds["targets"]]
    # 1. regret summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, prec in zip(axes, PRECISIONS):
        xpos = {("resolved", a): i for i, a in enumerate(ACTS)}
        xpos.update({("under-resolved", a): 4 + i for i, a in enumerate(ACTS)})
        rng = np.random.default_rng(0)
        for x in cells:
            if x["precision"] != prec or x["status"] == "unresolved" or not x["exact"]:
                continue
            xp = xpos[(x["status"], x["act"])]
            ax.plot(xp + rng.uniform(-0.25, 0.25), x["exact"]["regret"], "o", color="#d1352b", ms=5, alpha=0.7, mec="none")
            ax.plot(xp + rng.uniform(-0.25, 0.25), x["constant"]["regret"], "s", color="0.5", ms=4, alpha=0.5, mec="none")
        ax.axhline(2, color="k", lw=0.9, ls="--"); ax.axhline(10, color="k", lw=0.9, ls=":")
        ax.set_yscale("log"); ax.set_ylim(0.5, 1e4)
        ax.set_xticks(list(range(3)) + list(range(4, 7))); ax.set_xticklabels([f"{a}\nresolved" for a in ACTS] + [f"{a}\nunder-res." for a in ACTS])
        ax.set_title(prec, fontsize=12); ax.grid(alpha=0.3, which="both", axis="y")
    axes[0].set_ylabel("regret  E(lambda) / min E   (held-out targets)")
    handles = [Line2D([0], [0], marker="o", color="#d1352b", lw=0, ms=6, label="two-wall rule, exact band edge"),
               Line2D([0], [0], marker="s", color="0.5", lw=0, ms=5, label="constant rule A_K = eps (baseline S3)"),
               Line2D([0], [0], color="k", lw=0.9, ls="--", label="median criterion 2"),
               Line2D([0], [0], color="k", lw=0.9, ls=":", label="90th percentile criterion 10")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("expC10 -- regret of the pre-registered two-wall rule on 12 held-out targets x 4 widths", y=0.9, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = OUT_DIR / "figures" / "c10_prereg_regret.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  saved {p}")

    # 2. curves with the prediction marked: one figure per (precision, activation), 12 x 4
    pc = {(c["target"], c["act"], c["N"], c["precision"]): c for c in preds["cells"]}
    for prec in PRECISIONS:
        for act in ACTS:
            fig, axes = plt.subplots(12, 4, figsize=(15, 30), sharex=True, sharey=True)
            for i, tn in enumerate(tnames):
                for j, N in enumerate(WIDTHS):
                    ax = axes[i, j]
                    lam, rel = curve(rows, tn, act, N, prec)
                    c = pc[(tn, act, N, prec)]
                    if lam.size:
                        ax.loglog(lam, rel, "-", color="tab:blue", lw=1.3)
                    ax.axvline(c["constant"], color="k", lw=1.0, ls=":")
                    if c["exact"]["lambda_star"] is not None:
                        ax.axvline(c["exact"]["lambda_star"], color="#d1352b", lw=1.4, ls="--")
                        ax.plot(c["exact"]["lambda_star"], c["exact"]["E_star"], "o", color="#d1352b", ms=6, mec="k", mew=0.5, zorder=5)
                        for tag, mk in (("x3", "<"), ("div3", ">")):
                            if c[tag]["lambda_star"] is not None:
                                ax.plot(c[tag]["lambda_star"], c[tag]["E_star"], mk, color="#f2a65a", ms=5, mec="none")
                    ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
                    if i == 0:
                        ax.set_title(f"N = {N}", fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f"{tn}\nq@64={c['exact']['q']*64/N if N else 0:.2f}", fontsize=8)
                    if i == 11:
                        ax.set_xlabel(r"$\lambda$")
            handles = [Line2D([0], [0], color="tab:blue", lw=1.4, label=f"measured ({prec}, halo 32, gelsd)"),
                       Line2D([0], [0], color="#d1352b", lw=1.4, ls="--", label="two-wall lambda* (pre-registered)"),
                       Line2D([0], [0], marker="o", color="#d1352b", lw=0, ms=6, mec="k", label="predicted (lambda*, E*)"),
                       Line2D([0], [0], marker="<", color="#f2a65a", lw=0, ms=5, label="rule at 3x band edge"),
                       Line2D([0], [0], marker=">", color="#f2a65a", lw=0, ms=5, label="rule at band edge / 3"),
                       Line2D([0], [0], color="k", lw=1.0, ls=":", label="constant rule A_K = eps")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False, fontsize=9)
            fig.suptitle(f"expC10 -- {act}, {prec}: held-out lambda curves vs the locked prediction", y=0.982, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.972])
            p = OUT_DIR / "figures" / f"c10_prereg_curves_{act}_{prec}.png"
            fig.savefig(p, dpi=110); plt.close(fig); print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    preds_path = DATA / "c10_predictions.json"
    preds = json.loads(preds_path.read_text())
    h = hashlib.sha256(preds_path.read_bytes()).hexdigest()
    print(f"predictions.json sha256 {h}")
    targets = {t["name"]: t for t in make_targets()}
    assert all(t["omega"] == preds["targets"][i]["omega"] for i, t in enumerate(targets.values())), "targets changed"

    rows_path = DATA / "c10_rows.json"
    if args.score_only:
        rows = json.loads(rows_path.read_text())
    else:
        lam_grid, widths, acts, precs, tn = LAM_GRID, WIDTHS, ACTS, PRECISIONS, list(targets)
        if args.smoke:
            lam_grid, widths, acts, precs, tn = [0.1, 0.25, 0.5], [64], ["tanh"], ["fp64", "fp32"], ["trig_K2_0", "rat_J2_0"]
        jobs = sorted([{"act": a, "lam": l, "N": n, "precision": p, "target": targets[t]}
                       for a in acts for l in lam_grid for n in widths for p in precs for t in tn], key=lambda j: -j["N"])
        print(f"expC10 | {len(jobs)} solves")
        rows, t0 = [], time.time()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(solve_cell, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 1000 == 0 or i == len(jobs):
                    if not args.smoke:
                        rows_path.write_text(json.dumps(rows))
                    print(f"  [{i}/{len(jobs)}] ({time.time() - t0:.0f}s)", flush=True)
        if args.smoke:
            for r in sorted(rows, key=lambda r: (r["target"], r["precision"], r["lam"])):
                print(r)
            return
    cells, verdict = score(rows, preds)
    (DATA / "c10_scores.json").write_text(json.dumps({"cells": cells, "verdict": verdict, "predictions_sha256": h}, indent=1))
    print(json.dumps(verdict, indent=1))
    make_figures(rows, cells, preds)


if __name__ == "__main__":
    main()
