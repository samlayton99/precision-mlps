"""iteration_10 Phase 0c -- does a NON-Krylov first phase leave a better-shaped
leftover than CG does, and does handing off to the memoryless core then floor?

Phase 0b established: free depth is set by the spectral SHAPE of the error, not
its magnitude (head -> floor from 1e-2; flat -> a fixed ~5 orders; tail -> zero).
And cold CG stalls at 1e-6 with a TAIL-shaped leftover, which is why every
restart / refinement / damping arm was null -- CG poisons itself by spending the
head first and pushing the remainder down-spectrum.

A method that does not greedily minimise the residual should not do that. Adam
in particular takes per-coordinate steps of nearly uniform size, which has no
reason to align with the singular basis at all.

So: run a first phase to its own stall, MEASURE the spectral shape of its
leftover, then hand off to the memoryless <10-vector core and see where it
lands. Total state stays O(1) vectors -- this is one optimizer with two phases,
not a memory scheme.

Arms (first phase): adam, amsgrad-ish (frozen max-v), sgd_mom099, bb, cg (control).
Read: leftover shape (head/mid/tail energy fractions) and composite best.

Run: uv run python experiments/expD08_qi_init_nlcg/handoff10.py
Out: results/.../iteration_10/phase0/handoff_{rows.json,handoff.png}
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expD07_lstsq_optim_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from basin10 import memoryless  # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")

PROBLEMS = ["phi_sine_N256_r4", "phi_runge_N256_r4", "phi_sine_8pi_N256_r4",
            "syn_phispec_iso_N256_R1024"]
PHASE1 = ["adam", "adam_frozen", "sgd_mom099", "bb", "cg"]
P1_BUDGET = 8000        # matvec-pairs for phase 1
P2_BUDGET = 8000        # matvec-pairs for the memoryless hand-off
RCOND = 1e-15


def phase1(kind, A, y, budget, L, metric, checks):
    """First-phase optimizers, all O(1) state, all charged 1 matvec-pair/step."""
    R, m = A.shape
    sc = 2.0 / R
    v = np.zeros(m)
    traj, best, bestv = [], np.inf, v.copy()
    mom = np.zeros(m)
    v1 = np.zeros(m)
    vmax = np.zeros(m)
    b1, b2, eps = 0.9, 0.999, 1e-8
    prev_g, prev_v = None, None
    for t in range(1, budget + 1):
        r = A @ v - y
        g = sc * (A.T @ r)
        if not np.isfinite(g).all():
            break
        if kind in ("adam", "adam_frozen"):
            mom = b1 * mom + (1 - b1) * g
            v1 = b2 * v1 + (1 - b2) * g * g
            mh = mom / (1 - b1 ** t)
            if kind == "adam_frozen":
                vmax = np.maximum(vmax, v1 / (1 - b2 ** t))
                den = np.sqrt(vmax) + eps
            else:
                den = np.sqrt(v1 / (1 - b2 ** t)) + eps
            step = 1e-3 * mh / den
        elif kind == "sgd_mom099":
            mom = 0.99 * mom + g
            step = (1.0 / L) * mom
        elif kind == "bb":
            if prev_g is None:
                step = (1.0 / L) * g
            else:
                s, yy = v - prev_v, g - prev_g
                sy = float(s @ yy)
                a = float(s @ s) / sy if sy > 0 else 1.0 / L
                step = a * g
            prev_g, prev_v = g.copy(), v.copy()
        else:                                   # cg control
            if t == 1:
                d, gp = -g, g
            u = A @ d
            c = sc * float(u @ u)
            if not math.isfinite(c) or c <= 0:
                d = -g
                continue
            al = -float(g @ d) / c
            v = v + al * d
            gn = sc * (A.T @ (A @ v - y))
            den = float(gp @ gp)
            be = max(0.0, (float(gn @ gn) - float(gn @ gp)) / den) if den > 0 else 0.0
            d, gp = -gn + be * d, gn
            if t in checks:
                rel = metric(v)
                traj.append([t, rel])
                if rel < best:
                    best, bestv = rel, v.copy()
            continue
        v = v - step
        if not np.isfinite(v).all():
            break
        if t in checks:
            rel = metric(v)
            if math.isfinite(rel):
                traj.append([t, rel])
                if rel < best:
                    best, bestv = rel, v.copy()
    return bestv, best, traj


def shape_of(A, U, S, v, v_ref, n_eff):
    """Function-space energy fractions of the error, by spectral third."""
    e = v - v_ref
    f = np.abs(U.T @ (A @ e))
    tot = float(np.linalg.norm(f))
    if tot == 0.0:
        return [0.0, 0.0, 0.0]
    a, b = n_eff // 3, 2 * n_eff // 3
    return [float(np.linalg.norm(f[:a]) / tot),
            float(np.linalg.norm(f[a:b]) / tot),
            float(np.linalg.norm(f[b:n_eff]) / tot)]


def job_fn(job):
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    A_m, y_m = ((prob["A_ev"], prob["y_ev"]) if entry["kind"] == "phi"
                else (A, y))
    ny = float(np.linalg.norm(y_m))
    metric = lambda v: float(np.linalg.norm(A_m @ v - y_m) / ny)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > RCOND * S[0]
    n_eff = int(keep.sum())
    v_ref = Vt.T @ np.where(keep, (U.T @ y) / np.where(keep, S, 1.0), 0.0)
    floor = metric(v_ref)
    L = 2.0 * S[0] ** 2 / R
    ck1 = set(np.unique(np.logspace(0, math.log10(P1_BUDGET), 50)
                        .astype(int)).tolist())
    ck2 = set(np.unique(np.logspace(0, math.log10(P2_BUDGET), 50)
                        .astype(int)).tolist())
    out = []
    for kind in PHASE1:
        v1, b1, t1 = phase1(kind, A, y, P1_BUDGET, L, metric, ck1)
        sh = shape_of(A, U, S, v1, v_ref, n_eff)
        b2, t2 = memoryless(A, y, v1, P2_BUDGET, metric, ck2)
        out.append({"id": entry["id"], "phase1": kind, "p1_best": b1,
                    "composite": min(b1, b2), "floor": floor,
                    "shape": sh, "n_eff": n_eff,
                    "t1": t1, "t2": t2})
    return out


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = [p for p in PROBLEMS if any(r["id"] == p for r in rows)]
    fig, axes = plt.subplots(2, len(ids), figsize=(4.3 * len(ids), 7.4),
                             squeeze=False)
    col = {"adam": "#1f77b4", "adam_frozen": "#17becf",
           "sgd_mom099": "#ff7f0e", "bb": "#9467bd", "cg": "#d62728"}
    for i, pid in enumerate(ids):
        sel = [r for r in rows if r["id"] == pid]
        floor = sel[0]["floor"]
        ax = axes[0][i]
        for r in sel:
            t1 = np.array(r["t1"]) if r["t1"] else np.zeros((0, 2))
            t2 = np.array(r["t2"]) if r["t2"] else np.zeros((0, 2))
            if len(t1):
                ax.plot(t1[:, 0], np.minimum.accumulate(t1[:, 1]),
                        color=col[r["phase1"]], lw=1.5, label=r["phase1"])
            if len(t2):
                ax.plot(t2[:, 0] + P1_BUDGET, np.minimum.accumulate(t2[:, 1]),
                        color=col[r["phase1"]], lw=1.5, ls="--")
        ax.axvline(P1_BUDGET, color="#999999", lw=0.9, ls="-")
        ax.axhline(floor, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 2)
        ax.set_title(pid.replace("phi_", "").replace("syn_", ""), fontsize=9)
        ax.grid(alpha=.25, which="both", lw=.4)
        ax.set_xlabel("matvec-pairs (dashed = memoryless hand-off)")
        if i == 0:
            ax.set_ylabel("rel $L_2$ (running min)")
        ax = axes[1][i]
        ks = [r["phase1"] for r in sel]
        xs = np.arange(len(ks))
        bot = np.zeros(len(ks))
        for j, (lab, c) in enumerate((("head", "#1f77b4"), ("mid", "#c0a000"),
                                      ("tail", "#2ca02c"))):
            vals = np.array([r["shape"][j] for r in sel])
            ax.bar(xs, vals, 0.62, bottom=bot, color=c,
                   label=lab if i == 0 else None)
            bot += vals
        ax.set_xticks(xs)
        ax.set_xticklabels(ks, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.25, axis="y", lw=.4)
        if i == 0:
            ax.set_ylabel("leftover energy fraction\nby spectral third")
    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[1][0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=8, frameon=False, fontsize=9)
    fig.suptitle("Phase-1 optimizer -> memoryless hand-off: does a non-Krylov "
                 "first phase leave a better-shaped leftover?",
                 y=0.945, fontsize=10)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    man = {e["id"]: e for e in common.load_manifest()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for out in ex.map(job_fn, [{"entry": man[p]} for p in PROBLEMS]):
            rows.extend(out)
            print(f"\n{out[0]['id']}  floor {out[0]['floor']:.2e}  "
                  f"n_eff {out[0]['n_eff']}")
            for r in out:
                h, mi, ta = r["shape"]
                print(f"   {r['phase1']:<12s} phase1 {r['p1_best']:.2e}"
                      f"  -> composite {r['composite']:.3e}"
                      f"   x floor {r['composite'] / r['floor']:.1e}"
                      f"   leftover head/mid/tail "
                      f"{h:.2f}/{mi:.2f}/{ta:.2f}")
    (OUT_DIR / "handoff_rows.json").write_text(json.dumps(rows))
    figure(rows, OUT_DIR / "handoff.png")
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT_DIR}/handoff.png")


if __name__ == "__main__":
    main()
