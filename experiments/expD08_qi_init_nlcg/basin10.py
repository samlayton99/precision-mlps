"""iteration_10 Phase 0b -- the free-depth curve: how far does the MEMORYLESS
core descend, as a function of where it starts and what shape its error has?

iteration_9 [t9] asserted: "free depth is ~7 decades FROM WHEREVER YOU START,
provided the leftover is not adversarially tail-concentrated." That claim is
load-bearing for the whole campaign (it is the reason `qi_init_arm` floors with
<10 vectors while cold start caps at 1e-6) and it was never measured as a curve
-- only at three points, all of them real inits where magnitude and spectral
shape are confounded.

This decouples them on the frozen linear systems. Start the memoryless core at
v_lstsq + delta, with ||A_ev delta|| / ||y_ev|| = eps controlled, and delta's
FUNCTION-space energy shaped three ways:

  flat  equal energy per singular direction (the measured shape of a CG stall)
  head  energy only in the top 20% of directions (well-conditioned leftover)
  tail  energy only in the bottom 20% of effective directions (adversarial)

Sweep eps over 5 decades. Read the answer off one figure: achieved vs start.
Points on the diagonal = no free depth. Points on the floor = full recovery.

Run: uv run python experiments/expD08_qi_init_nlcg/basin10.py
Out: results/.../iteration_10/phase0/basin_{rows.json,free_depth.png}
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
import common  # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")

PROBLEMS = ["phi_sine_N256_r4", "phi_runge_N256_r4", "phi_sine_8pi_N256_r4",
            "syn_phispec_iso_N256_R1024"]
SHAPES = ["flat", "head", "tail"]
EPS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
BUDGET = 6000          # fp64 matvec-pairs; the free depth is spent well before
RCOND = 1e-15


def memoryless(A, y, v0, budget, metric, checks):
    """The campaign's slim core, no memory, NO fp64 re-quench (noB_nq: measured
    0.5 orders better than re-quenching). Carried residual, exact GN step from
    one JVP, PR+. Returns (best, trajectory)."""
    R, m = A.shape
    sc = 2.0 / R
    v = v0.copy()
    rhat = A @ v - y
    g = sc * (A.T @ rhat)
    d = -g
    cost, best = 1, np.inf
    traj = []
    while cost < budget:
        u = A @ d
        c = sc * float(u @ u)
        cost += 1
        if not math.isfinite(c) or c <= 0.0:
            d = -g
            continue
        alpha = -float(g @ d) / c
        if not math.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        g_new = sc * (A.T @ rhat)
        cost += 1
        if not np.isfinite(g_new).all():
            break
        den = float(g @ g)
        beta = max(0.0, (float(g_new @ g_new) - float(g_new @ g)) / den) \
            if den > 0 else 0.0
        d = -g_new + beta * d
        g = g_new
        if cost in checks:
            rel = metric(v)
            if math.isfinite(rel):
                traj.append([int(cost), rel])
                best = min(best, rel)
    return best, traj


def job_fn(job):
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    R, m = A.shape
    is_phi = entry["kind"] == "phi"
    if is_phi:
        A_m, y_m = prob["A_ev"], prob["y_ev"]
    else:
        A_m, y_m = A, y
    ny_m = float(np.linalg.norm(y_m))
    metric = lambda v: float(np.linalg.norm(A_m @ v - y_m) / ny_m)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > RCOND * S[0]
    v_ref = Vt.T @ np.where(keep, (U.T @ y) / np.where(keep, S, 1.0), 0.0)
    floor = metric(v_ref)
    n_eff = int(keep.sum())

    checks = set(np.unique(np.logspace(0, math.log10(BUDGET), 60)
                           .astype(int)).tolist()) | {BUDGET}

    rng = np.random.default_rng(job["seed"])
    out = []
    for shape in SHAPES:
        # function-space weights over the effective directions
        w = np.zeros(len(S))
        if shape == "flat":
            w[:n_eff] = 1.0
        elif shape == "head":
            w[:max(1, n_eff // 5)] = 1.0
        else:                                    # tail
            w[max(0, n_eff - n_eff // 5):n_eff] = 1.0
        c = rng.normal(size=len(S)) * w
        # delta with unit function-space norm: A delta = sum c_i sigma_i u_i
        fn_energy = float(np.linalg.norm(c * S))
        if fn_energy == 0.0:
            continue
        delta_unit = Vt.T @ (c / fn_energy)      # ||A delta_unit|| == 1
        for eps in EPS:
            v0 = v_ref + delta_unit * (eps * ny_m)
            start = metric(v0)
            best, traj = memoryless(A, y, v0, BUDGET, metric, checks)
            out.append({"id": entry["id"], "shape": shape, "eps": eps,
                        "start": start, "best": best, "floor": floor,
                        "n_eff": n_eff, "traj": traj})
    # references
    for tag, v0 in (("cold", np.zeros(m)), ("exact", v_ref.copy())):
        best, traj = memoryless(A, y, v0, BUDGET, metric, checks)
        out.append({"id": entry["id"], "shape": tag, "eps": None,
                    "start": metric(v0), "best": best, "floor": floor,
                    "n_eff": n_eff, "traj": traj})
    return out


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [p for p in PROBLEMS if any(r["id"] == p for r in rows)]
    col = {"flat": "#d62728", "head": "#1f77b4", "tail": "#2ca02c"}
    fig, axes = plt.subplots(1, len(ids), figsize=(4.5 * len(ids), 4.6),
                             squeeze=False)
    for i, pid in enumerate(ids):
        ax = axes[0][i]
        sel = [r for r in rows if r["id"] == pid]
        floor = sel[0]["floor"]
        lo = min(floor, 1e-16) * 0.3
        ax.plot([1e-16, 1], [1e-16, 1], color="#bbbbbb", lw=1.0, ls="--",
                label="no improvement", zorder=1)
        ax.axhline(floor, color="k", ls=":", lw=1.2, label="lstsq floor",
                   zorder=1)
        for shape in SHAPES:
            pts = sorted([r for r in sel if r["shape"] == shape],
                         key=lambda r: r["start"])
            if not pts:
                continue
            ax.plot([p["start"] for p in pts], [p["best"] for p in pts],
                    "o-", color=col[shape], lw=1.6, ms=4.5, label=shape,
                    zorder=3)
        for tag, mk in (("cold", "s"), ("exact", "*")):
            r = next((x for x in sel if x["shape"] == tag), None)
            if r:
                ax.plot([r["start"]], [r["best"]], mk, color="#111111",
                        ms=11 if tag == "exact" else 7, label=tag, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, 3.0)
        ax.set_ylim(lo, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.set_title(pid.replace("phi_", "").replace("syn_", ""), fontsize=9)
        ax.set_xlabel("starting rel $L_2$")
        if i == 0:
            ax.set_ylabel("best rel $L_2$ reached")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False)
    fig.suptitle("Free depth of the memoryless <10-vector core: "
                 "how far it descends vs where it starts and the "
                 "spectral shape of the starting error", y=0.91, fontsize=10)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.86))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    man = {e["id"]: e for e in common.load_manifest()}
    jobs = [{"entry": man[p], "seed": 10 + i} for i, p in enumerate(PROBLEMS)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for out in ex.map(job_fn, jobs):
            rows.extend(out)
            pid = out[0]["id"]
            print(f"\n{pid}  (floor {out[0]['floor']:.2e}, "
                  f"n_eff {out[0]['n_eff']})")
            for r in out:
                e = "cold/exact" if r["eps"] is None else f"{r['eps']:.0e}"
                gain = math.log10(r["start"] / max(r["best"], 1e-17))
                print(f"   {r['shape']:<6s} eps={e:>10s}  start {r['start']:.2e}"
                      f"  -> best {r['best']:.3e}   gain {gain:5.2f} orders"
                      f"   floor x{r['best'] / r['floor']:.1e}")
    (OUT_DIR / "basin_rows.json").write_text(json.dumps(rows))
    figure(rows, OUT_DIR / "basin_free_depth.png")
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT_DIR}/basin_free_depth.png")


if __name__ == "__main__":
    main()
