"""Budget-scaling study: does the Krylov phi stall yield to budget, and how
much orthogonalization MEMORY does 1e-10+ actually require?

Sam's bar: a DL-viable optimizer must drive the phi readout solve to at
least 1e-10. At 20k evals NLCG stalls ~1e-7 and plain CGLS ~1e-8 while full
reorthogonalization reaches the SVD floor -- so the two open questions are:
  (1) budget: does 10x budget (200k evals) move the plain-Krylov stall?
  (2) memory: does a WINDOWED reorthogonalization (last k directions only,
      L-BFGS-class memory, DL-deployable) recover the floor, or does the
      orthogonality loss come from long-range drift that only a full basis
      fixes?

Methods (on phi_{sine,runge,sine_mixture}_N256_r4, 200k evals):
  nlcg        NLCG-PR, exact line search, 2 evals/iter
  cgls_k0     plain CGLS
  cgls_k64    CGLS, reorthogonalize s against the last 64 basis vectors
  cgls_k256   ... last 256
  cgls_kfull  ... full basis (the round-4 floor reference)
  lsqr        LSQR

Separate outputs (does NOT touch runs.jsonl -- different budget):
  results/.../budget_scaling.json + budget_scaling.png

Usage: uv run python experiments/expD07_lstsq_optim_suite/budget_scaling.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import FIG_DIR, RESULTS_DIR, load_manifest, load_problem  # noqa: E402

PROBLEM_IDS = ["phi_sine_N256_r4", "phi_runge_N256_r4",
               "phi_sine_mixture_N256_r4"]
BUDGET = 200_000
METHODS = ["nlcg", "cgls_k0", "cgls_k64", "cgls_k256", "cgls_kfull", "lsqr"]
COLORS = {"nlcg": "tab:olive", "cgls_k0": "tab:gray", "cgls_k64": "tab:blue",
          "cgls_k256": "tab:purple", "cgls_kfull": "tab:red",
          "lsqr": "tab:green"}
OUT_JSON = RESULTS_DIR / "budget_scaling.json"
OUT_PNG = FIG_DIR / "budget_scaling.png"


def checkpoints(budget, n=80):
    return np.unique(np.round(np.logspace(0, np.log10(budget), n))).astype(int)


def run_method(method, entry):
    torch.set_num_threads(2)
    prob = load_problem(entry)
    A = torch.from_numpy(np.ascontiguousarray(prob["A"]))
    y = torch.from_numpy(prob["y"])
    A_ev = torch.from_numpy(np.ascontiguousarray(prob["A_ev"]))
    y_ev = torch.from_numpy(prob["y_ev"])
    y_ev_norm = float(torch.linalg.norm(y_ev))
    R, cols = A.shape
    ckpts = checkpoints(BUDGET)
    ptr, traj = [0], {"evals": [], "eval_rel_l2": []}

    def snap(evals, x, force=False):
        due = ptr[0] < len(ckpts) and evals >= ckpts[ptr[0]]
        if not (due or force):
            return
        while ptr[0] < len(ckpts) and ckpts[ptr[0]] <= evals:
            ptr[0] += 1
        with torch.no_grad():
            e = A_ev.mv(x) - y_ev
            traj["evals"].append(int(evals))
            traj["eval_rel_l2"].append(float(torch.linalg.norm(e)) / y_ev_norm)

    t0 = time.time()
    if method == "nlcg":
        x = torch.zeros(cols, dtype=torch.float64)
        r = -y.clone()
        g = (2.0 / R) * A.t().mv(r)
        d = -g
        gg = float(g.dot(g))
        it = 0
        while 2 * (it + 1) <= BUDGET:
            it += 1
            Ad = A.mv(d)
            denom = float(Ad.dot(Ad))
            if denom <= 0.0 or not np.isfinite(denom):
                break
            alpha = -float(r.dot(Ad)) / denom
            x = x + alpha * d
            r = r + alpha * Ad
            g_new = (2.0 / R) * A.t().mv(r)
            beta = max(0.0, float(g_new.dot(g_new - g)) / gg)
            d = -g_new + beta * d
            if float(g_new.dot(d)) >= 0.0:
                d = -g_new
            g, gg = g_new, float(g_new.dot(g_new))
            snap(2 * it, x)
            if gg == 0.0:
                break

    elif method.startswith("cgls_k"):
        k = {"cgls_k0": 0, "cgls_k64": 64, "cgls_k256": 256,
             "cgls_kfull": cols}[method]
        cost = 2 if k > 0 else 1          # reortho ~ 1 extra matvec-equiv
        x = torch.zeros(cols, dtype=torch.float64)
        r = y.clone()
        s = A.t().mv(r)
        basis = [s / torch.linalg.norm(s)] if k > 0 else []
        p = s.clone()
        gamma = float(s.dot(s))
        it = 0
        while cost * (it + 1) <= BUDGET:
            it += 1
            q = A.mv(p)
            qq = float(q.dot(q))
            if qq <= 0.0 or gamma <= 1e-300 or not np.isfinite(qq):
                snap(cost * it, x, force=True)
                break
            alpha = gamma / qq
            x = x + alpha * p
            r = r - alpha * q
            s = A.t().mv(r)
            if k > 0 and basis:
                B = torch.stack(basis[-k:])
                s = s - B.t().mv(B.mv(s))
                s = s - B.t().mv(B.mv(s))
            gamma_new = float(s.dot(s))
            if gamma_new <= 1e-300:
                snap(cost * it, x, force=True)
                break
            if k > 0:
                if len(basis) >= max(k, 1) and k < cols:
                    basis = basis[-(k - 1):] if k > 1 else []
                if len(basis) < cols:
                    basis.append(s / float(np.sqrt(gamma_new)))
            p = s + (gamma_new / gamma) * p
            gamma = gamma_new
            snap(cost * it, x)

    elif method == "lsqr":
        x = torch.zeros(cols, dtype=torch.float64)
        beta = float(torch.linalg.norm(y))
        u = y / beta
        v = A.t().mv(u)
        alpha = float(torch.linalg.norm(v))
        v = v / alpha
        w = v.clone()
        phibar, rhobar = beta, alpha
        for t in range(1, BUDGET + 1):
            u = A.mv(v) - alpha * u
            beta = float(torch.linalg.norm(u))
            if beta > 0:
                u = u / beta
            vv = A.t().mv(u) - beta * v
            alpha = float(torch.linalg.norm(vv))
            if alpha > 0:
                v = vv / alpha
            rho = float(np.hypot(rhobar, beta))
            c, s_rot = rhobar / rho, beta / rho
            theta = s_rot * alpha
            rhobar = -c * alpha
            phi = c * phibar
            phibar = s_rot * phibar
            x = x + (phi / rho) * w
            w = v - (theta / rho) * w
            snap(t, x)
            if beta == 0.0 or alpha == 0.0:
                snap(t, x, force=True)
                break
    else:
        raise ValueError(method)

    best = float(np.min(traj["eval_rel_l2"]))
    best20k = float(np.min([v for e, v in zip(traj["evals"],
                                              traj["eval_rel_l2"])
                            if e <= 20_000]))
    return {"problem_id": entry["id"], "method": method, "traj": traj,
            "best": best, "best_20k": best20k,
            "floor": entry["floor_eval_rel_l2"],
            "wall_s": round(time.time() - t0, 1)}


def make_figure(rows, entries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PROBLEM_IDS), figsize=(15.5, 4.8))
    for ax, pid in zip(axes, PROBLEM_IDS):
        for r in rows:
            if r["problem_id"] != pid:
                continue
            t = r["traj"]
            vals = np.minimum.accumulate(np.maximum(t["eval_rel_l2"], 1e-16))
            ax.loglog(t["evals"], vals, color=COLORS[r["method"]],
                      label=r["method"])
        ax.axhline(entries[pid]["floor_eval_rel_l2"], color="k", ls="--",
                   lw=1.4, label="SVD floor")
        ax.axvline(20_000, color="0.6", ls=":", lw=1.2)
        ax.set_ylim(1e-16, 1.0)
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("evals (dotted line = the 20k suite budget)")
        ax.set_ylabel("best eval rel $L_2$")
        ax.grid(True, alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle("expD07 budget scaling -- 200k evals, windowed vs full "
                 "reorthogonalization (memory buys depth?)", y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.94), ncol=7, frameon=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"saved {OUT_PNG}")


def main():
    manifest = {e["id"]: e for e in load_manifest()}
    entries = {pid: manifest[pid] for pid in PROBLEM_IDS}
    jobs = [(m, entries[pid]) for pid in PROBLEM_IDS for m in METHODS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(run_method, m, e) for m, e in jobs]
        for fut in futs:
            r = fut.result()
            rows.append(r)
            print(f"{r['problem_id']:28s} {r['method']:12s} "
                  f"20k={r['best_20k']:.2e}  200k={r['best']:.2e}  "
                  f"floor={r['floor']:.1e}  ({r['wall_s']}s)", flush=True)
    OUT_JSON.write_text(json.dumps(rows))
    make_figure(rows, entries)


if __name__ == "__main__":
    main()
