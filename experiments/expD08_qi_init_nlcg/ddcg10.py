"""iteration_10 Phase 0d -- THE isolation: is the wall memory, or is it precision?

Every O(1)-memory arm measured so far caps at ~1e-6 (5 mechanisms x 5 method
families x 8 systems). Orthogonalized rank memory floors. The campaign concluded
"the floor costs rank memory."

But the one arm that would separate memory from precision was never run cleanly.
iteration_9's `noB_ddmv` claimed "EVERYTHING double-double ... still stalls",
and it is the single data point behind "precision buys zero" -- yet it
(a) kept the step scalar alpha in fp64 (its own note says so), and
(b) re-quenched the carried residual with a PLAIN FP64 forward every 200 of
    10,000 iterations, zeroing the compensation word (`b_replacement.py:699`).

In exact arithmetic CG needs NO memory: it terminates in <= rank iterations.
So if the wall is precision, a CORRECT all-dd memoryless CG must floor with O(1)
state. If it still stalls, the wall is genuinely the memory/finite-termination
structure and no precision story can rescue it.

Arms:
  fp64        memoryless CG, all fp64 (control -- should stall ~1e-6)
  dd_state    dd state + dd dots, fp64 matvecs, NO re-quench
  dd_all      + dd matvecs (A d and A^T r), dd alpha/beta. The clean test.

Cost is NOT the question here -- reachability is. Iterations, not
matvec-equivalents, so a dd matvec is charged the same as an fp64 one.

Run: uv run python experiments/expD08_qi_init_nlcg/ddcg10.py
Out: results/.../iteration_10/phase0/ddcg_{rows.json,ddcg.png}
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
from refine import dd_gemv, dd_dot, dd_add, dd_axpy, dd_renorm  # noqa: E402

OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")

PROBLEMS = ["phi_sine_N256_r4", "phi_runge_N256_r4",
            "syn_phispec_iso_N256_R1024"]
ARMS = ["fp64", "dd_state", "dd_all"]
ITERS = 1200            # ~4x the effective rank (273) -- exact CG needs <= rank
RCOND = 1e-15


def dd_div(ah, al, bh, bl):
    """(ah,al)/(bh,bl) to ~u^2 (one Newton correction)."""
    q = ah / bh
    ph, pl = dd_axpy(np.array([0.0]), np.array([0.0]), q, 0.0,
                     np.array([bh]), np.array([bl]))
    rh, rl = dd_add(ah, al, -float(ph[0]), -float(pl[0]))
    return dd_renorm(q, (rh + rl) / bh)


def solve(arm, A, y, iters, metric, checks):
    R, m = A.shape
    sc = 2.0 / R
    dd_st = arm in ("dd_state", "dd_all")
    dd_mv = arm == "dd_all"
    Z_m, Z_R = np.zeros(m), np.zeros(R)

    v, ve = np.zeros(m), np.zeros(m)
    rhat, re = -y.copy(), np.zeros(R)          # exact at v = 0

    def gradient(rh, rl):
        if dd_mv:
            hh, hl = dd_gemv(A.T, rh, rl)
            return dd_renorm(sc * hh, sc * hl)
        return sc * (A.T @ (rh + rl if dd_st else rh)), np.zeros(m)

    def apply_A(dh, dl):
        if dd_mv:
            return dd_gemv(A, dh, dl)
        return A @ (dh + dl if dd_st else dh), np.zeros(R)

    g, gl = gradient(rhat, re)
    d, dl = -g.copy(), -gl.copy()
    best, traj = np.inf, []
    for it in range(1, iters + 1):
        u, ul = apply_A(d, dl)
        if dd_st:
            ch, cl = dd_dot(u, u, ae=ul, be=ul)
            ch, cl = sc * ch, sc * cl
            nh, nl = dd_dot(g, d, ae=gl, be=dl)
            if not (math.isfinite(ch) and ch > 0.0):
                d, dl = -g.copy(), -gl.copy()
                continue
            ah, al = dd_div(-nh, -nl, ch, cl)
        else:
            c = sc * float(u @ u)
            if not (math.isfinite(c) and c > 0.0):
                d = -g
                continue
            ah, al = -float(g @ d) / c, 0.0
        if not math.isfinite(ah):
            break
        if dd_st:
            v, ve = dd_axpy(v, ve, ah, al, d, dl)
            rhat, re = dd_axpy(rhat, re, ah, al, u, ul)
        else:
            v = v + ah * d
            rhat = rhat + ah * u
        gn, gnl = gradient(rhat, re)
        if not np.isfinite(gn).all():
            break
        if dd_st:
            dh_, _ = dd_dot(g, g, ae=gl, be=gl)
            n1h, _ = dd_dot(gn, gn, ae=gnl, be=gnl)
            n2h, _ = dd_dot(gn, g, ae=gnl, be=gl)
            beta = max(0.0, (n1h - n2h) / dh_) if dh_ > 0 else 0.0
        else:
            den = float(g @ g)
            beta = max(0.0, (float(gn @ gn) - float(gn @ g)) / den) \
                if den > 0 else 0.0
        if dd_st:
            p, pe = np.zeros(m), np.zeros(m)
            p, pe = dd_axpy(p, pe, beta, 0.0, d, dl)
            d, dl = dd_add(p, pe, -gn, -gnl)
        else:
            d = -gn + beta * d
        g, gl = gn, gnl
        if it in checks:
            rel = metric(v + ve if dd_st else v)
            if math.isfinite(rel):
                traj.append([it, rel])
                best = min(best, rel)
    return best, traj


def job_fn(job):
    entry = job["entry"]
    prob = common.load_problem(entry)
    A, y = prob["A"], prob["y"]
    A_m, y_m = ((prob["A_ev"], prob["y_ev"]) if entry["kind"] == "phi"
                else (A, y))
    ny = float(np.linalg.norm(y_m))
    metric = lambda v: float(np.linalg.norm(A_m @ v - y_m) / ny)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > RCOND * S[0]
    v_ref = Vt.T @ np.where(keep, (U.T @ y) / np.where(keep, S, 1.0), 0.0)
    floor = metric(v_ref)
    checks = set(np.unique(np.logspace(0, math.log10(ITERS), 60)
                           .astype(int)).tolist()) | {ITERS}
    out = []
    for arm in ARMS:
        t0 = time.time()
        best, traj = solve(arm, A, y, ITERS, metric, checks)
        out.append({"id": entry["id"], "arm": arm, "best": best,
                    "floor": floor, "n_eff": int(keep.sum()), "traj": traj,
                    "wall_s": round(time.time() - t0, 1)})
    return out


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = [p for p in PROBLEMS if any(r["id"] == p for r in rows)]
    col = {"fp64": "#8c8c8c", "dd_state": "#1f77b4", "dd_all": "#d62728"}
    fig, axes = plt.subplots(1, len(ids), figsize=(4.6 * len(ids), 4.4),
                             squeeze=False)
    for i, pid in enumerate(ids):
        ax = axes[0][i]
        sel = [r for r in rows if r["id"] == pid]
        for r in sel:
            t = np.array(r["traj"])
            if len(t):
                ax.plot(t[:, 0], np.minimum.accumulate(t[:, 1]),
                        color=col[r["arm"]], lw=1.7, label=r["arm"])
        ax.axhline(sel[0]["floor"], color="k", ls=":", lw=1.2,
                   label="lstsq floor")
        ax.axvline(sel[0]["n_eff"], color="#999999", lw=0.9, ls="--")
        ax.text(sel[0]["n_eff"] * 1.05, 3e-2, "rank", fontsize=8,
                color="#666666")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 2)
        ax.grid(alpha=.25, which="both", lw=.4)
        ax.set_title(pid.replace("phi_", "").replace("syn_", ""), fontsize=9)
        ax.set_xlabel("CG iteration")
        if i == 0:
            ax.set_ylabel("rel $L_2$ (running min)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(l), frameon=False)
    fig.suptitle("Memory or precision? Memoryless CG with the recurrence, "
                 "state and matvecs in double-double", y=0.90, fontsize=10)
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.85))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    man = {e["id"]: e for e in common.load_manifest()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for out in ex.map(job_fn, [{"entry": man[p]} for p in PROBLEMS]):
            rows.extend(out)
            print(f"\n{out[0]['id']}  floor {out[0]['floor']:.2e}  "
                  f"rank {out[0]['n_eff']}  ({ITERS} iters)")
            for r in out:
                print(f"   {r['arm']:<10s} best {r['best']:.3e}"
                      f"   x floor {r['best'] / r['floor']:.1e}"
                      f"   {r['wall_s']}s")
    (OUT_DIR / "ddcg_rows.json").write_text(json.dumps(rows))
    figure(rows, OUT_DIR / "ddcg.png")
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT_DIR}/ddcg.png")


if __name__ == "__main__":
    main()
