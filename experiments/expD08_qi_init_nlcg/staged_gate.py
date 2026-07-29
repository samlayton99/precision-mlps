"""Iteration 9 staged-recycling gate: one FIXED-width tanh network, machine
precision from a memoryless optimizer schedule.

The network is a single sum f(x) = sum_g sum_k v_gk tanh(gamma_g (x - c_gk))
+ c0 with three equal-width groups at gamma multipliers {1, 4, 16} (a
multi-scale geometry; total width fixed up front). The OPTIMIZER schedule:
train group A -> freeze -> train B -> freeze -> train C. By linearity of the
readout, training a group against the ordinary full loss IS fitting the
frozen groups' residual — no target arithmetic, no rescaling (slim is
scale-free). Each stage is the memoryless slim core (carried residual,
exact GN steps, PR+, refresh 200): <10 vectors of state, ~5F/step.

Arms:
  staged_cold   v = 0 start, three stages
  staged_qi     group A initialized from the fp64 QI construction (the
                expD08 story: QI leftover is tail content; do recycled
                sharp groups unstick it?)
  allatonce     all groups trained jointly, one run, same total budget
                (control: is the FREEZE schedule load-bearing?)
  A_only        group A alone, same total budget (control: no recycling)

Gate: staged arms reach composite eval rel L2 <= ~1e-12 at fixed width
where the controls stall 1e-6..1e-7.

Run: uv run python experiments/expD08_qi_init_nlcg/staged_gate.py
Out: results/.../expD08_qi_init_nlcg/iteration_9/staged_gate/
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = (REPO / "results" / "checkpoint_D_optimizers" / "expD08_qi_init_nlcg"
           / "iteration_9" / "staged_gate")

TARGETS = ["sine", "sine_8pi", "runge"]
N_GROUP = 256           # interior intervals per group
MU = [1.0, 4.0, 16.0]   # gamma multiplier ladder
SHIFT = [0.0, 0.5, 0.25]  # center shift (cells) per group
N_TRAIN = 6000
N_EVAL = 4001
ITERS_STAGE = 10_000
REFRESH = 200
LAMBDA_STAR = 0.30      # fp64 construction convention (group A = QI geometry)


def slim(A, y, iters, v0=None):
    """Memoryless slim core on a frozen linear system; returns best-by-train
    v and the train-rel trajectory (checkpoint every refresh)."""
    R, m = A.shape
    sc = 2.0 / R
    v = np.zeros(m) if v0 is None else v0.copy()
    rhat = A @ v - y
    yn = np.linalg.norm(y)
    g = sc * (A.T @ rhat)
    d = -g
    best_v = v.copy()
    best_tr = float(np.linalg.norm(rhat) / yn)
    traj = [best_tr]
    for it in range(1, iters + 1):
        if it % REFRESH == 0:
            rhat = A @ v - y
            tr = float(np.linalg.norm(rhat) / yn)
            traj.append(tr)
            if tr < best_tr:
                best_tr, best_v = tr, v.copy()
        u = A @ d
        c = sc * float(u @ u)
        if not np.isfinite(c) or c <= 0.0:
            d = -g
            continue
        alpha = -float(g @ d) / c
        if not np.isfinite(alpha):
            break
        v = v + alpha * d
        rhat = rhat + alpha * u
        g_new = sc * (A.T @ rhat)
        if not np.isfinite(g_new).all():
            break
        beta = max(0.0, float(g_new @ (g_new - g)) / float(g @ g))
        d = -g_new + beta * d
        g = g_new
    tr = float(np.linalg.norm(A @ v - y) / yn)
    if tr < best_tr:
        best_tr, best_v = tr, v.copy()
    return best_v, best_tr, traj


def build_network(qi_res):
    """Three groups sharing group A's QI geometry scale: group g has
    gamma_g = mu_g * gamma_A, centers on the same [-1,1] grid (with halo),
    shifted by SHIFT[g] cells. Returns per-group (gamma, centers)."""
    groups = []
    h = qi_res.h
    for mu, sh in zip(MU, SHIFT):
        centers = qi_res.centers + sh * h
        groups.append((mu * qi_res.gamma, centers))
    return groups


def phi_cols(x, groups):
    blocks = [np.tanh(g * (x[:, None] - c[None, :])) for (g, c) in groups]
    return np.hstack(blocks + [np.ones((x.size, 1))])


def run_arm(job):
    from src.construction.qi_mpmath import construct_qi
    from src.data.targets import get_target

    t = get_target(job["fn"])
    res = construct_qi(t.fn_numpy, t.deriv_numpy, N_GROUP, precision="fp64",
                       lambda_star=LAMBDA_STAR)
    groups = build_network(res)
    x_tr = np.linspace(-1.0, 1.0, N_TRAIN)
    x_ev = np.linspace(-1.0, 1.0, N_EVAL)
    y_tr, y_ev = t.fn_numpy(x_tr), t.fn_numpy(x_ev)
    Phi = phi_cols(x_tr, groups)
    Phi_ev = phi_cols(x_ev, groups)
    K = Phi.shape[1]
    gsize = groups[0][1].size
    slices = [np.arange(i * gsize, (i + 1) * gsize) for i in range(len(MU))]
    bias_col = K - 1
    yn_ev = np.linalg.norm(y_ev)

    v = np.zeros(K)
    arm = job["arm"]
    t0 = time.time()
    if arm == "staged_qi":
        v[slices[0]] = res.a_coeffs
        v[bias_col] = res.c0
    evals, stage_rows = [], []
    ev0 = float(np.linalg.norm(y_ev - Phi_ev @ v) / yn_ev)
    evals.append(ev0)

    if arm in ("staged_cold", "staged_qi"):
        for si, sl in enumerate(slices):
            act = np.concatenate([sl, [bias_col]]) if si == 0 else sl
            A_s = Phi[:, act]
            resid = y_tr - Phi @ v + A_s @ v[act]   # freeze others, keep own
            v_s, tr_s, _ = slim(A_s, resid, ITERS_STAGE, v0=v[act])
            v[act] = v_s
            ev = float(np.linalg.norm(y_ev - Phi_ev @ v) / yn_ev)
            evals.append(ev)
            stage_rows.append({"stage": si, "mu": MU[si],
                               "stage_train_rel": tr_s, "eval_rel": ev})
    elif arm == "allatonce":
        v, tr, _ = slim(Phi, y_tr, len(MU) * ITERS_STAGE, v0=v)
        ev = float(np.linalg.norm(y_ev - Phi_ev @ v) / yn_ev)
        evals.append(ev)
        stage_rows.append({"stage": "joint", "stage_train_rel": tr,
                           "eval_rel": ev})
    elif arm == "A_only":
        act = np.concatenate([slices[0], [bias_col]])
        v_s, tr, _ = slim(Phi[:, act], y_tr, len(MU) * ITERS_STAGE,
                          v0=v[act])
        v[act] = v_s
        ev = float(np.linalg.norm(y_ev - Phi_ev @ v) / yn_ev)
        evals.append(ev)
        stage_rows.append({"stage": "A", "stage_train_rel": tr,
                           "eval_rel": ev})

    # reference floors: full-union direct solve, on train and eval
    w, *_ = np.linalg.lstsq(Phi, y_tr, rcond=1e-15)
    floor_ev = float(np.linalg.norm(y_ev - Phi_ev @ w) / yn_ev)
    return {"fn": job["fn"], "arm": arm, "evals": evals,
            "stages": stage_rows, "floor_eval": floor_ev,
            "best_eval": min(evals), "wall_s": round(time.time() - t0, 1)}


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ARMS = {"staged_cold": ("staged, cold start", "#1f77b4", "-o"),
            "staged_qi": ("staged, QI init on A", "#2ca02c", "-s"),
            "allatonce": ("all-at-once (control)", "#888888", "--D"),
            "A_only": ("group A only (control)", "#bbbbbb", "--^")}
    by = {(r["fn"], r["arm"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, fn in zip(axes, TARGETS):
        for arm, (lab, col, st) in ARMS.items():
            r = by.get((fn, arm))
            if r is None:
                continue
            xs = np.arange(len(r["evals"]))
            ax.plot(xs, r["evals"], st, color=col, lw=1.6, ms=5, label=lab)
        ax.axhline(by[(fn, "staged_cold")]["floor_eval"], color="k",
                   ls=":", lw=1.2, label="union lstsq floor")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(fn)
        ax.set_xlabel("stage (0 = init)")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("composite eval rel L2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=5)
    fig.suptitle("Staged recycling on one fixed-width multi-scale network "
                 f"(3 x {N_GROUP}-interval groups, mu = {MU}, "
                 f"n = {N_TRAIN})", y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    p = OUT_DIR / "staged_gate.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [{"fn": fn, "arm": arm} for fn in TARGETS
            for arm in ["staged_cold", "staged_qi", "allatonce", "A_only"]]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_arm, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['fn']:10s} {r['arm']:12s} "
                  f"evals={['%.1e' % e for e in r['evals']]} "
                  f"floor={r['floor_eval']:.1e} ({r['wall_s']}s)",
                  flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    (OUT_DIR / "staged_gate_rows.json").write_text(json.dumps(rows))
    figure(rows)


if __name__ == "__main__":
    main()
