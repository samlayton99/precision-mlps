"""Basin geography under data resampling (Sam's partition experiment).

Question: when the data is resampled (new batch / new partition), does the
optimum actually MOVE -- is per-batch training chasing a moving target --
or is the target pinned and the failure purely dynamical?

Design. n = 4006 equispaced points on [-1, 1]. Four partitions into
halves of 2003: three seeded random splits + the even/odd interleave (a
low-discrepancy control). Each half is solved as its OWN full-batch
problem with the iteration_6 optimizer (3000 iters), plus one solve on
the union. Two conditions: CLEAN targets, and NOISY targets (iid y-noise,
sigma = 1e-6, one fixed realization on the union; each half inherits its
points' noise, so the two halves see genuinely different data).

Measured, per split (A, B):
  1. own-data train rel L2, and eval rel L2 on the dense 4001 grid;
  2. function-space distance ||f_A - f_B|| / ||f|| on the dense grid;
  3. parameter distance per block (max |dw|, max |db| vs gamma; rel dv);
  4. cross-objective excess: rel of theta_A evaluated on B's points,
     minus theta_B's own rel there;
  5. the linear interpolation path theta(t) = (1-t) theta_A + t theta_B,
     rel L2 on the UNION points at 21 values of t;
  6. the midpoint (t = 0.5) vs the union solution.

Predictions, stated before running:
  CLEAN -- the samples are exact values of one true function, so the two
  half-problems share their minimizer up to discretization at the
  ~1e-13 floor scale: function-space distance ~1e-13, cross-excess ~0,
  interpolation path flat near the floor. If confirmed, the target does
  NOT move under clean resampling, and per-batch failure is purely
  dynamical (operator scrambling during optimization, lesson 7), not
  target motion.
  NOISY -- the target genuinely moves at the noise scale: function-space
  distance ~ sigma-scale, and each half's dense-grid eval saturates near
  sigma/sqrt(n)-class. More data helps (1/sqrt n); this is the regime
  where "the basin shifts with each batch" is literally true.

Run:  uv run python experiments/expD08_qi_init_nlcg/basin_study.py
Outputs under results/checkpoint_D_optimizers/expD08_qi_init_nlcg/basin_study/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

BASIN_DIR = run.OUT_DIR.parent / "batching" / "basin_study"
N_UNION = 4006
NOISE = 1e-6
FNS = ["sine", "runge"]
NWIDTH = 256
SPLITS = ["rand0", "rand1", "rand2", "evenodd"]
ITERS = 3000


def make_split(name, n_union=N_UNION):
    idx = np.arange(n_union)
    if name == "evenodd":
        return idx[::2], idx[1::2]
    seed = int(name[4:])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_union)
    return np.sort(perm[:n_union // 2]), np.sort(perm[n_union // 2:])


def solve_case(job):
    """Full-batch iteration_6 solve on an arbitrary point set.

    Optional job fields (halo study): "domain" (default (-1, 1)) and
    "n_union" (default N_UNION) -- the union grid the split indexes into.
    The BEST-over-trajectory theta is returned (fix for the rand2
    final-iterate artifact), alongside the final one."""
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, cond, tag = job["fn"], job["cond"], job["tag"]
    lo, hi = job.get("domain", (-1.0, 1.0))
    n_union = job.get("n_union", N_UNION)
    f_vec = run.make_fn(run.TARGETS[fn_name])
    centers, gamma, _ = run.geometry(NWIDTH, run.LAM)
    W = centers.size
    theta0 = torch.tensor(np.concatenate([
        np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))

    x_all = np.linspace(lo, hi, n_union)
    y_all = f_vec(x_all)
    if cond == "noisy":
        y_all = y_all + NOISE * np.random.default_rng(999).standard_normal(
            n_union)
    sel = job["idx"]
    xt = torch.tensor(x_all[sel]).unsqueeze(1)
    yt = torch.tensor(y_all[sel])
    n = len(sel)
    x_ev = np.linspace(-1, 1, run.N_EVAL)
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))          # eval vs the TRUE function
    ye_norm = float(torch.linalg.norm(ye))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_jvp(th, dvec):
        _, jd = torch.func.jvp(f_pred, (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed):
        thr = th.detach().requires_grad_(True)
        (gg,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed)
        return gg.detach()

    best = {"rel": float("inf"), "theta": None}

    def eval_rel(th):
        with torch.no_grad():
            r = model(xe, th) - ye
            rel = float(torch.linalg.norm(r)) / ye_norm
        if rel < best["rel"]:
            best["rel"] = rel
            best["theta"] = th.detach().clone()
        return rel

    t0 = time.time()
    losses, rels, _, theta_fin, stats = run.train_iter6(
        theta0, f_resid, f_jvp, f_vjp, f_pred, n, ITERS, set(),
        lambda th: {}, eval_rel, float(torch.linalg.norm(yt)))
    theta_best = best["theta"] if best["theta"] is not None else theta_fin
    with torch.no_grad():
        train_rel = (float(torch.linalg.norm(model(xt, theta_best) - yt))
                     / float(torch.linalg.norm(yt)))
    return {"fn": fn_name, "cond": cond, "tag": tag,
            "theta": theta_best.numpy().tolist(),
            "theta_final": theta_fin.numpy().tolist(),
            "train_rel": train_rel, "eval_rel": float(np.min(rels)),
            "trips": len(stats["trips"]), "wall_s": round(time.time()-t0, 1)}


def analyze(rows, domain=(-1.0, 1.0), n_union=N_UNION, splits=None,
            fns=None):
    """Pairwise geometry of the half-solutions."""
    torch.set_default_dtype(torch.float64)
    out = []
    splits = splits or SPLITS
    fns = fns or FNS
    x_all = np.linspace(domain[0], domain[1], n_union)
    x_ev = np.linspace(-1, 1, run.N_EVAL)
    centers, gamma, _ = run.geometry(NWIDTH, run.LAM)
    W = centers.size
    xe = torch.tensor(x_ev).unsqueeze(1)
    xu = torch.tensor(x_all).unsqueeze(1)

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    by = {(r["fn"], r["cond"], r["tag"]): r for r in rows}
    for fn in fns:
        f_vec = run.make_fn(run.TARGETS[fn])
        for cond in ("clean", "noisy"):
            y_all = f_vec(x_all)
            if cond == "noisy":
                y_all = y_all + NOISE * np.random.default_rng(
                    999).standard_normal(n_union)
            yu = torch.tensor(y_all)
            yu_norm = float(torch.linalg.norm(yu))
            f_norm = float(torch.linalg.norm(torch.tensor(f_vec(x_ev))))
            for sp in splits:
                A = by.get((fn, cond, f"{sp}-A"))
                Br = by.get((fn, cond, f"{sp}-B"))
                if not (A and Br):
                    continue
                thA = torch.tensor(A["theta"])
                thB = torch.tensor(Br["theta"])
                with torch.no_grad():
                    fA = model(xe, thA)
                    fB = model(xe, thB)
                    fdist = float(torch.linalg.norm(fA - fB)) / f_norm
                    dw = float((thA[:W] - thB[:W]).abs().max())
                    db = float((thA[W:2*W] - thB[W:2*W]).abs().max())
                    dv = (float(torch.linalg.norm(
                        thA[2*W:3*W] - thB[2*W:3*W]))
                        / max(float(torch.linalg.norm(thB[2*W:3*W])),
                              1e-300))
                    iA, iB = make_split(sp, n_union)
                    xb = torch.tensor(x_all[iB]).unsqueeze(1)
                    yb = torch.tensor(y_all[iB])
                    ybn = float(torch.linalg.norm(yb))
                    cross = (float(torch.linalg.norm(model(xb, thA) - yb))
                             / ybn)
                    own = (float(torch.linalg.norm(model(xb, thB) - yb))
                           / ybn)
                    path = []
                    for t in np.linspace(0, 1, 21):
                        th = (1 - t) * thA + t * thB
                        path.append(float(torch.linalg.norm(
                            model(xu, th) - yu)) / yu_norm)
                out.append({"fn": fn, "cond": cond, "split": sp,
                            "fdist": fdist, "dw_over_gamma": dw / gamma,
                            "db_over_gamma": db / gamma, "dv_rel": dv,
                            "crossA_on_B": cross, "ownB_on_B": own,
                            "evalA": A["eval_rel"], "evalB": Br["eval_rel"],
                            "path": path})
    return out


def render(pairs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for i, cond in enumerate(("clean", "noisy")):
        for j, fn in enumerate(FNS):
            # metrics panel
            ax = axes[i, 2 * j]
            keys = ["evalA", "evalB", "fdist", "crossA_on_B", "ownB_on_B",
                    "dv_rel"]
            labels = ["eval A", "eval B", "f-dist", "A on B", "B on B",
                      "rel dv"]
            ps = [p for p in pairs if p["fn"] == fn and p["cond"] == cond]
            for k, (key, lb) in enumerate(zip(keys, labels)):
                vals = [max(p[key], 1e-17) for p in ps]
                ax.scatter([k] * len(vals), vals, s=22)
            ax.set_yscale("log")
            ax.set_ylim(1e-17, 1.0)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, fontsize=7)
            ax.set_title(f"{fn} -- {cond}: half-solution geometry",
                         fontsize=9)
            ax.grid(True, alpha=0.25)
            # interpolation panel
            ax = axes[i, 2 * j + 1]
            for p in ps:
                ax.semilogy(np.linspace(0, 1, 21), p["path"], lw=1.1,
                            label=p["split"])
            ax.set_ylim(1e-17, 1.0)
            ax.set_title(f"{fn} -- {cond}: rel on union along "
                         r"$(1{-}t)\theta_A + t\theta_B$", fontsize=9)
            ax.set_xlabel("t")
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="lower center",
                          bbox_to_anchor=(0.5, 1.15), ncol=4)
    fig.suptitle("basin geography under data resampling: does the target "
                 "move? (halves of n=4006, each solved full-batch, N=256)",
                 y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    if args.render_only:
        rows = json.loads((BASIN_DIR / "solutions.json").read_text())
        pairs = analyze(rows)
        (BASIN_DIR / "pairs.json").write_text(json.dumps(pairs))
        render(pairs, BASIN_DIR / "basin_geography.png")
        return

    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = []
    for fn in FNS:
        for cond in ("clean", "noisy"):
            for sp in SPLITS:
                iA, iB = make_split(sp)
                jobs.append({"fn": fn, "cond": cond, "tag": f"{sp}-A",
                             "idx": iA, "threads": threads})
                jobs.append({"fn": fn, "cond": cond, "tag": f"{sp}-B",
                             "idx": iB, "threads": threads})
            jobs.append({"fn": fn, "cond": cond, "tag": "union",
                         "idx": np.arange(N_UNION), "threads": threads})
    print(f"basin study | {len(jobs)} solves | n={N_UNION}, N={NWIDTH}, "
          f"splits={SPLITS}, noise={NOISE:g} (noisy cond)")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(solve_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['fn']:6s} {r['cond']:6s} "
                  f"{r['tag']:9s} train={r['train_rel']:.2e} "
                  f"eval={r['eval_rel']:.2e} trips={r['trips']} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"solves done in {time.time()-t0:.1f}s")
    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    (BASIN_DIR / "solutions.json").write_text(json.dumps(rows))
    pairs = analyze(rows)
    (BASIN_DIR / "pairs.json").write_text(json.dumps(pairs))
    render(pairs, BASIN_DIR / "basin_geography.png")


if __name__ == "__main__":
    main()
