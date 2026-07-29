"""expD08 halo study: do data-blind halo neurons make batching noise hard?

Sam's hunch: batching adds noise, and noise combined with a high density of
centers carrying NO data (the halo: |c_k| > 1 while training data stops at
[-1, 1]) led to wild solutions in the generalization experiments. Test:
give the halo data. Two arms, everything else identical:
  nohalo  x_train on [-1, 1]           (the archived batching/basin setup)
  halo    x_train on [c_min, c_max]    (the full halo support), SAME density

Design decisions, stated up front:
  * density-matched, not n-matched: the halo arm has n scaled by domain
    length (batching: 10000 -> 17969 at N=256), so local information per
    center is equal and only the COVERAGE differs. n changes as a
    consequence; floors are recomputed per arm and comparisons are made
    floor-relative as well as absolute.
  * eval is ALWAYS rel L2 on 4001 points of [-1, 1] vs the true function --
    the same metric as every other expD08 experiment.
  * nohalo batching rows/snaps are REUSED from the archived study
    (batching/slim, deterministic seed 0, code path unchanged --
    verified by --verify, which reruns one cell and requires an exact
    best_rel match). The basin nohalo arm is rerun (best-theta fix).

Parts (each caches JSON so --render works standalone):
  --verify   rerun sine B/16 nohalo, compare to the archived row
  --part1    halo-arm batching grid: 4 targets x {B, B/4, B/16, B/64,
             B/256} fresh (guards=(), fixed tether, slim core, snaps on)
             + carried-mode control at B/16 and B/64
  --part2    basin study both arms (best-theta fix): sine clean+noisy,
             runge clean; splits rand0/rand1/evenodd; halo union n=7198
  --nulldim  [Phi, 1] rank + tail spectrum vs data extent (no training)
  --render   all figures from cached rows

Run:  uv run python experiments/expD08_qi_init_nlcg/halo_study.py --part1 ...
Outputs under results/checkpoint_D_optimizers/expD08_qi_init_nlcg/batching/halo_study/
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import batching  # noqa: E402
import basin_study  # noqa: E402

HALO_DIR = run.OUT_DIR.parent / "batching" / "halo_study"
FIG_DIR = HALO_DIR / "figures"
N = 256
FNS = run.TARGETS3                    # sine, sine_8pi, runge, exp
LEVELS = [("B", 1), ("B/4", 4), ("B/16", 16), ("B/64", 64), ("B/256", 256)]
CARRIED_LEVELS = [("B/16", 16), ("B/64", 64)]
ITERS = 3000

_centers, _gamma, _halo = run.geometry(N, run.LAM)
CMAX = float(_centers.max())          # 1.7969 at N=256
DENS_B = batching.N_TRAIN_B / 2.0     # 5000 samples per unit length
N_TR_HALO = int(round(DENS_B * 2 * CMAX))          # 17969
DENS_U = basin_study.N_UNION / 2.0    # 2003 per unit length
N_UNION_HALO = int(round(DENS_U * 2 * CMAX)) // 2 * 2   # even: 7198
ARMS = {"nohalo": ((-1.0, 1.0), batching.N_TRAIN_B),
        "halo": ((-CMAX, CMAX), N_TR_HALO)}
SPLITS2 = ["rand0", "rand1", "evenodd"]
BASIN_CELLS = [("sine", "clean"), ("sine", "noisy"), ("runge", "clean")]


# ------------------------------- part 1 -------------------------------

def _batch_job(fn, blabel, div, guards, threads, save_snaps):
    return {"fn": fn, "N": N, "blabel": blabel, "div": div, "opt": "slim",
            "iters": ITERS, "threads": threads, "guards": guards,
            "domain": (-CMAX, CMAX), "n_train": N_TR_HALO,
            "save_snaps": save_snaps, "snap_dir": str(HALO_DIR / "snaps")}


def part1(workers):
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = []
    for fn in FNS:
        for blabel, div in LEVELS:
            jobs.append(_batch_job(fn, blabel, div, (), threads, True))
        for blabel, div in CARRIED_LEVELS:
            jobs.append(_batch_job(fn, blabel, div, ("carried",), threads,
                                   False))
    print(f"halo part1 | {len(jobs)} runs | n={N_TR_HALO} on "
          f"[-{CMAX:.4f},{CMAX:.4f}] | N={N} | iters={ITERS}")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(batching.batch_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            r.pop("snaps", None)
            r["arm"] = "halo"
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['fn']:10s} {r['blabel']:6s} "
                  f"guards={r['guards']} best={r['best_rel']:.2e} "
                  f"floor={r['floor']:.1e} ({r['wall_s']}s)", flush=True)
    HALO_DIR.mkdir(parents=True, exist_ok=True)
    (HALO_DIR / "batch_rows.json").write_text(json.dumps(rows))
    print(f"part1 done in {time.time()-t0:.1f}s -> batch_rows.json")


def verify(workers):
    """Determinism check: rerun one archived nohalo cell, require match."""
    threads = max(1, (os.cpu_count() or 4) // workers)
    job = {"fn": "sine", "N": N, "blabel": "B/16", "div": 16, "opt": "slim",
           "iters": ITERS, "threads": threads * workers, "guards": ()}
    r = batching.batch_case(job)
    arch = [x for x in json.loads(
        (batching.out_dir("slim") / "batching_rows.json").read_text())
        if x["fn"] == "sine" and x["N"] == N and x["blabel"] == "B/16"][0]
    match = np.isclose(r["best_rel"], arch["best_rel"], rtol=1e-12)
    print(f"verify: rerun best={r['best_rel']:.6e} "
          f"archived={arch['best_rel']:.6e} match={bool(match)}")
    if not match:
        raise SystemExit("archived nohalo rows are NOT reproducible -- "
                         "rerun the nohalo arm instead of reusing")


# ------------------------------- part 2 -------------------------------

def part2(workers):
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = []
    for arm, (dom, _) in ARMS.items():
        n_u = basin_study.N_UNION if arm == "nohalo" else N_UNION_HALO
        for fn, cond in BASIN_CELLS:
            for sp in SPLITS2:
                iA, iB = basin_study.make_split(sp, n_u)
                for tag, idx in ((f"{sp}-A", iA), (f"{sp}-B", iB)):
                    jobs.append({"fn": fn, "cond": cond, "tag": tag,
                                 "idx": idx, "threads": threads,
                                 "domain": dom, "n_union": n_u,
                                 "arm": arm})
            jobs.append({"fn": fn, "cond": cond, "tag": "union",
                         "idx": np.arange(n_u), "threads": threads,
                         "domain": dom, "n_union": n_u, "arm": arm})
    print(f"halo part2 | {len(jobs)} solves | nohalo n={basin_study.N_UNION}"
          f" halo n={N_UNION_HALO} | N={N}")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(basin_study.solve_case, j): j["arm"] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            r["arm"] = futs[fut]
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['arm']:6s} {r['fn']:6s} "
                  f"{r['cond']:6s} {r['tag']:9s} train={r['train_rel']:.2e} "
                  f"eval={r['eval_rel']:.2e} ({r['wall_s']}s)", flush=True)
    HALO_DIR.mkdir(parents=True, exist_ok=True)
    (HALO_DIR / "basin_solutions.json").write_text(json.dumps(rows))
    pairs = []
    for arm, (dom, _) in ARMS.items():
        n_u = basin_study.N_UNION if arm == "nohalo" else N_UNION_HALO
        sub = [r for r in rows if r["arm"] == arm]
        ps = basin_study.analyze(sub, domain=dom, n_union=n_u,
                                 splits=SPLITS2, fns=["sine", "runge"])
        for p in ps:
            p["arm"] = arm
        pairs.extend(ps)
    (HALO_DIR / "basin_pairs.json").write_text(json.dumps(pairs))
    print(f"part2 done in {time.time()-t0:.1f}s -> basin_solutions.json, "
          "basin_pairs.json")


# ------------------------------- nulldim -------------------------------

def nulldim():
    """Rank and tail spectrum of [Phi, 1] vs data extent, per width."""
    out = {"extents": {}, "spectra": {}}
    for n_width in (128, 256, 512):
        centers, gamma, _ = run.geometry(n_width, run.LAM)
        W = centers.size
        cmax = float(centers.max())
        dens = DENS_B
        rows = []
        for e in np.linspace(1.0, cmax, 9):
            n_e = int(round(dens * 2 * e))
            x = np.linspace(-e, e, n_e)
            A = np.hstack([run.build_phi(x, gamma, centers, "tanh"),
                           np.ones((n_e, 1))])
            s = np.linalg.svd(A, compute_uv=False)
            rank = int((s > run.RCOND * s[0]).sum())
            rows.append({"extent": float(e), "n": n_e, "cols": W + 1,
                         "rank": rank, "nulldim": W + 1 - rank})
            if n_width == N and float(e) in (1.0, cmax) or (
                    n_width == N and abs(e - (1 + cmax) / 2) < 0.06):
                out["spectra"][f"{e:.4f}"] = (s / s[0]).tolist()
        out["extents"][str(n_width)] = rows
        print(f"  N={n_width}: nulldim {rows[0]['nulldim']} at e=1.0 -> "
              f"{rows[-1]['nulldim']} at e={cmax:.3f}")
    HALO_DIR.mkdir(parents=True, exist_ok=True)
    (HALO_DIR / "nulldim.json").write_text(json.dumps(out))
    print("nulldim -> nulldim.json")


# ------------------------------- figures -------------------------------

def _load(name):
    return json.loads((HALO_DIR / name).read_text())


def render_batch_scaling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    halo_rows = _load("batch_rows.json")
    noh_rows = json.loads(
        (batching.out_dir("slim") / "batching_rows.json").read_text())
    noh_carr = json.loads((batching.BATCH_ROOT / "carried"
                           / "carried_rows.json").read_text())
    xs = np.arange(len(LEVELS))
    labels = [lb for lb, _ in LEVELS]

    def pick(rows, fn, blabel, guards=None):
        for r in rows:
            if (r["fn"] == fn and r.get("N", N) == N
                    and r["blabel"] == blabel
                    and (guards is None
                         or sorted(r.get("guards", [])) == sorted(guards))):
                return r
        return None

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for j, fn in enumerate(FNS):
        noh = [pick(noh_rows, fn, lb) for lb, _ in LEVELS]
        hal = [pick(halo_rows, fn, lb, guards=[]) for lb, _ in LEVELS]
        fn_noh = noh[0]["floor"]
        fn_hal = hal[0]["floor"]
        for i, rel in enumerate((False, True)):
            ax = axes[i, j]
            yn = [r["best_rel"] / (fn_noh if rel else 1.0) for r in noh]
            yh = [r["best_rel"] / (fn_hal if rel else 1.0) for r in hal]
            ax.plot(xs, yn, "o-", color="#888888",
                    label="no-halo data [-1,1]")
            ax.plot(xs, yh, "o-", color="#1f77b4",
                    label=f"halo data [-{CMAX:.2f},{CMAX:.2f}]")
            for lbl, div in CARRIED_LEVELS:
                k = labels.index(lbl)
                rn = pick(noh_carr, fn, lbl)
                rh = pick(halo_rows, fn, lbl, guards=["carried"])
                if rn:
                    ax.plot(k, rn["best_rel"] / (fn_noh if rel else 1.0),
                            "*", ms=13, color="#555555",
                            label="carried (no-halo)" if k == labels.index(
                                CARRIED_LEVELS[0][0]) and j == 0 else None)
                if rh:
                    ax.plot(k, rh["best_rel"] / (fn_hal if rel else 1.0),
                            "*", ms=13, color="#d62728",
                            label="carried (halo)" if k == labels.index(
                                CARRIED_LEVELS[0][0]) and j == 0 else None)
            if not rel:
                ax.axhline(fn_noh, color="#888888", ls=":", lw=1.0)
                ax.axhline(fn_hal, color="#1f77b4", ls=":", lw=1.0)
                ax.set_ylim(1e-16, 1e1)
                if j == 0:
                    ax.set_ylabel("best eval rel L2\n(dotted = lstsq floor)")
            else:
                ax.axhline(1.0, color="k", ls=":", lw=1.0)
                ax.set_ylim(1e-1, 1e15)
                if j == 0:
                    ax.set_ylabel("best rel / own floor")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(fn)
            if i == 1:
                ax.set_xticks(xs)
                ax.set_xticklabels(labels)
                ax.set_xlabel("batch size")
    handles, lbs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4)
    fig.suptitle("per-batch slim, halo-covered vs [-1,1] data   "
                 f"(N={N}, density-matched: n=10000 vs n={N_TR_HALO}, "
                 "epoch reshuffle, 3000 iters)", y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "batch_scaling_halo_vs_nohalo.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def render_batch_trajectories():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    halo_rows = _load("batch_rows.json")
    noh_rows = json.loads(
        (batching.out_dir("slim") / "batching_rows.json").read_text())

    def pick(rows, fn, blabel, guards=None):
        for r in rows:
            if (r["fn"] == fn and r.get("N", N) == N
                    and r["blabel"] == blabel
                    and (guards is None
                         or sorted(r.get("guards", [])) == sorted(guards))):
                return r
        return None

    show = ["B/4", "B/16", "B/64"]
    fig, axes = plt.subplots(len(show), 4, figsize=(16, 9), sharex=True,
                             sharey=True)
    for i, lb in enumerate(show):
        for j, fn in enumerate(FNS):
            ax = axes[i, j]
            rn = pick(noh_rows, fn, lb)
            rh = pick(halo_rows, fn, lb, guards=[])
            for r, col, lab in ((rn, "#888888", "no-halo"),
                                (rh, "#1f77b4", "halo")):
                if r:
                    y = np.array(r["rels"])
                    ax.plot(np.arange(1, len(y) + 1), y, color=col, lw=0.8,
                            label=lab)
                    ax.axhline(r["floor"], color=col, ls=":", lw=0.9)
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 1e2)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(fn)
            if j == 0:
                ax.set_ylabel(f"{lb}\neval rel L2")
            if i == len(show) - 1:
                ax.set_xlabel("iteration")
    handles, lbs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2)
    fig.suptitle("per-batch slim trajectories, halo vs no-halo data   "
                 f"(N={N}; dotted = own lstsq floor)", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = FIG_DIR / "batch_trajectories_halo_vs_nohalo.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def render_excursions():
    """Wildness metric from snaps: per frame, max |w - w0| / gamma over
    halo neurons vs interior neurons, both arms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    halo_mask = np.abs(_centers) > 1.0
    lb, tag = "B/16", "Bd16"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for j, fn in enumerate(FNS):
        zn = np.load(batching.out_dir("slim") / "snaps"
                     / f"{fn}--N{N}--{tag}.npz")
        zh = np.load(HALO_DIR / "snaps" / f"{fn}--N{N}--{tag}.npz")
        for i, (blk, ttl) in enumerate((("w", "inner weight |w - w0| / "
                                         "gamma"),
                                        ("v", "readout |v| magnitude"))):
            ax = axes[i, j]
            for z, col, arm in ((zn, "#888888", "no-halo"),
                                (zh, "#1f77b4", "halo")):
                steps = z["steps"].astype(int)
                X = z[blk]
                dev = (np.abs(X - X[0]) / _gamma if blk == "w"
                       else np.abs(X))
                ax.plot(steps, dev[:, halo_mask].max(axis=1), "-", lw=1.5,
                        color=col, label=f"{arm}: halo neurons")
                ax.plot(steps, dev[:, ~halo_mask].max(axis=1), "--", lw=1.1,
                        color=col, alpha=0.8,
                        label=f"{arm}: interior neurons")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(fn)
            if j == 0:
                ax.set_ylabel(f"max {ttl}")
            if i == 1:
                ax.set_xlabel("iteration")
    handles, lbs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4)
    fig.suptitle(f"parameter excursions at {lb}, halo vs interior neurons  "
                 f"(N={N}, per-batch slim; halo neuron = |center| > 1)",
                 y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = FIG_DIR / "excursions_halo_vs_nohalo.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def render_nulldim():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load("nulldim.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for nw, col in (("128", "#2ca02c"), ("256", "#1f77b4"),
                    ("512", "#d62728")):
        rows = data["extents"][nw]
        ax.plot([r["extent"] for r in rows], [r["nulldim"] for r in rows],
                "o-", color=col, label=f"N={nw}")
    ax.set_xlabel("data extent e  (train x in [-e, e])")
    ax.set_ylabel("null dimension of [Phi, 1]  (cols - rank at rcond 1e-15)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              borderaxespad=0)
    ax = axes[1]
    for key, col in zip(sorted(data["spectra"]),
                        ("#888888", "#1f77b4", "#d62728")):
        s = np.array(data["spectra"][key])
        ax.semilogy(np.arange(s.size), s, lw=1.2, color=col,
                    label=f"extent {key}")
    ax.axhline(run.RCOND, color="k", ls=":", lw=1.0)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma_i / sigma_0   (N=256; dotted = rcond)")
    ax.set_ylim(1e-18, 1.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              borderaxespad=0)
    fig.suptitle("how much null space is a data artifact: rank of "
                 "[Phi, 1] vs data extent (density 5000/unit)", y=1.06)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "nulldim_vs_extent.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def render_basin():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = _load("basin_pairs.json")
    cells = BASIN_CELLS
    keys = ["fdist", "dw_over_gamma", "db_over_gamma", "dv_rel",
            "crossA_on_B"]
    labels = ["f-dist", "max dw / gamma", "max db / gamma", "rel dv",
              "A on B's data"]
    fig, axes = plt.subplots(2, len(cells), figsize=(15, 8))
    for j, (fn, cond) in enumerate(cells):
        ax = axes[0, j]
        for arm, col, dx in (("nohalo", "#888888", -0.12),
                             ("halo", "#1f77b4", +0.12)):
            ps = [p for p in pairs if p["fn"] == fn and p["cond"] == cond
                  and p["arm"] == arm]
            for k, key in enumerate(keys):
                vals = [max(p[key], 1e-17) for p in ps]
                ax.scatter([k + dx] * len(vals), vals, s=26, color=col,
                           label=arm if k == 0 and j == 0 else None)
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1e1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, rotation=25, fontsize=7)
        ax.set_title(f"{fn} -- {cond}: half-solution geometry", fontsize=10)
        ax.grid(alpha=0.25)
        ax = axes[1, j]
        for arm, col in (("nohalo", "#888888"), ("halo", "#1f77b4")):
            ps = [p for p in pairs if p["fn"] == fn and p["cond"] == cond
                  and p["arm"] == arm]
            for p in ps:
                ax.semilogy(np.linspace(0, 1, 21), p["path"], lw=1.0,
                            color=col, alpha=0.8)
        ax.set_ylim(1e-17, 1.0)
        ax.set_xlabel("t")
        ax.set_title(r"rel on union along $(1{-}t)\theta_A + t\theta_B$",
                     fontsize=9)
        ax.grid(alpha=0.25)
    handles, lbs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbs, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2)
    fig.suptitle("basin geography with vs without halo data   (N=256, "
                 "halves solved full-batch, best-theta tracked; "
                 f"nohalo n={basin_study.N_UNION}, halo n={N_UNION_HALO})",
                 y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "basin_halo_vs_nohalo.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--part1", action="store_true")
    ap.add_argument("--part2", action="store_true")
    ap.add_argument("--nulldim", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.verify:
        verify(args.workers)
    if args.part1:
        part1(args.workers)
    if args.part2:
        part2(args.workers)
    if args.nulldim:
        nulldim()
    if args.render:
        render_nulldim()
        render_batch_scaling()
        render_batch_trajectories()
        render_excursions()
        render_basin()


if __name__ == "__main__":
    main()
