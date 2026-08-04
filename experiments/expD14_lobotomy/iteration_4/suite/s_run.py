"""iteration_4 suite runner -- every suite computation in one resumable queue.

Kinds:
  scale   width scaling: sine_8pi, 4 scenarios, N in {32,64,128,256}
  loss    6 targets at N=128 from stock-random init
  gif     4 scenarios with theta snapshots every 50 steps (for the gifs)
  twod    the four 2-D ridge cases, fgen vs adam, with mu/signal logging
  dl2     2-hidden-layer 1-D task (1->32->32->1 tanh), fgen vs adam
  dltest  expD07's real-data suite (airfoil, parkinsons, bike_sharing)

  uv run --extra dev python experiments/expD14_lobotomy/iteration_4/suite/s_run.py [workers]
"""
import importlib.util
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITERS = 4000


def _load():
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    s = importlib.util.spec_from_file_location("suite_common",
                                               HERE / "common.py")
    sc = importlib.util.module_from_spec(s)
    s.loader.exec_module(sc)
    return sc


def jobs_list():
    jobs = []
    # scaling
    for N in (32, 64, 128, 256):
        for case, seeds in (("qi", [0]), ("clustered", [0]),
                            ("datagap", [0, 1]), ("rand", [0, 1])):
            for seed in seeds:
                for arm in ("fgen", "adam"):
                    jobs.append(dict(kind="scale", fn="sine_8pi", case=case,
                                     N=N, seed=seed, arm=arm))
    # loss curves
    for fn in ("sine", "sine_8pi", "runge", "sine_mixture", "exp",
               "abs_cubed"):
        for arm in ("fgen", "adam"):
            jobs.append(dict(kind="loss", fn=fn, case="rand", N=128, seed=0,
                             arm=arm))
    # gif snapshot runs (fgen and adam, for side-by-side)
    for case in ("qi", "clustered", "datagap", "rand"):
        for arm in ("fgen", "adam"):
            jobs.append(dict(kind="gif", fn="sine_8pi", case=case, N=128,
                             seed=0, arm=arm))
    # 2-D
    for case in ("qi", "clustered", "datagap", "random"):
        for arm in ("fgen", "adam"):
            jobs.append(dict(kind="twod", fn="target2d", case=case, N=48,
                             seed=0, arm=arm))
    # dl2
    for arm in ("fgen", "adam"):
        jobs.append(dict(kind="dl2", fn="sine_mixture", case="dl2", N=32,
                         seed=0, arm=arm))
    # dltest
    for task in ("airfoil", "parkinsons", "bike_sharing"):
        for arm in ("fgen", "adam"):
            jobs.append(dict(kind="dltest", fn=task, case="real", N=64,
                             seed=0, arm=arm))
    return jobs


def _dl_env(sc, task):
    import torch
    d = torch.load(sc.REPO / "data" / "cache_all20" / f"{task}.pt",
                   weights_only=True)
    xtr, ytr = d["xtr"].double(), d["ytr"].double()
    xte, yte = d["xte"].double(), d["yte"].double()
    d_in, Wd = int(d["d_in"]), 64
    torch.manual_seed(0)
    import numpy as np
    sizes = [(Wd, d_in), (Wd,), (Wd, Wd), (Wd,), (1, Wd), (1,)]
    th0, groups, off = [], {}, 0
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    for nm, shp in zip(names, sizes):
        fan = shp[1] if len(shp) == 2 else shp[0]
        t = (torch.rand(*shp, dtype=torch.float64) * 2 - 1) / np.sqrt(fan)
        th0.append(t.reshape(-1))
        groups[nm] = np.arange(off, off + t.numel())
        off += t.numel()
    th0 = torch.cat(th0)

    def model_fn(X, th):
        i = 0
        W1 = th[groups["W1"]].reshape(Wd, d_in)
        b1 = th[groups["b1"]]
        W2 = th[groups["W2"]].reshape(Wd, Wd)
        b2 = th[groups["b2"]]
        W3 = th[groups["W3"]].reshape(1, Wd)
        b3 = th[groups["b3"]]
        h = torch.tanh(X @ W1.T + b1)
        h = torch.tanh(h @ W2.T + b2)
        return (h @ W3.T + b3).squeeze(-1)

    return sc.env_model(model_fn, th0, groups, xtr, ytr, xte, yte)


def _dl2_env(sc, fn):
    import torch
    import numpy as np
    f_vec = sc.core4.d08run.make_fn(sc.core4.d08run.TARGETS[fn])
    x = np.linspace(-1, 1, 1200)
    rngsplit = np.arange(1200)
    xtr = torch.tensor(x).unsqueeze(1)
    ytr = torch.tensor(f_vec(x))
    xe = np.linspace(-1, 1, 2401)
    xte = torch.tensor(xe).unsqueeze(1)
    yte = torch.tensor(f_vec(xe))
    Wd = 32
    torch.manual_seed(0)
    sizes = [(Wd, 1), (Wd,), (Wd, Wd), (Wd,), (1, Wd), (1,)]
    th0, groups, off = [], {}, 0
    for nm, shp in zip(["W1", "b1", "W2", "b2", "W3", "b3"], sizes):
        fan = shp[1] if len(shp) == 2 else shp[0]
        t = (torch.rand(*shp, dtype=torch.float64) * 2 - 1) / np.sqrt(fan)
        th0.append(t.reshape(-1))
        groups[nm] = np.arange(off, off + t.numel())
        off += t.numel()
    th0 = torch.cat(th0)

    def model_fn(X, th):
        W1 = th[groups["W1"]].reshape(Wd, 1)
        b1 = th[groups["b1"]]
        W2 = th[groups["W2"]].reshape(Wd, Wd)
        b2 = th[groups["b2"]]
        W3 = th[groups["W3"]].reshape(1, Wd)
        b3 = th[groups["b3"]]
        h = torch.tanh(X @ W1.T + b1)
        h = torch.tanh(h @ W2.T + b2)
        return (h @ W3.T + b3).squeeze(-1)

    return sc.env_model(model_fn, th0, groups, xtr, ytr, xte, yte)


def run_cell(job):
    sc = _load()
    t0 = time.time()
    kind, arm = job["kind"], job["arm"]
    iters = ITERS
    snap = 50 if kind == "gif" else 0
    if kind in ("scale", "loss", "gif"):
        env = sc.env_1d(job["fn"], job["N"], case=job["case"],
                        seed=job["seed"])
    elif kind == "twod":
        env = sc.env_2d(job["case"], seed=job["seed"], N=job["N"])
        iters = 2000
    elif kind == "dl2":
        env = _dl2_env(sc, job["fn"])
        iters = 4000
    elif kind == "dltest":
        env = _dl_env(sc, job["fn"])
        iters = 3000
    else:
        raise ValueError(kind)

    if arm == "fgen":
        res = sc.train_generic(env, iters, cool="fabsc", seed=job["seed"],
                               snap_every=snap)
    else:
        res = sc.train_adam(env, iters, snap_every=snap)

    rec = dict(job, iters=iters,
               floor0=env.get("floor0", float("nan")),
               best_rel=res["best_rel"], final_rel=res["final_rel"],
               floor_final=res["floor_final"],
               wall=round(time.time() - t0, 1),
               hist=res["hist"], fhist=res["fhist"])
    if snap:
        rec["snaps"] = res["snaps"]
        rec["W"] = env["W"]
    return rec


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sc = _load()
    out = sc.RESULTS / "suite.jsonl"
    done = {(r["kind"], r["fn"], r["case"], r["N"], r["seed"], r["arm"])
            for r in sc.load(out)}
    jobs = [j for j in jobs_list()
            if (j["kind"], j["fn"], j["case"], j["N"], j["seed"], j["arm"])
            not in done]
    print(f"{len(jobs)} suite cells to run ({len(done)} done)")
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            sc.append(out, rec)
            print(f"[{k}/{len(jobs)}] {rec['kind']:6s} {rec['fn']:12s} "
                  f"{rec['case']:9s} N{rec['N']:<3d} {rec['arm']:4s}  "
                  f"rel {rec['final_rel']:.2e}  floor {rec['floor_final']:.2e}"
                  f"  ({rec['wall']:.0f}s, total {(time.time()-t0)/60:.1f}m)",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
