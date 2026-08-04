"""iteration_4 / t8 -- the held-out handoff grid.

Arms: fgen (the assembled optimizer) and none (uncooled, same fit/holdout
harness so the comparison is clean). Same grid as t6/t7. Resumable.

  uv run --extra dev python experiments/expD14_lobotomy/iteration_4/t8_fgen.py [workers]
"""
import importlib.util
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

N = 128
ITERS = 4000
TARGETS = ("sine", "sine_8pi", "runge")
CASE_SEEDS = {"qi": [0], "clustered": [0],
              "datagap": [0, 1, 2], "rand": [0, 1, 2], "rand_scale": [0, 1, 2]}
ARM_NAMES = ("fgen", "none", "fabs", "fabsc")
OUT = "t8_fgen.jsonl"


def _load_core():
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    s = importlib.util.spec_from_file_location("d14_core4", HERE / "core4.py")
    c4 = importlib.util.module_from_spec(s)
    s.loader.exec_module(c4)
    return c4


def run_cell(job):
    arm, fn, case, seed = job
    c4 = _load_core()
    t0 = time.time()
    env = c4.build_case(fn, N, case=case, seed=seed)
    res = c4.train_v4(env, ITERS, seed=seed, **c4.ARMS[arm])
    return dict(arm=arm, fn=fn, case=case, seed=seed, N=N, iters=ITERS,
                floor0=env["floor0"], floor0_in=env["floor0_in"],
                floor0_out=env["floor0_out"],
                best_rel=res["best_rel"], final_rel=res["final_rel"],
                floor_final=res["floor_final"],
                floor_final_in=res["floor_final_in"],
                floor_final_out=res["floor_final_out"],
                passes=res["passes"],
                n_member_changes=res["n_member_changes"],
                wall=round(time.time() - t0, 1),
                hist=res["hist"], fhist=res["fhist"], thist=res["thist"])


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    c4 = _load_core()
    out = c4.RESULTS / OUT
    done = {(r["arm"], r["fn"], r["case"], r["seed"]) for r in c4.load(out)}
    jobs = [(arm, fn, case, seed)
            for arm in ARM_NAMES for fn in TARGETS
            for case, seeds in CASE_SEEDS.items() for seed in seeds
            if (arm, fn, case, seed) not in done]
    print(f"{len(jobs)} cells to run ({len(done)} done), {workers} workers")
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            c4.append(out, rec)
            print(f"[{k}/{len(jobs)}] {rec['arm']:5s} {rec['fn']:9s} "
                  f"{rec['case']:10s} s{rec['seed']}  "
                  f"rel {rec['final_rel']:.2e}  "
                  f"floor {rec['floor0']:.1e} -> {rec['floor_final']:.1e}  "
                  f"({rec['wall']:.0f}s, total {(time.time()-t0)/60:.1f}m)",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
