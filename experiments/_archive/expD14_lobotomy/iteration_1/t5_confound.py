"""iteration_1 / t5 -- the confound grid.

5 arms x 3 targets x {qi, clustered: 1 seed; datagap, rand, rand_scale: 3
seeds}, N = 128, 2500 steps. Every run logs the floor trajectory, coherent
travel, corrected SNR, region-split eval, and honest pass counts.

Resumable: completed (arm, fn, case, seed) cells found in the JSONL are
skipped, so the script can be re-run after an interruption.

  uv run --extra dev python experiments/expD14_lobotomy/iteration_1/t5_confound.py [workers]
"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

N = 128
ITERS = 2500
TARGETS = ("sine", "sine_8pi", "runge")
CASE_SEEDS = {"qi": [0], "clustered": [0],
              "datagap": [0, 1, 2], "rand": [0, 1, 2], "rand_scale": [0, 1, 2]}
ARM_NAMES = ("adam", "full", "ls", "lobo0", "tau1")
OUT = "t5_confound.jsonl"


def _load_core():
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    s = importlib.util.spec_from_file_location("d14_core1", HERE / "core1.py")
    c1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(c1)
    return c1


def run_cell(job):
    arm, fn, case, seed = job
    c1 = _load_core()
    t0 = time.time()
    env = c1.build_case(fn, N, case=case, seed=seed)
    res = c1.train_lobo1(env, ITERS, seed=seed, **c1.ARMS[arm])
    rec = dict(arm=arm, fn=fn, case=case, seed=seed, N=N, iters=ITERS,
               floor0=env["floor0"], floor0_in=env["floor0_in"],
               floor0_out=env["floor0_out"],
               best_rel=res["best_rel"], final_rel=res["final_rel"],
               floor_final=res["floor_final"],
               floor_final_in=res["floor_final_in"],
               floor_final_out=res["floor_final_out"],
               passes=res["passes"], B_final=res["B_final"],
               n_member_changes=res["n_member_changes"],
               n_discoveries=res["n_discoveries"],
               wall=round(time.time() - t0, 1),
               hist=res["hist"], fhist=res["fhist"], thist=res["thist"])
    return rec


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    c1 = _load_core()
    out = c1.RESULTS / OUT
    done = {(r["arm"], r["fn"], r["case"], r["seed"]) for r in c1.load(out)}
    jobs = [(arm, fn, case, seed)
            for arm in ARM_NAMES for fn in TARGETS
            for case, seeds in CASE_SEEDS.items() for seed in seeds
            if (arm, fn, case, seed) not in done]
    print(f"{len(jobs)} cells to run ({len(done)} already done), "
          f"{workers} workers")
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            c1.append(out, rec)
            print(f"[{k}/{len(jobs)}] {rec['arm']:6s} {rec['fn']:9s} "
                  f"{rec['case']:10s} s{rec['seed']}  "
                  f"rel {rec['final_rel']:.2e}  "
                  f"floor {rec['floor0']:.1e} -> {rec['floor_final']:.1e}  "
                  f"({rec['wall']:.0f}s, total {(time.time()-t0)/60:.1f}m)",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
