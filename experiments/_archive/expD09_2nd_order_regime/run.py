"""expD09 driver -- run every (problem, solver) pair, cache to runs.jsonl.

The cache is keyed (problem_id, solver_id, sweeps, seed): adding a solver to
`solvers.py` and re-running executes only the missing pairs. Figures are
regenerated from the cache every time.

Usage:
  uv run python experiments/expD09_2nd_order_regime/run.py            # run + plot
  uv run python experiments/expD09_2nd_order_regime/run.py --plot     # plot only
  uv run python experiments/expD09_2nd_order_regime/run.py --only bcd_contig_k64
  uv run python experiments/expD09_2nd_order_regime/run.py --force    # ignore cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import common  # noqa: E402
import solvers  # noqa: E402
from common import Metrics, append_run, load_manifest, load_problem, load_runs  # noqa: E402

SWEEPS = 400          # > rank (max 273) for every cell, so Krylov gets its
                      # exact-arithmetic termination budget and BCD gets room
SEED = 0


def run_pair(entry: dict, spec: dict, sweeps: int) -> dict:
    prob = load_problem(entry)
    met = Metrics(prob)
    t0 = time.time()
    for a in solvers.iterate(prob["A"], prob["y"], spec, sweeps, seed=SEED):
        met.record(a)
    ev = np.asarray(met.rows["eval_rel"])
    return {
        "problem": entry["id"], "fn": entry["fn"], "N": entry["N"],
        "solver": spec["id"], "sweeps": sweeps, "seed": SEED,
        "n": entry["n"], "d": entry["d"], "rank": entry["rank"],
        "floor_train": entry["floor_train_rel_l2"],
        "floor_eval": entry["floor_eval_rel_l2"],
        "flops_per_sweep": solvers.flops_per_sweep(spec),
        "state_floats": solvers.state_floats(spec, entry["d"], entry["rank"]),
        "best_eval_rel": float(ev.min()),
        "best_eval_sweep": int(ev.argmin()) + 1,
        "wall_s": round(time.time() - t0, 2),
        **met.rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="plot from cache only")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--only", nargs="*", default=None, help="solver ids")
    ap.add_argument("--sweeps", type=int, default=SWEEPS)
    args = ap.parse_args()

    manifest = load_manifest()
    specs = solvers.SOLVERS
    if args.only:
        specs = [s for s in specs if s["id"] in set(args.only)]

    if not args.plot:
        done = set() if args.force else {
            (r["problem"], r["solver"], r["sweeps"]) for r in load_runs()}
        todo = [(e, s) for s in specs for e in manifest
                if (e["id"], s["id"], args.sweeps) not in done]
        print(f"{len(todo)} (problem, solver) pairs to run "
              f"[{len(manifest)} problems x {len(specs)} solvers]")
        for i, (entry, spec) in enumerate(todo, 1):
            rec = run_pair(entry, spec, args.sweeps)
            append_run(rec)
            print(f"[{i:3d}/{len(todo)}] {rec['problem']:22s} "
                  f"{rec['solver']:26s} best eval {rec['best_eval_rel']:.2e} "
                  f"(floor {rec['floor_eval']:.2e}, "
                  f"sweep {rec['best_eval_sweep']}, {rec['wall_s']}s)")

    import plotting
    plotting.render_all()


if __name__ == "__main__":
    main()
