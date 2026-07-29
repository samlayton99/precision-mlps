"""Staircase study (iteration_6): the step-function loss is the trip cycle.

Diagnosis (iteration_6/staircase_diagnosis.png): iter5 (fixed tether) has
NO staircase; every iter6 step edge aligns with a tether trip. Between
trips the run sits at the drift equilibrium of the current T; the trip
ratchets T and resets the basis; the fresh deep solve drops several
orders; and trips STOP once drift < TRIP_FRAC*||r||, capping the final
precision ~2 orders above the drift. Three single-variable fixes:

  tight       trip_mode="noise": trip whenever drift exceeds the gauge's
              own noise floor -- ratchets continue until geometry damage
              reaches rounding (removes the equilibrium cap).
  bigratchet  ratchet 1e4 instead of 1e2 -- reach locking T in ~2 trips
              (fewer resets, shorter staircase).
  keepbasis   carry the basis across the metric change (rescale +
              re-orthonormalize) -- removes the rebuild cost per stair.
  all3        the three combined (run only to check composition).

Grid: 4 gif targets x N in {64, 256, 512}; baseline = iteration_6 rows.
Also --snap-s: rerun the 4 N=256 cells with tether-magnitude snapshots
for the S-gif (same deterministic runs, snaps enriched with S).

Run:  uv run python experiments/expD08_qi_init_nlcg/staircase.py [--snap-s]
Outputs under results/.../expD08_qi_init_nlcg/iteration_6/staircase/
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

STAIR_DIR = run.OUT_DIR / "staircase"
FNS = run.TARGETS3
NS = [64, 256, 512]
VARIANTS = {
    "tight": {"trip_mode": "noise"},
    "bigratchet": {"ratchet": 1e4},
    "keepbasis": {"keep_basis": True},
    "all3": {"trip_mode": "noise", "ratchet": 1e4, "keep_basis": True},
}


def stair_case(job):
    r = run.run_case({"fn": job["fn"], "N": job["N"], "iters": run.ITERS,
                      "threads": job["threads"], "save_snaps": False,
                      **VARIANTS[job["variant"]]})
    return {"variant": job["variant"], "fn": r["fn"], "N": r["N"],
            "floor": r["floor"], "best_rel": r["best_rel"],
            "rels": r["rels"], "trips": r["stats"]["trips"],
            "T_final": r["stats"]["T_final"], "wall_s": r["wall_s"]}


def snap_s_case(job):
    r = run.run_case({"fn": job["fn"], "N": 256, "iters": run.ITERS,
                      "threads": job["threads"], "save_snaps": True,
                      "snap_S": True})
    return {"fn": r["fn"], "best_rel": r["best_rel"], "wall_s": r["wall_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-s", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    threads = max(1, (os.cpu_count() or 4) // args.workers)

    if args.snap_s:
        jobs = [{"fn": f, "threads": threads} for f in FNS]
        with ProcessPoolExecutor(max_workers=min(4, args.workers)) as ex:
            for fut in as_completed([ex.submit(snap_s_case, j)
                                     for j in jobs]):
                r = fut.result()
                print(f"  snap_S {r['fn']:13s} best={r['best_rel']:.2e} "
                      f"({r['wall_s']}s)", flush=True)
        print(f"S-snapshots refreshed in iteration_{run.ITERATION}/snaps/")
        return

    jobs = [{"variant": v, "fn": f, "N": n, "threads": threads}
            for v in VARIANTS for f in FNS for n in NS]
    print(f"staircase study | {len(jobs)} runs | variants={list(VARIANTS)}")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(stair_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['variant']:10s} {r['fn']:13s} "
                  f"N={r['N']:<4d} best_rel={r['best_rel']:.3e} "
                  f"floor={r['floor']:.1e} trips={len(r['trips'])} "
                  f"T={r['T_final']:.0e} ({r['wall_s']}s)", flush=True)
    print(f"done in {time.time()-t0:.1f}s")
    STAIR_DIR.mkdir(parents=True, exist_ok=True)
    (STAIR_DIR / "staircase_rows.json").write_text(json.dumps(rows))

    base = {}
    for p in (run.OUT_DIR / "snaps").glob("*.npz"):
        z = np.load(p)
        fn, n_part = p.stem.split("--N")
        base[(fn, int(n_part))] = float(np.min(z["rels"]))
    print(f"\n{'variant':<11s} {'median orders vs iter6':>23s} "
          f"{'worst cell':>30s}")
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v
              and (r["fn"], r["N"]) in base]
        d = [np.log10(r["best_rel"] / base[(r["fn"], r["N"])]) for r in rs]
        wi = int(np.argmax(d))
        print(f"{v:<11s} {float(np.median(d)):>+23.2f} "
              f"{rs[wi]['fn']}@N{rs[wi]['N']}: "
              f"{rs[wi]['best_rel']:.1e} ({d[wi]:+.1f})")


if __name__ == "__main__":
    main()
