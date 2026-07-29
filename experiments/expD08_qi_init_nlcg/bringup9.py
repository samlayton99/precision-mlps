"""iteration_9 bring-up slice: one-variable arms before the 30-run grid.

Arms (each = one design decision, judged against iteration_7 rows):
  r1e4   ratchet 1e4 (iter7's), recycling on      -- the default candidate
  r1e2   ratchet 1e2, recycling on                -- cheap-trips hypothesis:
         with no basis to void, finer stairs may kill the late-trip
         excursions at no cost
  nrec   ratchet 1e4, recycling OFF               -- prices the recycler

Cells: TARGETS3 x N in {64, 256, 512}. Outputs under iteration_9/bringup/.

Run: uv run python experiments/expD08_qi_init_nlcg/bringup9.py
"""
from __future__ import annotations

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

BRING_DIR = run.OUT_DIR / "bringup"
FNS = run.TARGETS3
NS = [64, 256, 512]
ARMS = {
    "r1e4": {"ratchet": 1e4},
    "r1e2": {"ratchet": 1e2},
    "nrec": {"ratchet": 1e4, "no_recycle": True},
}


def one(job):
    r = run.run_case({"fn": job["fn"], "N": job["N"], "iters": run.ITERS,
                      "threads": job["threads"], "save_snaps": False,
                      **ARMS[job["arm"]]})
    s = r["stats"]
    return {"arm": job["arm"], "fn": r["fn"], "N": r["N"],
            "floor": r["floor"], "best_rel": r["best_rel"],
            "rels": r["rels"], "trips": s["trips"],
            "recycles": s.get("recycles", []),
            "recycle_info": s.get("recycle_info", []),
            "T_final": s["T_final"], "wall_s": r["wall_s"]}


def main():
    BRING_DIR.mkdir(parents=True, exist_ok=True)
    workers = 6
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = [{"arm": a, "fn": f, "N": n, "threads": threads}
            for a in ARMS for f in FNS for n in NS]
    print(f"bring-up | {len(jobs)} runs")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['arm']:5s} {r['fn']:13s} "
                  f"N={r['N']:<4d} best={r['best_rel']:.3e} "
                  f"floor={r['floor']:.1e} trips={len(r['trips'])} "
                  f"rec={len(r['recycles'])} T={r['T_final']:.0e} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    (BRING_DIR / "bringup_rows.json").write_text(json.dumps(rows))

    # score vs iteration_7
    i7 = {(r["fn"], r["N"]): r for r in json.loads(
        (run.OUT_DIR.parent / "iteration_7" / "expD08_rows.json").read_text())}
    print(f"\n{'arm':<6s} {'median vs iter7 (orders)':>26s} {'worst cell':>34s}")
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a and (r["fn"], r["N"]) in i7]
        d = [np.log10(max(r["best_rel"], 1e-300)
                      / max(i7[(r["fn"], r["N"])]["best_rel"], 1e-300))
             for r in rs]
        wi = int(np.argmax(d))
        print(f"{a:<6s} {float(np.median(d)):>+26.2f} "
              f"{rs[wi]['fn']}@N{rs[wi]['N']}: {rs[wi]['best_rel']:.1e} "
              f"({d[wi]:+.1f})")


if __name__ == "__main__":
    main()
