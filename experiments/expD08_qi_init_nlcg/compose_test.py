"""Composition test: tight-trip + big-ratchet, timer vs event-driven.

tb  = trip_mode="noise", ratchet=1e4, timer refresh (200)
tbe = the same + refresh_mode="exhaust" (no timer; recalibrate on the
      solver's own completion signal)

12-cell slice (4 gif targets x N in {64,256,512}); the winner becomes the
iteration_7 candidate. Run:
  uv run python experiments/expD08_qi_init_nlcg/compose_test.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

VARIANTS = {
    "tb": {"trip_mode": "noise", "ratchet": 1e4},
    "tbe": {"trip_mode": "noise", "ratchet": 1e4,
            "refresh_mode": "exhaust"},
}
FNS = run.TARGETS3
NS = [64, 256, 512]
OUT = run.OUT_DIR / "staircase" / "compose_rows.json"


def case(j):
    r = run.run_case({"fn": j["fn"], "N": j["N"], "iters": run.ITERS,
                      "threads": j["threads"], "save_snaps": False,
                      **VARIANTS[j["v"]]})
    return {"v": j["v"], "fn": r["fn"], "N": r["N"],
            "best_rel": r["best_rel"], "floor": r["floor"],
            "rels": r["rels"], "trips": r["stats"]["trips"],
            "T": r["stats"]["T_final"], "refreshes": r["stats"]["refresh"],
            "wall_s": r["wall_s"]}


def main():
    workers = int(os.environ.get("WORKERS", "6"))
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = [{"v": v, "fn": f, "N": n, "threads": threads}
            for v in VARIANTS for f in FNS for n in NS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['v']:4s} {r['fn']:13s} "
                  f"N={r['N']:<4d} best={r['best_rel']:.2e} "
                  f"floor={r['floor']:.1e} trips={len(r['trips'])} "
                  f"T={r['T']:.0e} refr={r['refreshes']} "
                  f"({r['wall_s']}s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
