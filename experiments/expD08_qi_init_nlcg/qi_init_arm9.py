"""iteration_9 companion arm: the memoryless core from the FULL fp64 QI
construction init (geometry + readout), vs the standard seed9 sweep.

NOT the standard sweep protocol (the success criterion excludes exact
constructive inits) — this arm evidences the Phase-0/staged-gate finding on
the real nonlinear trainer: from a construction-class state (benign
leftover) the <10-vector core reaches the machine floor with no memory
basis. Cells: 4 targets x N in {256, 512}.

Run: uv run python experiments/expD08_qi_init_nlcg/qi_init_arm9.py
Out: results/.../iteration_9/qi_init_arm/
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

ARM_DIR = run.OUT_DIR / "qi_init_arm"
FNS = ["sine", "sine_8pi", "runge", "exp"]
NS = [256, 512]


def one(job):
    r = run.run_case({"fn": job["fn"], "N": job["N"], "iters": run.ITERS,
                      "threads": job["threads"], "save_snaps": False,
                      "init": "qi_readout"})
    s = r["stats"]
    return {"fn": r["fn"], "N": r["N"], "floor": r["floor"],
            "best_rel": r["best_rel"], "rel0": r["rels"][0] if r["rels"]
            else None, "rels": r["rels"], "trips": s["trips"],
            "recycles": s.get("recycles", []), "T_final": s["T_final"],
            "wall_s": r["wall_s"]}


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["fn"], r["N"]): r for r in rows}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True,
                             sharey=True)
    for i, n in enumerate(NS):
        for j, fn in enumerate(FNS):
            ax = axes[i, j]
            r = by[(fn, n)]
            xs = np.arange(1, len(r["rels"]) + 1)
            ax.plot(xs, r["rels"], color="#2ca02c", lw=1.2,
                    label="iter9 core, QI-construction init")
            ax.axhline(r["floor"], color="k", ls=":", lw=1.0,
                       label="lstsq floor")
            for t in r["trips"]:
                ax.axvline(t, color="#d62728", lw=0.6, alpha=0.5)
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 1e0)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(fn, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N={n}\neval rel L2")
            if i == 1:
                ax.set_xlabel("iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle("Companion arm: <10-vector core from the full QI init "
                 "(no B anywhere; trips marked red)", y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = ARM_DIR / "qi_init_arm_trajectories.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    workers = 4
    threads = max(1, (os.cpu_count() or 4) // workers)
    jobs = [{"fn": f, "N": n, "threads": threads} for f in FNS for n in NS]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['fn']:10s} N={r['N']:<4d} "
                  f"start={r['rel0']:.1e} best={r['best_rel']:.3e} "
                  f"floor={r['floor']:.1e} trips={len(r['trips'])} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    (ARM_DIR / "qi_init_rows.json").write_text(json.dumps(rows))
    figure(rows)


if __name__ == "__main__":
    main()
