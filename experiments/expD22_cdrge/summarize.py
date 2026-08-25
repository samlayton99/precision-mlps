"""expD22 summary tables: per-variant medians across cells/seeds, vs expD16."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
D22 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD22_cdrge" / "data"
D16 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD16_optimizer_zoo" / "data"


def load(init):
    rows = []
    for p in sorted(D22.glob(f"trajectories_{init}*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    return rows


def main():
    for init in ["qi", "xavier"]:
        rows = load(init)
        if not rows:
            continue
        opts = sorted({r["opt"] for r in rows})
        print(f"\n=== {init} ({len(rows)} rows) ===")
        print(f"{'variant':22s} {'median':>10s} {'best':>10s} {'worst':>10s} "
              f"{'n_runs':>6s} {'med evals':>10s}")
        for opt in opts:
            rv = [r for r in rows if r["opt"] == opt]
            f = np.array([r["final_rel_l2"] for r in rv])
            ev = np.median([r.get("evals", np.nan) for r in rv])
            print(f"{opt:22s} {np.median(f):10.2e} {f.min():10.2e} "
                  f"{f.max():10.2e} {len(rv):6d} {ev:10.0f}")
        # per-cell medians over seeds for the headline variant
        print(f"\n  {init}: cdrge_adam_cos per-cell median over seeds "
              f"(rows=targets, cols=N):")
        for tgt in ["sine", "exp", "runge", "sine_8pi"]:
            vals = []
            for N in [64, 128, 256]:
                f = [r["final_rel_l2"] for r in rows
                     if r["opt"] == "cdrge_adam_cos"
                     and r["target"] == tgt and r["N"] == N]
                vals.append(f"{np.median(f):.1e}" if f else "--")
            print(f"    {tgt:9s} " + "  ".join(f"{v:>8s}" for v in vals))
        # expD16 comparison at matched cells (single-seed there)
        p16 = D16 / f"trajectories_{init if init != 'qi' else 'qi'}.jsonl"
        if p16.exists():
            r16 = [json.loads(l) for l in open(p16)]
            print(f"\n  expD16 medians ({init}): " + "  ".join(
                f"{o}={np.median([r['final_rel_l2'] for r in r16 if r['opt'] == o]):.1e}"
                for o in ["adam", "adam_ssbroyden", "adam_nncg", "adam_lbfgs", "spsa"]))


if __name__ == "__main__":
    main()
