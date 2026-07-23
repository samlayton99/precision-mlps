"""Free QI init under pure Adam (gradient descent, both layers) -- for the
overlay's runge and sine_mixture panels.

'Free QI init' = qi_geom_random_readout (target-independent QI geometry, readout
NOT handed over). Trained with pure full-batch Adam (no lstsq projection, no GN)
via expD05's run_case. This is the gradient-descent comparison line for the
init-slice overlay; it stalls ~1e-4 where VarPro+Adam->GN reaches the floor.

Usage: uv run --extra dev python experiments/expD04_varpro/free_qi_adam_probe.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_D05_PATH = REPO_ROOT / "experiments" / "expD05_scale_init_story" / "run.py"
_spec = importlib.util.spec_from_file_location("expd05_run", _D05_PATH)
d05 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["expd05_run"] = d05
_spec.loader.exec_module(d05)  # type: ignore[union-attr]

OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "free_qi_adam_summary.csv"

TARGETS = ("runge", "sine_mixture")
RESOLUTIONS = (32, 64, 128, 256, 512)
SEEDS = (0, 1, 2)
FAMILY = "qi_geom_random_readout"


def main():
    cfg = d05.RunConfig(
        targets=TARGETS, resolutions=RESOLUTIONS, seeds=SEEDS, families=(FAMILY,),
        steps=20000, eval_interval=250, learning_rate=1e-3,
        n_train=2003, n_eval=4001, mode="full",
    )
    rows = []
    total = len(TARGETS) * len(RESOLUTIONS) * len(SEEDS)
    i = 0
    for target in TARGETS:
        for resolution in RESOLUTIONS:
            for seed in SEEDS:
                i += 1
                summary, _, _ = d05.run_case(
                    d05.Case(target=target, resolution=resolution, seed=seed, family=FAMILY), cfg)
                rows.append({"target": target, "resolution": resolution,
                             "width": summary["width"], "seed": seed,
                             "adam_best_eval_rel_l2": summary["best_eval_rel_l2"]})
                print(f"[{i:2d}/{total}] {target:>12s} N={resolution:<4d} W={summary['width']:<4d} "
                      f"seed={seed} adam_best={summary['best_eval_rel_l2']:.2e}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
