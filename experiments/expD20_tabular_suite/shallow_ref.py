"""expD20 -- the shallow reference: was it the tasks, or was it our nets?

expD18 trained a 1-hidden-layer tanh MLP at width 256 and scored it on the
expF04 cache, whose targets are min-max normalized. Two things were wrong with
reading a low headroom off that:

  1. the metric. rel L2 = ||yhat-y||/||y|| with an OFFSET target gives free
     credit for predicting the mean (on bike_sharing, the mean-predictor scores
     0.52, so half the score is unearned), which compresses every ratio;
  2. the capacity. one hidden layer at width 256 is not a serious upper bound.

This script isolates (2): the SAME shallow architecture as expD18, scored under
the SAME standardized metric as the rest of expD20. The gap between this and
the best deep/GBDT model is pure underfitting, with the metric held fixed.

    uv run --with scikit-learn --extra dev python experiments/expD20_tabular_suite/shallow_ref.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EV = _load("expD20_eval_sr", HERE / "evaluate.py")
DS = _load("expD20_data_sr", HERE / "datasets.py")
OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD20_tabular_suite" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for name in DS.INCUMBENTS:
        Xtr, ytr, Xte, yte = DS.load_incumbent(name)
        Xtr, ytr, Xte, yte = EV.prep(Xtr, ytr.ravel(), Xte, yte.ravel())
        # expD18's architecture: ONE hidden layer, width 256.
        err, _ = EV.fit_mlps(Xtr, ytr, Xte, yte, archs=[(256,)], max_epochs=400)
        rows.append(dict(task=name, shallow_256=err))
        print(f"{name:20s} 1-layer W=256 (expD18 arch, expD20 metric): {err:.4f}")
    with open(OUT / "shallow_ref.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
