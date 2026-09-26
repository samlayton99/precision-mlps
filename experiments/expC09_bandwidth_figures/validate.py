"""Audit halo extension and doubled fitting/evaluation grids for a saved suite."""
from __future__ import annotations

import hashlib
import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from run import HERE, OUT, key, make_jobs, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    data = args.output / "data"
    manifest = json.loads((data / "predictions.json").read_text())
    config = manifest["config"]
    rows = [json.loads(line) for line in (data / "measurements.jsonl").read_text().splitlines()]
    jobs, predictions = make_jobs(config)
    lookup = {key(r): r for r in rows}
    assert len(rows) == len(lookup) == len(jobs)
    assert set(lookup) == set(jobs)
    assert all(r["width"] == r["N"] + 2*r["halo"] + 1 for r in rows)
    assert all(np.isfinite(r["relative_l2"]) and r["relative_l2"] > 0 for r in rows)
    assert all(r["train_points"] == config["train_points"] for r in rows)
    assert all(r["eval_points"] == config["eval_points"] for r in rows)
    halo_rows = []
    base_halo = config["halo_per_side"]
    for row in rows:
        if row["halo"] != base_halo:
            continue
        for halo in config["halo_checks"]:
            expanded_width = row["N"] + 2*halo + 1
            other = lookup.get((row["target"], expanded_width, halo, row["lambda"], row["p"]))
            if other is not None:
                halo_rows.append({"target": row["target"], "width": row["width"], "N": row["N"], "p": row["p"],
                                  "lambda": row["lambda"], "larger_halo": halo,
                                  "base_error": row["relative_l2"], "extended_error": other["relative_l2"],
                                  "improvement_factor": row["relative_l2"] / other["relative_l2"]})
    # Include both predictions for every displayed curve, and grid extremes.
    selected = {(pr["target"], pr["width"], base_halo, pr[rule]["lambda"], pr["p"])
                for pr in predictions for rule in ("basic", "refined")}
    for target in config["targets"]:
        selected.update({(target, min(config["bandwidth_widths"]), base_halo, config["lambda_min"], 53),
                         (target, max(config["bandwidth_widths"]), base_halo, config["lambda_max"], 53),
                         (target, config["refinement_width_start"], base_halo, config["refinement_lambda"], 53),
                         (target, config["precision_width"], base_halo, config["lambda_min"], 24)})
    checks = []
    with threadpool_limits(limits=1):
        for target, w, h, lam, p in sorted(selected):
            base = lookup[(target, w, h, lam, p)]
            evaluation = measure(w, h, lam, p, config["train_points"], 2*config["eval_points"]-1, target)
            denser = measure(w, h, lam, p, 2*config["train_points"]-1, 2*config["eval_points"]-1, target)
            checks.append({"target": target, "width": w, "p": p, "lambda": lam,
                           "base_error": base["relative_l2"],
                           "double_eval_error": evaluation["relative_l2"],
                           "double_train_eval_error": denser["relative_l2"],
                           "eval_ratio": evaluation["relative_l2"] / base["relative_l2"],
                           "train_eval_ratio": denser["relative_l2"] / base["relative_l2"]})
    above_floor = [r for r in halo_rows if r["base_error"] > 1e-12 and r["p"] == 53]
    near_rules = [r for r in halo_rows if any(r["target"] == pr["target"]
                  and r["width"] == pr["width"] and r["p"] == pr["p"]
                  and r["lambda"] in [pr[rule]["lambda"] for rule in ("basic", "refined")]
                  for pr in predictions)]
    result = {
        "complete_cells": len(rows), "halo_comparisons": len(halo_rows),
        "max_halo_improvement_above_1e_minus_12_fp64": max(r["improvement_factor"] for r in above_floor),
        "max_halo_improvement_at_rule_choices": max(r["improvement_factor"] for r in near_rules),
        "max_eval_grid_relative_change": max(abs(r["eval_ratio"]-1) for r in checks),
        "sampling_checks": checks, "halo_checks": halo_rows,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "run_sha256": hashlib.sha256((HERE/"run.py").read_bytes()).hexdigest()},
    }
    (data / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}, indent=2))
    print("Doubled fitting/evaluation grid ratios:")
    for r in checks:
        print(f"{r['target']}: W={r['width']:4}, p={r['p']}, lambda={r['lambda']:.4f}: "
              f"eval {r['eval_ratio']:.4f}, fit+eval {r['train_eval_ratio']:.3f}")


if __name__ == "__main__":
    main()
