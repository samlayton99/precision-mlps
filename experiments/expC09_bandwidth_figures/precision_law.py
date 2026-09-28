"""Measure chirp precision laws at W=512 using predicted and best sampled lambda."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import measure, predictions, selector, SELECTOR

BASE = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
OUT = BASE / "precision_law_W512"
CONFIG = {"target": "chirp", "width": 512, "halo_per_side": 24,
          "precisions": list(range(8, 54)), "train_points": 4801, "eval_points": 8001,
          "lambda_min": .05, "lambda_max": 3., "original_lambda_max": 1.5,
          "original_lambda_count": 85, "extension_lambda_count": 19}


def prediction(p):
    result = predictions(CONFIG["width"], CONFIG["halo_per_side"], p, CONFIG["target"])
    for rule in ("basic", "refined"):
        if result[rule]["status"] != "threshold":
            kwargs = {"omega_scale": result["omega_scale"]} if rule == "refined" else {}
            result[rule] = selector.choose_lambda(
                "tanh", spacing=2/result["N"], e_tol=2.**(1-p), interval=(.03, 3.), **kwargs)
        assert result[rule]["status"] == "threshold"
        theta = (2/result["N"])*result["omega_scale"] if rule == "refined" else None
        assert abs(selector.log_alias_score("tanh", result[rule]["lambda"], theta)
                   - np.log(2.**(1-p))) < 1e-11
    return result


def worker_init():
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def measure_job(job):
    p, lam = job
    return measure(CONFIG["width"], CONFIG["halo_per_side"], lam, p,
                   CONFIG["train_points"], CONFIG["eval_points"], CONFIG["target"])


def summarize(rows, pred):
    summary = []
    for pr in pred:
        cells = [r for r in rows if r["p"] == pr["p"]]
        rule = next(r for r in cells if r["lambda"] == pr["refined"]["lambda"])
        best = min(cells, key=lambda r: (r["relative_l2"], r["lambda"]))
        assert best["relative_l2"] <= rule["relative_l2"]
        summary.append({"p": pr["p"], "width": CONFIG["width"],
                        "rule_lambda": rule["lambda"], "rule_error": rule["relative_l2"],
                        "best_lambda": best["lambda"], "best_error": best["relative_l2"],
                        "points": len(cells),
                        "best_at_search_boundary": best["lambda"] in
                        (CONFIG["lambda_min"], CONFIG["lambda_max"])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pred = [prediction(p) for p in CONFIG["precisions"]]
    grid = np.unique(np.r_[np.geomspace(.05, 1.5, 85), np.geomspace(1.5, 3., 19)])
    jobs = {(pr["p"], float(lam)) for pr in pred
            for lam in [*grid, pr["basic"]["lambda"], pr["refined"]["lambda"]]}
    seed_path = BASE / "combined/data/extra_bandwidth/measurements.jsonl"
    seed_manifest = BASE / "combined/data/extra_bandwidth/predictions.json"
    original = json.loads(seed_manifest.read_text())
    sources = [Path(__file__), Path(__file__).with_name("run.py"), SELECTOR,
               Path(__file__).with_name("additional_targets.py"),
               Path(__file__).with_name("targets.py"), seed_path, seed_manifest]
    manifest = {"config": CONFIG, "predictions": pred, "base_lambda_grid": grid.tolist(),
                "precision_model": original["precision_model"],
                "best_definition": "Minimum measured relative L2 over the common grid plus each precision's basic/refined prediction points",
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}}
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "config.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Source/config changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    if not path.exists():
        seeds = [r for r in map(json.loads, seed_path.read_text().splitlines())
                 if r["width"] == CONFIG["width"] and (r["p"], r["lambda"]) in jobs]
        path.write_text("".join(json.dumps(r)+"\n" for r in seeds))
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    done = {(r["p"], r["lambda"]) for r in rows}
    assert len(done) == len(rows) and done <= jobs
    pending = sorted(jobs-done)
    if args.check_only:
        assert not pending, f"Missing {len(pending)} measurements"
    print(f"Reusing {len(rows)} points; measuring {len(pending)} at W={CONFIG['width']}", flush=True)
    start = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool, path.open("a") as output:
            for i, row in enumerate(pool.map(measure_job, pending, chunksize=1), 1):
                output.write(json.dumps(row)+"\n")
                output.flush()
                rows.append(row)
                if i == 1 or i % 100 == 0 or i == len(pending):
                    print(f"Measured {i}/{len(pending)}; p={row['p']}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    assert len(rows) == len(jobs)
    for r in rows:
        assert r["target"] == CONFIG["target"] and r["width"] == CONFIG["width"]
        assert r["halo"] == CONFIG["halo_per_side"] and r["N"] + 2*r["halo"] + 1 == r["width"]
        assert r["train_points"] == CONFIG["train_points"] and r["eval_points"] == CONFIG["eval_points"]
        assert np.isfinite(r["relative_l2"]) and r["relative_l2"] > 0
    summary = summarize(rows, pred)
    (data / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (data / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (data / "validation.json").write_text(json.dumps({
        "complete": True, "measurements": len(rows), "precisions": len(summary),
        "all_prediction_thresholds_verified": True,
        "best_at_search_boundary": [r["p"] for r in summary if r["best_at_search_boundary"]]
    }, indent=2)+"\n")
    print(f"Validated {len(rows)} measurements at {len(summary)} precisions", flush=True)


if __name__ == "__main__":
    main()
