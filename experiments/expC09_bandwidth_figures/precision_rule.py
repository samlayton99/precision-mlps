"""Measure the predicted-bandwidth precision law at W=1024."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import measure, predictions, selector, SELECTOR

BASE = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
DATA = BASE / "precision_law_W1024/data"
CONFIG = {"target": "chirp", "width": 1024, "halo_per_side": 24,
          "precisions": list(range(8, 54)), "train_points": 4801, "eval_points": 8001,
          "selection": "refined_rule_only", "selector_search_interval": [.03, 3.]}


def main():
    pred = []
    for p in CONFIG["precisions"]:
        pr = predictions(CONFIG["width"], CONFIG["halo_per_side"], p, CONFIG["target"])
        if pr["refined"]["status"] != "threshold":
            pr["refined"] = selector.choose_lambda(
                "tanh", spacing=2/pr["N"], e_tol=2.**(1-p), omega_scale=pr["omega_scale"],
                interval=tuple(CONFIG["selector_search_interval"]))
        assert pr["refined"]["status"] == "threshold"
        theta = 2/pr["N"]*pr["omega_scale"]
        assert abs(selector.log_alias_score("tanh", pr["refined"]["lambda"], theta)
                   - np.log(2.**(1-p))) < 1e-11
        pred.append(pr)
    seed_path = BASE / "combined/data/extra_bandwidth/measurements.jsonl"
    seed_manifest = BASE / "combined/data/extra_bandwidth/predictions.json"
    sources = [Path(__file__), Path(__file__).with_name("run.py"), SELECTOR,
               Path(__file__).with_name("additional_targets.py"),
               Path(__file__).with_name("targets.py"), seed_path, seed_manifest]
    manifest = {"config": CONFIG, "predictions": pred,
                "precision_model": json.loads(seed_manifest.read_text())["precision_model"],
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}}
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "config.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Source/config changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    jobs = {(pr["p"], pr["refined"]["lambda"]) for pr in pred}
    path = DATA / "measurements.jsonl"
    if not path.exists():
        seeds = [r for r in map(json.loads, seed_path.read_text().splitlines())
                 if r["width"] == CONFIG["width"] and (r["p"], r["lambda"]) in jobs]
        path.write_text("".join(json.dumps(r)+"\n" for r in seeds))
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    done = {(r["p"], r["lambda"]) for r in rows}
    assert len(done) == len(rows) and done <= jobs
    pending = sorted(jobs-done)
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, (p, lam) in enumerate(pending, 1):
            row = measure(CONFIG["width"], CONFIG["halo_per_side"], lam, p,
                          CONFIG["train_points"], CONFIG["eval_points"], CONFIG["target"])
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            if i == 1 or i % 10 == 0 or i == len(pending):
                print(f"Measured {i}/{len(pending)} new precisions at W=1024", flush=True)
    assert len(rows) == len(jobs)
    for row in rows:
        assert row["width"] == CONFIG["width"] and row["halo"] == CONFIG["halo_per_side"]
        assert row["N"] + 1 + 2*row["halo"] == row["width"]
        assert row["target"] == CONFIG["target"]
        assert row["train_points"] == CONFIG["train_points"] and row["eval_points"] == CONFIG["eval_points"]
        assert np.isfinite(row["relative_l2"]) and row["relative_l2"] > 0
    summary = [{"p": r["p"], "width": r["width"], "rule_lambda": r["lambda"],
                "rule_error": r["relative_l2"]} for r in sorted(rows, key=lambda r: r["p"])]
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (DATA / "validation.json").write_text(json.dumps({
        "complete": True, "measurements": len(rows), "all_prediction_thresholds_verified": True,
        "selection": "refined_rule_only"}, indent=2)+"\n")
    print(f"Validated {len(rows)} precision measurements", flush=True)


if __name__ == "__main__":
    main()
