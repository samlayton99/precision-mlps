"""Extend the combined figure's measured chirp sweep to W=512 and W=1024."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import measure, predictions, SELECTOR

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
DATA = OUT / "combined/data/extra_bandwidth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = OUT / "additional_targets/data/predictions.json"
    original = json.loads(source.read_text())
    config = {key: original["config"][key] for key in
              ("halo_per_side", "train_points", "eval_points", "lambda_min",
               "lambda_max", "lambda_count")}
    config.update(target="chirp", bandwidth_widths=[512, 1024], p=53)
    pred = [predictions(w, config["halo_per_side"], config["p"], config["target"])
            for w in config["bandwidth_widths"]]
    sources = [source, Path(__file__), SELECTOR,
               Path(__file__).with_name("run.py"),
               Path(__file__).with_name("additional_targets.py"),
               Path(__file__).with_name("targets.py")]
    manifest = {"config": config, "predictions": pred,
                "precision_model": original["precision_model"],
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sources}}
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "predictions.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Source/config changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")

    grid = np.geomspace(config["lambda_min"], config["lambda_max"], config["lambda_count"])
    jobs = {(pr["width"], float(lam)) for pr in pred
            for lam in [*grid, pr["basic"]["lambda"], pr["refined"]["lambda"]]}
    path = DATA / "measurements.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {(r["width"], r["lambda"]) for r in rows}
    assert len(done) == len(rows) and done <= jobs
    pending = sorted(jobs-done)
    if args.check_only:
        assert not pending, f"Missing {len(pending)} measurements"
    start = time.monotonic()
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, (width, lam) in enumerate(pending, 1):
            row = measure(width, config["halo_per_side"], lam, config["p"],
                          config["train_points"], config["eval_points"], config["target"])
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            if i == 1 or i % 10 == 0 or i == len(pending):
                print(f"Measured {i}/{len(pending)} new points; W={width}; "
                      f"elapsed={time.monotonic()-start:.1f}s", flush=True)
    assert len(rows) == len(jobs)
    for row in rows:
        assert row["target"] == config["target"] and row["p"] == config["p"]
        assert row["halo"] == config["halo_per_side"]
        assert row["N"] + 1 + 2*row["halo"] == row["width"]
        assert row["train_points"] == config["train_points"]
        assert row["eval_points"] == config["eval_points"]
        assert np.isfinite(row["relative_l2"]) and row["relative_l2"] > 0
        assert np.isclose(row["gamma"] * (2/row["N"]), row["lambda"])
    summary = {}
    for pr in pred:
        cells = [r for r in rows if r["width"] == pr["width"]]
        best = min(cells, key=lambda r: r["relative_l2"])
        predicted = next(r for r in cells if r["lambda"] == pr["refined"]["lambda"])
        summary[str(pr["width"])] = {
            "points": len(cells), "minimum": best, "predicted": predicted}
    (DATA / "validation.json").write_text(json.dumps({
        "complete": True, "measurements": len(rows), "curves": summary}, indent=2)+"\n")
    print(f"Validated {len(rows)} measurements", flush=True)


if __name__ == "__main__":
    main()
