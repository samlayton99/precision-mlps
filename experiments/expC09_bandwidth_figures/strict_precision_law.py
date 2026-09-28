"""Rerun all 46 precisions with correctly rounded full forward inference."""
import hashlib
import json
from pathlib import Path
import sys
import time

import gmpy2
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.strict_precision import measure, OUT, MODEL
from experiments.expC09_bandwidth_figures.run import selector, SELECTOR


def main():
    source = OUT.parent / "precision_law_W1024/data/config.json"
    old = json.loads(source.read_text())
    config = {**old["config"], "inference": "MPFR p-bit centered tanh, all parameters and operations"}
    pred = old["predictions"]
    for pr in pred:
        theta = 2/pr["N"]*pr["omega_scale"]
        assert abs(selector.log_alias_score("tanh", pr["refined"]["lambda"], theta)
                   - np.log(2.**(1-pr["p"]))) < 1e-11
    sources = [source, Path(__file__), Path(__file__).with_name("strict_precision.py"),
               Path(__file__).with_name("strict_arithmetic.c"), Path(__file__).with_name("run.py"),
               Path(__file__).with_name("additional_targets.py"), SELECTOR]
    manifest = {"config": config, "predictions": pred, "precision_model": MODEL,
                "gmpy2": gmpy2.version(), "mpfr": gmpy2.mpfr_version(),
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    models = data / "models"
    models.mkdir(exist_ok=True)
    manifest_path = data / "config.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Source/config changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["p"] for r in rows}
    assert len(done) == len(rows) and done <= set(config["precisions"])
    pending = [pr for pr in pred if pr["p"] not in done]
    start = time.monotonic()
    with threadpool_limits(limits=1), path.open("a") as output:
        for i, pr in enumerate(pending, 1):
            row = measure(config["width"], config["halo_per_side"], pr["refined"]["lambda"], pr["p"],
                          config["train_points"], config["eval_points"], config["target"],
                          save_model=models / f"p{pr['p']}.npz")
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            print(f"Measured {i}/{len(pending)}: p={pr['p']}, error={row['relative_l2']:.4g}, "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)
    rows.sort(key=lambda r: r["p"])
    assert [r["p"] for r in rows] == config["precisions"]
    for r in rows:
        assert r["precision_model"] == MODEL and r["width"] == config["width"]
        assert np.isfinite(r["relative_l2"]) and r["relative_l2"] > 0
    summary = [{"p": r["p"], "width": r["width"], "rule_lambda": r["lambda"],
                "rule_error": r["relative_l2"]} for r in rows]
    (data / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (data / "validation.json").write_text(json.dumps({
        "complete": True, "measurements": len(rows), "all_prediction_thresholds_verified": True,
        "all_forward_values_p_bit": True, "precision_model": MODEL}, indent=2)+"\n")


if __name__ == "__main__":
    main()
