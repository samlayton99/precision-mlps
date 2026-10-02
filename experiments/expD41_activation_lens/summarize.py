"""Summarize the completed activation comparison without rerunning training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_D_optimizers/expD41_activation_lens"


def summarize(out=OUT):
    cfg = json.loads((out / "config.json").read_text())
    bandwidth = json.loads((out / "bandwidth.json").read_text())
    activations = cfg["activations"] + cfg["supplementary_activations"]
    expected = {(a, t, init, seed) for a in activations for t in cfg["targets"]
                for init, seeds in (("qi", [0]), ("xavier", cfg["xavier_seeds"])) for seed in seeds}
    runs = {}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        key = (r["activation"], r["target"], r["initialization"], r["seed"])
        assert key not in runs and r["complete"] and r["history"][-1]["step"] == cfg["steps"], path
        assert r["signature"]["config"] == cfg, path
        assert r["signature"]["lambda"] == bandwidth["selected"][r["activation"]]["lambda"], path
        assert hashlib.sha256(json.dumps(r["signature"], sort_keys=True).encode()).hexdigest() == r["fingerprint"], path
        for name, digest in r["signature"]["sources"].items():
            assert hashlib.sha256((ROOT / "experiments/expD41_activation_lens" / name).read_bytes()).hexdigest() == digest, name
        for name, digest in r["signature"].get("local_sources", {}).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        assert path.with_suffix(".pt").exists(), path
        with np.load(path.with_suffix(".npz")) as saved:
            assert saved["step"].tolist() == [h["step"] for h in r["history"]], path
            for name in ("w", "b", "v", "solved_v"):
                assert saved[name].shape == (len(r["history"]), cfg["interior_centers"] + 2 * cfg["halo_per_side"]), path
                assert np.isfinite(saved[name]).all(), path
        runs[key] = r
    assert set(runs) == expected, (expected - set(runs), set(runs) - expected)
    rows = []
    for target in cfg["targets"]:
        for activation in activations:
            qi = runs[(activation, target, "qi", 0)]
            xs = [runs[(activation, target, "xavier", s)] for s in cfg["xavier_seeds"]]
            rows.append({"target": target, "activation": activation,
                "lambda": bandwidth["selected"][activation]["lambda"],
                "qi_initial": qi["history"][0], "qi_final": qi["history"][-1],
                "xavier_final_by_seed": {str(r["seed"]): r["history"][-1] for r in xs},
                "xavier_final_medians": {field: float(np.median([r["history"][-1][field] for r in xs]))
                    for field in ("actual_rel_l2", "floor_rel_l2")}})
    validation = json.loads((out / "validation.json").read_text())
    assert len(validation["rows"]) == len(expected)
    verified = validation["rows"]
    cutoff_factors = []
    for r in verified:
        errors = [v["relative_l2"] for v in r["cutoffs"].values()]
        cutoff_factors.append({"run": r["run"], "factor": max(errors) / min(errors),
                               "min_error": min(errors), "max_error": max(errors)})
    summary = {"completed_runs": len(runs), "steps_per_run": cfg["steps"],
        "total_adam_steps": len(runs) * cfg["steps"], "protocol_and_artifacts_verified": True,
        "rows": rows, "validation_summary": {
            "actual_grid_ratio_range": [min(r["actual_grid_ratio"] for r in verified), max(r["actual_grid_ratio"] for r in verified)],
            "floor_grid_ratio_range": [min(r["floor_grid_ratio"] for r in verified), max(r["floor_grid_ratio"] for r in verified)],
            "cutoff_sensitivity": sorted(cutoff_factors, key=lambda r: r["factor"], reverse=True)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(runs)} complete runs and saved summary.json")
    for r in rows:
        print(r["target"], r["activation"], "QI", r["qi_final"]["actual_rel_l2"],
              "QI LS", r["qi_final"]["floor_rel_l2"], "Xavier medians", r["xavier_final_medians"])
    print(summary["validation_summary"])


if __name__ == "__main__":
    summarize()
