"""Retrospective chirp prediction using each original sweep's measured minimum."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import geometry, measure, selector
from experiments.expC09_bandwidth_figures.sine24_panels import draw

BASE = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures"
OUT = BASE / "chirp_min_budget"


def main():
    source = BASE / "additional_targets/data/measurements.jsonl"
    config_source = BASE / "additional_targets/data/predictions.json"
    config = json.loads(config_source.read_text())["config"]
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    rows = [r for r in rows if r["target"] == "chirp" and r["halo"] == config["halo_per_side"]]
    curves = [(w, 53) for w in config["bandwidth_widths"]]
    curves += [(config["precision_width"], p) for p in config["precisions"] if p != 53]
    rows = [r for r in rows if (r["width"], r["p"]) in curves]
    predictions, comparisons, probes = {}, [], []
    with threadpool_limits(limits=1):
        for w, p in curves:
            cells = [r for r in rows if r["width"] == w and r["p"] == p]
            best = min(cells, key=lambda r: r["relative_l2"])
            _, h, _ = geometry(w, config["halo_per_side"])
            # Expand only the selector's search to locate out-of-sweep roots.
            # The original sweep and its measured minimum stay fixed.
            prediction = selector.choose_lambda("tanh", spacing=h, e_tol=best["relative_l2"],
                                                omega_scale=16*np.pi, interval=(.03, 10.))
            assert prediction["status"] == "threshold"
            prediction["e_tol"] = best["relative_l2"]
            prediction["outside_original_sweep"] = not config["lambda_min"] <= prediction["lambda"] <= config["lambda_max"]
            predictions[f"{w}:{p}"] = prediction
            probe = measure(w, config["halo_per_side"], prediction["lambda"], p,
                            config["train_points"], config["eval_points"], "chirp")
            probes.append(probe)
            comparisons.append({"width": w, "N": best["N"], "p": p,
                                "e_tol": best["relative_l2"], "sweep_min_lambda": best["lambda"],
                                "predicted_lambda": prediction["lambda"],
                                "predicted_point_error": probe["relative_l2"],
                                "error_over_sweep_min": probe["relative_l2"]/best["relative_l2"],
                                "outside_original_sweep": prediction["outside_original_sweep"]})
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "predictions": predictions,
                "e_tol_definition": "Minimum measured relative L2 error of each original curve; frozen before new probes",
                "chirp_frequency_summary": "16*pi, mean local frequency; heuristic",
                "selector_search_interval": [.03, 10.],
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source": str(source.relative_to(ROOT)),
                "comparison_type": "retrospective; tolerance uses observed sweep outcome"}
    (data / "predictions.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (data / "comparison.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    (data / "prediction_measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in probes))
    # Retain every original observation and add only the directly measured marker locations.
    lookup = {(r["width"], r["p"], r["lambda"]): r for r in rows+probes}
    draw(list(lookup.values()), predictions, output_path=OUT / "figures/chirp_min_budget.png")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
