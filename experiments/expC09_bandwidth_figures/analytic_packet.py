"""Measure an entire oscillatory Gaussian for the combined figure's eighth curve."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.convergence import geometry, widths

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/analytic_packet_8"
LABEL = "Oscillatory Gaussian"
FORMULA = "exp(-8*(x-0.2)**2) * sin(8*pi*x)"


def target(x):
    return np.exp(-8*(x-.2)**2)*np.sin(8*np.pi*x)


def evaluate(centers, gamma, coef, x):
    truth = target(x)
    residual = np.empty_like(x)
    for start in range(0, len(x), 1024):
        stop = min(start+1024, len(x))
        features = np.tanh(gamma*(x[start:stop, None]-centers))
        residual[start:stop] = features @ coef[:-1] + coef[-1] - truth[start:stop]
    return float(np.linalg.norm(residual)/np.linalg.norm(truth))


def measure(width, config):
    n, h, centers = geometry(width, config["halo_per_side"])
    gamma = config["lambda"]/h
    nt = max(config["minimum_train_points"], config["train_points_per_width"]*width+1)
    ne = max(config["minimum_eval_points"], config["eval_points_per_width"]*width+1)
    x = np.linspace(-1, 1, nt)
    a = np.column_stack((np.tanh(gamma*(x[:, None]-centers)), np.ones(nt)))
    # Same independent SVD solve and chunked vector evaluation as convergence.py.
    coef, _, rank, singular = lstsq(a, target(x), cond=2.**-52, lapack_driver="gelsd")
    del a
    error = evaluate(centers, gamma, coef, np.linspace(-1, 1, ne))
    row = {"target": "analytic_packet", "width": width, "N": n,
           "halo_per_side": config["halo_per_side"], "lambda": config["lambda"],
           "p": 53, "train_points": nt, "eval_points": ne, "relative_l2": error,
           "rank": int(rank), "sigma_max": float(singular[0]),
           "coefficient_norm": float(np.linalg.norm(coef))}
    if width in (256, 512, 1024, 2048):
        # Independent shifted grid, twice as dense, with the fitted readout frozen.
        dense_x = -1+(np.arange(2*ne)+.5)*2/(2*ne)
        row["dense_shifted_relative_l2"] = evaluate(centers, gamma, coef, dense_x)
    return row, coef


def main():
    original = OUT.parent / "eight_target_convergence/data/config.json"
    config = json.loads(original.read_text())["config"]
    assert config["precision_bits"] == 53 and config["eval_chunk_size"] == 1024
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest = {"formula": FORMULA, "config": dict(config, targets=["analytic_packet"]),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "base_config_sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
    manifest_path = data / "config.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Saved configuration differs"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = data / "measurements.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["width"] for r in rows}
    with threadpool_limits(limits=1), path.open("a") as output:
        for w in widths(config):
            if w in done:
                continue
            row, coef = measure(w, config)
            assert np.isfinite(row["relative_l2"])
            np.savez_compressed(data / f"coefficients_W{w}.npz", coefficients=coef)
            output.write(json.dumps(row)+"\n")
            output.flush()
            print(f"W={w}: {row['relative_l2']:.4g}", flush=True)


if __name__ == "__main__":
    main()
