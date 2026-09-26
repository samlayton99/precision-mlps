"""Runge 25 comparison using the unchanged expC11 true-p model and solver.

Run: .venv/bin/python experiments/expC11_true_precision_law/runge_comparison.py
Resumes completed precision levels, then renders a separate comparison PNG.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC11_true_precision_law import run as baseline
from experiments.expC09_bandwidth_figures.run import predictions, selector, SELECTOR
from experiments.expC09_bandwidth_figures.targets import target_values
from src.precision import pbit

OUT = baseline.OUT / "runge25_comparison"
DATA = OUT / "data"
CONFIG = {**baseline.CONFIG, "target": "runge25", "formula": "1/(1+25*x**2)",
          "cutoff_factors": [2], "omega_scale": 5.0, "lambda_interval": [.03, 3.],
          "bandwidth": "same refined rule as chirp, with Runge's frequency scale 5"}


def prediction(p):
    pr = predictions(CONFIG["width"], CONFIG["halo_per_side"], p, "runge")
    rule = pr["refined"]
    if rule["status"] != "threshold":
        rule = selector.choose_lambda("tanh", spacing=2/pr["N"], e_tol=2.**(1-p),
                                      omega_scale=pr["omega_scale"], interval=(.03, 3.))
        pr["refined"] = rule
    assert rule["status"] in ("threshold", "upper_search_limit")
    score = selector.log_alias_score("tanh", rule["lambda"], 2/pr["N"]*pr["omega_scale"])
    budget = np.log(2.**(1-p))
    if rule["status"] == "threshold":
        assert abs(score - budget) < 1e-11
    else:
        assert rule["lambda"] == CONFIG["lambda_interval"][1] and score < budget
    return pr


def measure(p, lam):
    n, halo = baseline.geometry()
    x = np.linspace(-1, 1, CONFIG["train_points"])
    xe = np.linspace(-1, 1, CONFIG["eval_points"])
    truth = target_values(xe, "runge")
    fmt = pbit.panel_format(p)
    start = time.monotonic()
    r = pbit.run(x, target_values(x, "runge"), xe, n, halo, lam, fmt)
    for key in ("weights", "fit", "sigma"):
        assert np.isfinite(r[key]).all()
        assert np.array_equal(pbit.round_p(r[key], fmt), r[key])
    row = {"target": "runge25", "p": p, "format": list(fmt), "width": CONFIG["width"],
           "N": n, "halo_per_side": halo, "lambda": lam, "lambda_p": r["lambda_p"],
           "relative_l2": float(np.linalg.norm(r["fit"]-truth)/np.linalg.norm(truth)),
           "linf": float(np.max(np.abs(r["fit"]-truth))), "rank": r["rank"],
           "rcond": 2.**(1-p), "events": r["events"], "seconds": time.monotonic()-start}
    model = {key: r[key] for key in ("weights", "fit", "sigma")}
    return row, model


def main(workers=8):
    # Do not mix a changed arithmetic implementation with the saved chirp curve.
    original = json.loads((baseline.DATA / "config.json").read_text())
    for source, digest in original["sources"].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    pred = [prediction(p) for p in CONFIG["precisions"]]
    sources = [*baseline.SOURCES, Path(__file__), SELECTOR,
               ROOT/"experiments/expC09_bandwidth_figures/run.py",
               ROOT/"experiments/expC09_bandwidth_figures/targets.py",
               ROOT/"experiments/expC09_bandwidth_figures/additional_targets.py"]
    manifest = {"config": CONFIG, "predictions": pred,
                "sources": {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest()
                            for s in sources}}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "models").mkdir(exist_ok=True)
    path = DATA / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Runge sources/config changed"
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = DATA / "measurements.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["p"] for r in rows}
    assert len(done) == len(rows) and done <= set(CONFIG["precisions"])
    pbit.round_p([1.], pbit.panel_format(24))  # Build once before starting workers.
    start = time.monotonic()
    with ProcessPoolExecutor(workers) as pool, path.open("a") as output:
        jobs = [pool.submit(measure, pr["p"], pr["refined"]["lambda"])
                for pr in pred if pr["p"] not in done]
        for job in as_completed(jobs):
            row, model = job.result()
            np.savez(DATA / "models" / f"p{row['p']}.npz", **model)
            output.write(json.dumps(row)+"\n")
            output.flush()
            rows.append(row)
            print(f"{len(rows)}/46: p={row['p']}, error={row['relative_l2']:.6e}, "
                  f"rank={row['rank']}, elapsed={time.monotonic()-start:.0f}s", flush=True)
    rows.sort(key=lambda r: r["p"])
    assert [r["p"] for r in rows] == CONFIG["precisions"]
    xe = np.linspace(-1, 1, CONFIG["eval_points"])
    truth = target_values(xe, "runge")
    replayed = []
    for row in rows:
        p, fmt = row["p"], tuple(row["format"])
        with np.load(DATA / "models" / f"p{p}.npz") as saved:
            for key in ("weights", "fit", "sigma"):
                assert np.array_equal(pbit.round_p(saved[key], fmt), saved[key])
            error = float(np.linalg.norm(saved["fit"]-truth)/np.linalg.norm(truth))
            assert error == row["relative_l2"]
            if p in (8, 24, 53):
                fit = pbit.evaluate(saved["weights"], xe, row["N"], row["halo_per_side"],
                                    row["lambda"], fmt)
                assert np.array_equal(fit, saved["fit"])
                replayed.append(p)
    summary = [{"p": r["p"], "width": r["width"], "rule_lambda": r["lambda"],
                "rule_error": r["relative_l2"]} for r in rows]
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (DATA / "validation.json").write_text(json.dumps({
        "complete": True, "precisions": len(rows), "all_saved_values_p_bit": True,
        "all_saved_errors_recomputed": True, "bitwise_model_replay_precisions": replayed,
        "baseline_sources_unchanged": True, "prediction_statuses_verified": True,
        "bandwidth_at_upper_limit_precisions": [pr["p"] for pr in pred
                                               if pr["refined"]["status"] == "upper_search_limit"],
    }, indent=2)+"\n")
    from experiments.expC09_bandwidth_figures.combined import main as render
    render("rule", 1024, true_precision_data=baseline.DATA, runge_precision_data=DATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
