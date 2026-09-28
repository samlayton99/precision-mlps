"""Compare FP64-constructed, p-bit-quantized QUILL and Mhaskar networks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.linalg import lstsq, norm
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC12_mhaskar_comparison.construction import (
    TanhNetwork, chebyshev_coefficients, chirp, construct_mhaskar, round_bits,
)
from experiments.expC09_bandwidth_figures.run import geometry, predictions, selector, SELECTOR

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/checkpoint_C_geometry/expC12_mhaskar_comparison"
PROTOCOL = {
    "construction": "FP64 throughout, including Chebyshev coefficients, derivative recurrence and SVD",
    "parameters": "Every affine hidden weight/bias, readout weight and output bias rounded to p significant binary bits",
    "evaluation": "FP64 input grid and all intermediate operations; final predictions rounded to p bits",
    "metric": "FP64 relative discrete L2 norm on a held-out uniform grid",
    "exponent": "FP64 exponent range, not native FP16/FP32 exponent limits",
    "not_claimed": "This is parameter/output quantization, not p-bit arithmetic or a precision lower bound",
}


def relative_l2(predicted, truth):
    if not np.all(np.isfinite(predicted)):
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        # scipy's vector norm uses scaled BLAS nrm2, avoiding avoidable square overflow.
        return float(norm(predicted-truth) / norm(truth))


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def run():
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    bits = np.arange(config["precision_min"], config["precision_max"]+1)
    degrees = np.array(config["degrees"])
    steps = np.geomspace(config["step_min"], config["step_max"], config["step_count"])
    assert 2*degrees.max()+1 <= config["width_budget"]
    data, models = OUT / "data", OUT / "models"
    data.mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    xval = -1 + (np.arange(config["validation_points"])+.5)*2/config["validation_points"]
    xe = np.linspace(-1., 1., config["evaluation_points"])
    xv_truth, truth = chirp(xval), chirp(xe)
    # Validation midpoints are disjoint from the reporting grid.
    assert not np.intersect1d(xval, xe).size
    coefficients = chebyshev_coefficients(chirp, config["chebyshev_points"])
    np.save(data / "chebyshev_coefficients.npy", coefficients)
    np.savez(data / "grids.npz", validation=xval, evaluation=xe, truth=truth)
    manifest = {"config": config, "precision_model": PROTOCOL,
                "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [HERE/"construction.py", HERE/"run.py", HERE/"config.yaml", SELECTOR]},
                "paper": {"doi": "10.1162/neco.1996.8.1.164", "construction": "Lemma 3.2, normalized centered tanh specialization",
                          "adaptation": "Discrete Chebyshev truncation; validation-selected degree and step, not unspecified theorem constants"},
                "numpy_version": np.__version__}
    json_write(data / "config.json", manifest)
    polynomial_errors = [relative_l2(chebval(xe, coefficients[:d+1]), truth) for d in degrees]
    validation = np.full((len(steps), len(degrees), len(bits)), np.inf)
    valid_models = np.zeros((len(steps), len(degrees)), dtype=bool)
    best = [None for _ in bits]
    start = time.monotonic()
    # Cache only one step's feature arrays. Every candidate is evaluated by
    # the same ordinary FP64 matrix-vector product as the final stored model.
    for hi, step in enumerate(steps):
        candidates = []
        for di, degree in enumerate(degrees):
            try:
                model = construct_mhaskar(coefficients[:degree+1], step, config["activation_bias"])
            except FloatingPointError:
                continue
            valid_models[hi, di] = True
            candidates.append((di, int(degree), model))
        largest = max(d for _, d, _ in candidates)
        slopes = .5*step*np.arange(-largest, largest+1, dtype=np.float64)
        for pi, p in enumerate(bits):
            features = np.tanh(xval[:, None]*round_bits(slopes, int(p))
                               + round_bits(config["activation_bias"], int(p)))
            for di, degree, model in candidates:
                # The indices identify shared *unrounded* slopes; distinct
                # neurons remain distinct if quantization makes slopes equal.
                indices = np.rint(model.slope/(.5*step)).astype(int)+largest
                try:
                    q = model.quantized(int(p))
                except FloatingPointError:
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    # Match the standalone evaluator's C-contiguous layout;
                    # BLAS accumulation order matters for cancelling readouts.
                    estimate = round_bits(np.ascontiguousarray(features[:, indices]) @ q.readout
                                          + q.offset, int(p))
                error = relative_l2(estimate, xv_truth)
                validation[hi, di, pi] = error
                if best[pi] is None or error < best[pi]["validation_error"]:
                    best[pi] = {"validation_error": error, "degree": degree,
                                "step": float(step), "model": model, "hi": hi, "di": di}
        if hi % 8 == 0 or hi == len(steps)-1:
            print(f"Mhaskar steps {hi+1}/{len(steps)}; {time.monotonic()-start:.1f}s", flush=True)
    np.savez_compressed(data / "mhaskar_search.npz", steps=steps, degrees=degrees, bits=bits,
                        validation_error=validation, finite_models=valid_models,
                        polynomial_error=polynomial_errors)
    rows = []
    quill_path = data / "quill.jsonl"
    with quill_path.open("w") as stream:
        for pi, p in enumerate(bits):
            pr = predictions(config["width_budget"], config["halo_per_side"], int(p), "chirp")
            if pr["refined"]["status"] != "threshold":
                pr["refined"] = selector.choose_lambda(
                    "tanh", spacing=2/pr["N"], e_tol=2.**(1-int(p)),
                    omega_scale=pr["omega_scale"], interval=(.03, 3.))
            assert pr["refined"]["status"] == "threshold"
            lam = float(pr["refined"]["lambda"])
            theta = 2/pr["N"]*pr["omega_scale"]
            assert abs(selector.log_alias_score("tanh", lam, theta)-np.log(2.**(1-int(p)))) < 1e-11
            _, spacing, centers = geometry(config["width_budget"], config["halo_per_side"])
            slope = np.full(centers.size, lam/spacing)
            bias = -slope*centers
            xt = np.linspace(-1., 1., config["train_points"])
            matrix = np.column_stack((np.tanh(xt[:, None]*slope+bias), np.ones(xt.size)))
            # Preserve the precision-dependent truncation used by panel (c).
            # The solve itself remains FP64; this is a rank-selection policy.
            weights, _, rank, _ = lstsq(matrix, chirp(xt), cond=2.**(1-int(p)),
                                       lapack_driver="gelsd")
            quill = TanhNetwork(slope, bias, weights[:-1], weights[-1])
            quill_estimate = quill.evaluate(xe, int(p))
            quill.quantized(int(p)).save(models / f"quill_p{p}.npz")
            picked = best[pi]
            mhaskar = picked["model"]
            # Recheck the cached-feature selection against the standalone evaluator.
            checked = relative_l2(mhaskar.evaluate(xval, int(p)), xv_truth)
            if not np.isclose(checked, picked["validation_error"], rtol=1e-10, atol=1e-12):
                raise AssertionError("Selection and standalone model evaluations differ")
            mhaskar_estimate = mhaskar.evaluate(xe, int(p))
            mhaskar.quantized(int(p)).save(models / f"mhaskar_p{p}.npz")
            row = {"p": int(p), "width_budget": config["width_budget"],
                   "quill_width": quill.width, "quill_lambda": lam, "quill_rank": int(rank),
                   "quill_error": relative_l2(quill_estimate, truth),
                   "quill_fp64_error": relative_l2(quill.evaluate(xe), truth),
                   "quill_max_parameter": float(max(np.max(abs(a)) for a in (slope, bias, weights))),
                   "mhaskar_width": mhaskar.width, "mhaskar_degree": picked["degree"],
                   "mhaskar_step": picked["step"], "mhaskar_validation_error": checked,
                   "mhaskar_error": relative_l2(mhaskar_estimate, truth),
                   "mhaskar_fp64_error": relative_l2(mhaskar.evaluate(xe), truth),
                   "mhaskar_polynomial_error": polynomial_errors[picked["di"]],
                   "mhaskar_readout_l1": float(norm(mhaskar.readout, 1)),
                   "mhaskar_max_parameter": float(max(np.max(abs(a)) for a in
                                                       (mhaskar.slope, mhaskar.bias, mhaskar.readout))),
                   "mhaskar_step_boundary": picked["hi"] in (0, len(steps)-1)}
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            rows.append(row)
            if pi % 8 == 0 or pi == len(bits)-1:
                print(f"QUILL and held-out errors {pi+1}/{len(bits)}; {time.monotonic()-start:.1f}s", flush=True)
    json_write(data / "summary.json", rows)
    # Recompute FP64 step sweeps on the held-out grid only for diagnostics;
    # these values never influence model selection above.
    diagnostic_degrees = [4, 8, 16, 32, 64, 128]
    errors = np.full((len(diagnostic_degrees), len(steps)), np.inf)
    for di, degree in enumerate(diagnostic_degrees):
        for hi, step in enumerate(steps):
            try:
                net = construct_mhaskar(coefficients[:degree+1], step, config["activation_bias"])
                errors[di, hi] = relative_l2(net.evaluate(xe), truth)
            except FloatingPointError:
                pass
    np.savez(data / "diagnostics.npz", degrees=diagnostic_degrees, steps=steps, network_error=errors)
    validation_report = {
        "complete": True, "precisions": len(rows), "candidate_count_per_precision": len(steps)*len(degrees),
        "finite_parameter_candidates": int(valid_models.sum()),
        "unrepresentable_parameter_candidates": int((~valid_models).sum()),
        "selected_models_obey_width_budget": all(r["mhaskar_width"] <= config["width_budget"] for r in rows),
        "validation_and_reporting_grids_disjoint": True,
        "selected_evaluations_match_cached_search": True,
        "selection_uses_reporting_grid": False,
        "higher_precision_arithmetic": False,
        "elapsed_seconds": time.monotonic()-start,
    }
    json_write(data / "validation.json", validation_report)
    print(json.dumps(validation_report), flush=True)
    return rows


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
    from experiments.expC12_mhaskar_comparison.plot import main
    main()
