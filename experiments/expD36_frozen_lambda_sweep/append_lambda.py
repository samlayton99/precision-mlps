"""Train an additional lambda sequentially in each arm, preserving saved cases."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run
import ablations
from plot import plot_all

CASE_AXES = {"c": 0, "adam_m": 0, "adam_v": 0, "physical_c": 0,
             "train_rel_l2": 2, "eval_rel_l2": 2, "coefficient_snapshots": 1,
             "best_recorded_index": 1}
STATIC_REFERENCE = {"centers", "x_train", "x_eval", "y_train", "y_eval", "alpha", "readout_map"}


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def concatenate_cases(old, new, axis, order):
    return np.take(np.concatenate((old, new), axis=axis), order, axis=axis)


def merge_trajectory(old, new, old_lambdas, new_lambdas, cfg):
    for name in ("step", "steps", "coefficient_steps"):
        np.testing.assert_array_equal(old[name], new[name])
    combined = np.r_[old_lambdas, new_lambdas]
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Cannot append an existing lambda")
    order = np.argsort(combined)
    old_positions = np.argsort(order)[:len(old_lambdas)]
    result = {}
    for key, value in old.items():
        if key in CASE_AXES:
            result[key] = concatenate_cases(value, new[key], CASE_AXES[key], order)
            np.testing.assert_array_equal(np.take(result[key], old_positions, axis=CASE_AXES[key]), value)
        else:
            result[key] = value
    if "physical_c" not in result:
        # Original raw baseline predates the explicit physical_c field.
        if cfg["coordinates"] != "raw":
            raise ValueError("Mapped checkpoint is missing its physical coefficients")
        result["physical_c"] = result["c"].copy()
    result["config_sha256"] = np.asarray(run.fingerprint(cfg))
    np.testing.assert_array_equal(result["best_recorded_index"], result["eval_rel_l2"].argmin(axis=0))
    return result, order, old_positions


def extend_arm(folder, arm, value, baseline_folder):
    meta_path = folder / "data/metadata.json"
    meta = json.loads(meta_path.read_text())
    original = arrays(folder / "data/trajectory.npz")
    reference = arrays(folder / "data/reference.npz")
    if value in reference["lambdas"]:
        print(f"{arm}: lambda {value:g} is already saved; skipping training", flush=True)
        return
    horizon = int(original["step"])
    subset_cfg = dict(meta["config"], coordinates="raw", lambda_values=[value], steps=horizon)
    p = run.make_problem(subset_cfg)
    if arm != "raw":
        raw_reference = arrays(baseline_folder / "data/reference.npz")
        where = np.flatnonzero(raw_reference["lambdas"] == value)
        if len(where) != 1:
            raise ValueError("Extend the baseline first")
        p.reference = {key: raw_reference[key][where].copy() for key in p.reference}
        p = ablations.transform_problem(p, arm)
    for key, actual in (("centers", p.centers.numpy()), ("x_train", p.x_train.numpy()),
                        ("x_eval", p.x_eval.numpy()), ("y_train", p.y_train.numpy()),
                        ("y_eval", p.y_eval.numpy())):
        np.testing.assert_array_equal(reference[key], actual)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".lambda_extension_", dir=folder / "data") as scratch:
        scratch = Path(scratch)
        run.record_problem(p, scratch)
        state, _ = run.train(p, scratch, horizon)
        addition = arrays(scratch / "data/trajectory.npz")
        new_reference = arrays(scratch / "data/reference.npz")
        updated_cfg = dict(meta["config"])
        if "lambda_values" in updated_cfg:
            updated_cfg["lambda_values"] = sorted(reference["lambdas"].tolist()+[value])
        else:
            updated_cfg["additional_lambdas"] = sorted(set(updated_cfg.get("additional_lambdas", [])+[value]))
        merged, order, old_positions = merge_trajectory(original, addition, reference["lambdas"],
                                                       p.lambdas, updated_cfg)
        merged_reference = {}
        for key, old in reference.items():
            if key in STATIC_REFERENCE:
                np.testing.assert_array_equal(old, new_reference[key])
                merged_reference[key] = old
            else:
                merged_reference[key] = concatenate_cases(old, new_reference[key], 0, order)
                np.testing.assert_array_equal(merged_reference[key][old_positions], old)
        # All consistency checks happen before replacing any saved artifact.
        run.save_npz(folder / "data/trajectory.npz", **merged)
        run.save_npz(folder / "data/reference.npz", **merged_reference)
        meta.update(config=updated_cfg, config_sha256=run.fingerprint(updated_cfg),
                    lambdas=merged_reference["lambdas"].tolist(),
                    gammas=merged_reference["gammas"].tolist(),
                    gd_rates=merged_reference["gd_rates"].tolist(),
                    grid="original eight values preserved; additional lambda values appended independently")
        audit = {"lambda": value, "gamma": float(p.gammas[0]), "steps": horizon,
                 "new_runs": 2*len(p.cfg["targets"]), "existing_lambda_count": len(reference["lambdas"]),
                 "original_cases_array_equal": True, "seconds": time.perf_counter()-started}
        if p.readout_map is not None:
            native = p.mapped_E @ state["c"]
            physical = p.E @ run.physical_coefficients(p, state)
            audit["prediction_discrepancy_relative_to_target"] = float(
                (torch.linalg.vector_norm(native-physical, dim=1)
                 / torch.linalg.vector_norm(p.y_eval_pair, dim=0)).max())
        meta.setdefault("lambda_extensions", []).append(audit)
        run.save_json(meta_path, meta)
        if p.readout_map is not None:
            validation_path = folder / "data/map_validation.json"
            validation = json.loads(validation_path.read_text())
            validation["maximum_final_prediction_discrepancy_relative_to_target"] = max(
                validation["maximum_final_prediction_discrepancy_relative_to_target"],
                audit["prediction_discrepancy_relative_to_target"])
            validation["lambda_count"] = len(merged_reference["lambdas"])
            run.save_json(validation_path, validation)
        status_path = folder / "data/status.json"
        status = json.loads(status_path.read_text())
        status.update(lambda_count=len(merged_reference["lambdas"]),
                      last_lambda_extension=audit, complete=True)
        run.save_json(status_path, status)
        # Read back and verify old trajectories, coefficients, and moments exactly.
        committed = arrays(folder / "data/trajectory.npz")
        for key, axis in CASE_AXES.items():
            if key in original:
                np.testing.assert_array_equal(np.take(committed[key], old_positions, axis=axis), original[key])
        print(f"{arm}: appended lambda {value:g}; original cases unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("value", type=float)
    parser.add_argument("--baseline", type=Path, default=run.RESULTS)
    args = parser.parse_args()
    meta = json.loads((args.baseline / "data/metadata.json").read_text())
    torch.set_num_threads(meta["config"]["torch_threads"])
    torch.set_num_interop_threads(1)
    arms = [(args.baseline, "raw")]+[(args.baseline / "ablations" / name, name) for name in ablations.DESCRIPTIONS]
    with threadpool_limits(limits=meta["config"]["torch_threads"]):
        for folder, arm in arms:
            extend_arm(folder, arm, args.value, args.baseline)
        source = json.loads((args.baseline / "data/metadata.json").read_text())
        digest = hashlib.sha256((args.baseline / "data/trajectory.npz").read_bytes()).hexdigest()
        for folder, arm in arms:
            if arm != "raw":
                path = folder / "data/metadata.json"
                arm_meta = json.loads(path.read_text())
                arm_meta.update(baseline_config_sha256=source["config_sha256"], baseline_trajectory_sha256=digest)
                run.save_json(path, arm_meta)
            plot_all(folder, baseline_output=None if arm == "raw" else args.baseline)


if __name__ == "__main__":
    main()
