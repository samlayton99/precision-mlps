"""The 88-run crossed initialization/map experiment, with a separate GPU budget."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
import optax

from . import core
from .run import Case, run_batch, write_json


def manifest():
    g = core.geometry(512)
    records = []
    families = ["xavier", "xavier_a_uniform", "xavier_a_reference", "envelope"]
    pairs = [(1e-4, 1e-3), (1e-3, 1e-3), (1e-3, 1e-2)]
    for family in families:
        for arm in ["uniform", "both"]:
            for rr, rg in pairs:
                for seed in [2, 3]:
                    case = Case(arm=arm, initialization=family, seed=seed,
                                rate_r=rr if arm == "uniform" else rr * np.sqrt(g.h / g.ordinary_alpha),
                                rate_g=rg, epsilon_mode="native", bias_rate=rr * np.sqrt(g.h), dense_validation=True)
                    records.append({"case": asdict(case), "key": case.key, "kind": "primary",
                                    "uniform_readout_lr": rr, "lambda_lr": rg})
                    if (rr, rg) == (1e-3, 1e-2):
                        physical = replace(case, epsilon_mode="physical")
                        records.append({"case": asdict(physical), "key": physical.key, "kind": "epsilon_control",
                                        "uniform_readout_lr": rr, "lambda_lr": rg})
        for rate in [1e-4, 1e-3, 1e-2]:
            for seed in [2, 3]:
                case = Case(arm="raw", initialization=family, seed=seed, rate_r=rate, rate_g=rate,
                            epsilon_mode="native", bias_rate=rate, dense_validation=True)
                records.append({"case": asdict(case), "key": case.key, "kind": "shared_lr_control"})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Run the GPU worker inside an allocated srun step with its GPU mask intact.")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]], text=True)
    step_id = f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
    step = subprocess.check_output(["scontrol", "show", "step", step_id], text=True)
    if "JobState=RUNNING" not in allocation or "State=RUNNING" not in step or "gpu" not in step:
        raise RuntimeError("Scheduler did not confirm an active GPU step.")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1 or not jax.config.x64_enabled:
        raise RuntimeError("The worker requires exactly one allocated GPU and JAX FP64.")
    root = args.root
    output = root / "runs" / "focused"
    output.mkdir(parents=True, exist_ok=True)
    records = [r for r in manifest() if r["case"]["seed"] == args.shard + 2]
    write_json(root / f"manifest_focused_{args.shard}.json", records)
    environment = {"jax": jax.__version__, "optax": optax.__version__, "x64": jax.config.x64_enabled,
                   "devices": str(jax.devices()), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "allocation": allocation, "step": step,
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                   "pip_freeze": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)}
    write_json(root / f"environment_focused_{os.environ['SLURM_JOB_ID']}.json", environment)
    print(json.dumps({"start": True, "shard": args.shard, "cases": len(records), "devices": environment["devices"],
                      "cuda_visible_devices": environment["cuda_visible_devices"], "x64": True}), flush=True)
    ledger_path = root / f"budget_focused_{args.shard}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"seconds_used": 0.}
    if "active_since" in ledger:
        ledger["seconds_used"] += max(0., time.time() - ledger["active_since"])
    started = time.monotonic()
    # Two 55-minute worker caps plus the allocation's startup/cleanup fit in 2 GPU-hours.
    deadline = started + max(0., 3300 - ledger["seconds_used"])
    ledger.update(active_since=time.time(), limit_seconds=3300, slurm_job_id=os.environ["SLURM_JOB_ID"])
    write_json(ledger_path, ledger)
    try:
        for frontier in [20000, 80000, 160000, 320000]:
            groups = defaultdict(list)
            for record in records:
                case = Case(**record["case"])
                latest_path = output / case.key / "latest.json"
                latest = json.loads(latest_path.read_text()) if latest_path.exists() else {"step": 0, "status": "continuing"}
                if latest["status"] == "nonfinite" or latest["step"] >= frontier:
                    continue
                groups[latest["step"]].append(case)
            for cases in groups.values():
                for index in range(0, len(cases), 4):
                    if time.monotonic() >= deadline - 120:
                        return
                    batch = cases[index:index + 4]
                    before = time.monotonic()
                    results = run_batch(batch, output, frontier, deadline - 60)
                    print(json.dumps({"frontier": frontier, "seconds": time.monotonic() - before,
                                      "cases": [c.key for c in batch], "steps": [r["step"] for r in results],
                                      "status": [r["status"] for r in results]}), flush=True)
                    if any(r["step"] < frontier and r["status"] != "nonfinite" for r in results):
                        return
            print(json.dumps({"frontier_complete": frontier, "shard": args.shard}), flush=True)
    finally:
        ledger["seconds_used"] += time.monotonic() - started
        ledger.pop("active_since", None)
        write_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
