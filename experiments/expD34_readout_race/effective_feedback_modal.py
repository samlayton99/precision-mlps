"""Single-GPU Modal launch for an explicitly budgeted effective-feedback case.

Use ``modal run .../effective_feedback_modal.py --inputs ... --run-id ...``.
This creates no public endpoint. Each invocation needs its own budget reservation.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("effective_feedback.py", "effective_feedback_kernel.py",
                "adam_forces.py", "adam_run.py", "targets.py", "transport.py",
                "run.py", "core.py", "references.py", "effective_feedback_modal.py")
app = modal.App("d34-effective-feedback")
volume = modal.Volume.from_name("d34-effective-feedback", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("jax[cuda12]==0.11.1", "numpy==2.5.1", "scipy==1.18.0")
         .env({"JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cuda",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": "/code",
               "D34_EXECUTION_PLATFORM": "modal"}))
if modal.is_local():
    for name in SOURCE_FILES:
        source = HERE / name
        image = image.add_local_file(source, f"/code/experiments/expD34_readout_race/{source.name}")


@app.function(image=image, gpu=["H200", "H100"], cpu=4, memory=16384,
              volumes={"/evidence": volume}, min_containers=0, max_containers=4,
              buffer_containers=0, scaledown_window=2, retries=0, timeout=3600,
              single_use_containers=True)
def run_case(run_id: str, input_hash: str, source_hashes: dict, arm: str,
             eta: float, degree: int, horizon: int, max_seconds: float,
             reservation: str, indices: str, prediction_hash: str) -> dict:
    import jax

    if not reservation.strip() or not 0 < max_seconds <= 3300:
        raise ValueError("A reservation and bounded runner duration are required")
    root = Path("/evidence") / run_id
    inputs = root / "inputs.npz"
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != input_hash:
        raise ValueError("Uploaded inputs do not match the reserved invocation")
    prediction_path = root / "predictions.json"
    if prediction_hash and hashlib.sha256(prediction_path.read_bytes()).hexdigest() != prediction_hash:
        raise ValueError("Uploaded predictions differ from the reserved invocation")
    for name, digest in source_hashes.items():
        path = Path("/code/experiments/expD34_readout_race") / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash differs: {name}")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected one allocated GPU; got {devices}")
    command = [sys.executable, "-m", "experiments.expD34_readout_race.effective_feedback",
               "run", "--inputs", str(inputs), "--output", str(root / "run"),
               "--arm", arm, "--eta", str(eta), "--degree", str(degree),
               "--horizon", str(horizon), "--max-seconds", str(max_seconds),
               "--backend", "modal"]
    if indices:
        command.extend(["--indices", indices])
    if prediction_hash:
        command.extend(["--predictions", str(prediction_path)])
    record = dict(reservation=reservation, input_sha256=input_hash,
                  prediction_sha256=prediction_hash,
                  source_sha256=source_hashes, command=command, device=str(devices[0]),
                  device_kind=devices[0].device_kind, started_unix=time.time())
    launch_path = root / f"launch-{time.time_ns()}.json"
    launch_path.write_text(json.dumps(record, indent=2) + "\n")
    (root / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    volume.commit()
    try:
        with (root / "stdout.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=max_seconds + 180, check=False)
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Runner exited {result.returncode}; see {run_id}/stdout.log")
        return record
    finally:
        record["finished_unix"] = time.time()
        record["process_wall_seconds"] = record["finished_unix"] - record["started_unix"]
        launch_path.write_text(json.dumps(record, indent=2) + "\n")
        (root / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        volume.commit()


def upload_run(inputs: str, run_id: str, predictions: str = "", resume: bool = False) -> dict:
    """Upload an immutable input capsule, or verify it exactly before resume."""
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run-id must be a unique alphanumeric/underscore/hyphen label")
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in SOURCE_FILES}
    path = Path(inputs)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prediction_hash = hashlib.sha256(Path(predictions).read_bytes()).hexdigest() if predictions else ""
    capsule = dict(run_id=run_id, input_hash=digest, source_hashes=source_hashes,
                   prediction_hash=prediction_hash)
    if resume:
        stored = json.loads(b"".join(volume.read_file(f"/{run_id}/capsule.json")))
        if stored != capsule:
            raise ValueError("Resume inputs, predictions, or source files differ")
        for name, expected in (("inputs.npz", digest), ("predictions.json", prediction_hash)):
            if expected and hashlib.sha256(b"".join(volume.read_file(f"/{run_id}/{name}"))).hexdigest() != expected:
                raise ValueError(f"Stored upload differs: {name}")
        return capsule
    with volume.batch_upload(force=False) as batch:
        batch.put_file(path, f"/{run_id}/inputs.npz")
        if predictions:
            batch.put_file(Path(predictions), f"/{run_id}/predictions.json")
        batch.put_file(io.BytesIO(json.dumps(capsule, sort_keys=True).encode()), f"/{run_id}/capsule.json")
    return capsule


@app.local_entrypoint()
def main(inputs: str, run_id: str, reservation: str, arm: str = "joint",
         eta: float = .002, degree: int = 65, horizon: int = 20000,
         max_seconds: float = 1200, indices: str = "", predictions: str = "",
         resume: bool = False):
    if not reservation.strip():
        raise ValueError("An explicit shared-budget reservation identifier is required")
    if arm not in ("joint", "freeze_map", "clamp_residual"):
        raise ValueError("Unknown feedback arm")
    if not 0 < max_seconds <= 3300 or horizon <= 0:
        raise ValueError("Require a positive horizon and at most 3300 runner seconds")
    capsule = upload_run(inputs, run_id, predictions, resume)
    function = run_case.with_options(timeout=int(max_seconds + 240))
    print(json.dumps(function.remote(run_id, capsule['input_hash'], capsule['source_hashes'],
                                    arm, eta, degree, horizon, max_seconds, reservation,
                                    indices, capsule['prediction_hash']), indent=2))
