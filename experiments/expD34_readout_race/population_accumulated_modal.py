"""CPU-only Modal audit of three cached CSVs; never load parameter checkpoints.

Run with .venv-modal/bin/modal run experiments/expD34_readout_race/
population_accumulated_modal.py --output <new-local-evidence-directory>.
Only the six explicitly named files below are uploaded. Tests and numerical
analysis run remotely, with a 4 GiB hard memory limit and a ten-minute timeout.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
EVIDENCE = Path("results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence")
HELPER = Path("experiments/expD34_readout_race/population_accumulated_audit.py")
TEST = Path("tests/test_population_accumulated_audit.py")
SOURCES = (EVIDENCE / "force_moment_archive/states.csv",
           EVIDENCE / "force_rotation_dilations_final/states.csv")
MOTION = EVIDENCE / "dilation_final_summary/states.csv"
FILES = (*SOURCES, MOTION, HELPER, TEST,
         Path("experiments/expD34_readout_race/population_accumulated_modal.py"))

app = modal.App("d34-population-accumulated-audit")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.2.6", "scipy==1.15.3", "matplotlib==3.10.3", "pytest==8.4.0")
         .env({"PYTHONPATH": "/work", "MPLBACKEND": "Agg",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}))
if modal.is_local():
    for relative in FILES:
        image = image.add_local_file(ROOT / relative, str(Path("/work") / relative))


@app.function(image=image, cpu=2, memory=(1024, 4096), timeout=600,
              max_containers=1, retries=0)
def audit() -> bytes:
    import importlib.metadata
    import resource

    started = time.time()
    root, output = Path("/work"), Path("/tmp/accumulated-audit")
    hashes = {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in FILES}
    commands = [[sys.executable, "-m", "pytest", str(TEST), "-q", "-p", "no:cacheprovider"]]
    command = [sys.executable, "-m", "experiments.expD34_readout_race.population_accumulated_audit"]
    for source in SOURCES:
        command.extend(["--source", str(source)])
    command.extend(["--motion-source", str(MOTION), "--output", str(output)])
    commands.append(command)
    logs = []
    for command in commands:
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=240, check=False)
        print(result.stdout, flush=True)
        logs.append(dict(command=command, returncode=result.returncode, stdout=result.stdout))
        if result.returncode:
            raise RuntimeError(f"Remote command failed with status {result.returncode}")
    record = dict(platform="Modal CPU", memory_request_mib=1024, memory_hard_limit_mib=4096,
                  cpu=2, gpu=None, started_unix=started, finished_unix=time.time(),
                  child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
                  sources=hashes, commands=logs,
                  versions={p: importlib.metadata.version(p)
                            for p in ("numpy", "scipy", "matplotlib", "pytest")})
    (output / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    artifacts = sorted(output.iterdir())
    if sum(p.stat().st_size for p in artifacts) > 32 * 1024**2:
        raise RuntimeError("Unexpected output size: do not download more than 32 MiB")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in artifacts:
            archive.write(path, path.name)
    print(json.dumps({k: record[k] for k in ("platform", "child_peak_rss_mib", "finished_unix")}), flush=True)
    return packed.getvalue()


@app.local_entrypoint()
def main(output: str):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Use a new output directory to preserve earlier evidence")
    result = audit.remote()
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 32 * 1024**2:
            raise ValueError("Remote artifact limit exceeded")
        if any(Path(item.filename).name != item.filename for item in archive.infolist()):
            raise ValueError("Only flat output artifacts are expected")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    print(f"Downloaded {len(result)} compressed bytes to {destination}")
