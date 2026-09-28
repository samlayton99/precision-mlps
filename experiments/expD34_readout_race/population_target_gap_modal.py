"""Bounded Modal post-processing: two CSVs and two small archives, no training."""
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
BASE = Path("results/checkpoint_D_optimizers/expD34_readout_race")
SOURCE = BASE/"population_output/evidence/force_moment_archive/states.csv"
DENSE = BASE/"population_output/evidence/feedback_flow_100k/states.csv"
CAPSULE = BASE/"mechanism_refinement/energy_baseline_arb/inputs/N512_seed30_20k.npz"
ENDPOINTS = BASE/"population_output/evidence/feedback_flow_100k/endpoints.npz"
HELPER = Path("experiments/expD34_readout_race/population_target_gap.py")
TEST = Path("tests/test_population_target_gap.py")
FILES = (SOURCE, DENSE, CAPSULE, ENDPOINTS, HELPER, TEST,
         Path("experiments/expD34_readout_race/population_balance_probe.py"),
         Path("tests/test_population_balance_probe.py"),
         Path("experiments/expD34_readout_race/population_target_gap_modal.py"),
         Path("experiments/expD34_readout_race/targets.py"),
         Path("experiments/expD34_readout_race/effective_feedback_holdout.py"))
app = modal.App("d34-target-gap-audit")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.2.6", "matplotlib==3.10.3", "pytest==8.4.0")
         .env({"PYTHONPATH": "/work", "MPLBACKEND": "Agg",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}))
if modal.is_local():
    for path in FILES:
        image = image.add_local_file(ROOT/path, str(Path("/work")/path))


@app.function(image=image, cpu=2, memory=(1024, 4096), timeout=600,
              max_containers=1, retries=0)
def run(study: str = "gap") -> bytes:
    import resource
    import importlib.metadata
    started = time.time()
    output = Path("/tmp/target-gap")
    commands = [[sys.executable, "-m", "pytest", str(TEST), "tests/test_population_balance_probe.py", "-q", "-p", "no:cacheprovider"],
                [sys.executable, "-m", "experiments.expD34_readout_race.population_target_gap",
                 "--source", str(SOURCE), "--dense", str(DENSE), "--capsule", str(CAPSULE),
                 "--output", str(output)]]
    if study == "balance":
        commands[-1] = [sys.executable, "-m", "experiments.expD34_readout_race.population_balance_probe",
                        "--capsule", str(CAPSULE), "--endpoints", str(ENDPOINTS), "--output", str(output)]
    elif study != "gap":
        raise ValueError("Unknown study")
    logs = []
    for command in commands:
        result = subprocess.run(command, cwd="/work", text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=260, check=False)
        print(result.stdout, flush=True)
        logs.append(dict(command=command, returncode=result.returncode, stdout=result.stdout))
        if result.returncode:
            raise RuntimeError("Remote verification failed")
    record = dict(platform="Modal CPU", study=study, memory_hard_limit_mib=4096,
                  child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                  started_unix=started, finished_unix=time.time(), commands=logs,
                  sources={str(p): hashlib.sha256((Path("/work")/p).read_bytes()).hexdigest() for p in FILES},
                  versions={p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "pytest")})
    (output/"execution.json").write_text(json.dumps(record, indent=2)+"\n")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, path.name)
    if len(packed.getvalue()) > 16*1024**2:
        raise RuntimeError("Unexpected artifact size")
    return packed.getvalue()


@app.local_entrypoint()
def main(output: str, study: str = "gap"):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Use a fresh evidence directory")
    data = run.remote(study)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(p.file_size for p in archive.infolist()) > 16*1024**2:
            raise ValueError("Artifact cap exceeded")
        if any(Path(p.filename).name != p.filename for p in archive.infolist()):
            raise ValueError("Expected flat artifacts")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    print(f"Downloaded {len(data)} bytes to {destination}")
