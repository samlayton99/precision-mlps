"""Resumable 144-case, float64 full-batch GD sweep on the local compute host."""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precisionmlps-mpl")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expD56_scale_acquisition.model import (
    TARGETS, diagnostics, gd_chunk, initial_parameters, midpoint_grid, target,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/checkpoint_D_optimizers/expD56_scale_acquisition"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj):
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def load_config():
    return yaml.safe_load((HERE/"config.yaml").read_text())


def case_id(name, multiplier, lr):
    return f"{name}__init{multiplier:g}__lr{lr:.0e}"


def all_cases(config):
    names = [name for group in config["targets"].values() for name in group]
    assert len(names)==len(set(names))==16 and set(names)==set(TARGETS)
    assert config["width"]==196
    # Interleave targets, so no function difficulty group waits for all other groups.
    groups = list(config["targets"].values())
    names = [group[i] for i in range(max(map(len, groups))) for group in groups if i<len(group)]
    return [(name, multiplier, lr) for name in names
            for multiplier in config["initialization_multipliers"] for lr in config["learning_rates"]]


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def manifest(config):
    reference = config["quill_reference"]
    assert reference["interior_intervals"]+1+2*reference["halo_each_side"]==config["width"]
    return {
        "created_utc": utc_now(), "host": platform.node(), "python": sys.version,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "config": config, "config_sha256": config_digest(config),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in HERE.glob("*.py")},
        "target_formulas": {name: {"difficulty": v[0], "formula": v[2]} for name,v in TARGETS.items()},
        "run_count": len(all_cases(config)),
        "initialization_note": "Default float64 torch.nn.Linear(1,196) and Linear(196,1). First-layer weight and bias multiplied by 1, 2, or 4. Identical readout and random draw across all targets and rates. All parameters then trained freely.",
        "halo_note": "Standard initialization has random centers, not prescribed halo neurons. There are 196 total units; the QUILL reference uses 148 interior centers plus 48 halo centers within this same budget.",
        "quill_gamma_reference": reference["lambda"]*reference["interior_intervals"]/2,
        "uncertainty_note": "One paired seed. Percentile bands are across 16 targets, not confidence intervals or seed uncertainty.",
        "diagnostic_scope": "Structural diagnostics at logged checkpoints; not continuous assumption verification or a theorem certificate. q_generic_current is a current-state output allowance, not a forward growth bound.",
    }


def save_checkpoint(directory, theta, history, snapshots, step, digest):
    keys = list(history[0])
    temp = directory/"checkpoint.tmp.npz"
    np.savez_compressed(temp, theta=theta.numpy(), step=step,
                        keys=np.array(keys), history=np.array([[r[k] for k in keys] for r in history]),
                        snapshots=np.array(snapshots), config_sha256=np.array(digest))
    temp.replace(directory/"checkpoint.npz")


def log_steps(config):
    end = config["steps"]
    return sorted({0, end, *[s for s in (1,2,5,10,20,50,100,200) if s<=end],
                   *range(config["log_every"], end+1, config["log_every"])})


def train_case(case, config, output):
    name, multiplier, lr = case
    torch.set_num_threads(config["threads_per_worker"])
    directory = Path(output)/"runs"/case_id(*case)
    directory.mkdir(parents=True, exist_ok=True)
    digest = config_digest(config)
    status_path = directory/"status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["config_sha256"] != digest:
            raise RuntimeError(f"Configuration mismatch in {directory}")
        if previous["state"] in ("complete", "nonfinite"):
            return previous
    x = midpoint_grid(config["train_points"])
    xe = midpoint_grid(config["eval_points"])
    raw = target(name, x)
    norm = float(np.sqrt(np.mean(raw**2)))
    y, ye = raw/norm, target(name, xe)/norm
    h_ref = 2/config["quill_reference"]["interior_intervals"]
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    theta = initial_parameters(config["width"], config["seed"], multiplier)
    initial_path = directory/"initial.npz"
    if not initial_path.exists():
        np.savez_compressed(initial_path, theta=theta.numpy(), train_x=x, train_y=y,
                            eval_x=xe, eval_y=ye, target_rms=norm)
    history, snapshots, step = [], [], 0
    checkpoint = directory/"checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            if data["config_sha256"].item()!=digest: raise RuntimeError("Checkpoint configuration mismatch")
            theta = torch.from_numpy(data["theta"].copy())
            step = int(data["step"])
            keys = data["keys"].tolist()
            history = [dict(zip(keys, row.tolist())) for row in data["history"]]
            snapshots = list(data["snapshots"].copy())
    started = time.monotonic()
    start_step = step
    metadata = {"target": name, "difficulty": TARGETS[name][0], "initialization_multiplier": multiplier,
                "learning_rate": lr, "seed": config["seed"], "width": config["width"],
                "target_rms_normalizer": norm, "pid": os.getpid(), "config_sha256": digest}

    def record():
        row = {"step": step, **diagnostics(theta.numpy(), x, y, xe, ye, h_ref)}
        history.append(row)
        snapshots.append(theta.numpy().copy())

    def publish(state):
        elapsed = time.monotonic()-started
        result = {**metadata, "state": state, "step": step, "total_steps": config["steps"],
                  "updated_utc": utc_now(), "elapsed_seconds_this_session": elapsed,
                  "steps_per_second": (step-start_step)/elapsed if elapsed>0 else 0,
                  "gamma_rms": history[-1]["gamma_rms"], "eval_rel_l2": history[-1]["eval_rel_l2"]}
        write_json(status_path, result)
        return result

    if not history: record()
    save_checkpoint(directory, theta, history, snapshots, step, digest)
    publish("running")
    print(f"START {case_id(*case)} step={step} gamma={history[-1]['gamma_rms']:.6g}", flush=True)
    for endpoint in log_steps(config):
        if endpoint<=step: continue
        theta, executed, finite = gd_chunk(theta, tx, ty, float(lr), endpoint-step)
        step += executed
        if not finite:
            # Keep the last finite state and mark the failed update. Never hide this case.
            if history[-1]["step"]!=step: record()
            save_checkpoint(directory, theta, history, snapshots, step, digest)
            result = publish("nonfinite")
            print(f"NONFINITE {case_id(*case)} at update {step+1}", flush=True)
            return result
        record()
        if step % config["checkpoint_every"]==0 or step==config["steps"]:
            save_checkpoint(directory, theta, history, snapshots, step, digest)
            result = publish("complete" if step==config["steps"] else "running")
            print(f"PROGRESS {case_id(*case)} {step}/{config['steps']} "
                  f"gamma={result['gamma_rms']:.5g} err={result['eval_rel_l2']:.5g} "
                  f"speed={result['steps_per_second']:.1f}/s", flush=True)
    return publish("complete")


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["config_sha256"]!=config_digest(config): raise RuntimeError("Existing manifest differs")
        for name, digest in existing["source_sha256"].items():
            if name not in ("run.py", "model.py"):
                continue  # Figure layout can evolve without changing training.
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f"Source changed since launch: {name}; use a new output directory")
    else:
        write_json(manifest_path, manifest(config))
        source = output/"provenance"
        source.mkdir(exist_ok=True)
        for p in [*HERE.glob("*.py"), HERE/"config.yaml"]: shutil.copy2(p, source/p.name)
    write_json(output/"sweep_status.json", {"state":"running", "pid":os.getpid(), "updated_utc":utc_now()})
    failures = []
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_case, case, config, str(output)):case for case in all_cases(config)}
        for future in as_completed(futures):
            case = futures[future]
            try:
                result=future.result()
                if result["state"]!="complete": failures.append({"case":case_id(*case), "state":result["state"]})
            except Exception as exc:
                failures.append({"case":case_id(*case), "state":"error", "error":repr(exc)})
                directory=output/"runs"/case_id(*case)
                directory.mkdir(parents=True,exist_ok=True)
                path=directory/"status.json"
                previous=json.loads(path.read_text()) if path.exists() else {
                    "target":case[0],"initialization_multiplier":case[1],"learning_rate":case[2],
                    "step":0,"config_sha256":config_digest(config)}
                write_json(path,{**previous,"state":"error","error":repr(exc),"updated_utc":utc_now()})
                print(f"ERROR {case}: {exc!r}", flush=True)
            write_json(output/"sweep_status.json", {"state":"running", "pid":os.getpid(),
                       "finished_futures":sum(f.done() for f in futures), "total":len(futures),
                       "failures":failures, "updated_utc":utc_now()})
    write_json(output/"sweep_status.json", {"state":"complete" if not failures else "finished_with_failures",
               "total":len(all_cases(config)), "failures":failures, "updated_utc":utc_now()})
    subprocess.run([sys.executable, str(HERE/"plot.py"), "--results", str(output)], check=True)


def status(output):
    rows=[json.loads(p.read_text()) for p in sorted((output/"runs").glob("*/status.json"))]
    counts={s:sum(r["state"]==s for r in rows) for s in ("running","complete","nonfinite","error")}
    print(json.dumps({"output":str(output), "counts":counts, "started":len(rows),
                      "updated_steps":{r["target"]+f"/x{r['initialization_multiplier']:g}/{r['learning_rate']:g}":r["step"]
                                       for r in rows if r["state"]=="running"}}, indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot","run","launch","status"))
    parser.add_argument("--output", type=Path, default=OUT/"sweep")
    args=parser.parse_args()
    config=load_config()
    if args.mode=="status": status(args.output); return
    if args.mode=="pilot":
        torch.set_num_threads(1)
        x=torch.from_numpy(midpoint_grid(config["train_points"]))
        y=torch.from_numpy(target("chirp", x.numpy()))
        base=initial_parameters(config["width"],config["seed"],1.)
        gd_chunk(base, x, y, .001, 5)
        start=time.monotonic()
        _,n,ok=gd_chunk(base,x,y,.001,1000)
        elapsed=time.monotonic()-start
        report={"updates":n,"finite":ok,"seconds":elapsed,"steps_per_second":n/elapsed,
                "serial_training_hours_estimate":elapsed/n*config["steps"]*len(all_cases(config))/3600}
        print(json.dumps(report, indent=2))
        return
    if args.mode=="launch":
        args.output.mkdir(parents=True,exist_ok=True)
        pidfile=args.output/"launcher.json"
        if pidfile.exists():
            old=json.loads(pidfile.read_text())
            try: os.kill(old["pid"],0)
            except ProcessLookupError: pass
            else: raise RuntimeError(f"Existing coordinator PID {old['pid']} is still alive")
        with (args.output/"run.log").open("a") as log:
            process=subprocess.Popen([sys.executable,str(HERE/"run.py"),"run","--output",str(args.output)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(pidfile,{"pid":process.pid,"launched_utc":utc_now(),"log":str(args.output/"run.log")})
        print(json.dumps({"pid":process.pid,"output":str(args.output)},indent=2))
    else: run(config,args.output)


if __name__=="__main__": main()
