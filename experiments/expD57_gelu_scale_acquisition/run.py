"""Matched two-target GELU sweeps, with persistent GD/Adam state and live plots."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key,"1")
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.model import TARGETS, initial_parameters, midpoint_grid, target
from experiments.expD56_scale_acquisition.run import utc_now, write_json, config_digest, case_id
from experiments.expD57_gelu_scale_acquisition.model import train_chunk, predict

HERE=Path(__file__).resolve().parent
OUT=ROOT/"results/checkpoint_D_optimizers/expD57_gelu_scale_acquisition"


def load_config():
    return yaml.safe_load((HERE/"config.yaml").read_text())


def cases(config):
    names=[name for group in config["targets"].values() for name in group]
    return [(optimizer,name,multiplier,lr) for multiplier in config["initialization_multipliers"]
            for name in names for lr in config["learning_rates"] for optimizer in config["optimizers"]]


def source_paths():
    return [HERE/"run.py",HERE/"model.py",ROOT/"experiments/expD56_scale_acquisition/model.py",
            ROOT/"experiments/expD56_scale_acquisition/run.py"]


def prepare(config,output):
    output.mkdir(parents=True,exist_ok=True)
    digest=config_digest(config)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()}
    path=output/"manifest.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["config_sha256"]!=digest or old["training_sources"]!=hashes:
            raise RuntimeError("Existing sweep has different configuration or training source")
    else:
        metadata={"config":config,"config_sha256":digest,"created_utc":utc_now(),
                  "training_sources":hashes,"runs":len(cases(config)),"torch_version":torch.__version__,
                  "activation":"Exact GELU: x Phi(x), no tanh approximation",
                  "quill_reference_note":"GELU lambda=0.707 follows expD19 / expC07; this is a construction-scale reference, not a necessary accuracy threshold.",
                  "diagnostic_scope":"No tanh-specific theorem bounds are applied to GELU or Adam.",
                  "target_formulas":{n:TARGETS[n][2] for group in config["targets"].values() for n in group},
                  "band":"1st to 99th percentile of absolute slopes across 196 neurons within each run"}
        write_json(path,metadata)
        for p in [*source_paths(),HERE/"config.yaml"]:
            destination=output/"provenance"/p.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,destination)
    for optimizer in config["optimizers"]:
        folder=output/optimizer
        folder.mkdir(exist_ok=True)
        subconfig={**config,"optimizer":optimizer,"optimizer_label":"GD" if optimizer=="gd" else "Adam",
                   "figure_title":f"Scale Acquisition Ablation: 2 functions (GELU, {'GD' if optimizer=='gd' else 'Adam'})"}
        write_json(folder/"manifest.json",{"config":subconfig,"config_sha256":digest})


def run_round(case, config, output, endpoint):
    torch.set_num_threads(1)
    optimizer,name,multiplier,lr=case
    directory=Path(output)/optimizer/"runs"/case_id(name,multiplier,lr)
    directory.mkdir(parents=True,exist_ok=True)
    status_path=directory/"status.json"
    digest=config_digest(config)
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old["config_sha256"]!=digest: raise RuntimeError("Mismatched run configuration")
        if old["state"] in ("complete","nonfinite","error"): return old
    x,xe=map(torch.from_numpy,(midpoint_grid(config["train_points"]),midpoint_grid(config["eval_points"])))
    raw=target(name,x.numpy()); normalizer=float(np.sqrt(np.mean(raw**2)))
    y=torch.from_numpy(raw/normalizer)
    ye=torch.from_numpy(target(name,xe.numpy())/normalizer)
    theta=initial_parameters(config["width"],config["seed"],multiplier)
    m,v=torch.zeros_like(theta),torch.zeros_like(theta)
    if not (directory/"initial.npz").exists():
        np.savez_compressed(directory/"initial.npz",theta=theta.numpy(),train_x=x.numpy(),train_y=y.numpy(),
                            eval_x=xe.numpy(),eval_y=ye.numpy(),target_rms=normalizer)
    step=0
    history,snapshots=[],[]
    checkpoint=directory/"checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            if data["config_sha256"].item()!=digest: raise RuntimeError("Mismatched checkpoint")
            theta=torch.from_numpy(data["theta"].copy())
            m=torch.from_numpy(data["adam_first"].copy())
            v=torch.from_numpy(data["adam_second"].copy())
            step=int(data["step"])
            keys=data["keys"].tolist()
            history=[dict(zip(keys,row.tolist())) for row in data["history"]]
            snapshots=list(data["snapshots"].copy())
    initial_step=step
    start=time.monotonic()

    def record():
        a=theta[:config["width"]].numpy()
        errors=[float(torch.linalg.vector_norm(predict(theta,grid)-truth)/torch.linalg.vector_norm(truth))
                for grid,truth in ((x,y),(xe,ye))]
        if not np.all(np.isfinite(errors)): return False
        history.append({"step":step,"gamma_rms":float(np.sqrt(np.mean(a*a))),
                        "gamma_median":float(np.median(np.abs(a))),
                        "gamma_p01":float(np.quantile(np.abs(a),.01)),
                        "gamma_p99":float(np.quantile(np.abs(a),.99)),
                        "gamma_max":float(np.max(np.abs(a))),
                        "train_rel_l2":errors[0],"eval_rel_l2":errors[1]})
        snapshots.append(theta.numpy().copy())
        return True

    if not history:
        if not record(): raise RuntimeError("Nonfinite initial evaluation")
    stop=min(endpoint,config["steps"])
    times=sorted({stop,*[s for s in (1,2,5,10,20,50,100,200) if step<s<=stop],
                  *range(((step//config["log_every"])+1)*config["log_every"],stop+1,config["log_every"])})
    finite=True
    adam=config["adam"]
    for until in times:
        if until<=step: continue
        theta,m,v,n,finite=train_chunk(theta,m,v,x,y,float(lr),until-step,step,optimizer=="adam",
                                      float(adam["beta1"]),float(adam["beta2"]),float(adam["epsilon"]))
        step+=n
        if not finite: break
        if not record():
            finite=False
            break
    keys=list(history[0])
    tmp=directory/"checkpoint.tmp.npz"
    # Snapshot history includes only finite evaluated states; exact optimizer state is separate.
    np.savez_compressed(tmp,theta=theta.numpy(),adam_first=m.numpy(),adam_second=v.numpy(),step=step,
                        keys=np.array(keys),history=np.array([[r[k] for k in keys] for r in history]),
                        snapshots=np.array(snapshots),config_sha256=np.array(digest))
    tmp.replace(checkpoint)
    elapsed=time.monotonic()-start
    result={"optimizer":optimizer,"activation":"gelu","target":name,"initialization_multiplier":multiplier,
            "learning_rate":lr,"width":config["width"],"seed":config["seed"],"step":step,
            "total_steps":config["steps"],"last_plottable_step":history[-1]["step"],
            "state":("complete" if step>=config["steps"] else "running") if finite else "nonfinite",
            "gamma_rms":history[-1]["gamma_rms"],"eval_rel_l2":history[-1]["eval_rel_l2"],
            "round_seconds":elapsed,"steps_per_second":(step-initial_step)/elapsed,
            "updated_utc":utc_now(),"config_sha256":digest}
    write_json(status_path,result)
    return result


def summarize(output):
    rows=[json.loads(p.read_text()) for p in output.glob("*/runs/*/status.json")]
    return {optimizer:{"counts":{state:sum(r["optimizer"]==optimizer and r["state"]==state for r in rows)
                                  for state in ("complete","running","nonfinite","error")},
                       "min_logged_step":min([r.get("last_plottable_step",0) for r in rows if r["optimizer"]==optimizer],default=0),
                       "max_step":max([r["step"] for r in rows if r["optimizer"]==optimizer],default=0)}
            for optimizer in ("gd","adam")}


def run(config,output):
    prepare(config,output)
    with ProcessPoolExecutor(max_workers=config["workers"],mp_context=multiprocessing.get_context("spawn")) as pool:
        for endpoint in range(config["round_steps"],config["steps"]+config["round_steps"],config["round_steps"]):
            endpoint=min(endpoint,config["steps"])
            jobs={pool.submit(run_round,case,config,str(output),endpoint):case for case in cases(config)}
            for future in as_completed(jobs):
                case=jobs[future]
                try:
                    result=future.result()
                    print(f"{case} step={result['step']} {result['state']} "
                          f"gamma={result['gamma_rms']:.6g} error={result['eval_rel_l2']:.6g} "
                          f"speed={result['steps_per_second']:.1f}/s",flush=True)
                except Exception as exc:
                    optimizer,name,multiplier,lr=case
                    p=output/optimizer/"runs"/case_id(name,multiplier,lr)/"status.json"
                    p.parent.mkdir(parents=True,exist_ok=True)
                    old=json.loads(p.read_text()) if p.exists() else {"step":0,"target":name,
                        "initialization_multiplier":multiplier,"learning_rate":lr,"width":config["width"]}
                    write_json(p,{**old,"optimizer":optimizer,"state":"error","error":repr(exc),"config_sha256":config_digest(config)})
                    print(f"ERROR {case}: {exc!r}",flush=True)
            summary=summarize(output)
            failed=any(s["counts"]["nonfinite"] or s["counts"]["error"] for s in summary.values())
            state=("finished_with_failures" if failed else "complete") if endpoint==config["steps"] else "running"
            write_json(output/"sweep_status.json",{"state":state,
                       "round_endpoint":endpoint,"updated_utc":utc_now(),"optimizers":summary})
            print(f"ROUND {endpoint}: {json.dumps(summary)}",flush=True)
            if all(s["counts"]["error"]==18 for s in summary.values()):
                raise RuntimeError("All workers failed; stopping rather than repeating an infrastructure failure")
            # Refresh both real-data figures every round, without delaying the next training round.
            subprocess.run([sys.executable,str(HERE/"plot.py"),"--results",str(output)],check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("run","launch","status","pilot"))
    parser.add_argument("--output",type=Path,default=OUT/"sweep")
    args=parser.parse_args()
    config=load_config()
    if args.mode=="status": print(json.dumps(summarize(args.output),indent=2)); return
    if args.mode=="pilot":
        config={**config,"steps":100,"round_steps":100}
        run(config,OUT/"pilot_verified")
        return
    if args.mode=="launch":
        args.output.mkdir(parents=True,exist_ok=True)
        pidfile=args.output/"launcher.json"
        if pidfile.exists():
            try: os.kill(json.loads(pidfile.read_text())["pid"],0)
            except ProcessLookupError: pass
            else: raise RuntimeError("Existing sweep process is still alive")
        with (args.output/"run.log").open("a") as log:
            proc=subprocess.Popen([sys.executable,str(HERE/"run.py"),"run","--output",str(args.output)],
                                  cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(pidfile,{"pid":proc.pid,"launched_utc":utc_now()})
        print(json.dumps({"pid":proc.pid,"output":str(args.output)},indent=2))
    else: run(config,args.output)


if __name__=="__main__": main()
