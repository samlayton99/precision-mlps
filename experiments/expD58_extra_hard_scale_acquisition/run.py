"""Extra-hard tanh-GD runs; reuses D56's tested update and diagnostics."""
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
import scipy
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.model import gd_chunk,initial_parameters,midpoint_grid,diagnostics,prediction
from experiments.expD56_scale_acquisition.run import utc_now,write_json,config_digest,case_id,save_checkpoint
from experiments.expD58_extra_hard_scale_acquisition.targets import TARGETS,target,special_probe_points

HERE=Path(__file__).resolve().parent
OUT=ROOT/"results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition"


def load_config(): return yaml.safe_load((HERE/"config.yaml").read_text())


def cases(config):
    names=[n for group in config["targets"].values() for n in group]
    assert len(names)==8 and set(names)==set(TARGETS) and config["width"]==196
    return [(name,multiplier,lr) for multiplier in config["initialization_multipliers"]
            for name in names for lr in config["learning_rates"]]


def prepare(config,output):
    output.mkdir(parents=True,exist_ok=True)
    sources=[HERE/"run.py",HERE/"targets.py",ROOT/"experiments/expD56_scale_acquisition/model.py",
             ROOT/"experiments/expD56_scale_acquisition/run.py"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    metadata={"config":config,"config_sha256":config_digest(config),"created_utc":utc_now(),
              "training_sources":hashes,"targets":TARGETS,"run_count":len(cases(config)),
              "torch_version":torch.__version__,"scipy_version":scipy.__version__,
              "scope":"Exploratory extra-hard label, not verified optimization difficulty. Nonsmooth targets are outside the analytic construction theorem; empirical scale acquisition is the question.",
              "jump_error_note":"Continuous tanh networks cannot uniformly approximate a jump below half its jump height. Reported independent-grid RMS and jump-probe maximum errors are distinct metrics."}
    path=output/"manifest.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["config_sha256"]!=metadata["config_sha256"] or old["training_sources"]!=hashes:
            raise RuntimeError("Configuration or training source mismatch")
    else:
        write_json(path,metadata)
        for p in [*sources,HERE/"config.yaml"]:
            q=output/"provenance"/p.relative_to(ROOT)
            q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)


def final_audit(theta,name,config,normalizer,directory):
    x=midpoint_grid(config["dense_eval_points"])
    y=target(name,x)/normalizer
    # Bound temporary matrix memory while retaining the independent dense grid.
    pred=np.concatenate([prediction(theta,chunk) for chunk in np.array_split(x,16)])
    probes=special_probe_points(); truth=target(name,probes)/normalizer
    probe_pred=prediction(theta,probes)
    np.savez_compressed(directory/"final_dense_audit.npz",x=x,y=y,prediction=pred,
                        probe_x=probes,probe_y=truth,probe_prediction=probe_pred)
    write_json(directory/"final_dense_audit.json",{
        "points":len(x),"relative_l2":float(np.linalg.norm(pred-y)/np.linalg.norm(y)),
        "sampled_max_absolute_error":float(np.max(np.abs(pred-y))),
        "special_probe_max_absolute_error":float(np.max(np.abs(probe_pred-truth))),
        "normalizer":normalizer,"uniform_error_certificate":False})


def run_round(case,config,output,endpoint):
    torch.set_num_threads(1)
    name,multiplier,lr=case
    directory=Path(output)/"runs"/case_id(*case)
    directory.mkdir(parents=True,exist_ok=True)
    digest=config_digest(config); status_path=directory/"status.json"
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old["config_sha256"]!=digest: raise RuntimeError("Run configuration mismatch")
        if old["state"] in ("complete","nonfinite","error"): return old
    x,xe=midpoint_grid(config["train_points"]),midpoint_grid(config["eval_points"])
    raw=target(name,x); normalizer=float(np.sqrt(np.mean(raw*raw)))
    y,ye=raw/normalizer,target(name,xe)/normalizer
    theta=initial_parameters(config["width"],config["seed"],multiplier)
    if not (directory/"initial.npz").exists():
        np.savez_compressed(directory/"initial.npz",theta=theta.numpy(),train_x=x,train_y=y,
                            eval_x=xe,eval_y=ye,target_rms=normalizer)
    history,snapshots,step=[],[],0
    checkpoint=directory/"checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as d:
            if d["config_sha256"].item()!=digest: raise RuntimeError("Checkpoint configuration mismatch")
            theta=torch.from_numpy(d["theta"].copy()); step=int(d["step"])
            history=[dict(zip(d["keys"].tolist(),row.tolist())) for row in d["history"]]
            snapshots=list(d["snapshots"].copy())
    start=time.monotonic(); start_step=step
    def record():
        history.append({"step":step,**diagnostics(theta.numpy(),x,y,xe,ye,2/config["quill_reference"]["interior_intervals"])})
        snapshots.append(theta.numpy().copy())
    if not history: record()
    stop=min(endpoint,config["steps"])
    times=sorted({stop,*[s for s in (1,2,5,10,20,50,100,200) if step<s<=stop],
                  *range((step//config["log_every"]+1)*config["log_every"],stop+1,config["log_every"])})
    finite=True; tx,ty=torch.from_numpy(x),torch.from_numpy(y)
    for until in times:
        if until<=step: continue
        theta,n,finite=gd_chunk(theta,tx,ty,float(lr),until-step)
        step+=n
        if not finite: break
        record()
    save_checkpoint(directory,theta,history,snapshots,step,digest)
    if finite and step==config["steps"]:
        final_audit(theta.numpy(),name,config,normalizer,directory)
    elapsed=time.monotonic()-start
    result={"target":name,"difficulty":"extra_hard","optimizer":"gd","activation":"tanh",
            "initialization_multiplier":multiplier,"learning_rate":lr,"width":config["width"],"seed":config["seed"],
            "step":step,"total_steps":config["steps"],"last_plottable_step":history[-1]["step"],
            "state":("complete" if step==config["steps"] else "running") if finite else "nonfinite",
            "gamma_rms":history[-1]["gamma_rms"],"eval_rel_l2":history[-1]["eval_rel_l2"],
            "config_sha256":digest,"updated_utc":utc_now(),"steps_per_second":(step-start_step)/elapsed}
    write_json(status_path,result)
    return result


def summary(output):
    rows=[json.loads(p.read_text()) for p in output.glob("runs/*/status.json")]
    return {"started":len(rows),"counts":{s:sum(r["state"]==s for r in rows) for s in ("running","complete","nonfinite","error")},
            "min_logged_step":min([r.get("last_plottable_step",0) for r in rows],default=0),
            "max_step":max([r["step"] for r in rows],default=0)}


def run(config,output):
    prepare(config,output)
    write_json(output/"sweep_status.json",{"state":"running","updated_utc":utc_now(),**summary(output)})
    with ProcessPoolExecutor(max_workers=config["workers"],mp_context=multiprocessing.get_context("spawn")) as pool:
        for endpoint in range(config["round_steps"],config["steps"]+config["round_steps"],config["round_steps"]):
            endpoint=min(endpoint,config["steps"])
            futures={pool.submit(run_round,case,config,str(output),endpoint):case for case in cases(config)}
            for future in as_completed(futures):
                case=futures[future]
                try:
                    r=future.result()
                    print(f"{case} step={r['step']} {r['state']} gamma={r['gamma_rms']:.5g} "
                          f"error={r['eval_rel_l2']:.5g} speed={r['steps_per_second']:.1f}/s",flush=True)
                except Exception as exc:
                    folder=output/"runs"/case_id(*case); folder.mkdir(parents=True,exist_ok=True)
                    path=folder/"status.json"
                    old=json.loads(path.read_text()) if path.exists() else {"step":0,"width":config["width"],
                        "target":case[0],"initialization_multiplier":case[1],"learning_rate":case[2]}
                    write_json(path,{**old,"state":"error","error":repr(exc),"config_sha256":config_digest(config)})
                    print(f"ERROR {case}: {exc!r}",flush=True)
            s=summary(output); failed=s["counts"]["nonfinite"]+s["counts"]["error"]
            state=("finished_with_failures" if failed else "complete") if endpoint==config["steps"] else "running"
            write_json(output/"sweep_status.json",{"state":state,"round_endpoint":endpoint,"updated_utc":utc_now(),**s})
            if s["counts"]["error"]==len(cases(config)): raise RuntimeError("All workers failed")
            subprocess.run([sys.executable,str(HERE/"plot.py"),"--results",str(output)],check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("pilot","launch","run","status"))
    p.add_argument("--output",type=Path,default=OUT/"sweep"); a=p.parse_args(); config=load_config()
    if a.mode=="status": print(json.dumps(summary(a.output),indent=2)); return
    if a.mode=="pilot": run({**config,"steps":100,"round_steps":100},OUT/"pilot"); return
    if a.mode=="launch":
        a.output.mkdir(parents=True,exist_ok=True); pidfile=a.output/"launcher.json"
        if pidfile.exists():
            try: os.kill(json.loads(pidfile.read_text())["pid"],0)
            except ProcessLookupError: pass
            else: raise RuntimeError("Sweep coordinator is already running")
        with (a.output/"run.log").open("a") as log:
            proc=subprocess.Popen([sys.executable,str(HERE/"run.py"),"run","--output",str(a.output)],cwd=ROOT,
                 stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(pidfile,{"pid":proc.pid,"launched_utc":utc_now()})
        print(json.dumps({"pid":proc.pid,"output":str(a.output)},indent=2))
    else: run(config,a.output)


if __name__=="__main__": main()
