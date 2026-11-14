"""Shared fixed-geometry QUILL fits for every smooth D56/D58 target."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key,"1")
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from docs.qi_practical_guide.reference import fit_qi, MLP, default_halo
from experiments.expD56_scale_acquisition.model import target as original_target, TARGETS as ORIGINAL, midpoint_grid
from experiments.expD58_extra_hard_scale_acquisition.targets import target as extra_target, TARGETS as EXTRA

HERE=Path(__file__).resolve().parent
OUT=ROOT/"results/checkpoint_D_optimizers/expD59_quill_smooth_scaling"


def load_config(): return yaml.safe_load((HERE/"config.yaml").read_text())


def target(name,x):
    return original_target(name,x) if name in ORIGINAL else extra_target(name,x)


def values(names,x,normalizers):
    return np.column_stack([target(name,x)/normalizers[i] for i,name in enumerate(names)])


def write_json(path,obj):
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def audit_points():
    eps=np.geomspace(1e-12,.1,100)
    edges=np.r_[-1.,1.,-1+eps,1-eps]
    features=np.concatenate([np.linspace(c-.08,c+.08,401) for c in (0.,.15,.2,-.35,.35)])
    return np.unique(np.r_[edges,features])


def score(pred,truth):
    residual=pred-truth
    return np.linalg.norm(residual,axis=0)/np.linalg.norm(truth,axis=0),np.max(np.abs(residual),axis=0)


def write_csv(rows,path):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    config=load_config();names=config["targets"]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"models").mkdir(exist_ok=True)
    source_paths=[Path(__file__),HERE/"config.yaml",ROOT/"docs/qi_practical_guide/reference.py",
                  ROOT/"docs/qi_practical_guide/activations.py",ROOT/"experiments/expD56_scale_acquisition/model.py",
                  ROOT/"experiments/expD58_extra_hard_scale_acquisition/targets.py"]
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    xn=midpoint_grid(config["normalization_points"])
    normalizers=np.array([np.sqrt(np.mean(target(name,xn)**2)) for name in names])
    xeval=midpoint_grid(config["evaluation_points"])
    xr=np.random.default_rng(config["random_seed"]).uniform(-1,1,config["random_audit_points"])
    xp=audit_points()
    truth=values(names,xeval,normalizers);truth_random=values(names,xr,normalizers);truth_probe=values(names,xp,normalizers)
    np.savez_compressed(OUT/"evaluation_data.npz",targets=np.array(names),normalizers=normalizers,
                        x=xeval,y=truth,random_x=xr,random_y=truth_random,probe_x=xp,probe_y=truth_probe)
    metadata={"config":config,"source_sha256":source_hashes,"numpy_version":np.__version__,"scipy_version":scipy.__version__,
              "normalizers":dict(zip(names,normalizers.tolist())),
              "normalization":"Fixed per-target RMS on D56's 1024 midpoints; output centering/scaling inside fit_qi is reversed in the saved model.",
              "width_convention":"N interior intervals + 1 + 2*max(20,ceil(sqrt(N))) halo-inclusive neurons",
              "evaluation":"Actual exported ordinary float64 affine-tanh-affine MLP; no anchored or surrogate evaluation",
              "smooth_bump":"C-infinity but nonanalytic; included by explicit request",
              "gamma_domain":"Gamma argument is in [0.05,5]. Fitting uses only requested-domain labels; no exterior target evaluation."}
    manifest=OUT/"manifest.json"
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous["source_sha256"]!=source_hashes or previous["config"]!=config:
            raise RuntimeError("Existing experiment differs; preserve it and use a new output path")
    else: write_json(manifest,metadata)
    rows=[];started=time.monotonic()
    for n in config["intervals"]:
        t=time.monotonic();m=max(config["train_points_minimum"],config["train_points_per_interior_center"]*(n+1))
        x=np.linspace(-1,1,m);y=values(names,x,normalizers)
        model,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=n,halo=None,lam=config["lambda"],
                          activation="tanh",rcond=config["rcond"],solver=config["solver"])
        fit_seconds=time.monotonic()-t
        path=OUT/"models"/f"N{n:04d}.npz";model.save(path)
        # Score the saved artifact, not merely the in-memory fitted object.
        exported=MLP.load(path)
        pred=exported(xeval)
        relative,maximum=score(pred,truth)
        random_relative,_=score(exported(xr),truth_random)
        _,probe_maximum=score(exported(xp),truth_probe)
        # Nested half-density grids check resolution without another fit.
        rel_a,_=score(pred[::2],truth[::2]);rel_b,_=score(pred[1::2],truth[1::2])
        train_relative,_=score(exported(x),y)
        width=len(model.hidden_bias)
        assert width==n+1+2*default_halo(n)
        assert np.all(np.isfinite(relative)) and np.all(relative>0)
        for i,name in enumerate(names):
            rows.append({"target":name,"intervals":n,"width":width,"halo_per_side":default_halo(n),
                         "gamma":config["lambda"]*n/2,"lambda":config["lambda"],"train_points":m,
                         "relative_l2":float(relative[i]),"sampled_max_absolute_error":float(maximum[i]),
                         "random_relative_l2":float(random_relative[i]),"probe_max_absolute_error":float(probe_maximum[i]),
                         "half_grid_even_relative_l2":float(rel_a[i]),"half_grid_odd_relative_l2":float(rel_b[i]),
                         "train_relative_l2":float(train_relative[i]),"readout_l1":info["readout_l1"][i],
                         "rank":info["rank"],"rcond":config["rcond"],"solver":config["solver"],
                         "shared_fit_seconds":fit_seconds})
        write_json(OUT/"models"/f"N{n:04d}_diagnostics.json",info)
        write_csv(rows,OUT/"scaling.csv")
        write_json(OUT/"status.json",{"state":"running","completed_widths":len(rows)//len(names),
                   "total_widths":len(config["intervals"]),"latest_width":width,"elapsed_seconds":time.monotonic()-started})
        print(f"N={n:4d} W={width:4d} fit={fit_seconds:.2f}s total_width={time.monotonic()-t:.2f}s "
              f"error range={relative.min():.2e}..{relative.max():.2e} "
              f"worst={names[int(np.argmax(relative))]}",flush=True)
    best={name:min((r for r in rows if r["target"]==name),key=lambda r:r["relative_l2"]) for name in names}
    write_json(OUT/"best_observed.json",best)
    write_json(OUT/"status.json",{"state":"complete","targets":len(names),"widths":len(config["intervals"]),
               "fits":len(rows),"elapsed_seconds":time.monotonic()-started})
    print(OUT/"scaling.csv",flush=True)


if __name__=="__main__":run()
