"""Score each saved readout as a scalar ordinary MLP, reusing only its features."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key,"1")
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from docs.qi_practical_guide.reference import MLP
from experiments.expD59_quill_smooth_scaling.run import OUT,score,write_csv,write_json


def evaluate_scalar_columns(model,x,chunk=512):
    x=np.asarray(x,dtype=np.float64).reshape(-1,1)
    output=np.empty((len(x),model.readout.shape[1]))
    for begin in range(0,len(x),chunk):
        features=np.tanh(x[begin:begin+chunk]@model.weight.T+model.hidden_bias)
        for i in range(model.readout.shape[1]):
            output[begin:begin+chunk,i]=(features@model.readout[:,i:i+1]+model.output_bias[i]).ravel()
    return output


def main():
    with (OUT/"scaling.csv").open() as f:rows=list(csv.DictReader(f))
    with np.load(OUT/"evaluation_data.npz") as d:
        x,y,xr,yr,xp,yp=(d[k] for k in ("x","y","random_x","random_y","probe_x","probe_y"))
        names=d["targets"].tolist()
    new=[]
    for n in sorted({int(r["intervals"]) for r in rows}):
        model=MLP.load(OUT/"models"/f"N{n:04d}.npz")
        pred=evaluate_scalar_columns(model,x)
        relative,maximum=score(pred,y)
        rand,_=score(evaluate_scalar_columns(model,xr),yr)
        _,probes=score(evaluate_scalar_columns(model,xp),yp)
        even,_=score(pred[::2],y[::2]);odd,_=score(pred[1::2],y[1::2])
        records={r["target"]:r for r in rows if int(r["intervals"])==n}
        for i,name in enumerate(names):
            row=records[name]
            row["shared_output_relative_l2"]=float(row["relative_l2"])
            row.update(relative_l2=float(relative[i]),sampled_max_absolute_error=float(maximum[i]),
                       random_relative_l2=float(rand[i]),probe_max_absolute_error=float(probes[i]),
                       half_grid_even_relative_l2=float(even[i]),half_grid_odd_relative_l2=float(odd[i]))
            new.append(row)
        print(f"Scalar evaluation N={n}, max relative L2={relative.max():.3e}",flush=True)
    write_csv(new,OUT/"scaling_scalar.csv")
    best={name:min((r for r in new if r["target"]==name),key=lambda r:float(r["relative_l2"])) for name in names}
    write_json(OUT/"best_scalar_observed.json",best)
    write_json(OUT/"scalar_audit_provenance.json",{
        "source_sha256":hashlib.sha256((OUT/"scaling.csv").read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation":"For each target, an independent feature-matrix times single-column readout; same operation as scalar MLP.__call__. Features are cached across functions only.",
        "fits_repeated":False,"rows":len(new)})


if __name__=="__main__":main()
