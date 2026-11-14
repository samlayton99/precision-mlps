"""Evaluate Theorem F.1's slope-growth recurrence on saved D56/D58 runs.

No networks are trained. q = B/sqrt(W) is the theorem's RMS-slope ceiling.
All four force alternatives from F.4 are retained; the minimum is used at
every actual GD update. The ceiling is never reset to measured parameters.

Three allowance policies:
  interpolated: chi and tracking linearly interpolated between checkpoints;
                E0 is the largest saved non-affine residual over the suffix.
  checkpoint_max: constant maxima of the saved chi/tracking/residual records.
  optimistic: chi=1, R=0, E0=the starting non-affine residual. This is a LOWER
              envelope of any admissible F.6 bound, NOT a bound on training.
              If even this recurrence crosses the reference, any admissible
              theorem recurrence must cross by then. No interpolation needed.

Saved-checkpoint interpolation/maxima do not certify the assumptions between
checkpoints. Floating-point evaluation is not outward-rounded certification.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCES=[ROOT/"results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep",
         ROOT/"results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition/sweep"]
OUT=SOURCES[1].parent/"combined_24_functions/slope_growth_audit"
MOMENTS=(4,6,10,14)


def constants(x):
    u2=x*x-np.mean(x*x)
    u3=x**3-np.mean(x**4)/np.mean(x*x)*x
    s2,s3=np.mean(u2*u2),np.mean(u3*u3)
    alpha={3:1/3,5:2/15,7:17/315}
    a={p:alpha[p]*2**(p/2)*p**(p/2)/(p+1)**((p+1)/2) for p in alpha}
    c={p:(p+1)*a[p] for p in a}
    return {"A3":a[3],"A5":a[5],"A7":a[7],"C3":c[3],"C5":c[5],"C7":c[7],
            "B3":math.sqrt(s2/64+3*s3/256),"D3":math.sqrt(8*s2/27+3*s3/16)}


def force_terms(q,width,chi,e0,tau3,tau5,k):
    """Return the four F.4 force allowances divided by sqrt(W)."""
    chi4,chi6,chi10,chi14=chi
    q2=q*q;q3=q2*q;q4=q2*q2;q5=q3*q2;q6=q3*q3;q7=q5*q2
    output=width*min(k["A3"]*chi4*q4,k["B3"]*chi4*q4+k["A5"]*chi6*q6)
    cubic=math.sqrt(chi6)*q3
    fifth=math.sqrt(chi10)*q5
    seventh=math.sqrt(chi14)*q7
    generic=k["C3"]*e0*cubic
    projected=e0*(k["D3"]*cubic+k["C5"]*fifth)
    target3=k["D3"]*cubic*(tau3+output)+k["C5"]*e0*fifth
    target5=k["D3"]*cubic*tau3+k["C5"]*fifth*tau5+(k["D3"]*cubic+k["C5"]*fifth)*output+k["C7"]*e0*seventh
    return generic,projected,target3,target5


def integrate(history,status,start_step,policy,k,reference=18.375):
    steps=history["step"].astype(int)
    indices=np.flatnonzero(steps==start_step)
    if len(indices)!=1: raise ValueError(f"Missing restart {start_step}")
    j=int(indices[0]);width=status["width"];lr=status["learning_rate"]
    chi=np.column_stack([history[f"chi_{p}"] for p in MOMENTS])
    tracking=history["tracking_norm"]
    residual=history["nonaffine_residual_rms"]
    if not (np.isfinite(chi[j:]).all() and np.isfinite(tracking[j:]).all() and np.isfinite(residual[j:]).all()):
        raise ValueError("Nonfinite premise diagnostics")
    if np.any(chi[j:]<1-1e-10) or np.any(tracking[j:]<0): raise ValueError("Invalid structural records")
    q=math.sqrt(history["parameter_energy"][j]/width)
    q0=q;actual0=float(history["gamma_rms"][j])
    tau3=float(history["tau_3"][j]);tau5=float(history["tau_5"][j])
    e0=float(residual[j] if policy=="optimistic" else np.max(residual[j:]))
    max_chi=np.max(chi[j:],axis=0)
    max_tracking=float(np.max(tracking[j:]))
    crossing=None
    double_start=0 if q>=2*actual0 else None
    choices=[0]*4
    trace=[(start_step,q,actual0)]
    violations=0
    norm_violations=0
    for segment in range(j,len(steps)-1):
        n0,n1=int(steps[segment]),int(steps[segment+1]);dt=n1-n0
        if policy=="interpolated":
            cs=chi[segment];dc=(chi[segment+1]-cs)/dt
            rs=float(tracking[segment]);dr=float(tracking[segment+1]-tracking[segment])/dt
        elif policy=="checkpoint_max":
            cs=max_chi;dc=np.zeros(4);rs=max_tracking;dr=0.
        elif policy=="optimistic":
            cs=np.ones(4);dc=np.zeros(4);rs=0.;dr=0.
        else: raise ValueError(policy)
        for offset in range(dt):
            current_chi=cs+offset*dc
            r=rs+offset*dr
            terms=force_terms(q,width,current_chi,e0,tau3,tau5,k)
            which=min(range(4),key=terms.__getitem__)
            choices[which]+=1
            q+=lr*(terms[which]+r/math.sqrt(width))
            now=n0+offset+1
            if double_start is None and q>=2*actual0: double_start=now-start_step
            if q>=reference:
                crossing=now-start_step
                actual=float(np.interp(now,steps,history["gamma_rms"]))
                trace.append((now,q,actual))
                break
            if now-start_step<=100 or (now-start_step<1000 and (now-start_step)%10==0):
                actual=float(np.interp(now,steps,history["gamma_rms"]))
                trace.append((now,q,actual))
        if crossing is not None: break
        actual=float(history["gamma_rms"][segment+1])
        if policy!="optimistic":
            violations+=int(q+1e-10<actual)
            norm_violations+=int(q+1e-10<math.sqrt(history["parameter_energy"][segment+1]/width))
        if trace[-1][0]!=n1:trace.append((n1,q,actual))
    useful=crossing is None
    actual_end=float(history["gamma_rms"][-1])
    return {"start_step":start_step,"policy":policy,"initial_ceiling":q0,"initial_rms_slope":actual0,
            "E0":e0,"reference":reference,"steps_to_reference":crossing,
            "absolute_reference_crossing":None if crossing is None else start_step+crossing,
            "steps_to_twice_starting_rms":double_start,"below_reference_through_end":useful,
            "end_ceiling":q if useful else None,"actual_end_rms":actual_end,
            "actual_max_rms_suffix":float(np.max(history["gamma_rms"][j:])),
            "end_ceiling_over_actual":q/actual_end if useful else None,
            "allowance_choices":choices,"recorded_slope_violations":violations,
            "recorded_parameter_norm_violations":norm_violations},trace


def load_runs():
    records=[]
    for root in SOURCES:
        for path in sorted((root/"runs").glob("*/checkpoint.npz")):
            status=json.loads((path.parent/"status.json").read_text())
            assert status["state"]=="complete" and status["step"]==200000
            with np.load(path) as data:
                h={key:data["history"][:,i] for i,key in enumerate(data["keys"].tolist())}
            with np.load(path.parent/"initial.npz") as data:x=data["train_x"]
            records.append((path,status,h,constants(x)))
    assert len(records)==216
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument("--starts",type=int,nargs="+",default=[0,25000,125000])
    p.add_argument("--policies",nargs="+",default=["optimistic","interpolated","checkpoint_max"])
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();records=load_runs();results=[];traces={};validation=[]
    for index,(path,status,h,k) in enumerate(records):
        # Independent check that F.4 really upper-bounds the recorded fine gradient
        # when evaluated at the measured combined norm and residual.
        width=status["width"]
        for j in range(len(h["step"])):
            q=math.sqrt(h["parameter_energy"][j]/width)
            chi=[h[f"chi_{p}"][j] for p in MOMENTS]
            force=min(force_terms(q,width,chi,h["nonaffine_residual_rms"][j],h["tau_3"][j],h["tau_5"][j],k))*math.sqrt(width)
            if force+1e-10<h["fine_norm"][j]:raise AssertionError((path,j,force,h["fine_norm"][j]))
        validation.append({"case":path.parent.name,"checkpoints":len(h["step"]),
                           "min_coarse_eigenvalue":float(np.min(h["coarse_kernel_min_eigenvalue"]))})
        for restart in args.starts:
            for policy in args.policies:
                result,trace=integrate(h,status,restart,policy,k)
                result.update(target=status["target"],initialization_multiplier=status["initialization_multiplier"],
                              learning_rate=status["learning_rate"],case=path.parent.name)
                results.append(result)
                traces[f"{path.parent.name}/{restart}/{policy}"]=trace
        if (index+1)%24==0: print(f"Audited {index+1}/216 runs; elapsed {time.monotonic()-started:.1f}s",flush=True)
    (OUT/"runs.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    (OUT/"traces.json").write_text(json.dumps(traces,allow_nan=False)+"\n")
    (OUT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    summary=[]
    for restart in args.starts:
        for policy in args.policies:
            for scale in (1.,2.,4.):
                for lr in (.01,.001,.0001):
                    group=[r for r in results if r["start_step"]==restart and r["policy"]==policy
                           and r["initialization_multiplier"]==scale and r["learning_rate"]==lr]
                    crossed=[r["steps_to_reference"] for r in group if r["steps_to_reference"] is not None]
                    survivors=[r for r in group if r["below_reference_through_end"]]
                    summary.append({"start_step":restart,"policy":policy,"initialization_multiplier":scale,
                        "learning_rate":lr,"runs":len(group),"below_reference_through_end":len(survivors),
                        "min_steps_to_reference":min(crossed) if crossed else None,
                        "median_steps_to_reference":float(np.median(crossed)) if crossed else None,
                        "max_steps_to_reference":max(crossed) if crossed else None,
                        "survivor_max_end_ceiling":max([r["end_ceiling"] for r in survivors],default=None)})
    with (OUT/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    metadata={"source":"Theorem F.1 / equations F.4-F.7 in the supplied ICLR manuscript",
        "runs":216,"restarts":args.starts,"policies":args.policies,"constants":records[0][3],
        "analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_checkpoints_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path,_,_,_ in records},
        "elapsed_seconds":time.monotonic()-started,
        "units":"q=B/sqrt(W), an upper ceiling on RMS slope, not median or maximum individual slope",
        "reference":"18.375 is a comparison scale, not a necessary accuracy threshold",
        "limits":"Interpolated/checkpoint-max policies do not certify inter-checkpoint premises. Optimistic policy is a lower envelope of theorem ceilings, never asserted to bound actual slopes. No training was performed."}
    (OUT/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    for restart in args.starts:
        for policy in args.policies:
            group=[r for r in results if r["start_step"]==restart and r["policy"]==policy]
            survived=[r for r in group if r["below_reference_through_end"]]
            crossed=[r["steps_to_reference"] for r in group if r["steps_to_reference"] is not None]
            print(restart,policy,"below reference",len(survived),"/",len(group),
                  "crossing steps min/median/max",(min(crossed),float(np.median(crossed)),max(crossed)) if crossed else None,flush=True)
    print(OUT,flush=True)


if __name__=="__main__":main()
