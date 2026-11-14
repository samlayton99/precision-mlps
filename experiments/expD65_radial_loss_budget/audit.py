"""Read-only calibration/held-out audit of a radial-loss-budget condition for native GD."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import json,sys,math,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/checkpoint_D_optimizers/expD65_radial_loss_budget'
SOURCES=[ROOT/'results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep/runs',ROOT/'results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition/sweep/runs']
CAL_START=20000;RESTART=30000;FACTOR=2.

def diagnostic(theta,x,y,lr):
    a,b,c=np.split(theta[:-1],3);W=len(a);h=np.tanh(x[:,None]*a+b);e=h@c+theta[-1]-y
    sens=(1-h*h)*c
    ga=sens.T@(e*x)/len(x);gb=sens.T@e/len(x);gc=h.T@e/len(x);gd=e.mean();g=np.r_[ga,gb,gc,gd]
    g2=float(g@g);E=float(np.sqrt(np.mean(e*e)));A=float(a@a);M=float(theta[:-1]@theta[:-1])
    kA=E*max(0.,-float(a@ga))/g2 if g2 else 0.
    kM=E*max(0.,-float(theta[:-1]@g[:-1]))/g2 if g2 else 0.
    # Loss Hessian bound over the actual proposed GD segment; no loss subtraction.
    cpath=float(np.linalg.norm(c))+lr*float(np.linalg.norm(gc));mpath=float(np.max(np.abs(c)))+lr*float(np.max(np.abs(gc)))
    J2=W+1+2*cpath*cpath;Epath=E+math.sqrt(J2)*lr*math.sqrt(g2)
    smoothness=J2+Epath*(4*mpath+math.sqrt(2))
    return [E,A,M,g2,kA,kM,lr*smoothness]

def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];start=time.monotonic()
    for root in SOURCES:
        for path in sorted(root.glob('*__init1__lr*/checkpoint.npz')):
            status=json.loads((path.parent/'status.json').read_text());lr=status['learning_rate']
            if lr not in [.001,.0001]:continue
            with np.load(path.parent/'initial.npz') as d:x=d['train_x'];y=d['train_y']
            with np.load(path) as d:
                steps=d['history'][:,d['keys'].tolist().index('step')];snap=d['snapshots'].copy()
            keep=steps>=CAL_START;steps=steps[keep];snap=snap[keep]
            vals=np.array([diagnostic(theta,x,y,lr) for theta in snap])
            records.append({'name':path.parent.name,'target':status['target'],'lr':lr,'steps':steps,'values':vals,
                            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            if len(records)%12==0:print('Audited',len(records),'runs in',round(time.monotonic()-start,1),'seconds',flush=True)
    assert len(records)==48
    K_A=FACTOR*max(float(r['values'][r['steps']<=RESTART,4].max()) for r in records)
    K_M=FACTOR*max(float(r['values'][r['steps']<=RESTART,5].max()) for r in records)
    summary=[];arrays={}
    for r in records:
        s=r['steps'];v=r['values'];i=int(np.flatnonzero(s==RESTART)[0]);E0,A0,M0=v[i,:3];post=s>RESTART
        KpostA=float(v[post,4].max());KpostM=float(v[post,5].max());lr=r['lr']
        slope_cap=math.sqrt((A0+4*K_A*E0+lr*E0*E0)/196)
        norm_cap=math.sqrt(M0+4*K_M*E0+lr*E0*E0)
        slope_curve=np.sqrt(np.maximum(0,(A0+4*K_A*(E0-v[i:,0])+lr*(E0*E0-v[i:,0]**2))/196))
        measured=np.sqrt(v[i:,1]/196)
        row={'case':r['name'],'target':r['target'],'lr':lr,'K_calibration_A':float(v[s<=RESTART,4].max()),
             'K_heldout_A':KpostA,'K_heldout_M':KpostM,'A_condition_pass':KpostA<=K_A,'M_condition_pass':KpostM<=K_M,
             'maximum_eta_hessian_bound':float(v[:,6].max()),'sufficient_decrease_check_pass':bool(np.max(v[:,6])<=1),
             'restart_E':float(E0),'restart_slope_rms':math.sqrt(A0/196),'final_slope_rms':float(measured[-1]),
             'prospective_slope_cap':slope_cap,'prospective_parameter_norm_cap':norm_cap,
             'maximum_actual_slope_rms':float(measured.max()),'budget_curve_violation':float(np.max(measured-slope_curve)),
             'source_sha256':r['source_sha256']}
        summary.append(row);arrays[r['name']]=np.column_stack([s,v])
    report={'scope':'48 default-initialization GD runs, 24 functions x rates .001/.0001, width 196. No training.',
            'calibration_steps':[CAL_START,RESTART],'heldout_steps':[RESTART+500,200000],'margin':FACTOR,
            'exploratory_note':'The normalized radial condition was developed after inspecting a few saved states; this is temporal holdout evidence, not preregistered independent confirmation.',
            'K_A':K_A,'K_M':K_M,'A_condition_pass_count':sum(r['A_condition_pass'] for r in summary),
            'M_condition_pass_count':sum(r['M_condition_pass'] for r in summary),
            'descent_check_pass_count':sum(r['sufficient_decrease_check_pass'] for r in summary),
            'records_per_run':len(records[0]['steps']),'columns':['step','E','slope_energy','hidden_energy','gradient_squared','K_A','K_M','eta_hessian_bound'],
            'cases':summary}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'diagnostics.npz',**arrays)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    for r in sorted(summary,key=lambda r:r['K_heldout_A'],reverse=True)[:8]:print(r,flush=True)
if __name__=='__main__':main()
