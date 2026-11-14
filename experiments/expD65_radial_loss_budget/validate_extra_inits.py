"""Test the already calibrated threshold on 2x/4x saved initializations, without retuning."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import sys,json,math,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.audit import diagnostic,OUT,SOURCES,RESTART
from experiments.expD62_joint_training_diagnostics.loss_descent_bound import evaluate

def main():
    K=json.loads((OUT/'results.json').read_text())['K_A'];rows=[];arrays={};start=time.monotonic()
    for root in SOURCES:
        for path in sorted(root.glob('*/checkpoint.npz')):
            status=json.loads((path.parent/'status.json').read_text());lr=status['learning_rate'];mult=status['initialization_multiplier']
            if lr not in [.001,.0001] or mult not in [2.,4.]:continue
            with np.load(path.parent/'initial.npz') as d:x=d['train_x'];y=d['train_y']
            with np.load(path) as d:
                steps=d['history'][:,d['keys'].tolist().index('step')];snap=d['snapshots'].copy()
            keep=steps>=RESTART;steps=steps[keep];snap=snap[keep]
            vals=np.array([diagnostic(theta,x,y,lr) for theta in snap]);E0,A0=vals[0,:2]
            stepcheck=evaluate(snap[0],x,y,196,lr,200000-RESTART)
            cap=math.sqrt((A0+4*K*E0+lr*E0**2)/196)
            actual=np.sqrt(vals[:,1]/196);budget=np.sqrt(np.maximum(0,(A0+4*K*(E0-vals[:,0])+lr*(E0**2-vals[:,0]**2))/196))
            row={'case':path.parent.name,'target':status['target'],'lr':lr,'initialization':mult,
                 'K_max':float(vals[:,4].max()),'K_condition_pass':bool(vals[:,4].max()<=K),
                 'pointwise_descent_check_pass':bool(vals[:,6].max()<=1),
                 'starting_state_horizon_descent_check_pass':stepcheck['step_condition_passes'],
                 'starting_state_eta_L_hat':stepcheck['eta_L_bound'],
                 'prospective_slope_cap':cap,'maximum_actual_slope_rms':float(actual.max()),
                 'budget_curve_violation':float((actual-budget).max()),'restart_E':float(E0),
                 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(row);arrays[path.parent.name]=np.column_stack([steps,vals])
            if len(rows)%24==0:print('Additional cases',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
    assert len(rows)==96
    report={'K_fixed_from_default_calibration':K,'no_retuning':True,'cases':rows}
    (OUT/'extra_initializations.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'extra_diagnostics.npz',**arrays)
    for mult in [2.,4.]:
        selected=[r for r in rows if r['initialization']==mult]
        print(mult,'passes',sum(r['K_condition_pass'] for r in selected),'/',len(selected),'K max',max(r['K_max'] for r in selected),'cap max',max(r['prospective_slope_cap'] for r in selected),
              'starting descent passes',sum(r['starting_state_horizon_descent_check_pass'] for r in selected),'pointwise descent passes',sum(r['pointwise_descent_check_pass'] for r in selected),flush=True)
    for row in rows:
        if not row['K_condition_pass']:print('FAILED_K',row,flush=True)
if __name__=='__main__':main()
