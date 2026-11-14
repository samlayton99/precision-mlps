"""Finite-horizon GD movement bound with a starting-state tanh step-size check.

See loss_descent_bound.md for the mathematical derivation. These calculations
use floating-point starting states, not outward-rounded arithmetic certificates.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_D_optimizers/expD62_joint_training_diagnostics'

def evaluate(theta,x,y,width,eta,horizon):
    a,b,w=np.split(theta[:-1],3)
    residual=np.tanh(x[:,None]*a+b)@w+theta[-1]-y
    E=float(np.sqrt(np.mean(residual**2)))
    D=E*np.sqrt(eta*horizon)
    c=float(np.linalg.norm(w));m=float(np.max(np.abs(w)))
    A=np.sqrt(width+1+2*(c+D)**2)
    radius=D+eta*A*E
    J2=width+1+2*(c+radius)**2
    Esegment=E*(1+eta*J2)
    smoothness=J2+Esegment*(4*(m+radius)+np.sqrt(2))
    return {'initial_residual_rms':E,'initial_loss':.5*E**2,'distance_bound':float(D),
            'eta_L_bound':float(eta*smoothness),'step_condition_passes':bool(eta*smoothness<=1),
            'initial_norm':float(np.linalg.norm(theta[:-1])),
            'final_norm_bound':float(np.linalg.norm(theta[:-1])+D),
            'initial_gamma_rms':float(np.linalg.norm(a)/np.sqrt(width)),
            'final_gamma_rms_bound':float((np.linalg.norm(a)+D)/np.sqrt(width)),
            'step_check_constants':{'A':float(A),'radius':float(radius),'J_squared':float(J2),
                                   'segment_residual_bound':float(Esegment),'L_hat':float(smoothness)}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=30000);a=p.parse_args()
    rows=[]
    for path in sorted(OUT.glob('width*/runs/*/checkpoint.npz')):
        meta=json.loads((path.parent/'status.json').read_text());W=meta['width'];eta=meta['learning_rate']
        with np.load(path) as z,np.load(path.parent/'initial.npz') as ini:
            steps=z['history'][:,z['keys'].tolist().index('step')].astype(int)
            j=int(np.flatnonzero(steps==a.start)[0]);snap=z['snapshots'][j:];x=ini['train_x'];y=ini['train_y']
            assert np.max(np.abs(x))<=1
            horizon=int(steps[-1]-a.start);result=evaluate(snap[0],x,y,W,eta,horizon)
            elapsed=steps[j:]-a.start
            bounds=result['initial_residual_rms']*np.sqrt(eta*elapsed)
            distances=np.linalg.norm(snap-snap[0],axis=1)
            observed_gamma=np.linalg.norm(snap[:,:W],axis=1)/np.sqrt(W)
            slope_bounds=result['initial_gamma_rms']+bounds/np.sqrt(W)
            violations=int(np.sum(distances>bounds+1e-10))
            if result['step_condition_passes']:
                assert violations==0
                assert np.all(observed_gamma<=slope_bounds+1e-10)
            result.update(width=W,target=meta['target'],learning_rate=eta,start=a.start,horizon=horizon,
                          observed_distance_final=float(distances[-1]),observed_gamma_rms_final=float(observed_gamma[-1]),
                          recorded_distance_bound_violations=violations)
            rows.append(result)
    assert len(rows)==24
    report={'description':'Finite-horizon movement bound from loss descent, with a sufficient step-size condition using only the starting state and horizon.',
            'qualification':'Numerical evaluation of analytic inequalities in float64, not an interval-arithmetic certificate.',
            'start':a.start,'cases':rows}
    path=OUT/f'loss_descent_bound_start{a.start}.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print('Step condition passes:',sum(r['step_condition_passes'] for r in rows),'of 24')
    print('Maximum eta*L_hat:',max(r['eta_L_bound'] for r in rows))
    for r in rows:
        if r['width']==196 and r['target']=='sine_4pi' and r['learning_rate']==.001:print(json.dumps(r,indent=2))
    print(path)

if __name__=='__main__':main()
