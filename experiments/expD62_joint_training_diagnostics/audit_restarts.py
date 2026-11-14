"""Read-only GD restart audit: reconstruct B0 from saved parameters, then evaluate F.6."""
import json,sys,math,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD62_joint_training_diagnostics import plot
OUT=plot.OUT

def main():
    rows=[]
    for path in sorted(OUT.glob('width*/runs/*/checkpoint.npz')):
        status=json.loads((path.parent/'status.json').read_text());w=status['width'];lr=status['learning_rate']
        with np.load(path) as d:
            h={key:d['history'][:,i] for i,key in enumerate(d['keys'].tolist())};snap=d['snapshots'].copy()
        with np.load(path.parent/'initial.npz') as d:k=plot.constants(d['train_x'])
        for restart in [0,1000,30000]:
            i=int(np.flatnonzero(h['step']==restart)[0]);theta=snap[i];parts=[float(np.dot(v,v)) for v in np.split(theta[:-1],3)]
            b=math.sqrt(sum(parts));np.testing.assert_allclose(b*b,h['parameter_energy'][i],rtol=1e-14)
            plot.RESTART=restart;result=plot.ceiling(h,w,lr,k)
            assert abs(b-result['initial_B'])<1e-12
            E=result['E0'];chi=[h[f'chi_{p}'][i] for p in [4,6,10,14]]
            forces=np.array(plot.force_terms(b/math.sqrt(w),w,chi,E,h['tau_3'][i],h['tau_5'][i],k))*math.sqrt(w)
            # Independent physical-B evaluation at restart, not normalized-q code.
            S={p:chi[j]*b**p/w**(p/2-1) for j,p in enumerate([4,6,10,14])}
            Q=min(k['A3']*S[4],k['B3']*S[4]+k['A5']*S[6]);r6,r10,r14=[math.sqrt(S[p]) for p in [6,10,14]]
            direct=[k['C3']*E*r6,E*(k['D3']*r6+k['C5']*r10),
                    k['D3']*r6*(h['tau_3'][i]+Q)+k['C5']*E*r10,
                    k['D3']*r6*h['tau_3'][i]+k['C5']*r10*h['tau_5'][i]+(k['D3']*r6+k['C5']*r10)*Q+k['C7']*E*r14]
            np.testing.assert_allclose(forces,direct,rtol=2e-14)
            row=dict(width=w,target=status['target'],lr=lr,restart=restart,B0=b,energy=b*b,
                slope_energy=parts[0],bias_energy=parts[1],readout_energy=parts[2],gamma_rms=math.sqrt(parts[0]/w),
                initial_slope_ceiling=b/math.sqrt(w),E0=E,chi6=chi[1],R=h['tracking_norm'][i],
                force_min=float(min(forces)),force_choice=int(np.argmin(forces)),
                initial_ceiling_increment=lr*(min(forces)+h['tracking_norm'][i]),
                updates_to_B1000=result['cap_crossing_step']-restart if result['cap_crossing_step'] else None,
                force_choices=result['force_choices'],actual_norm_final=math.sqrt(h['parameter_energy'][-1]))
            rows.append(row)
    report={'runs':24,'restarts':[0,1000,30000],'B0_definition':'sqrt(sum_j(gamma_j^2+beta_j^2+w_j^2)) at the chosen restart; output bias excluded',
            'checks':'All 72 restart energies match saved raw parameters; direct physical-B forces match normalized implementation.',
            'allowances':'Same existing retrospective interpolation; E0=max saved nonaffine residual from restart to 50k.',
            'rows':rows}
    (OUT/'restart_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'restart_audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for r in rows:
        if r['width']==196 and r['lr']==.001:
            print(json.dumps(r))
    for restart in [1000,30000]:
        shifts=[]
        for r in rows:
            if r['restart']==restart:
                initial=next(a for a in rows if (a['width'],a['target'],a['lr'],a['restart'])==(r['width'],r['target'],r['lr'],0))
                shifts.append(100*(r['B0']/initial['B0']-1))
        print('restart',restart,'B0 percentage change range',min(shifts),max(shifts))
if __name__=='__main__':main()
