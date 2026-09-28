"""FP64 usefulness audit of proved conditional constants; no GD certificate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(root, output):
    rows=[]; sources={}
    for nref in (128,512,1024):
        path=root/'width_inputs'/f'N{nref}_fork20000.npz'
        data=np.load(path); sources[str(path)]=digest(path)
        cases=json.loads(str(data['cases'])); x=data['x']; variance=float(np.mean(x*x))
        if not np.allclose(x,-x[::-1],atol=2e-15,rtol=0):
            raise ValueError('Symmetric grid required for this audit')
        basis,_=np.linalg.qr(np.vander(x,4,increasing=True))
        for p,y,case in zip(data['p'],data['y'],cases):
            width=(len(p)-1)//3; aa,bb,cc=p[:-1].reshape(3,width)
            initial=np.sqrt(width)*np.array([max(abs(aa)),max(abs(bb)),max(abs(cc))])
            cnorm=float(np.linalg.norm(cc)); kappa=variance*cnorm**2/4
            yh=y-basis[:,:2]@(basis[:,:2].T@y)
            low=basis[:,2:4]@(basis[:,2:4].T@yh)
            eps=float(np.linalg.norm(low)/np.sqrt(len(x)))
            Y=float(np.linalg.norm(yh)/np.sqrt(len(x)))
            Yhigh=float(np.linalg.norm(yh-low)/np.sqrt(len(x)))
            for margin in (.10,.25,.50):
                A=initial*(1+margin); ac=A[2]; u=A[0]+A[1]
                lh=np.sqrt(2*ac*ac*u**4+u**6/9)
                d=np.array([ac*u*u,ac*u*u,u**3/3])
                q=np.array([ac,ac,u]); e=Y+ac*u**3/(3*width)
                K=e*(d+q*lh/np.sqrt(kappa))
                c_budget=.5*cnorm-lh/(np.sqrt(variance)*width)
                modes=[('generic',K,width)]
                if case['target']=='moment5':
                    # Global exact cubic remainder inequalities give these
                    # constants; no sampled derivative maximum is used.
                    rem=np.array([ac*(2/3)*u**4,ac*(2/3)*u**4,(2/15)*u**5])
                    cj=float(np.linalg.norm(rem)); bf=ac*u**3/3
                    gs=lh*bf+cj*Yhigh
                    star=d*bf+rem*Yhigh+q*gs/np.sqrt(kappa)
                    star+=eps*width*(d+q*lh/np.sqrt(kappa))
                    modes.append(('high_mode_with_measured_low_load',star,width**2))
                for name,force,scale in modes:
                    parameter_limits=scale/.002*(A-initial)/force
                    coarse_limit=scale/.002*c_budget/force[2]
                    sufficient=max(0.,min(float(min(parameter_limits)),float(coarse_limit)))
                    rows.append(dict(target=case['target'],seed=case['seed'],nref=nref,
                        width=width,fork=20000,margin=margin,model=name,
                        kappa=kappa,readout_norm=cnorm,coarse_motion_budget=c_budget,
                        initial_conditioning_test_closes=bool(c_budget>=0),
                        A_a=float(A[0]),A_b=float(A[1]),A_c=float(A[2]),
                        L_H=float(lh),fine_target_norm=Y,low_mode_load=eps,
                        K_a=float(force[0]),K_b=float(force[1]),K_c=float(force[2]),
                        parameter_limit_a=float(parameter_limits[0]),
                        parameter_limit_b=float(parameter_limits[1]),
                        parameter_limit_c=float(parameter_limits[2]),
                        coarse_conditioning_limit=float(coarse_limit),
                        sufficient_updates=int(np.floor(sufficient))))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'constants.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for name in sorted({r['model'] for r in rows}):
        for margin in (.1,.25,.5):
            group=[r for r in rows if r['model']==name and r['margin']==margin]
            summary.append(dict(model=name,margin=margin,cases=len(group),
                closes_initial=sum(r['initial_conditioning_test_closes'] for r in group),
                positive_horizons=sum(r['sufficient_updates']>0 for r in group),
                max_updates=max(r['sufficient_updates'] for r in group),
                median_updates=float(np.median([r['sufficient_updates'] for r in group]))))
    (output/'summary.json').write_text(json.dumps(dict(source_sha256=digest(__file__),
        inputs=sources,assumptions='Tracking is set to zero in this best-case diagnostic. FP64 empirical projections and arithmetic; not a rounding-controlled certificate. No constants fitted to future motion.',
        cases=36,summary=summary),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.root,args.output)
