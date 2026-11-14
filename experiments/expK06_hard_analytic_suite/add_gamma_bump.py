"""Append two requested QUILL-only cases without altering the existing suite."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mpmath as mp
from docs.qi_practical_guide.reference import fit_qi,MLP
from experiments.expK06_hard_analytic_suite.common import OUT,NS,halo,training_grid,quadrature,save
from experiments.expK06_hard_analytic_suite.tune_quill import predictions
from experiments.expD58_extra_hard_scale_acquisition.targets import target

ADDED_NAMES=['gamma_shifted','smooth_bump']
ADDED_LABELS={
    'gamma_shifted':r'$\Gamma(0.05+2.475(x+1))$',
    'smooth_bump':r'$e^{-1/(1-(x/0.35)^2)}$',
}
ADDED_DESCRIPTIONS={
    'gamma_shifted':'Shifted Gamma function',
    'smooth_bump':r'Compact bump; zero for $|x|\geq0.35$',
}
DEST=OUT/'gamma_bump'


def main():
    torch.set_num_threads(1);DEST.mkdir(parents=True,exist_ok=True)
    sizes=NS+[4096]
    spec=dict(targets={
        'gamma_shifted':dict(formula='Gamma(0.05+2.475*(x+1))',regularity='analytic on a neighborhood of [-1,1]; pole outside at -1-2/99'),
        'smooth_bump':dict(formula='exp(-1/(1-(x/.35)^2)) for |x|<.35; zero otherwise',regularity='C-infinity but nonanalytic at +/-0.35')},
        domain=[-1,1],N=sizes,halo='max(20,ceil(sqrt(N)))',lambda_value=.25,
        solver='gelsy',rcond=float(np.finfo(float).eps),dtype='float64',training='sample values inside [-1,1] only',
        scope='QUILL-only supplement; no changes to the four-function trained comparison',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(DEST/'specification.json',spec)
    rows=[]
    for n in sizes:
        folder=DEST/f'N{n}';folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            rows+=json.loads((folder/'summary.json').read_text());continue
        x=training_grid(n);y=np.column_stack([target(name,x) for name in ADDED_NAMES]);start=time.monotonic()
        m,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=n,lam=.25,solver='gelsy',rcond=np.finfo(float).eps)
        m.save(folder/'model.npz');save(folder/'diagnostics.json',info)
        errors={}
        for q in [32,128,256]:
            xx,ww=quadrature(q);truth=np.column_stack([target(name,xx) for name in ADDED_NAMES]);pred=predictions(m,xx)
            errors[str(q)]=np.sqrt(np.sum(ww[:,None]*(pred-truth)**2,axis=0)/np.sum(ww[:,None]*truth**2,axis=0))
        current=[]
        for k,name in enumerate(ADDED_NAMES):
            e={q:float(v[k]) for q,v in errors.items()}
            current.append(dict(target=name,method='QUILL',N=n,width=len(m.weight),parameters=3*len(m.weight)+1,
                                validation=e['32'],relative_l2=e['256'],quadrature=e,
                                quadrature_stable=abs(e['128']-e['256'])<=max(2e-13,1e-5*e['256']),
                                model=str((folder/'model.npz').relative_to(ROOT)),output_index=k,
                                solver='gelsy',rcond=float(np.finfo(float).eps),lambda_value=.25,
                                halo=halo(n),training_samples=len(x),seconds=time.monotonic()-start))
        rows+=current;save(folder/'summary.json',current);save(OUT/'quill_extension_verified.json',rows)
        print('W',len(m.weight),{r['target']:r['relative_l2'] for r in current},flush=True)
    save(OUT/'quill_extension_verified.json',rows)
    selected=[]
    for name in ADDED_NAMES:
        r=min((r for r in rows if r['target']==name),key=lambda r:r['validation']);m=MLP.load(ROOT/r['model'])
        # Dense independent samples, including bump support boundaries and the Gamma endpoint.
        x=np.unique(np.r_[np.linspace(-1,1,16385),-.35,.35,-1+np.geomspace(1e-12,.1,257)])
        yp=predictions(m,x)[:,r['output_index']];truth=target(name,x)
        np.savez(DEST/f'{name}_selected_predictions.npz',x=x,truth=truth,prediction=yp)
        xx=np.array([-1.,-.999,-.95,-.351,-.35,-.349,-.1,0,.1,.349,.35,.351,.9,1.])
        with mp.workdps(70):
            exact=[]
            for z in xx:
                z=mp.mpf(float(z))
                if name=='gamma_shifted':exact.append(float(mp.gamma(mp.mpf('.05')+mp.mpf('2.475')*(z+1))))
                elif abs(z)>=mp.mpf('.35'):exact.append(0.)
                else:exact.append(float(mp.exp(-1/(1-(z/mp.mpf('.35'))**2))))
        reference_difference=float(np.max(abs(target(name,xx)-np.asarray(exact))))
        selected.append({**r,'maximum_dense_absolute_error':float(abs(yp-truth).max()),'reference_spot_maximum_difference':reference_difference})
    save(DEST/'selected.json',selected)
    print('Selected:',[(r['target'],r['width'],r['relative_l2']) for r in selected],flush=True)
if __name__=='__main__':main()
