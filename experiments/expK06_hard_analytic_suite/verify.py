"""Reload ordinary networks and measure all methods with identical Torch inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import *
from experiments.expK06_hard_analytic_suite.models import build
from experiments.expK06_hard_analytic_suite.train import prediction,score
from docs.qi_practical_guide.reference import MLP
from experiments.expK06_hard_analytic_suite.integration import zero_crossings,zero_refined_grid


def load_row(r):
    if r['method']=='QUILL':
        q=MLP.load(ROOT/r['model']);k=r['output_index'];w=r['width']
        m=torch.nn.Sequential(torch.nn.Linear(1,w,dtype=torch.float64),torch.nn.Tanh(),torch.nn.Linear(w,1,dtype=torch.float64))
        with torch.no_grad():
            m[0].weight.copy_(torch.from_numpy(q.weight));m[0].bias.copy_(torch.from_numpy(q.hidden_bias))
            m[2].weight.copy_(torch.from_numpy(q.readout[:,k][None,:]));m[2].bias.fill_(float(q.output_bias[k]))
        return m,1.,0.
    checkpoint=torch.load(ROOT/r['run_directory']/'selected.pt',weights_only=False)
    m=build(r['method'],r['width'],r['variant'],r['seed']);m.load_state_dict(checkpoint['model'])
    return m,checkpoint['amplitude'],checkpoint['offset']


def measure(r,repeat=False):
    model_path=ROOT/r['model'] if r['method']=='QUILL' else ROOT/r['run_directory']/'selected.pt'
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    path=OUT/'verified_final'/f"{r['target']}__{r['method']}__N{r['N']}__{digest}.json"
    if path.exists():return json.loads(path.read_text())
    m,amp,offset=load_row(r)
    errors={}
    for q in [128,256]:
        x,w=quadrature(q);y=target(r['target'],x)
        errors[str(q)]=score(m,x,y,w,amp,offset)
    stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256'])
    if not stable and r['method']!='NPU':
        x,w=quadrature(512);y=target(r['target'],x);errors['512']=score(m,x,y,w,amp,offset)
        stable=abs(errors['256']-errors['512'])<=max(2e-13,1e-5*errors['512'])
    final_error=errors[str(max(map(int,errors)))];zero_errors={};roots=[]
    if r['method']=='NPU':
        roots=zero_crossings(m)
        if roots:
            for order in [64,128]:
                x,w=zero_refined_grid(roots,order);y=target(r['target'],x)
                zero_errors[str(order)]=score(m,x,y,w,amp,offset)
            final_error=zero_errors['128']
            stable=abs(zero_errors['64']-zero_errors['128'])<=max(2e-13,1e-5*final_error)
            if not stable:
                x,w=zero_refined_grid(roots,256);y=target(r['target'],x)
                zero_errors['256']=score(m,x,y,w,amp,offset);final_error=zero_errors['256']
                stable=abs(zero_errors['128']-zero_errors['256'])<=max(2e-13,1e-5*final_error)
    row={**r,'original_test_relative_l2':r.get('relative_l2'),'relative_l2':final_error,
         'independent_torch_errors':errors,'quadrature_stable':stable,
         'zero_crossings':roots,'zero_refined_errors':zero_errors,
         'total_stored_parameters':sum(p.numel() for p in m.parameters()),'repeat':repeat}
    if r['N']==NS[-1]:
        rng=np.random.default_rng(8319)
        # Endpoint-inclusive plotting grid, plus independent random and endpoint probes.
        x=np.linspace(-1,1,8193);yp=prediction(m,x,amp,offset);true=target(r['target'],x)
        xq,wq=quadrature(128);norm=np.sqrt(np.dot(wq,target(r['target'],xq)**2)/2)
        destination=OUT/'predictions'/f"{r['target']}__{r['method']}__seed{r.get('seed',0)}.npz"
        destination.parent.mkdir(exist_ok=True)
        np.savez(destination,x=x,truth=true,prediction=yp,residual=yp-true,normalized_residual=(yp-true)/norm,rms=norm)
        row['predictions']=str(destination.relative_to(ROOT))
        probes=np.unique(np.r_[x,rng.uniform(-1,1,8192),-1+np.geomspace(1e-12,.1,257),1-np.geomspace(1e-12,.1,257)])
        diff=prediction(m,probes,amp,offset)-target(r['target'],probes)
        row['maximum_sampled_absolute']=float(np.max(abs(diff)))
        row['maximum_sampled_normalized']=float(np.max(abs(diff))/norm)
        row['dense_probe_count']=len(probes)
        if r['method']=='QUILL':
            export=OUT/'quill'/f"{r['target']}__selected_torch.pt"
            torch.save(dict(model=m.state_dict(),target=r['target'],N=r['N'],width=r['width'],domain=[-1,1]),export)
    save(path,row)
    return row


def main():
    torch.set_num_threads(1)
    spec=json.loads((OUT/'specification_main50k.json').read_text())
    for path,digest in spec['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    qi=json.loads((OUT/'quill_tuned_selected.json').read_text())
    full=json.loads((OUT/'main50k_summary.json').read_text())
    assert len(qi)>=len(NAMES)*len(NS) and len(full)==48
    assert all(r['steps']==50000 and r['selected_phase']=='Adam' for r in full)
    rows=[]
    for r in qi+full:
        rows.append(measure(r));save(OUT/'verified_summary.json',rows)
    selected=[]
    for name in MAIN:
        for method in ['QUILL','Adam','NPU']:
            r=min((a for a in rows if a['target']==name and a['method']==method and a['width'] in WIDTHS),key=lambda a:a['validation'])
            m,amp,offset=load_row(r)
            x=np.linspace(-1,1,8193);true=target(name,x);yp=prediction(m,x,amp,offset)
            xq,wq=quadrature(128);norm=np.sqrt(np.dot(wq,target(name,xq)**2)/2)
            path=OUT/'predictions'/f"{name}__{method}__best_validated_N{r['N']}.npz"
            np.savez(path,x=x,truth=true,prediction=yp,residual=yp-true,normalized_residual=(yp-true)/norm,rms=norm)
            selected.append({**r,'predictions':str(path.relative_to(ROOT))})
    save(OUT/'selected_fits.json',selected)
    for name in NAMES:
        for method in (['QUILL','Adam','NPU'] if name in MAIN else ['QUILL']):
            assert sorted(r['width'] for r in rows if r['target']==name and r['method']==method and r['width'] in WIDTHS)==WIDTHS
    failed=[r for r in rows if not r['quadrature_stable']]
    data_checks=[]
    for n in NS:
        train=training_grid(n);val,_=quadrature(32);test,_=quadrature(128)
        data_checks.append(dict(N=n,samples=len(train),training_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                                train_validation_overlap=len(np.intersect1d(train,val)),train_test_overlap=len(np.intersect1d(train,test))))
    fields=['target','method','N','width','parameters','relative_l2','validation','quadrature_stable','selected_phase','variant','lr','best_adam_step','steps','training_samples']
    with (OUT/'scaling.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    save(OUT/'verification.json',dict(primary_models=len(rows),trained_models=48,repeat_models=0,quadrature_flags=failed,
                                     data_checks=data_checks,source_hashes_match=True,scoring='ordinary float64 Torch for every method'))
    print('Verified',len(rows),'models; 48 trained baselines;',len(failed),'integration flags',flush=True)
if __name__=='__main__':main()
