"""Resumable paired pilots, full training, and fresh-seed main-panel checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import *
from experiments.expK06_hard_analytic_suite.train import run_one,identifier
from experiments.expK06_hard_analytic_suite.accurate_full import full_run,refine_pilot


def configs():
    for name in NAMES:
        for n,w in zip(NS,WIDTHS):
            for method in ['Adam','NPU']:
                variants=['pytorch','xavier'] if method=='Adam' else ['native','positive','composed']
                rates=[.001,.01,.03] if method=='Adam' else [.001,.01]
                for variant in variants:
                    for lr in rates:
                        yield dict(target=name,N=n,width=w,method=method,variant=variant,lr=lr,seed=0)

def batch(jobs,stage):
    rows=[];errors=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        worker=run_one if stage=='pilots' else full_run
        pending={pool.submit(worker,**job):job for job in jobs}
        for fut in as_completed(pending):
            job=pending[fut]
            try:rows.append(fut.result())
            except Exception as exc:errors.append(dict(config=job,error=repr(exc)))
            save(OUT/f'{stage}_summary.json',rows)
            save(OUT/'status.json',dict(stage=stage,completed=len(rows),planned=len(jobs),failures=errors,elapsed=time.monotonic()-start,coordinator_pid=os.getpid()))
            if len(rows)%16==0 or errors:print(stage,len(rows),'/',len(jobs),'failures',len(errors),flush=True)
    if errors:raise RuntimeError(errors)
    return rows

def main():
    all_configs=list(configs());source=[Path(__file__).with_name(s) for s in ['targets.py','common.py','models.py','train.py','construction.py','integration.py','accurate_full.py']]
    spec=dict(targets=LABELS,main=MAIN,extra=EXTRA,domain=[-1,1],N=NS,widths=WIDTHS,
              training_samples='linspace(-1,1,max(2003,4*(N+1))); same for every method',
              data_access='function values only; no exterior labels; no derivatives',
              pilot_steps=10000,full_steps=100000,batch_size=100,seed=0,repeat_seed=1,
              schedule='constant until 50k; cosine to 1e-6 by 100k; no warmup',
              polishing='up to 500 full-batch LBFGS iterations, then gelsd readout LS at rcond=1e-14; accept only validation improvements',
              quill='guide fit_qi, automatic halo, lambda .2/.25/.3 selected on validation, gelsd rcond 1e-14',
              selection='per target, method, width: best pilot validation; continue to 100k and polish; independent test does not select',
              validation_refinement='NPU pilot best/final checkpoints re-ranked by order-32 log panels at input zeros before full-run selection; full-run validation uses this resolved physical-dx metric; test uses order 128',
              config_count=len(all_configs),sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    path=OUT/'specification.json'
    if path.exists():assert json.loads(path.read_text())==spec,'Freeze changed; audit before resume'
    else:save(path,spec)
    pilots=batch([dict(c=c,steps=10000,stage='pilots') for c in all_configs],'pilots')
    with ProcessPoolExecutor(max_workers=4) as pool:
        pilots=list(pool.map(refine_pilot,pilots))
    save(OUT/'refined_pilot_summary.json',pilots)
    selected=[]
    for name in NAMES:
        for n in NS:
            for method in ['Adam','NPU']:
                r=min((r for r in pilots if r['target']==name and r['N']==n and r['method']==method),key=lambda r:r['validation'])
                selected.append(r)
    save(OUT/'selected_pilots.json',selected)
    keys=all_configs[0].keys()
    full=batch([dict(c={k:r[k] for k in keys},steps=100000,stage='full',resume=r['run_directory'],polish=True) for r in selected],'full')
    # Two seeds at the largest width in the four main cases; repeats are shown,
    # not silently used to replace a less favorable seed-0 result.
    repeats=[]
    for r in full:
        if r['target'] in MAIN and r['N']==NS[-1]:
            c={k:r[k] for k in keys};c['seed']=1
            repeats.append(dict(c=c,steps=100000,stage='repeats',polish=True))
    batch(repeats,'repeats')
    save(OUT/'status.json',dict(stage='complete',pilots=len(pilots),full=len(full),repeats=len(repeats)))
if __name__=='__main__':main()
