"""User-scoped comparison: four functions, six widths, two methods, 50k steps."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import MAIN
from experiments.expK06_hard_analytic_suite.accurate_full import full_run

def main():
    pilots=json.loads((OUT/'refined_pilot_summary.json').read_text())
    selected=[];keys=['target','method','N','width','variant','lr','seed']
    for name in MAIN:
        for n in NS:
            for method in ['Adam','NPU']:
                selected.append(min((r for r in pilots if r['target']==name and r['N']==n and r['method']==method),key=lambda r:r['validation']))
    sources=[Path(__file__).with_name(s) for s in ['targets.py','common.py','models.py','train.py','integration.py','accurate_full.py']]
    spec=dict(main=MAIN,N=NS,widths=WIDTHS,steps=50000,scheduled_decay_start=25000,final_lr=1e-6,
              training_runs=48,selection='reasonable settings from saved validation-ranked pilots; no new baseline sweep',
              resume='10k pilot optimizer/RNG state; first 10k steps match the 50k schedule exactly',
              polishing=False,repeats=False,additional_functions='QUILL only, all 16 targets',
              sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    path=OUT/'specification_main50k.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:save(path,spec)
    rows=[];failures=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures=[]
        for r in selected:
            c={k:r[k] for k in keys};c['schedule_steps']=50000
            futures.append(pool.submit(full_run,c,50000,'main50k',r['run_directory'],False))
        for f in as_completed(futures):
            try:rows.append(f.result())
            except Exception as exc:failures.append(repr(exc))
            save(OUT/'main50k_summary.json',rows)
            save(OUT/'status_main50k.json',dict(completed=len(rows),planned=48,failures=failures,elapsed=time.monotonic()-start,coordinator_pid=os.getpid()))
            print('main50k',len(rows),'/48',failures,flush=True)
    if failures:raise RuntimeError(failures)
if __name__=='__main__':main()
