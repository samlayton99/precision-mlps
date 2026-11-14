"""Six matched small-initialization GD traces for the restart-norm comparison."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import sys,json,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD60_feature_bound_reconstruction.run import train,CONFIG
from experiments.expD56_scale_acquisition.run import write_json,utc_now
OUT=ROOT/'results/checkpoint_D_optimizers/expD62_joint_training_diagnostics/matched_small_initialization'
TARGETS=['square','sine_4pi','runge_100'];RATES=[.001,.0001]

def cfg(lr):
    return {**CONFIG,'width':196,'interior_intervals':147,'train_points':1024,'eval_points':4096,
            'learning_rate':lr,'steps':50000,'restart':30000}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'experiments/expD60_feature_bound_reconstruction/run.py',ROOT/'experiments/expD60_feature_bound_reconstruction/bound.py',ROOT/'experiments/expD56_scale_acquisition/model.py']
    manifest={'targets':TARGETS,'configs':[cfg(lr) for lr in RATES],
              'initialization':'Slopes, hidden biases, readouts uniform +/-sqrt(6/(W+1)); output bias zero. Same family as D60; width 196 and seed components [30,147] and [30,147,24].',
              'purpose':'Matched second row for measured restart B0; smaller-init family is not itself a guarantee of a useful bound on every new target.',
              'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write_json(path,manifest)
    write_json(OUT/'sweep_status.json',{'state':'running','pid':os.getpid(),'started_utc':utc_now()})
    failures=[];completed=0
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(train,('glorot',name),cfg(lr),OUT/f'lr{lr:.0e}'):(name,lr) for lr in RATES for name in TARGETS}
        for job in as_completed(jobs):
            try:job.result();completed+=1
            except Exception as exc:failures.append({'case':jobs[job],'error':repr(exc)})
            write_json(OUT/'sweep_status.json',{'state':'running','completed':completed,'failures':failures,'updated_utc':utc_now()})
    write_json(OUT/'sweep_status.json',{'state':'complete' if not failures else 'failed','completed':completed,'failures':failures,'updated_utc':utc_now()})
    if failures:raise RuntimeError(failures)
if __name__=='__main__':main()
