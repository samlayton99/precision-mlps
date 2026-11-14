"""Resume the original unchanged D60 trainer, validating its JSON manifest.

The original entry point compares Python tuple cases with JSON list cases and
rejects an unchanged manifest on resume. Normalize that representation here;
keep the original training and bound source files byte-for-byte unchanged.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD60_feature_bound_reconstruction.run import OUT,CONFIG,CASES,train,write_json,utc_now

def main():
    path=OUT/'manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config']==CONFIG
    assert manifest['cases']==[list(case) for case in CASES]
    for relative,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    write_json(OUT/'resume_manifest.json',{'original_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'resume_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'resumed_utc':utc_now()})
    write_json(OUT/'status.json',{'state':'running','pid':os.getpid(),'cases':len(CASES),'updated_utc':utc_now()})
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(train,case,CONFIG,str(OUT)) for case in CASES]
        for job in as_completed(jobs):print('COMPLETE',job.result(),flush=True)
    write_json(OUT/'status.json',{'state':'complete','cases':len(CASES),'updated_utc':utc_now()})

if __name__=='__main__':main()
