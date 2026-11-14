"""Matched 10k comparison: reuse verified existing rates, execute the new rates."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,sys
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD55_qi_adam_drift.run import HERE,OUT,run_job,write_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--min-exponent',type=int,default=12)
 p.add_argument('--workers',type=int,default=3);args=p.parse_args()
 if args.min_exponent<7:raise ValueError('Expanded sweep must include at least 1e-7')
 output=OUT/f'expanded_lr_1e-{args.min_exponent}_10k';output.mkdir(parents=True,exist_ok=True)
 base=json.loads((HERE/'config.json').read_text());rates=[10.**(-k) for k in range(2,args.min_exponent+1)]
 manifest={'steps':10000,'warmup_steps':1000,'learning_rates':rates,'targets':base['targets'],
           'rows':['standard','qi'],'runs':{},'initials':{},'source_configs':{},
           'description':'Matched 10k comparison; prior 50k standard-initialization runs remain preserved separately.'}
 jobs=[]
 for init,dirname in [('standard','standard_init_warmup1000'),('qi','warmup1000')]:
  source=OUT/dirname;original=json.loads((source/'config.json').read_text())
  for key in ['steps','warmup_steps','train_points','eval_points','adam','dtype','seed','threads']:
   assert original[key]==base[key],key
  new_rates=[lr for lr in rates if lr not in original['learning_rates']]
  config=dict(base,learning_rates=new_rates,initialization=init,width=269,
              checkpoint_every=2500,progress_every=10000)
  dest=output/init;dest.mkdir(parents=True,exist_ok=True);write_json(dest/'config.json',config)
  manifest['source_configs'][init]={'existing':str((source/'config.json').relative_to(ROOT)),
                                   'new':str((dest/'config.json').relative_to(ROOT))}
  for target in base['targets']:
   key=f'{init}/{target}';manifest['runs'][key]={};manifest['initials'][key]=str((source/target/'initial.json').relative_to(ROOT))
   for lr in rates:
    run_dir=source if lr in original['learning_rates'] else dest
    manifest['runs'][key][f'{lr:.0e}']=str((run_dir/target/f'lr_{lr:.0e}.json').relative_to(ROOT))
   jobs.append((target,config,dest/target))
 write_json(output/'manifest.json',manifest)
 write_json(output/'provenance.json',{'runner_sha256':hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest(),
                                    'orchestrator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    'torch':torch.__version__,'numpy':np.__version__,'complete':False})
 with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_job,jobs))
 # Check every new initial state against the originals, plus all expected histories.
 checks=[]
 for init,dirname in [('standard','standard_init_warmup1000'),('qi','warmup1000')]:
  for target in base['targets']:
   source=OUT/dirname/target;dest=output/init/target
   old=torch.load(source/'initial.pt',weights_only=True);new=torch.load(dest/'initial.pt',weights_only=True)
   assert all(torch.equal(old[k],new[k]) for k in old)
   a=np.load(source/('dataset.npz' if (source/'dataset.npz').exists() else 'qi.npz'));b=np.load(dest/'dataset.npz')
   for k in b.files:np.testing.assert_array_equal(a[k],b[k])
   initial=json.loads((source/'initial.json').read_text())['eval_rel_l2']
   for lr,path in manifest['runs'][f'{init}/{target}'].items():
    r=json.loads((ROOT/path).read_text());assert r['history'][0]['eval_rel_l2']==initial
    assert r['history'][-1]['step']==10000
    checks.append({'initialization':init,'target':target,'lr':float(lr),'final_rel_l2':r['summary']['final_rel_l2']})
 write_json(output/'validation.json',{'identical_initial_states_and_data':True,'runs':checks})
 provenance=json.loads((output/'provenance.json').read_text());provenance['complete']=True;write_json(output/'provenance.json',provenance)
 print(output,flush=True)

if __name__=='__main__':main()
