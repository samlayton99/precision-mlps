"""Second validation-only iteration: separate angular allocation from slope.

Original centered banks:21/22centers, lambda=.25, normalized slope2.5/2.625.
64directions:8centers,lambda=.25,normalized slope.875.
soft24 uses lambda=.0875, matching .875 for21centers (.91875 for22).
sharp64 uses lambda=5/7, matching normalized slope2.5 exactly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from experiments.expD39_qi_init_theory import run as parent

EXTRA = {'soft24':dict(centered=True,lam=.0875),
         'sharp64':dict(centered=True,n_dirs=64,lam=5/7)}


def config():
    cfg=parent.config()
    cfg['d39_followup_sha256']=hashlib.sha256((parent.HERE/'followup.py').read_bytes()).hexdigest()
    cfg['d39_resolved_overrides']=EXTRA
    return cfg


def collect():
    rows={}
    recipes=json.loads((parent.BASE_OUT/'selected_recipe.json').read_text())
    for task in ['airfoil','kin8nm','sarcos','superconductivity']:
        for arm in EXTRA:
            lr=recipes[task]['lr']
            path=parent.OUT/'followup/data/pilot'/f'{task}_{arm}_w512_lr{lr:g}_seed0_steps10000.json'
            result=json.loads(path.read_text())
            expected=dict(task=task,scheme=arm,lr=lr,seed=0,steps=10000,width=512,phase='pilot',config=config())
            if task in expected['config'].get('task_data_protocols',{}):
                expected['data_protocol']=expected['config']['task_data_protocols'][task]
            assert parent.base.same_identity(result['identity'],expected)
            assert result.get('complete') and result['trace'][-1]['step']==10000
            assert all(set(q['learned'])=={'train','val'} for q in result['trace'])
            assert [q['step'] for q in result['trace']]==[0,1,10,30,100,300,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
            rows[task,arm]=(result,path)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['screen','confirm'])
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    parent.VARIANTS.update(EXTRA)
    parent.base.make_model=parent.make_model
    recipes=json.loads((parent.BASE_OUT/'selected_recipe.json').read_text())
    if a.phase=='screen':
        parent.base.OUT=parent.OUT/'followup'
        tasks=['airfoil','kin8nm','sarcos','superconductivity']
        arms=list(EXTRA)
        seeds=[0]
        steps,phase=10000,'pilot'
    else:
        parent.base.OUT=parent.OUT
        tasks=list(recipes)
        arms=json.loads((parent.OUT/'selection.json').read_text())['confirm_variants']
        assert all(arm in EXTRA for arm in arms)
        seeds=[0,1,2]
        steps,phase=20000,'compare'
    jobs=[(t,arm,recipes[t]['lr'],s,steps,phase,config(),512) for t in tasks for s in seeds for arm in arms]
    for i,job in enumerate(jobs):
        if i%a.shards==a.shard:
            parent.base.run_one(job)


if __name__=='__main__':
    main()
