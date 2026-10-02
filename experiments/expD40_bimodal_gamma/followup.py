"""Separate, recorded 10:1 mode-range iteration; original sources unchanged."""
from __future__ import annotations
import argparse
import json
from experiments.expD40_bimodal_gamma import run as parent

ARMS = ['mild_mix_both','mild_rms_both','mild_mix_last','mild_rms_last']


def config():
    cfg=parent.config()
    cfg['d40_design']['high_multiplier']=1.
    cfg['d40_followup_override']={'low_multiplier':.1,'high_multiplier':1.,'high_fraction':.5}
    cfg['d40_followup_sha256']=parent.digest(parent.HERE/'followup.py')
    cfg['d40_followup_plan_sha256']=parent.digest(parent.OUT/'followup_plan.md')
    return cfg


def make_model(d,width,seed,scheme,train_x,cfg):
    is_mild=scheme.startswith('mild_')
    if is_mild:
        assert cfg==config()
    model,info=parent.make_model(d,width,seed,scheme.removeprefix('mild_'),train_x,cfg)
    info['scheme']=scheme
    return model,info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['screen','confirm'])
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    spec=parent.design()
    if args.phase=='screen':
        pairs=[(t,a) for t in spec['tasks'] for a in ARMS]
        steps,phase,seeds=10000,'pilot',[0]
        parent.base.OUT=parent.OUT/'followup'
    else:
        selection=json.loads((parent.OUT/'selection.json').read_text())
        assert selection['source_sha256']==parent.config()['d40_source_sha256']
        assert selection['followup_sha256']==config()['d40_followup_sha256']
        assert selection['followup_plan_sha256']==config()['d40_followup_plan_sha256']
        for path,sha in selection['screen_sha256'].items():assert parent.digest(parent.ROOT/path)==sha
        pairs=selection['confirmation_pairs']
        steps,phase,seeds=20000,'compare',[0,1,2]
        parent.base.OUT=parent.OUT
    parent.base.make_model=make_model
    recipes=json.loads((parent.BASE_OUT/'selected_recipe.json').read_text())
    jobs=[(t,a,recipes[t]['lr'],s,steps,phase,config() if a.startswith('mild_') else parent.config(),512)
          for t,a in pairs for s in seeds]
    for i,job in enumerate(jobs):
        if i%args.shards==args.shard:parent.base.run_one(job)


if __name__=='__main__':
    main()
