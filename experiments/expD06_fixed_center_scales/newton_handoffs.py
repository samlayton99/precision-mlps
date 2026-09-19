"""Four-GPU-hour, two-seed Newton/Adam-handoff and SSB scaling comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import joint_conditioning as joint, difference_training as first, ratio, run


def adam_case(seed):
    return joint.case('adam','parameter_scale',eta=.003,seed=seed,native_epsilon=1e-15)


def manifest(root):
    rows=[]
    for seed in (0,1):
        key=first.case_key(adam_case(seed));checkpoint=root.parent/'joint_conditioning'/key/'checkpoint_005300000.npz'
        warm=dict(root='../joint_conditioning',source_key=key,step=5300000,
                  sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        def add(opt,coord='parameter_scale',start=None,**settings):
            config=joint.case(opt,coord,seed=seed,followup='newton_handoffs_v1',diagnostics=True,**settings)
            if start:config['warm_start']=start
            rows.append(config)
        # Finish the new scientific comparisons before repeating known controls.
        add('newton',start=warm)
        add('newton')
        add('gn',start=warm,damping_floor=1e-30)
        for epsilon in (1e-30,float(np.finfo(float).eps)):
            add('ssbroyden',start=warm,curvature_epsilon=epsilon)
        add('ssbroyden','physical',curvature_epsilon=float(np.finfo(float).eps))
        for epsilon in (float(np.finfo(float).eps),1e-30):
            add('ssbroyden',curvature_epsilon=epsilon)
    return rows


def continue_adam(root,seed,deadline):
    """Copy a full Adam state into an isolated root and keep its original clock."""
    config=adam_case(seed);key=first.case_key(config)
    origin=root.parent/'joint_conditioning'/key;dest=root/'adam_continuation'/key
    if not (dest/'latest.json').exists():
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('case.json','reference.npz','checkpoint_005300000.npz','state_005300000.pkl'):
            shutil.copy2(origin/name,dest/name)
        with np.load(dest/'checkpoint_005300000.npz') as cp:
            run.write_json(dest/'latest.json',dict(step=5300000,completed_updates=5300000,status='continuing',
                           train_mse=float(cp['train_mse']),validation_mse=float(cp['validation_mse'])))
        run.write_json(dest/'continuation_origin.json',dict(step=5300000,source_key=key,
            state_sha256=hashlib.sha256((origin/'state_005300000.pkl').read_bytes()).hexdigest()))
    first.advance_group(root/'adam_continuation',[config],5400000,deadline)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',type=int,choices=(0,1))
    parser.add_argument('--seconds',type=float,default=1800)
    parser.add_argument('--frontier',type=int,default=20000)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--only',choices=('newton','others'))
    parser.add_argument('--continue-blocks',action='store_true',help='Continue viable Newton and Adam-to-GN runs in equal 10k increments')
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--ssbroyden-source',type=Path)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        run.write_json(args.root/'cases.json',manifest(args.root));return
    if args.worker is None:parser.error('A worker seed is required')
    if args.require_gpu:ratio.gpu_environment(args.root)
    deadline=time.monotonic()+args.seconds-45
    cases=[c for c in json.loads((args.root/'cases.json').read_text()) if c['seed']==args.worker]
    if args.continue_blocks:
        cases=[c for c in cases if c['optimizer']=='newton' or (c['optimizer']=='gn' and c.get('warm_start'))]
        while time.monotonic()<deadline:
            active=[]
            for config in cases:
                latest=json.loads((args.root/first.case_key(config)/'latest.json').read_text())
                if latest['status']!='continuing':continue
                if latest['completed_updates']<20000:raise ValueError('Complete the mandatory horizon before continuation')
                active.append((config,latest['completed_updates']))
            if not active:return
            frontier=(min(step for _,step in active)//10000+1)*10000
            for config,_ in active:
                if time.monotonic()>=deadline:return
                joint.advance_higher(args.root,config,frontier,deadline,args.ssbroyden_source)
        return
    for config in cases:
        if args.only=='newton' and config['optimizer']!='newton':continue
        if args.only=='others' and config['optimizer']=='newton':continue
        if time.monotonic()>=deadline:return
        joint.advance_higher(args.root,config,100 if args.benchmark else args.frontier,deadline,args.ssbroyden_source)
    if not args.benchmark and args.only!='newton' and time.monotonic()<deadline:
        continue_adam(args.root,args.worker,deadline)


if __name__=='__main__':main()
