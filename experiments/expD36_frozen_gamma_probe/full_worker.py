"""Two fixed Slurm workers partitioning the complete declared training matrix."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import jax
from . import core, full_train as t


def tasks(worker,cfg):
    maps=(['raw','collective_neighbor','individual_neighbor'] if worker==0
          else ['collective','individual'])
    rows=[]
    for name in maps:
        rows.extend([('primary',cfg['n'],name),('initialization',cfg['n'],name)])
    width_map='raw' if worker==0 else 'collective_neighbor'
    rows.extend(('width',n,width_map) for n in cfg['widths'] if n!=cfg['n'])
    rows.append(('polynomial',cfg['n'],width_map))
    if worker==0:
        rows.append(('rate',cfg['n'],'raw'))
        rows.append(('coordinate',cfg['n'],'uniform_sqrt_h'))
    else:
        rows.extend(('coordinate',cfg['n'],name) for name in cfg['coordinate_controls'] if name!='uniform_sqrt_h')
        rows.extend(('joint',n,'joint') for n in cfg['widths'])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',type=int,choices=[0,1],required=True)
    parser.add_argument('--seconds',type=float,default=3200)
    args=parser.parse_args(); root=args.root
    manifest=json.loads((root/'manifest.json').read_text()); assert manifest['complete']
    cfg=manifest['config']; deadline=time.monotonic()+args.seconds
    t.train.verify_gpu(root,f'full_worker_{args.worker}')
    record=dict(worker=args.worker,tasks=tasks(args.worker,cfg),completed=[],complete=False)
    destination=root/'validation'/f'worker_{args.worker}.json'
    for stage,n,name in record['tasks']:
        if stage=='primary':
            t.primary(root,cfg,n,name,cfg['gammas'],cfg['targets'],deadline)
        elif stage=='width':
            t.primary(root,cfg,n,name,[1,4,n/8],cfg['robust_targets'],deadline)
        elif stage=='initialization':
            t.robustness(root,cfg,name,deadline)
        elif stage=='joint':
            t.joint(root,cfg,n,deadline)
        else:
            t.controls(root,cfg,name,stage,deadline)
        record['completed'].append([stage,n,name]); core.write_json(destination,record)
        jax.clear_caches(); gc.collect()
    record['complete']=True; core.write_json(destination,record)


if __name__=='__main__':
    main()
