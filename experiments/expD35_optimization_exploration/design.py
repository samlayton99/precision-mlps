"""Curated stage manifests, selected using the final fifth of validation history."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from . import run


def rank(root, minimum=20000):
    rows=[]
    for path in sorted(root.glob('*/case.json')):
        folder=path.parent
        if not (folder/'latest.json').exists(): continue
        latest=json.loads((folder/'latest.json').read_text())
        if latest['step']<minimum or latest['failed_update']: continue
        values=[]
        for evaluation in folder.glob('evaluation_*.json'):
            e=json.loads(evaluation.read_text())
            if .8*latest['step']<=e['step']<=latest['step'] and e['validation_relative_mse'] is not None:
                values.append(e['validation_relative_mse'])
        if not values or not np.all(np.isfinite(values)): continue
        rows.append(dict(id=folder.name,config=json.loads(path.read_text()),step=latest['step'],
                         score=float(np.mean(values)),median=float(np.median(values)),
                         upper=float(np.quantile(values,.9)),best=float(min(values))))
    return sorted(rows,key=lambda r:r['score'])


def baseline_rows(rows):
    return [r for r in rows if r['config']==run.case(**{k:r['config'][k] for k in
        ('n','seed','target','optimizer','coordinates','eta')})]


def winners(rows, count=1, coordinates=None):
    groups=defaultdict(list)
    for r in rows:
        c=r['config']
        if coordinates and c['coordinates'] not in coordinates: continue
        group=(c['n'],c['target'],c['optimizer'],c['coordinates'])
        if c['seed']==0 and len(groups[group])<count: groups[group].append(r)
    return [r for group in groups.values() for r in group]


def build(root,stage):
    rows=rank(root,100000 if stage in ('ema','initialization','agreement') else 20000)
    controls=winners(baseline_rows(rows),2 if stage=='baseline_promote' else 1,
                     None if stage=='baseline_promote' else ('individual','neighbor'))
    cases=[]
    if stage=='baseline_promote':
        cases=[dict(r['config'],seed=s) for r in controls for s in (0,1)]
    elif stage=='ema':
        for r in controls:
            c=r['config'];eta=c['eta'];cases.append(c)
            for alpha,strength in [(a,2.) for a in (.9,.98,.995,.999)]+[(.98,.5),(.98,8.)]:
                for factor in (.3,1.,3.):
                    cases.append(dict(c,ema_alpha=alpha,ema_strength=strength,eta=eta*factor))
            cases.append(dict(c,eta=eta*3))
            cases.append(dict(c,ema_strength=2.,ema_location=2))
            if c['optimizer']=='gd':
                cases.extend([dict(c,ema_strength=2.,eta=eta/3),dict(c,ema_strength=2.,ema_normalized=True)])
            else:
                cases.append(dict(c,ema_strength=2.,ema_normalized=True,epsilon=c['epsilon']/3))
    elif stage=='initialization':
        for r in controls:
            c=r['config']
            for init in ('reference_xavier','individual_gaussian'):
                for slope in ('physical_xavier','lambda_xavier'):
                    for factor in (.3,1.,3.):
                        cases.append(dict(c,initialization=init,slope_initialization=slope,eta=c['eta']*factor))
    elif stage=='agreement':
        for r in controls:
            c=r['config'];cases.append(c)
            for sampling in ('full','stratified'):
                for threshold in (.5,.9):
                    cases.append(dict(c,sampling=sampling,schedule='agreement',agreement_threshold=threshold,eta_ceiling=c['eta']))
                cases.append(dict(c,sampling=sampling,schedule='decay',eta_ceiling=c['eta']))
                cases.append(dict(c,sampling=sampling))
    else: raise ValueError(stage)
    return list({run.key(c):c for c in cases}.values()),controls


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',required=True)
    args=p.parse_args();cases,selected=build(args.root,args.stage)
    if not cases: raise RuntimeError('No eligible completed controls')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    run.write_json(args.out,cases)
    run.write_json(args.out.with_suffix('.selection.json'),dict(stage=args.stage,selected=selected,cases=len(cases)))
    print(json.dumps(dict(cases=len(cases),selected=[dict(id=r['id'],score=r['score'],eta=r['config']['eta']) for r in selected])))


if __name__=='__main__':main()
