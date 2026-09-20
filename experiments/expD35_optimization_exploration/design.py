"""Curated stage manifests, selected using the final fifth of validation history."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from . import run
from . import history


def rank(root, minimum=20000, horizon=None):
    rows=[]
    for path in sorted(root.glob('*/case.json')):
        folder=path.parent
        if not (folder/'latest.json').exists(): continue
        latest=json.loads((folder/'latest.json').read_text())
        endpoint=horizon if horizon is not None else latest['step']
        if latest['step']<max(minimum,endpoint): continue
        if latest['failed_update'] and latest['failed_update']<=endpoint:continue
        values=[];evaluations=history.evaluations(folder)
        if not any(e['step']==endpoint for e in evaluations):continue
        for e in evaluations:
            if .8*endpoint<=e['step']<=endpoint and e['validation_relative_mse'] is not None:
                values.append(e['validation_relative_mse'])
        if not values or not np.all(np.isfinite(values)): continue
        rows.append(dict(id=folder.name,config=json.loads(path.read_text()),step=endpoint,
                         score=float(np.mean(values)),median=float(np.median(values)),
                         upper=float(np.quantile(values,.9)),best=float(min(values))))
    return sorted(rows,key=lambda r:r['score'])


def baseline_rows(rows):
    return [r for r in rows if r['config']==run.case(**{k:r['config'][k] for k in
        ('n','seed','target','optimizer','coordinates','eta')})]


def paired_recipes(rows,root=None):
    """Rank recipes only when both selection seeds completed the same horizon."""
    groups=defaultdict(list)
    def recipe(config):
        c=dict(config);c.pop('seed')
        if 'origin' in c and root is not None:
            origin=dict(c['origin']);checkpoint=Path(origin.pop('checkpoint'));origin.pop('sha256')
            parent=json.loads((root/checkpoint.parent.name/'case.json').read_text())
            c['origin']=dict(origin,checkpoint=checkpoint.name,parent_recipe=recipe(parent))
        return c
    for row in rows:
        c=recipe(row['config'])
        groups[json.dumps(c,sort_keys=True)].append(row)
    paired=[]
    for config,group in groups.items():
        selected=[r for r in group if r['config']['seed'] in (0,1)]
        if {r['config']['seed'] for r in selected}!={0,1}:continue
        if len({r['step'] for r in selected})!=1:continue
        original=dict(next(r['config'] for r in selected if r['config']['seed']==0));original.pop('seed')
        paired.append(dict(config=original,score=float(np.mean([r['score'] for r in selected])),
                           members=selected,step=selected[0]['step']))
    return sorted(paired,key=lambda r:r['score'])


def winners(rows, count=1, coordinates=None):
    groups=defaultdict(list)
    for r in rows:
        c=r['config']
        if coordinates and c['coordinates'] not in coordinates: continue
        group=(c['n'],c['target'],c['optimizer'],c['coordinates'])
        if c['seed']==0 and len(groups[group])<count: groups[group].append(r)
    return [r for group in groups.values() for r in group]


def build(root,stage):
    if stage=='intervention_promote': return promote(root)
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


def promote(root):
    """Promote separate mechanisms and their paired controls before widening."""
    rows=rank(root,horizon=20000);selected=[];cases=[]
    directory=Path(__file__).parent/'manifests'
    def members(name):
        identities={run.key(c) for c in json.loads((directory/(name+'.json')).read_text())}
        return [r for r in rows if r['id'] in identities]
    ema=winners([r for r in members('ema_screen') if r['config']['ema_strength']>0
                 and not r['config']['ema_normalized'] and r['config']['ema_location']==1],2)
    selected.extend(ema)
    for r in ema:
        c=r['config'];cases.extend([c,dict(c,ema_strength=0.)])
        if c['optimizer']=='gd':
            cases.extend([dict(c,eta=c['eta']/(1+c['ema_strength'])),
                          dict(c,ema_strength=0.,eta=c['eta']*(1+c['ema_strength']))])
        else:
            cases.extend([dict(c,ema_normalized=True,epsilon=c['epsilon']/(1+c['ema_strength'])),
                          dict(c,ema_location=2)])
    initial=members('initialization_screen')
    for initialization in ('reference_xavier','individual_gaussian'):
        for slope in ('physical_xavier','lambda_xavier'):
            best=winners([r for r in initial if r['config']['initialization']==initialization
                          and r['config']['slope_initialization']==slope])
            selected.extend(best);cases.extend(r['config'] for r in best)
    for r in members('agreement_screen'):
        c=r['config']
        useful=c['optimizer']=='adam' or (c['coordinates']=='neighbor' and c['target']=='sine')
        if useful and c.get('agreement_threshold',.9)==.9:cases.append(c)
    # Combine the best screened initialization with the best gradient-memory recipe.
    by_group=lambda c:(c['optimizer'],c['coordinates'],c['target'])
    best_ema={by_group(r['config']):r['config'] for r in reversed(ema)}
    for r in winners(initial):
        c=r['config'];e=best_ema[by_group(c)]
        for factor in (.3,1.,3.):
            cases.append(dict(c,ema_alpha=e['ema_alpha'],ema_strength=e['ema_strength'],eta=c['eta']*factor))
    cases=[dict(c,seed=seed) for c in cases for seed in (0,1)]
    return list({run.key(c):c for c in cases}.values()),selected


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
