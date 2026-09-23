import csv,json
from pathlib import Path
import numpy as np
root=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/persistence_1bf7138')
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def stats(rows,key):
    a=[float(r[key]) for r in rows if r.get(key) not in ('',None)]
    return dict(min=min(a),median=float(np.median(a)),max=max(a)) if a else None
path=read(root/'path_audit/path_diagnostics.csv')
result=dict(paths=[],late_sine=[],matching=[],physical=[],halving=[])
for n in (128,512,1024):
    g=[r for r in path if int(r['nref'])==n and r['arm']=='natural' and int(r['horizon'])==20000 and not r['panel'].endswith('late')]
    result['paths'].append(dict(nref=n,count=len(g),**{k:stats(g,k) for k in ('k_initial','k_arm','force_cos_initial','slope_force_cos_initial','slope_fraction','fine_residual_relative_change')}))
result['late_sine']=[r for r in path if r['target']=='sine' and r['arm']=='natural']
matching=[];paired=[];halving=[]
for directory in root.glob('matching_*'):
    matching+=read(directory/'matching.csv')
for directory in root.glob('physical_*_paired'):
    paired+=read(directory/'paired.csv')
for directory in root.glob('kicks_*'):
    for case in json.loads((directory/'diagnostics.json').read_text()):
        for key,values in case['halving_ratios'].items():
            for v in values:halving.append(dict(key=key,value=v))
for amplitude in (.01,.005,.0025):
    g=[r for r in matching if abs(float(r['amplitude']))==amplitude]
    result['matching'].append(dict(amplitude=amplitude,count=len(g),slopes_exact=sum(r['slopes_unchanged']=='True' for r in g),tracking_exceeds_force=sum(float(r['tracking_over_current_force'])>1 for r in g),**{k:stats(g,k) for k in ('slope_gradient_change_over_baseline_ga','force_norm_change_over_baseline_force','tracking_over_current_force','slope_tracking_over_current_slope_force')}))
    g=[r for r in paired if float(r['amplitude'])==amplitude and int(r['horizon'])==20000]
    for r in g:
        r['mean_sign_correct']=np.sign(float(r['actual_mean_lambda_response']))==np.sign(float(r['predicted_mean_lambda_response']))
        r['q_sign_correct']=np.sign(float(r['actual_q_change_response']))==np.sign(float(r['predicted_q_change_response']))
    result['physical'].append(dict(amplitude=amplitude,count=len(g),slope_error=stats(g,'slope_relative_error'),readout_error=stats(g,'readout_relative_error'),halving_slope=stats(g,'slope_halving_response_difference'),mean_sign_correct=sum(r['mean_sign_correct'] for r in g),q_sign_correct=sum(r['q_sign_correct'] for r in g)))
for key in sorted({r['key'] for r in halving}):
    result['halving'].append(dict(key=key,**stats([r for r in halving if r['key']==key],'value')))
def convert(o):
    if isinstance(o,np.generic):return o.item()
    raise TypeError(type(o).__name__)
(root/'retrospective.json').write_text(json.dumps(result,indent=2,default=convert))
print(json.dumps(result,indent=2,default=convert))
