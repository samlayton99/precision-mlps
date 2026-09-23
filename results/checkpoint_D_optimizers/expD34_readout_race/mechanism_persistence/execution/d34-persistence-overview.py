import csv, json
from pathlib import Path
import numpy as np
base=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/persistence_1bf7138')
def read(name):
    with (base/'feedback_summary'/name).open() as f:
        return list(csv.DictReader(f))
def val(row,key):
    return float(row[key])
def stats(rows,key):
    a=[val(r,key) for r in rows if r.get(key) not in ('',None)]
    return {'min':min(a),'median':float(np.median(a)),'max':max(a)} if a else None
rows=read('natural_20k.csv'); contrasts=read('contrasts_20k.csv'); audits=read('checkpoint_audits.csv')
result={'natural':[],'contrasts':[],'examples':[],'audits':[]}
for n in (128,512,1024):
    group=[r for r in rows if val(r,'nref')==n and not r['panel'].endswith('late')]
    result['natural'].append(dict(nref=n,count=len(group),positive_k=sum(val(r,'k0')>0 for r in group),
        scalar_beats_constant=sum(r['slope_scalar_beats_constant']=='True' for r in group),
        **{key:stats(group,key) for key in ('q_actual_over_q0','q_forecast_relative_error','constant_relative_error','amplification_relative_error','D_over_q_squared','C_over_q_squared','loaded_over_q_squared','mean_lambda_change','mean_positive_travel')}))
    for arm in ('projected_constant','no_geometry','double_geometry','no_relaxation','double_relaxation'):
        gg=[r for r in contrasts if val(r,'nref')==n and not r['panel'].endswith('late') and r['arm']==arm]
        valid=[r for r in gg if r['observed_mean_resolved']=='True']
        result['contrasts'].append(dict(nref=n,arm=arm,count=len(gg),resolved=len(valid),
          correct=sum(r['resolved_sign_correct']=='True' for r in valid),
          skill=stats(gg,'resolved_skill'),q_ratio=stats(gg,'q_ratio_to_natural'),positive_travel_ratio=stats(gg,'positive_ratio_to_natural')))
for r in rows:
    result['examples'].append({k:r[k] for k in ('target','seed','nref','panel','q0','k0','D_over_q_squared','C_over_q_squared','C_generated_over_q_squared','C_target_over_q_squared','loaded_over_q_squared','q_actual_over_q0','q_forecast_relative_error','constant_relative_error','amplification_relative_error','mean_lambda_change','mean_positive_travel','newly_ever_hit')})
for panel in ('audit_development','audit_confirmation'):
    g=[r for r in audits if r['audit_panel']==panel]
    result['audits'].append(dict(panel=panel,count=len(g),positive_k=sum(val(r,'k')>0 for r in g),
        **{key:stats(g,key) for key in ('k','D_over_q_squared','C_over_q_squared','loaded_over_q_squared')}))
result['all_new_hits']=sum(val(r,'newly_ever_hit') for r in read('states_20k.csv'))
result['contrast_resolved']=sum(r['observed_mean_resolved']=='True' for r in contrasts)
result['contrast_correct']=sum(r['resolved_sign_correct']=='True' for r in contrasts)
result['contrast_positive_skill']=sum(float(r['resolved_skill'])>0 for r in contrasts if r['resolved_skill'])
(base/'overview.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
