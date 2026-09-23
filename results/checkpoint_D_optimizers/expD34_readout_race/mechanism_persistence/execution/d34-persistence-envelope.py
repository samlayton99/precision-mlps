import csv,json
from pathlib import Path
import numpy as np
root=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/persistence_1bf7138')
rows=[]
for pred in sorted(root.glob('feedback_*')):
    if not (pred/'fork_diagnostics.csv').exists():continue
    pack=dict(np.load(pred/'inputs.npz')); fc=dict(np.load(pred/'forecasts.npz'))
    cases=json.loads(str(pack['cases']))
    with (pred/'fork_diagnostics.csv').open() as f:diags=list(csv.DictReader(f))
    w=(pack['p'].shape[-1]-1)//3
    natural=[i for i,c in enumerate(cases) if c['arm']=='natural']
    ratios={i:[] for i in natural}
    for path in sorted(pred.with_name(pred.name+'_run').joinpath('snapshots').glob('*.npz')):
        n=int(path.stem); hi=list(fc['horizons']).index(n)
        state=dict(np.load(path))
        for i in natural:ratios[i].append(float(state['sampled'][i,0]/fc['q'][i,hi]))
    final=dict(np.load(pred.with_name(pred.name+'_run')/'snapshots/000020000.npz'))
    for i in natural:
        c=cases[i]; q0=float(diags[i]['q']); k=float(diags[i]['k_actual_arm']); z=.002*k
        qtravel=.002*q0*(20000 if z==0 else np.expm1(20000*z)/np.expm1(z))
        initial_max=c['h']*np.max(np.abs(pack['p'][i,:w]))
        rem=final['tracking_travel'][i]
        gap=(.25-initial_max)/c['h']
        rows.append(dict(c,panel=pred.name,sampled_one_sided_multiplier=max(ratios[i]),
            q_reference_travel=float(qtravel),observed_tracking_travel=float(rem),
            initial_max_lambda=float(initial_max),
            candidate_multiplier=2.,candidate_full_travel=float(2*qtravel+rem),
            candidate_ever_fraction=float(min(1.,(2*qtravel+rem)**2/(w*gap*gap))) if gap>0 else 1.,
            single_neuron_multiplier_margin=float((gap-rem)/qtravel),
            actual_effective_travel=float(final['effective_travel'][i]),
            candidate_travel_covers_observed=bool(2*qtravel>=final['effective_travel'][i])))
out=root/'envelope_audit';out.mkdir()
with (out/'cases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
summary=dict(status='Retrospective diagnostic only; factor2 was selected after viewing outcomes; sampled speed ratios are not an all-step bound; the tracking travel is observed, not a proved allowance.',
    cases=len(rows),max_sampled_one_sided_multiplier=max(r['sampled_one_sided_multiplier'] for r in rows),
    multiplier2_covers_sampled=sum(r['sampled_one_sided_multiplier']<=2 for r in rows),
    multiplier2_covers_observed_force_travel=sum(r['candidate_travel_covers_observed'] for r in rows),
    minimum_single_neuron_multiplier_margin=min(r['single_neuron_multiplier_margin'] for r in rows),
    max_conditional_ever_fraction=max(r['candidate_ever_fraction'] for r in rows),
    interpretation='Candidate population allowances become theorems only after regional one-sided amplification, tracking, and first-exit hypotheses are proved.',
    groups=[dict(panel=panel,max_sampled_multiplier=max(r['sampled_one_sided_multiplier'] for r in rows if r['panel']==panel),minimum_single_neuron_multiplier_margin=min(r['single_neuron_multiplier_margin'] for r in rows if r['panel']==panel)) for panel in sorted({r['panel'] for r in rows})])
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
