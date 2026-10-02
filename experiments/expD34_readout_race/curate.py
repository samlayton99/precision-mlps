"""Reduce completed analysis tables into reviewable evidence; never write prose."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .analyze import table
from .run import write_json

PLOT_COLUMNS=np.array([0,1,3,4,5,9,11,12,16,17,25,28,31,32])


def number(row,key):
    value=row.get(key,'')
    return float(value) if value else float('nan')


def reduce_table(paths,name,keys,metrics):
    groups={}
    for path in paths:
        source=path/name
        stream=source.open() if source.exists() else gzip.open(str(source)+'.gz','rt')
        with stream:
            for row in csv.DictReader(stream):
                key=tuple(row[k] for k in keys)
                group=groups.setdefault(key,dict(**{k:row[k] for k in keys},count=0))
                group['count']+=1
                if int(row['step'])>=group.get('last_step',-1):
                    group['last_step']=int(row['step'])
                    for metric in metrics: group['last_'+metric]=number(row,metric)
                for metric in metrics:
                    value=number(row,metric)
                    if np.isfinite(value): group['max_abs_'+metric]=max(abs(value),group.get('max_abs_'+metric,0.))
                if name=='reference_metrics.csv' and number(row,'gradient_relative_error')>.05:
                    group.setdefault('first_gradient_error_above_5pct',int(row['step']))
                if name=='sample_probes.csv':
                    for order in (2,4,6,8):
                        error=number(row,f'pointwise_p{order}_error');bound=number(row,f'pointwise_p{order}_bound')
                        if np.isfinite(error) and np.isfinite(bound):
                            label=f'pointwise_p{order}'
                            group[label+'_checks']=group.get(label+'_checks',0)+1
                            group[label+'_violations']=group.get(label+'_violations',0)+int(error>bound+1e-13*max(1.,bound))
                            if bound>0: group[label+'_max_error_bound_ratio']=max(error/bound,group.get(label+'_max_error_bound_ratio',0.))
    return list(groups.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads((path/'analysis_manifest.json').read_text()) for path in args.inputs]
    assert len({m['end'] for m in manifests})==1,'Do not mix endpoint horizons'
    rows=[]
    for path in args.inputs:
        rows.extend(json.loads((path/'summary.json').read_text()))
        for source in path.glob('*_curves.npz'):
            with np.load(source) as data:
                arrays={k:data[k] for k in data.files}
            for key in arrays:
                if key.endswith('_trace'): arrays[key]=arrays[key][:,:,PLOT_COLUMNS]
            np.savez_compressed(out/source.name,plot_trace_columns=PLOT_COLUMNS,**arrays)
    assert len({(r['bundle'],r['degree'],r['target'],r['kappa']) for r in rows})==len(rows)
    write_json(out/'summary.json',rows);table(out/'summary.csv',rows)
    keys=('bundle','target','seed','n','degree','kappa')
    reference_metrics=('gradient_absolute_error','gradient_relative_error','signed_change_actual',
        'signed_change_reference','signed_change_absolute_error','slope_parameter_l2_error',
        'signed_force_actual','signed_force_reference','residual_moment_l2_error','median_gamma_error')
    table(out/'reference_audit.csv',reduce_table(args.inputs,'reference_metrics.csv',keys,reference_metrics))
    table(out/'matched_target_audit.csv',reduce_table(args.inputs,'matched_target_metrics.csv',
        ('bundle','n','seed','degree','left','right','kappa'),
        ('signed_mean_gamma_difference','slope_gradient_difference','slope_parameter_difference','coarse_residual_difference')))
    table(out/'probe_audit.csv',reduce_table(args.inputs,'sample_probes.csv',keys,
        ('loss_discrepancy','sample_gradient_error','fraction_preactivation_above_radius',
         'coarse_discrete_derivative_error','tanh_at_reference_slope_gradient_defect')))
    probes=[];contrasts=[]
    for path in args.inputs:
        with (path/'sample_probes.csv').open() as stream:
            probes.extend(r for r in csv.DictReader(stream) if r['seed']=='0' and r['n']=='128' and r['degree']=='0')
        with (path/'rate_contrasts.csv').open() as stream: contrasts.extend(csv.DictReader(stream))
    table(out/'sample_probes.csv',probes);table(out/'rate_contrasts.csv',contrasts)
    write_json(out/'evidence_manifest.json',dict(inputs=[str(p) for p in args.inputs],analyses=manifests,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='evidence_manifest.json'},
        audit_scope='All analyzed rows; max_abs fields are descriptive maxima; 5% is a post-observation summary threshold',
        curve_scope='Only columns identified by plot_trace_columns are included; all full traces remain at source',
        probe_scope='Full sample probe curves retained for width 177 seed 0; other probes summarized and retained at source'))
    print(json.dumps(dict(rows=len(rows),output=str(out))),flush=True)


if __name__=='__main__':main()
