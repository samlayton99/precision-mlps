"""Evaluate queued continuations and paired probes, preserving partial outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from . import adam_analyze as aa, adam_forces as af, plateau


def endpoint(p,target):
    row={};w=(len(p)-1)//3
    for m,label in ((2048,'train'),(8192,'eval')):
        x,y,_,_=af.data(target,m);(a,b,c),d=af.unpack(p)
        residual=np.tanh(x[:,None]*a+b) @ c+d-y
        row[label+'_relative_mse']=float(np.mean(residual**2)/np.mean(y**2))
    row.update(mean_gamma=float(np.mean(abs(p[:w]))),max_gamma=float(np.max(abs(p[:w]))))
    for cutoff in (1,4,16):row[f'fraction_gamma_ge_{cutoff}']=float(np.mean(abs(p[:w])>=cutoff))
    return row


def run(root,output):
    output.mkdir(parents=True,exist_ok=True);rows=[];predictions=[];inventory=[];derivatives=[]
    folders=[*sorted((root/'long').glob('*')), *sorted((root/'probes').glob('*')),
             *sorted((root/'confirm').glob('*')), *sorted((root/'confirm-probes').glob('*')),
             *sorted((root/'checks').glob('*'))]
    for folder in folders:
        if not (folder/'state.npz').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text());status=json.loads((folder/'status.json').read_text())
        inventory.append(dict(path=str(folder),status=status));f=np.load(folder/'state.npz');sn=np.load(folder/'snapshots.npz')
        probe='offset' in f;cursor=int(f['offset' if probe else 'cursor']);stage=folder.parent.name
        for i,c in enumerate(manifest['cases']):
            p=f['p'][i];p0=sn['p'][i,0];w=(len(p)-1)//3
            key=dict(stage=stage,bundle=folder.name,target=c['target'],seed=c['seed'],
                optimizer=c.get('optimizer','gd'),arm=c.get('arm','joint'),weight=c.get('weight',1.),
                fork=c.get('start',manifest.get('initial_step',0)),cursor=cursor,
                complete=bool(status['complete']),failed=bool(f['failed'][i]))
            row=dict(key);row.update(endpoint(p,c['target']))
            row.update(gamma_change=float(np.mean(abs(p[:w])-abs(p0[:w]))),
                slope_displacement=float(np.linalg.norm(p[:w]-p0[:w])),
                readout_displacement=float(np.linalg.norm(p[2*w:]-p0[2*w:])))
            if probe:
                signed=f['signed'][i];crossing=f['crossing'][i]
                row['force_coherence']=float(np.linalg.norm(f['force_sum'][i])/max(float(f['force_norm_sum'][i]),1e-300))
                row['slope_path']=float(f['path'][i]);unresolved=int(f['unresolved'][i])
            else:
                signed=f['signed_channels'][i]-sn['signed_channels'][i,0]
                crossing=f['crossing'][i]-sn['crossing'][i,0]
                row['slope_path']=float(f['path'][i]-sn['path'][i,0]);unresolved=int(f['unresolved_steps'][i])
            for k,name in enumerate(af.CHANNELS):row['signed_'+name]=float(signed[k])
            row['crossing']=float(crossing);row['unresolved_steps']=unresolved
            row['motion_identity']=abs(row['gamma_change']-sum(signed)-float(crossing))
            x,y,_,_=af.data(c['target']);xx=jnp.asarray(x);yy=jnp.asarray(y)
            # Derivative claims here apply only to the ordinary optimizer.
            if not probe:
                settings=jnp.array([c[k] for k in ('eta','beta1','beta2','epsilon','adaptive')])
                d=jax.device_get(plateau.diagnostics(*[jnp.asarray(f[k][i]) for k in ('p','m','v','channel_m','count')],yy,xx,settings))
                dr=dict(key)|{k:float(v) for k,v in d.items() if np.ndim(v)==0}
                for k,name in enumerate(plateau.DRIVERS):dr['rate_'+name]=float(d['log_rates'][k])
                derivatives.append(dr)
            g,r,jc,ec=af.field(jnp.asarray(p),xx,yy);channels,_=af.split(g,jc,ec)
            row['reference_effective_norm']=float(jnp.linalg.norm(channels[0,:w]))
            row['reference_tracking_norm']=float(jnp.linalg.norm(channels[1,:w]))
            g0,r0,jc0,ec0=af.field(jnp.asarray(p0),xx,yy);ch0,_=af.split(g0,jc0,ec0)
            row['effective_force_ratio']=row['reference_effective_norm']/max(float(jnp.linalg.norm(ch0[0,:w])),1e-300)
            row['effective_force_cosine']=float(channels[0,:w] @ ch0[0,:w])/max(row['reference_effective_norm']*float(jnp.linalg.norm(ch0[0,:w])),1e-300)
            rows.append(row)
        if stage=='confirm':
            for path in sorted(folder.glob('forecast_*.npz')):
                forecast=np.load(path);end=int(forecast['end_step']);issued=int(forecast['issued_step'])
                if end not in sn['steps']:continue
                si=int(np.flatnonzero(sn['steps']==end)[0])
                for j,i in enumerate(forecast['indices']):
                    c=manifest['cases'][i];p=sn['p'][i,si];x,y,_,_=af.data(c['target'])
                    r=plateau.prediction(jnp.asarray(p),jnp.asarray(x))-y
                    actual=np.asarray(plateau.effective(jnp.asarray(p),r,jnp.asarray(x)))[:177]
                    for method in ('frozen','constant'):
                        pred=forecast[method+'_force'][j]
                        a=forecast['frozen_p'][j,:177] if method=='frozen' else forecast['constant_a'][j]
                        predictions.append(dict(target=c['target'],seed=c['seed'],issued=issued,end=end,method=method,
                            relative_vector_error=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),
                            predicted_norm=float(np.linalg.norm(pred)),actual_norm=float(np.linalg.norm(actual)),
                            predicted_mean_gamma=float(np.mean(abs(a))),actual_mean_gamma=float(np.mean(abs(p[:177]))),
                            slope_vector_error=float(np.linalg.norm(a-p[:177]))))
    contrasts=[]
    for r in rows:
        if r['stage'] not in ('probes','confirm-probes','checks'):continue
        base=next((v for v in rows if v['stage']==('probes' if r['stage']=='checks' else r['stage'])
            and v['seed']==r['seed'] and v['target']==r['target'] and v['fork']==r['fork'] and
            v['arm']==r['arm'] and v['weight']==r['weight']),None) if r['stage']=='checks' else next(
            (v for v in rows if v['bundle']==r['bundle'] and v['stage']==r['stage'] and v['target']==r['target'] and v['arm']=='joint'),None)
        if base is None or base['cursor']!=r['cursor']:continue
        contrasts.append({k:r[k] for k in ('stage','bundle','target','seed','fork','arm','weight','cursor','complete','failed')}|
            {k+'_difference':r[k]-base[k] for k in ('gamma_change','eval_relative_mse','reference_effective_norm','slope_displacement')})
    for name,data in [('endpoints',rows),('derivatives',derivatives),('forecasts',predictions),('contrasts',contrasts)]:
        if data:aa.write_csv(output/f'{name}.csv',data)
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(bundles=len(inventory),cases=len(rows),forecasts=len(predictions))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.root,args.output)
