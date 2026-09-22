"""Resolve force-vector oscillations hidden by scalar norms in saved trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from . import adam_analyze as aa, adam_forces as af, adam_run as ar, targets


def kernel(x, modes, length):
    def one(state, y, settings):
        def step(old, _):
            new, row=ar.one_step(old,x,y,modes,settings)
            g,_,jc,ec=af.field(old['p'],x,y); channels,_=af.split(g,jc,ec)
            eta,b1,b2,eps,adaptive=settings
            inverse=jnp.where(adaptive,1/(jnp.sqrt(new['v']/(1-b2**new['count']))+eps),1.)
            steps=-eta*inverse*new['channel_m']/(1-b1**new['count'])
            w=(len(g)-1)//3
            vectors=jnp.stack((channels[0,:w],channels[1,:w],steps[0,:w],steps[1,:w]))
            return new,(vectors,row)
        return jax.lax.scan(step,state,None,length=length)
    return jax.jit(jax.vmap(one,in_axes=(0,0,0)))


def run(source, output, length=512, seed_limit=5):
    output.mkdir(parents=True,exist_ok=True)
    x=targets.grid(2048); modes=np.polynomial.legendre.legvander(x,9) @ targets.polynomial_map(x)
    advance=kernel(jnp.asarray(x),jnp.asarray(modes),length)
    summaries=[]; samples=[]; checks=[]; hashes={}
    names=('raw_effective','raw_tracking','step_effective','step_tracking')
    for seed in range(seed_limit):
        folder=source/f'primary_{seed}'; cases=json.loads((folder/'manifest.json').read_text())['cases']
        f=np.load(folder/'snapshots.npz')
        hashes[str(folder/'snapshots.npz')]=hashlib.sha256((folder/'snapshots.npz').read_bytes()).hexdigest()
        yy=jnp.asarray(np.stack([af.data(c['target'])[1] for c in cases]))
        settings=jnp.asarray([[c[k] for k in ('eta','beta1','beta2','epsilon','adaptive')] for c in cases])
        for start in (100000,600000):
            si=int(np.flatnonzero(f['steps']==start)[0])
            state={k:jnp.asarray(f[k][:,si]) for k in f.files if k!='steps'}
            final,(vectors,rows)=jax.device_get(advance(state,yy,settings))
            if np.any(final['failed']): raise ValueError('Nonfinite dense continuation')
            for i,c in enumerate(cases):
                for k,name in enumerate(names):
                    v=vectors[i,:,k]; norm=np.linalg.norm(v,axis=1)
                    cos=lambda lag:np.sum(v[lag:]*v[:-lag],axis=1)/(norm[lag:]*norm[:-lag])
                    summaries.append(dict(target=c['target'],optimizer=c['optimizer'],seed=seed,start=start,
                        updates=length,channel=name,norm_mean=norm.mean(),norm_cv=norm.std()/norm.mean(),
                        norm_min=norm.min(),norm_max=norm.max(),lag1_cosine=np.median(cos(1)),lag2_cosine=np.median(cos(2)),
                        vector_coherence=np.linalg.norm(v.sum(axis=0))/norm.sum(),
                        even_odd_mean_ratio=np.linalg.norm(v[::2].mean(axis=0)-v[1::2].mean(axis=0))/norm.mean(),
                        mean_gamma_change=np.mean(abs(final['p'][i,:177])-abs(np.asarray(state['p'])[i,:177]))))
                # Retain every scalar sample; vectors are reduced to exact window statistics.
                for j,row in enumerate(rows[i]):
                    samples.append(dict(target=c['target'],optimizer=c['optimizer'],seed=seed,start=start,offset=j,
                        **{key:float(row[ar.METRICS.index(key)]) for key in ('raw_effective_norm','raw_tracking_norm',
                            'step_effective_norm','step_tracking_norm','actual_gamma_increment')}))
            checks.append(dict(seed=seed,start=start,maximum_identity_error=float(final['identity_max'].max()),
                               failed=int(np.count_nonzero(final['failed'])),unresolved=int(np.max(final['unresolved_steps']))))
            print(json.dumps(checks[-1]),flush=True)
    aa.write_csv(output/'windows.csv',summaries); aa.write_csv(output/'samples.csv.gz',samples)
    (output/'audit.json').write_text(json.dumps(dict(checks=checks,input_hashes=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        length=length,description='Unchanged actual optimizer, contiguous updates, complete moment state preserved'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,default=512);p.add_argument('--seed-limit',type=int,default=5)
    args=p.parse_args();run(args.source,args.output,args.length,args.seed_limit)
