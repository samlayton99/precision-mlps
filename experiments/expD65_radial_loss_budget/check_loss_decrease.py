"""One native GD update from each saved checkpoint; stable actual loss-decrease ratios."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import json,sys,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD65_radial_loss_budget.audit import SOURCES,OUT as PARENT
from experiments.expD56_scale_acquisition.model import gd_chunk
OUT=PARENT/'loss_decrease'

@torch.jit.script
def native_step(theta:torch.Tensor,x:torch.Tensor,y:torch.Tensor,lr:float):
    W=(theta.numel()-1)//3
    a,b,c,d=theta[:W],theta[W:2*W],theta[2*W:3*W],theta[-1]
    h=torch.tanh(a[:,None]*x[None,:]+b[:,None])
    e=torch.matmul(c,h)+d-y;scaled=e/x.numel()
    dh=(1-h*h)*c[:,None]*scaled[None,:]
    ga=torch.sum(dh*x[None,:],dim=1);gb=torch.sum(dh,dim=1);gc=torch.matmul(h,scaled);gd=torch.sum(scaled)
    g=torch.cat((ga,gb,gc,gd.reshape(1)))
    nxt=torch.cat((a-lr*ga,b-lr*gb,c-lr*gc,(d-lr*gd).reshape(1)))
    return nxt,g,h,e

def stable_drop(theta,nxt,h,e,x):
    W=(len(theta)-1)//3;delta=nxt-theta;dc=delta[2*W:3*W]
    du=x[:,None]*delta[:W]+delta[W:2*W]
    th=np.tanh(du)
    dh=(1-h*h)*th/(1+h*th)
    dq=h@dc+dh@nxt[2*W:3*W]+delta[-1]
    # Avoid subtracting two nearly identical O(1) losses.
    drop=math.fsum((-e*dq-.5*dq*dq).tolist())/len(x)
    return drop

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    rows=[];arrays={};worst_naive=[];started=time.monotonic();matched=0;max_match_error=0.
    for root in SOURCES:
        for path in sorted(root.glob('*/checkpoint.npz')):
            status=json.loads((path.parent/'status.json').read_text());lr=status['learning_rate']
            if lr not in [.001,.0001]:continue
            with np.load(path.parent/'initial.npz') as d:x=d['train_x'];y=d['train_y']
            tx=torch.from_numpy(x);ty=torch.from_numpy(y)
            with np.load(path) as d:
                steps=d['history'][:,d['keys'].tolist().index('step')];snap=d['snapshots'].copy()
            result=[]
            for j,n in enumerate(steps):
                if n>=200000:continue # Last probe is 199500 -> 199501, within the saved horizon.
                theta=snap[j];tt=torch.from_numpy(theta.copy());nxt,g,h,e=native_step(tt,tx,ty,lr)
                nn=nxt.numpy();gg=g.numpy();hh=h.numpy().T;ee=e.numpy();denom=lr*math.fsum((gg*gg).tolist())
                loss=.5*math.fsum((ee*ee).tolist())/len(x)
                drop=stable_drop(theta,nn,hh,ee,x)
                W=(len(theta)-1)//3
                next_e=np.tanh(x[:,None]*nn[:W]+nn[W:2*W])@nn[2*W:3*W]+nn[-1]-y
                next_loss=.5*math.fsum((next_e*next_e).tolist())/len(x)
                ratio=drop/denom if denom>0 else np.nan
                naive=(loss-next_loss)/denom if denom>0 else np.nan
                if n in [0,1]:
                    original,count,finite=gd_chunk(tt,tx,ty,lr,1)
                    np.testing.assert_array_equal(nn,original.numpy());assert count==1 and finite
                    k=int(np.flatnonzero(steps==n+1)[0]);err=float(np.max(np.abs(nn-snap[k])))
                    np.testing.assert_allclose(nn,snap[k],atol=2e-14,rtol=2e-14)
                    matched+=1;max_match_error=max(max_match_error,err)
                result.append([n,loss,denom,drop,ratio,naive])
            a=np.array(result);finite=np.isfinite(a[:,4]);assert finite.all(),path
            # A roundoff screen flags badly conditioned update/denominator cases for review, not as failures.
            naive_disagreement=np.abs(a[:,4]-a[:,5]);i=int(np.argmax(naive_disagreement))
            worst_naive.append({'case':path.parent.name,'step':int(a[i,0]),'stable_c':float(a[i,4]),'naive_c':float(a[i,5]),'difference':float(naive_disagreement[i]),'denominator':float(a[i,2])})
            post=a[:,0]>=30000
            row={'case':path.parent.name,'target':status['target'],'initialization':status['initialization_multiplier'],
                 'lr':lr,'checks':len(a),'minimum_c':float(a[:,4].min()),'maximum_c':float(a[:,4].max()),
                 'minimum_step':int(a[np.argmin(a[:,4]),0]),'minimum_c_after_30k':float(a[post,4].min()),
                 'failures_below_half':int(np.sum(a[:,4]<.5)),'min_denominator':float(a[:,2].min()),
                 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(row);arrays[path.parent.name]=a
            if len(rows)%24==0:print('Checked',len(rows),'runs;',round(time.monotonic()-started,1),'seconds',flush=True)
    assert len(rows)==144
    report={'scope':'24 functions x 3 initialization multipliers x 2 learning rates; native full-batch tanh GD, width 196, seed 30.',
            'method':'Independent one-step probes from saved checkpoints with the original simultaneous float64 GD arithmetic; no checkpoints overwritten and no continued training.',
            'loss_difference':'-mean(e*delta_q)-0.5*mean(delta_q^2), using the exact tanh addition identity for delta_q; avoids subtracting near-equal losses.',
            'checkpoint_spacing':'Early checkpoints 0,1,2,5,10,20,50,100,200; then every 500; checkpoints strictly below 200k.',
            'coverage_limitation':'Tests one adjacent GD step per saved checkpoint, not every intervening update.',
            'columns':['step','loss','eta_gradient_squared','stable_loss_drop','c','naive_c'],
            'runs':len(rows),'checks':sum(r['checks'] for r in rows),'failures_below_half':sum(r['failures_below_half'] for r in rows),
            'global_minimum_c':min(r['minimum_c'] for r in rows),'global_maximum_c':max(r['maximum_c'] for r in rows),
            'matched_saved_early_updates':matched,'max_saved_update_difference':max_match_error,
            'worst_naive_disagreements':sorted(worst_naive,key=lambda r:r['difference'],reverse=True)[:12],'cases':rows}
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');np.savez_compressed(OUT/'checks.npz',**arrays)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','worst_naive_disagreements']},indent=2))
    print('Worst naive differences:',report['worst_naive_disagreements'][:4])
if __name__=='__main__':main()
