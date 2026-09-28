"""Richardson combinations of the even-step Mhaskar approximation family.

The final object is still one flat affine tanh network, with all coefficients
assembled and stored at p bits; this is not a factored or higher-precision evaluator.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import json,time
import gmpy2 as g
import numpy as np
from scipy.linalg import norm
from experiments.expC12_mhaskar_comparison import pbit,robust
from experiments.expC12_mhaskar_comparison.construction import TanhNetwork,chirp,round_bits
from experiments.expC12_mhaskar_comparison.run import json_write,relative_l2


def coefficients(level,p):
    with g.context(precision=p,round=g.RoundToNearest):
        c=[g.mpfr(1)]
        for k in range(1,level+1):
            factor=g.mpfr(4)**k
            new=[g.mpfr(0) for _ in range(len(c)+1)]
            for j,a in enumerate(c):
                new[j]-=a/(factor-1)
                new[j+1]+=factor*a/(factor-1)
            c=new
    return np.array(c,dtype=float)


def construct(poly,taylor,degree,step,bias,p,level):
    scales=coefficients(level,p)
    pieces=[robust.construct(poly,taylor,degree,step/2.**j,bias,p) for j in range(level+1)]
    # Integer indices are exact bookkeeping; every parameter calculation is p-bit.
    locations=sorted({int(k)*2**(level-j) for j in range(level+1) for k in range(-degree,degree+1)})
    if len(locations)>1024:raise FloatingPointError('width budget')
    index={k:i for i,k in enumerate(locations)}
    contributions=np.zeros((len(locations),level+1))
    slopes=np.empty(len(locations))
    for j,model in enumerate(pieces):
        for k,(s,w) in enumerate(zip(model.slope,model.readout)):
            i=index[(k-degree)*2**(level-j)]
            contributions[i,j]=w;slopes[i]=s
    weights=robust.readout(contributions,scales,0.,p,'dot2')
    return TanhNetwork(slopes,np.full(len(slopes),bias),weights,0.)


def main():
    data=robust.OUT/'data'
    cfg=json.loads((data/'config.json').read_text())['config']
    xv=np.load(pbit.OUT/'data/grids.npz')['validation'];xs=xv[::8]
    xe=np.linspace(-1,1,8001)
    yv,ys,ye=chirp(xv),chirp(xs),chirp(xe)
    rows=[];start=time.monotonic()
    for p in [24,53]:
        saved=np.load(data/f'construction_p{p}.npz')
        poly,t,b=saved['monomial'],saved['taylor'],float(saved['bias'])
        best=None
        observations=[]
        for level in [1,2,3]:
            for d in cfg['degrees']:
                if len({k*2**(level-j) for j in range(level+1) for k in range(-d,d+1)})>1024:continue
                for h in np.geomspace(cfg['step_min'],cfg['step_max'],cfg['step_count']):
                    try:model=construct(poly[d],t,d,float(h),b,p,level)
                    except (FloatingPointError,ValueError):continue
                    coarse=robust.evaluate(model,xs,p)
                    bound=relative_l2(coarse,ys)*norm(ys)/norm(yv)
                    if best is not None and bound>best['validation_error']*(1+1e-12):continue
                    full=robust.evaluate(model,xv,p)
                    np.testing.assert_array_equal(full[::8],coarse)
                    error=relative_l2(full,yv)
                    observations.append([level,d,float(h),error])
                    if best is None or error<best['validation_error']:
                        best={'p':p,'level':level,'degree':d,'step':float(h),'width':model.width,
                              'validation_error':error,'model':model}
            print(f'Richardson p={p}, level {level} complete; {time.monotonic()-start:.0f}s',flush=True)
        model=best.pop('model')
        predicted=robust.evaluate(model,xe,p)
        best['error']=relative_l2(predicted,ye)
        model.save(robust.OUT/f'models/extrapolated_p{p}.npz')
        for a in [model.slope,model.bias,model.readout,model.offset,predicted]:
            np.testing.assert_array_equal(a,round_bits(a,p))
        rows.append(best)
        json_write(data/f'extrapolation_validation_p{p}.json',observations)
        json_write(data/'extrapolation.json',rows)
        print(best,flush=True)
    plot()


def plot():
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/precisionmlps-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((robust.OUT/'data/extrapolation.json').read_text())
    base=json.loads((pbit.OUT/'data/summary.json').read_text())
    rescue=json.loads((robust.OUT/'data/summary.json').read_text())
    p=[r['p'] for r in rows]
    fig,ax=plt.subplots(figsize=(7.4,4.9));fig.subplots_adjust(left=.13,right=.98,bottom=.16,top=.72)
    ax.plot(p,[next(r['mhaskar_error'] for r in base if r['p']==b) for b in p],'o-',label='Original')
    ax.plot(p,[next(r['rescued_error'] for r in rescue if r['p']==b) for b in p],'s-',label='Symmetric + compensated')
    ax.plot(p,[r['error'] for r in rows],'D-',label='Also cancel leading step errors')
    ax.set(xlim=(22,55),ylim=(.9,1.),xlabel='Working precision p (bits)',ylabel='Relative L2 error',title='Higher-order finite-difference rescue')
    ax.set_xticks(p);ax.grid(alpha=.2);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.13),frameon=False)
    fig.savefig(robust.OUT/'figures/extrapolation_check.png',dpi=240);plt.close(fig)


if __name__=='__main__':main()
