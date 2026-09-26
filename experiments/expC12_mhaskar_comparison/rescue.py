"""Measure summation rescues and compensated/symmetric Mhaskar construction."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/precisionmlps-mpl")
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.linalg import norm
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC12_mhaskar_comparison import pbit,robust
from experiments.expC12_mhaskar_comparison.construction import TanhNetwork,chirp,round_bits
from experiments.expC12_mhaskar_comparison.run import relative_l2,json_write


def load_model(path):
    a=np.load(path)
    return TanhNetwork(a['slope'],a['bias'],a['readout'],a['offset'])


def main():
    out=robust.OUT
    data=out/'data';data.mkdir(parents=True,exist_ok=True)
    models=out/'models';models.mkdir(exist_ok=True)
    base=json.loads((pbit.OUT/'data/summary.json').read_text())
    cfg=json.loads((pbit.OUT/'data/config.json').read_text())['config']
    grids=np.load(pbit.OUT/'data/grids.npz')
    xe,xv=grids['evaluation'],grids['validation']
    ye,yv=chirp(xe),chirp(xv)
    xs=xv[::8];ys=chirp(xs)
    src=[Path(__file__),Path(__file__).with_name('robust.py'),Path(__file__).with_name('robust_kernels.c'),
         Path(__file__).with_name('pbit_kernels.c'),pbit.OUT/'data/config.json',pbit.OUT/'data/summary.json']
    manifest={'config':cfg,'search_precisions':[8,16,24,32,40,48,53],
              'arithmetic':'Every primitive, including correction registers, at p bits; stored model remains one p-bit value per parameter; no FMA',
              'sources':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in src},
              'references':['https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf'],
              'selection':'Same full validation grid with certified partial-residual pruning; reporting grid excluded'}
    json_write(data/'config.json',manifest)
    start=time.monotonic()
    reevaluated=[]
    for b in base:
        p=b['p'];model=load_model(pbit.OUT/f'models/mhaskar_p{p}.npz')
        phi=pbit.features(xe,model.slope,model.bias,p)
        errors={mode:relative_l2(robust.readout(phi,model.readout,model.offset,p,mode),ye) for mode in robust.MODES}
        np.testing.assert_allclose(errors['sequential'],b['mhaskar_error'],rtol=1e-14)
        reevaluated.append({'p':p,**errors})
    json_write(data/'readout_only.json',reevaluated)
    print(f"All six readout algorithms tested on 46 fixed models; {time.monotonic()-start:.0f}s",flush=True)
    nodes=np.cos(np.pi*(np.arange(cfg['chebyshev_points'])+.5)/cfg['chebyshev_points'])
    labels=chirp(nodes)
    degrees=np.array(cfg['degrees'])
    steps=np.geomspace(cfg['step_min'],cfg['step_max'],cfg['step_count'])
    results=[]
    last_update=time.monotonic()
    for p in manifest['search_precisions']:
        c,poly,t,bias=robust.polynomial_data(nodes,labels,int(degrees[-1]),p)
        np.savez(data/f'construction_p{p}.npz',chebyshev=c,monomial=poly,taylor=t,bias=bias)
        val=np.full((len(steps),len(degrees)),np.nan)
        screen=np.full_like(val,np.inf);bound=np.full_like(val,np.inf)
        pruned=np.zeros_like(val,dtype=bool);valid=np.zeros_like(val,dtype=bool)
        best=None
        for hi,h in enumerate(steps):
            candidates=[]
            for di,d in enumerate(degrees):
                try:model=robust.construct(poly[d],t,int(d),float(h),bias,p)
                except FloatingPointError:continue
                valid[hi,di]=True;candidates.append((di,int(d),model))
            if not candidates:continue
            largest=max(candidates,key=lambda a:a[1])
            common=pbit.features(xs,largest[2].slope,largest[2].bias,p)
            for di,d,model in candidates:
                offset=largest[1]-d
                phi=np.ascontiguousarray(common[:,offset:offset+model.width])
                pred=robust.readout(phi,model.readout,model.offset,p,'dot2')
                screen[hi,di]=relative_l2(pred,ys)
                bound[hi,di]=screen[hi,di]*norm(ys)/norm(yv)
                if best is not None and bound[hi,di]>best['error']*(1+1e-12):
                    pruned[hi,di]=True;continue
                full=robust.evaluate(model,xv,p)
                np.testing.assert_array_equal(full[::8],pred)
                error=relative_l2(full,yv);val[hi,di]=error
                if best is None or error<best['error']:
                    best={'error':error,'model':model,'degree':d,'step':float(h)}
            if time.monotonic()-last_update>20:
                print(f"Rescued p={p}, steps {hi+1}/{len(steps)}; elapsed {time.monotonic()-start:.0f}s",flush=True)
                last_update=time.monotonic()
        assert np.all(bound[pruned]>best['error'])
        np.savez_compressed(data/f'search_p{p}.npz',degrees=degrees,steps=steps,validation_error=val,
                            screen_error=screen,lower_bound=bound,pruned=pruned,finite_model=valid)
        model=best['model'];model.save(models/f'mhaskar_p{p}.npz')
        prediction=robust.evaluate(model,xe,p)
        np.testing.assert_array_equal(prediction,robust.evaluate(load_model(models/f'mhaskar_p{p}.npz'),xe,p))
        for a in (model.slope,model.bias,model.readout,model.offset,prediction):
            np.testing.assert_array_equal(a,round_bits(a,p))
        old=next(r for r in base if r['p']==p)
        phi=pbit.features(xe,model.slope,model.bias,p)
        per_mode={mode:relative_l2(robust.readout(phi,model.readout,model.offset,p,mode),ye) for mode in robust.MODES}
        # Equal access to Dot2 for QUILL, with its fixed saved model.
        quill=load_model(pbit.OUT/f'models/quill_p{p}.npz')
        quill_dot2=relative_l2(robust.evaluate(quill,xe,p),ye)
        row={'p':p,'baseline_error':old['mhaskar_error'],'rescued_error':relative_l2(prediction,ye),
             'validation_error':best['error'],'degree':best['degree'],'step':best['step'],'width':model.width,
             'readout_l1':float(norm(model.readout,1)),'rescued_model_readout_variants':per_mode,
             'quill_baseline':old['quill_error'],'quill_dot2':quill_dot2,
             'improvement_factor':old['mhaskar_error']/relative_l2(prediction,ye)}
        results.append(row);json_write(data/'summary.json',results)
        print(f"p={p}: old={row['baseline_error']:.6g}, rescued={row['rescued_error']:.6g}, degree={row['degree']}; {time.monotonic()-start:.0f}s",flush=True)
    json_write(data/'validation.json',{'complete':True,'fixed_models':46,'readout_methods':list(robust.MODES),
                                      'construction_search_precisions':manifest['search_precisions'],
                                      'all_saved_models_replayed_exactly':True,'all_parameters_outputs_pbit':True,
                                      'pruning_certified':True,'elapsed_seconds':time.monotonic()-start})
    plot()


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=robust.OUT/'data';figures=robust.OUT/'figures';figures.mkdir(exist_ok=True)
    readout=json.loads((data/'readout_only.json').read_text())
    results=json.loads((data/'summary.json').read_text())
    old=json.loads((pbit.OUT/'data/summary.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.labelsize':13,'axes.titlesize':15})
    colors=plt.get_cmap('viridis')
    fig,axes=plt.subplots(1,3,figsize=(17,5.4))
    fig.subplots_adjust(left=.06,right=.985,bottom=.16,top=.73,wspace=.3)
    for mode,color in zip(robust.MODES,colors(np.linspace(.05,.9,len(robust.MODES)))):
        axes[0].plot([r['p'] for r in readout],[r[mode] for r in readout],lw=1.7,color=color,label=mode.replace('_',' '))
    axes[0].set(title='Readout changes only',ylim=(.94,1.02))
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.13),ncol=3,fontsize=9,frameon=False)
    axes[1].plot([r['p'] for r in old],[r['mhaskar_error'] for r in old],color=colors(.12),lw=2,label='Original construction')
    axes[1].plot([r['p'] for r in results],[r['rescued_error'] for r in results],'o-',color=colors(.65),lw=2,label='Symmetric + compensated')
    axes[1].set(title='Construction and readout changes',ylim=(.94,1.02))
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.13),fontsize=10,frameon=False)
    axes[2].semilogy([r['p'] for r in old],[r['quill_error'] for r in old],color=colors(.45),lw=2,label='QUILL: sequential')
    axes[2].semilogy([r['p'] for r in results],[r['quill_dot2'] for r in results],'o--',color=colors(.7),lw=2,label='QUILL: compensated dot')
    axes[2].semilogy([r['p'] for r in results],[r['rescued_error'] for r in results],'s-',color=colors(.12),lw=2,label='Mhaskar: symmetric + compensated')
    axes[2].set(title='Precision comparison',ylim=(1e-16,10))
    axes[2].legend(loc='lower center',bbox_to_anchor=(.5,1.13),fontsize=9,frameon=False)
    for ax in axes:
        ax.set(xlim=(8,53),xlabel=r'Working precision $p$ (bits)',ylabel=r'Relative $L^2$ error')
        ax.set_xticks([8,16,24,32,40,48,53]);ax.grid(alpha=.2)
    fig.savefig(figures/'numerical_rescue_comparison.png',dpi=240)
    plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
