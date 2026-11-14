"""Raw inverse square root: fixed standard uniform QUILL halo recipe.

Sample-fitted tanh features on [1e-4,1], no exterior labels or extension,
no coordinate warp, adaptive centers, derivatives or parameter tuning.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import sys,json,hashlib,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import roots_legendre
from docs.qi_practical_guide.reference import fit_qi,MLP,default_halo

OUT=ROOT/'results/checkpoint_K_practical_recipes/expK07_raw_inverse_sqrt'
A,B=1e-4,1.
NS=[64,128,256,512,1024,2048,4096]

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def quadrature(order):
    z,w=roots_legendre(order);edges=np.geomspace(A,B,65)
    return (np.concatenate([(a+b)/2+(b-a)*z/2 for a,b in zip(edges[:-1],edges[1:])]),
            np.concatenate([(b-a)*w/2 for a,b in zip(edges[:-1],edges[1:])]))

@torch.no_grad()
def predict(model,x):
    w=torch.from_numpy(model.weight);b=torch.from_numpy(model.hidden_bias)
    v=torch.from_numpy(model.readout.T.copy());c=torch.from_numpy(model.output_bias)
    return torch.cat([torch.nn.functional.linear(torch.nn.functional.linear(z,w,b).tanh(),v,c)
                      for z in torch.from_numpy(x[:,None]).split(512)]).numpy().ravel()

def run(n):
    folder=OUT/f'N{n}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    x=np.linspace(A,B,max(2003,4*(n+1)));y=1/np.sqrt(x)
    start=time.monotonic()
    model,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=n,halo=default_halo(n),lam=.25,
                      origin=(A+B)/2,input_scale=(B-A)/2,solver='gelsy',rcond=np.finfo(float).eps)
    seconds=time.monotonic()-start;model.save(folder/'model.npz');save(folder/'diagnostics.json',info)
    np.savez(folder/'training_data.npz',x=x,y=y)
    errors={}
    for order in [64,128,256]:
        xx,ww=quadrature(order);truth=1/np.sqrt(xx)
        assert np.isclose(np.dot(ww,truth**2),np.log(B/A),rtol=3e-14)
        errors[str(order)]=float(np.sqrt(np.dot(ww,(predict(model,xx)-truth)**2)/np.log(B/A)))
    stable=abs(errors['128']-errors['256'])<=max(2e-13,1e-5*errors['256'])
    if not stable:
        xx,ww=quadrature(512);truth=1/np.sqrt(xx)
        errors['512']=float(np.sqrt(np.dot(ww,(predict(model,xx)-truth)**2)/np.log(B/A)))
        stable=abs(errors['256']-errors['512'])<=max(2e-13,1e-5*errors['512'])
    xx=np.unique(np.r_[np.linspace(A,B,16385),np.geomspace(A,B,16385)])
    truth=1/np.sqrt(xx);yp=predict(model,xx)
    np.savez(folder/'predictions.npz',x=xx,truth=truth,prediction=yp,residual=yp-truth)
    row=dict(intervals=n,width=len(model.weight),halo_per_side=default_halo(n),lambda_value=.25,
             solver='gelsy',rcond=float(np.finfo(float).eps),training_samples=len(x),rank=info['rank'],
             relative_l2=errors[str(max(map(int,errors)))],quadrature_errors=errors,quadrature_stable=stable,
             maximum_sampled_absolute_error=float(np.max(abs(yp-truth))),
             maximum_sampled_relative_error=float(np.max(abs(yp/truth-1))),
             left_endpoint_prediction=float(yp[0]),right_endpoint_prediction=float(yp[-1]),
             fit_seconds=seconds,readout_l1=info['readout_l1'][0],
             model=str((folder/'model.npz').relative_to(ROOT)))
    save(folder/'summary.json',row)
    print('N',n,'W',row['width'],'L2',row['relative_l2'],'fit_s',seconds,'stable',stable,flush=True)
    return row

def finish(rows):
    for i,r in enumerate(rows):
        r['effective_width_order']=None if i==0 else float(np.log(rows[i-1]['relative_l2']/r['relative_l2'])/np.log(r['width']/rows[i-1]['width']))
        r['digits_gained']=None if i==0 else float(np.log10(rows[i-1]['relative_l2']/r['relative_l2']))
    save(OUT/'summary.json',rows)
    fields=['intervals','width','halo_per_side','training_samples','relative_l2','effective_width_order','digits_gained','fit_seconds','quadrature_stable']
    with (OUT/'scaling.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'mathtext.fontset':'dejavusans'})
    fig,ax=plt.subplots(figsize=(6.3,4.2));fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.88)
    ax.plot([r['width'] for r in rows],[r['relative_l2'] for r in rows],color='#1f77b4',lw=2.6,
            marker='o',ms=5,mfc='white',mew=1.2)
    ax.set(yscale='log',xlabel='Hidden width $W$',ylabel=r'Relative $L^2$ error',xlim=(0,max(r['width'] for r in rows)*1.04))
    ax.set_title('Raw inverse-square-root scaling',fontsize=13,fontweight='bold',pad=10)
    ax.set_xticks([0,1000,2000,3000,4000],['0','1k','2k','3k','4k'])
    ax.grid(axis='y',color='#dddddd',lw=.45);ax.set_axisbelow(True)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'raw_inverse_sqrt_scaling.{ext}',dpi=350,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

if __name__=='__main__':
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol.json',dict(description=__doc__,domain=[A,B],origin=(A+B)/2,input_scale=(B-A)/2,
         intervals=NS,halo='max(20,ceil(sqrt(N))) per side',lambda_value=.25,solver='gelsy',rcond=float(np.finfo(float).eps),
         training_grid='uniform in physical x; max(2003,4*(N+1)) samples',
         evaluation='ordinary scalar Float64 Torch forward; physical-dx relative L2; log-panel Gauss integration',
         no_exterior_labels=True,no_extension=True,no_parameter_sweep=True,
         sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'docs/qi_practical_guide/reference.py']}))
    rows=[]
    for n in NS:rows.append(run(n));save(OUT/'summary.json',rows)
    finish(rows)
