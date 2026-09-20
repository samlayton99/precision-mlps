"""Offline phase-dynamics predictions from saved initial states, with no fitted rates.

Early: freeze the Jacobian projected onto constant/linear residual modes.
Fitted: exact fixed-geometry readout GD, then evaluate instantaneous slope probes.
Neither model consumes future states; saved trajectories are validation only.
"""
from pathlib import Path
import sys,json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import nudge_correlation as n
cfg=n.late.config();eta=cfg['learning_rate'];torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
x=n.base.original.midpoint_grid(cfg['n_train']);y=n.base.original.matched.target_values('runge',x,cfg);nt=len(x);m=177
source=np.load(n.RESULTS/'data/nudge_correlation_5000.npz')
selected=source['early__selected_neurons']
def first_state(name):
 if name=='early': initial=n.base.original.initial_state('xavier',cfg)
 else:
  with np.load(n.RESULTS/'data/late_freeze_runge.npz') as s:initial={k:s['warmup__'+k][-1] for k in n.KEYS}
 state=n.independent_fork(initial,torch.tensor(x),torch.tensor(y),eta,set())
 np.testing.assert_array_equal(abs(state['a'][selected]),source[name+'__gamma'][0])
 np.testing.assert_array_equal(abs(state['v'][selected]),source[name+'__coefficient'][0])
 return state
def metrics(pred,actual):
 return {'relative_l2':float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),'max_abs':float(np.max(abs(pred-actual)))}
with threadpool_limits(limits=2):
 initial=first_state('early');a,b,v=(initial[k] for k in n.KEYS);c=v[:-1]
 hidden=np.tanh(x[:,None]*a+b);q=1-hidden**2;res=hidden@c+v[-1]-y
 J=np.column_stack((q*c*x[:,None],q*c,hidden,np.ones(nt)))
 U=np.column_stack((np.ones(nt),x/np.sqrt(np.mean(x*x))))
 B=U.T@J/nt;K=B@B.T;rho=U.T@res/nt
 ev,R=np.linalg.eigh(K);logs=np.log1p(-eta*ev);t=np.arange(5001)
 modes=np.exp(t[:,None]*logs)*(R.T@rho);rho_t=modes@R.T
 g=rho_t@B
 cumul=(-np.expm1(t[:,None]*logs)/ev)*(R.T@rho)
 theta=np.concatenate((a,b,v))[None,:]-(cumul@R.T)@B
 predg=abs(theta[:5000,selected]-eta*g[:5000,selected])-abs(theta[:5000,selected])
 predc=abs(theta[:5000,2*m+selected]-eta*g[:5000,2*m+selected])-abs(theta[:5000,2*m+selected])
 out={'early':{'K':K.tolist(),'rates':ev.tolist(),'efold_steps':(-1/logs).tolist()}}
 for T in (200,500,1000,2000,5000):
  out['early'][T]={'gamma':metrics(predg[:T],source['early__delta_gamma'][:T]),'c':metrics(predc[:T],source['early__delta_coefficient'][:T]),'moments':metrics(rho_t[:T],np.column_stack((source['early__residual_mean'],source['early__residual_first_moment']/np.sqrt(np.mean(x*x))))[:T])}
 out['early']['last_1000']={'gamma':metrics(predg[-1000:],source['early__delta_gamma'][-1000:]),'c':metrics(predc[-1000:],source['early__delta_coefficient'][-1000:])}
 early_pred=(predg,predc,rho_t)
 initial=first_state('fitted');a,b,v=(initial[k] for k in n.KEYS);hidden=np.tanh(x[:,None]*a+b);q=1-hidden**2
 A=np.column_stack((hidden,np.ones(nt)))/np.sqrt(nt);r0=A@v-y/np.sqrt(nt)
 u,s,vh=np.linalg.svd(A,full_matrices=False);alpha=u.T@r0
 logs=np.log1p(-eta*s*s);rates=np.exp(t[:,None]*logs);dm=np.expm1(t[:,None]*logs)*alpha
 drift=-np.expm1(t[:,None]*logs)*alpha/s
 v_t=v[None,:]-drift@vh
 T=(x[:,None]*q)/np.sqrt(nt);pair0=T.T@r0;pair_t=pair0[None,:]+dm@(u.T@T)
 predg=abs(a[None,selected]-eta*v_t[:5000,selected]*pair_t[:5000,selected])-abs(a[None,selected])
 predc=abs(v_t[1:,selected])-abs(v_t[:-1,selected])
 out['fitted']={'max_eta_curvature':float(eta*s[0]**2)}
 for tcheck in (0,1,10):
  vv=v_t[tcheck];rr=A@vv-y/np.sqrt(nt)
  np.testing.assert_allclose(v_t[tcheck+1],vv-eta*(A.T@rr),rtol=1e-12,atol=1e-15)
  np.testing.assert_allclose(pair_t[tcheck],T.T@rr,rtol=1e-7,atol=1e-15)
 for TT in (200,1000,5000):
  out['fitted'][TT]={'gamma':metrics(predg[:TT],source['fitted__delta_gamma'][:TT]),'c':metrics(predc[:TT],source['fitted__delta_coefficient'][:TT])}
 print(json.dumps(out,indent=2))
 np.savez_compressed(n.RESULTS/'data/nudge_flow_prediction.npz',early_g=early_pred[0],early_c=early_pred[1],early_moments=early_pred[2],fitted_g=predg,fitted_c=predc)
 (n.RESULTS/'data/nudge_flow_prediction.json').write_text(json.dumps(out,indent=2)+'\n')

# Compare the fitted-run phase plots using identical axes and time colors.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
fig,axes=plt.subplots(1,2,figsize=(14,7.7),dpi=180,sharex=True,sharey=True)
panels=[(source['fitted__delta_gamma'],source['fitted__delta_coefficient'],
         'Observed: ordinary joint GD'),
        (predg,predc,'Predicted: fixed geometry, evolving readout')]
color=np.repeat(np.arange(1,5001),len(selected))
for ax,(xx,yy,title) in zip(axes,panels):
 ax.scatter(xx.ravel(),yy.ravel(),c=color,cmap='viridis',norm=Normalize(1,5000),
            s=1,alpha=.5,linewidths=0,rasterized=True)
 ax.axhline(0,color='.8',lw=.7);ax.axvline(0,color='.8',lw=.7)
 ax.set_title(title,fontsize=14,pad=18)
 ax.set_xlabel(r'Proposed scale change $\Delta|\gamma_j|$',fontsize=12,labelpad=10)
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 for axis in (ax.xaxis,ax.yaxis):
  fmt=ScalarFormatter(useOffset=False,useMathText=True);fmt.set_powerlimits((0,0));axis.set_major_formatter(fmt)
axes[0].set_ylabel(r'Proposed readout change $\Delta|c_j|$',fontsize=12,labelpad=10)
fig.suptitle('Readout dynamics predict the fitted-run phase pattern',fontsize=18,y=.96)
fig.subplots_adjust(left=.09,right=.89,top=.81,bottom=.29,wspace=.13)
cax=fig.add_axes([.925,.3,.012,.43]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,5000),cmap='viridis'),cax=cax)
cb.set_label('Step within the fitted window',labelpad=9)
errors=out['fitted'][5000]
fig.text(.49,.105,
         f"Prediction discrepancy across all 640,000 pairs: scale {100*errors['gamma']['relative_l2']:.3f}%, "
         f"readout {100*errors['c']['relative_l2']:.3f}% (relative Euclidean norm, each coordinate separately).\n"
         "Prediction uses only the starting state and the fixed readout matrix's GD dynamics; no decay rates are fitted.\n"
         "Scale changes on the right are instantaneous probes; they are calculated but never applied to the predicted geometry.\n"
         "Same Runge target, starting near 1% error with gamma about 16; eta=0.002, 128 selected neurons, 5,000 steps.",
         ha='center',va='center',fontsize=10)
fig.savefig(n.RESULTS/'nudge_flow_prediction.png');plt.close(fig)
