"""Numerical multi-resolvent diagnostic; grid values are not certified bounds.

Use the frozen input pairs from this pilot, not later certificate iterations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
p=argparse.ArgumentParser()
p.add_argument('--inputs',type=Path,default=Path(__file__).with_name('multi_resolvent_inputs.json'))
args=p.parse_args()
for row in json.loads(args.inputs.read_text()):
 n,cap,certs,baseline=(row[k] for k in ['n','cap','certificates','baseline'])
 shifts=np.unique(np.concatenate([c['beta']*np.geomspace(1e-4,1e3,24) for c in certs]))
 moments=np.max(np.array([c['delta']**2*shifts/(c['beta']+shifts) for c in certs]),axis=0)
 def value(steps):
  grid=np.unique(np.r_[0.,1.,np.geomspace(max(min(shifts.min(),1/steps)*1e-3,1e-30),1,4096)])
  f=np.exp(2*steps*np.log1p(-.5*grid))
  q=shifts[None,:]/(shifts[None,:]+grid[:,None])
  lp=linprog(-np.r_[1.,moments],A_ub=np.column_stack([np.ones(len(grid)),q]),b_ub=f,bounds=[(None,None)]+[(0,1e5)]*len(shifts),method='highs')
  return -lp.fun if lp.success else -1
 low=max(1,baseline); high=low*2
 while value(high)>.0001:high*=2
 while high-low>max(1,low*.001):
  mid=(low+high)//2
  if value(mid)>.0001:low=mid
  else:high=mid
 print(n,cap,'witnesses',len(certs),'baseline',baseline,'multi_estimate',low,'ratio',low/max(1,baseline),flush=True)
