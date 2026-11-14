from __future__ import annotations
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK02_inverse_sqrt_extension.construction import Extension,build,metrics
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
 config={'epsilon':1e-4,'multiscale_orders':[4,8,12,16],'multiscale_intervals':[32,64,128,256,512],
         'uniform_order':16,'uniform_intervals':[256,512,1024,2048,4096,8192,16384],
         'halo':160,'Kc':160,'lambda':.25,'construction_precision_decimal_digits':30,
         'evaluation_dtype':'float64','l2_quadrature_per_log_panel':256,
         'note':'Exploratory sweep after pilots; not preregistered. No learned coefficients or least-squares fits.'}
 (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 rows=[]
 jobs=[('multiscale',p,n) for p in config['multiscale_orders'] for n in config['multiscale_intervals']]
 jobs += [('uniform',config['uniform_order'],n) for n in config['uniform_intervals']]
 dense=np.unique(np.r_[np.geomspace(1e-4,1,8193),np.linspace(1e-4,1,4097)])
 for method,p,n in jobs:
  start=time.monotonic();ext=Extension(order=p);net=build(ext,n,OUT/'cache',method=method)
  stem=f'{method}_p{p}_N{n}';net.save(OUT/'models'/f'{stem}.npz')
  l2=metrics(net,ext.eps,config['l2_quadrature_per_log_panel']);pred=net.predict(dense);true=dense**-.5
  rel=np.abs((pred-true)/true)
  row=dict(method=method,order=p,N=n,width=net.width,relative_l2=l2,
           max_sampled_relative=float(rel.max()),max_sampled_absolute=float(np.max(abs(pred-true))),
           left_error=float(pred[0]-100),right_error=float(pred[-1]-1),
           readout_l1=float(np.sum(abs(net.readout))),seconds=time.monotonic()-start,
           model=f'models/{stem}.npz')
  rows.append(row);(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(row,flush=True)
 (OUT/'provenance.json').write_text(json.dumps({'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},'numpy':np.__version__},indent=2)+'\n')

if __name__=='__main__':main()
