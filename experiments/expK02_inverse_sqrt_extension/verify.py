from __future__ import annotations
import json,sys,math,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK02_inverse_sqrt_extension.construction import Extension,Network,build,metrics,quadrature
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'

def main():
 ext=Extension(order=16);cache=OUT/'cache';rows=[]
 # Fixed halo 32 is selected for stability, not the accidental minimum at halo 16.
 for n in [32,64,128,256,512]:
  net=build(ext,n,cache,halo=32);net.save(OUT/'models'/f'multiscale_p16_N{n}_R32.npz')
  rows.append(dict(N=n,halo=32,width=net.width,relative_l2=metrics(net,ext.eps),model=f'models/multiscale_p16_N{n}_R32.npz'))
 (OUT/'compact_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
 halo=[]
 for n in [128,256]:
  for r in [0,8,16,32,64,160]:
   net=build(ext,n,cache,halo=r)
   halo.append(dict(N=n,halo=r,width=net.width,relative_l2=metrics(net,ext.eps)))
 (OUT/'halo_sweep.json').write_text(json.dumps(halo,indent=2)+'\n')
 net=Network.load(OUT/'models/multiscale_p16_N256_R32.npz')
 info={'selected_model':'models/multiscale_p16_N256_R32.npz','width':net.width,'scales':ext.scales.tolist(),
       'epsilon':ext.eps,'center':ext.center,'radius':ext.radius,'centered_singularity':-ext.center/ext.radius,
       'relative_l2_quadrature':{str(q):metrics(net,ext.eps,q) for q in [64,128,256,512]},
       'readout_l1':float(abs(net.readout).sum()),'output_bias':net.output_bias,
       'max_hidden_weight':float(abs(net.weight).max())}
 x=np.unique(np.r_[np.geomspace(ext.eps,1,32769),np.linspace(ext.eps,1,8193),
                    np.exp(np.random.default_rng(7341).uniform(np.log(ext.eps),0,4096))])
 truth=x**-.5;pred=net.predict(x);rel=(pred-truth)/truth
 info.update(dense_points=len(x),max_sampled_relative=float(abs(rel).max()),
             max_sampled_absolute=float(abs(pred-truth).max()),max_error_x=float(x[np.argmax(abs(rel))]))
 np.savez(OUT/'selected_predictions.npz',x=x,truth=truth,prediction=pred,relative_error=rel)
 # Ordinary torch Linear/Tanh/Linear export, no feature preprocessing beyond affine normalization.
 torch.set_num_threads(1)
 model=torch.nn.Sequential(torch.nn.Linear(1,net.width,dtype=torch.float64),torch.nn.Tanh(),
                           torch.nn.Linear(net.width,1,dtype=torch.float64))
 with torch.no_grad():
  model[0].weight.copy_(torch.tensor(net.weight[:,None]));model[0].bias.copy_(torch.tensor(net.hidden_bias))
  model[2].weight.copy_(torch.tensor(net.readout[None,:]));model[2].bias.fill_(net.output_bias)
  xq,wq=quadrature(ext.eps,512);z=ext.centered(xq)
  yp=torch.cat([model(torch.tensor(z[i:i+256,None],dtype=torch.float64)) for i in range(0,len(z),256)]).numpy().ravel()
  rr=(yp-xq**-.5)*np.sqrt(xq)
  info['torch_relative_l2']=float(np.sqrt(np.dot(wq,rr**2)/np.log(1/ext.eps)))
 torch.save({'model':model.state_dict(),'epsilon':ext.eps,'input_center':ext.center,'input_scale':ext.radius},OUT/'selected_torch.pt')
 # Check that the implemented extension derivative equals the exact derivative on the target domain.
 grid=np.geomspace(ext.eps,1,4097)
 part=np.zeros_like(grid)
 for s in ext.scales:
  u=grid/s
  part+=s**-1.5*ext.prototype_derivative((u-2)/2)/2
 info['partition_derivative_max_relative']=float(np.max(abs((part+.5*grid**-1.5)/(.5*grid**-1.5))))
 info['extension_value_max_relative_roundoff']=float(np.max(abs((ext.value(grid)-grid**-.5)*np.sqrt(grid))))
 # High-precision evaluation of the saved fp64 network: isolate arithmetic from model error.
 import mpmath as mp
 mp.mp.dps=60
 xp=np.geomspace(ext.eps,1,25)
 ww=list(map(mp.mpf,net.weight));bb=list(map(mp.mpf,net.hidden_bias));vv=list(map(mp.mpf,net.readout))
 high=[];start=time.monotonic()
 for xx in xp:
  xm=mp.mpf(xx);zz=(xm-mp.mpf(ext.eps))/mp.mpf(ext.radius)-1
  terms=[]
  for a,b,v in zip(ww,bb,vv):
   t=a*zz+b
   activation=mp.sign(t) if abs(t)>45 else mp.tanh(t)
   terms.append(v*activation)
  prediction=mp.mpf(net.output_bias)+mp.fsum(terms)
  high.append(float((prediction-mp.sqrt(1/xm))/mp.sqrt(1/xm)))
 info['mpmath_saved_network_check']={'decimal_digits':60,'points':len(xp),
   'maximum_relative_error':float(np.max(np.abs(high))),'saturated_tanh_cutoff':45,
   'saturation_absolute_error_bound':float(2*np.exp(-90)*sum(abs(net.readout))),
   'seconds':time.monotonic()-start}
 # Exact cutoff/primitive construction: coefficients scale with epsilon, prototype stays fixed.
 robustness=[]
 for eps in [1e-3,1e-4,1e-5]:
  other=Extension(eps=eps,order=16);nn=build(other,256,cache,halo=32)
  robustness.append({'epsilon':eps,'scales':len(other.scales),'width':nn.width,'relative_l2':metrics(nn,eps)})
 info['epsilon_robustness']=robustness
 (OUT/'verification.json').write_text(json.dumps(info,indent=2)+'\n')
 selected={'target':'1/sqrt(x)','domain':[ext.eps,1.0],'input_center':ext.center,'input_scale':ext.radius,
           'method':'15 scaled cardinal-QI blocks, exported as one tanh MLP','cutoff_order':16,
           'extension_smoothness':'C17','prototype_interval':[0,4],'N_per_block':256,
           'halo_per_side_per_block':32,'Kc':160,'lambda':.25,'construction_decimal_digits':30,
           'inference_dtype':'float64','hidden_neurons':net.width,
           'model':info['selected_model'],'torch_model':'selected_torch.pt'}
 (OUT/'selected_config.json').write_text(json.dumps(selected,indent=2)+'\n')
 print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
