"""Validation-tuned Adam/NPU, same function-value samples as fitted QUILL."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,math,time,json,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.expK06_hard_analytic_suite.common import *
from experiments.expK06_hard_analytic_suite.targets import NAMES,MAIN,target
from experiments.expK06_hard_analytic_suite.models import build

def identifier(c):return f"{c['target']}__{c['method']}__N{c['N']}__{c['variant']}__lr{c['lr']:.0e}__s{c['seed']}"

def data(c):
    x=training_grid(c['N']);y=target(c['target'],x);amp=max(float(abs(y).max()),1e-30);offset=float((y/amp).mean())
    return torch.from_numpy(x[:,None]),torch.from_numpy((y/amp-offset)[:,None]),amp,offset

@torch.no_grad()
def prediction(m,x,amp,offset):
    return torch.cat([m(p) for p in torch.from_numpy(x[:,None]).split(512)]).numpy().ravel()*amp+amp*offset

@torch.no_grad()
def score(m,x,y,w,amp,offset):
    yp=prediction(m,x,amp,offset)
    return float(np.sqrt(np.dot(w,(yp-y)**2)/np.dot(w,y*y)))

def run_one(c,steps,stage='pilot',resume=None,polish=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=OUT/stage/identifier(c);folder.mkdir(parents=True,exist_ok=True)
    summary=folder/'summary.json'
    if summary.exists():return json.loads(summary.read_text())
    save(folder/'config.json',{**c,'requested_steps':steps,'polish':polish})
    start=time.monotonic();m=build(c['method'],c['width'],c['variant'],c['seed'])
    x,y,amp,offset=data(c);xv,wv=quadrature(32);yv=target(c['target'],xv)
    opt=torch.optim.Adam(m.parameters(),lr=c['lr'],foreach=False)
    rng=np.random.default_rng(12700+c['seed']);first=1;history=[]
    best=score(m,xv,yv,wv,amp,offset);best_step=0;best_state=copy.deepcopy(m.state_dict())
    torch.save(dict(model=m.state_dict(),config=c,amplitude=amp,offset=offset),folder/'initial.pt')
    if resume:
        state=torch.load(ROOT/resume/'last.pt',weights_only=False)
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng']
        best=state['best'];best_step=state['best_step'];best_state=state['best_state'];first=state['step']+1
    status='completed';step=first-1
    for step in range(first,steps+1):
        schedule_steps=c.get('schedule_steps',100000)
        phase=max(0.,(step-schedule_steps/2)/(schedule_steps/2))
        lr=1e-6+.5*(c['lr']-1e-6)*(1+math.cos(math.pi*phase))
        for group in opt.param_groups:group['lr']=lr
        idx=rng.integers(0,len(x),size=100)
        opt.zero_grad(set_to_none=True);loss=(m(x[idx])-y[idx]).square().mean()
        if not torch.isfinite(loss):status=f'nonfinite_loss_at_{step}';break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad):status=f'nonfinite_gradient_at_{step}';break
        opt.step()
        if step%1000==0 or step==steps:
            val=score(m,xv,yv,wv,amp,offset)
            history.append(dict(step=step,validation=val if math.isfinite(val) else None))
            if math.isfinite(val) and val<best:best=val;best_step=step;best_state=copy.deepcopy(m.state_dict())
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,step=step,
                    best=best,best_step=best_step,best_state=best_state,config=c,amplitude=amp,offset=offset),folder/'last.pt')
    m.load_state_dict(best_state)
    adam_best=best;selected_phase='Adam';calls=0;polish_status='not_requested'
    torch.save(dict(model=best_state,config=c,amplitude=amp,offset=offset,step=best_step),folder/'adam_best.pt')
    if polish:
        optimizer=torch.optim.LBFGS(m.parameters(),lr=1,max_iter=5,max_eval=12,history_size=50,
                                   tolerance_grad=1e-14,tolerance_change=0.,line_search_fn='strong_wolfe')
        polish_status='completed';unchanged=0
        def closure():
            nonlocal calls
            calls+=1;optimizer.zero_grad(set_to_none=True)
            # Exact full training loss; chunks bound memory, gradients accumulate.
            total=0.
            for xx,yy in zip(x.split(512),y.split(512)):
                loss=(m(xx)-yy).square().sum()/len(x)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite polish')
                loss.backward();total+=float(loss.detach())
            return torch.tensor(total,dtype=torch.float64)
        for k in range(100):
            before=copy.deepcopy(m.state_dict())
            try:optimizer.step(closure)
            except (FloatingPointError,RuntimeError) as exc:polish_status=str(exc);m.load_state_dict(before);break
            val=score(m,xv,yv,wv,amp,offset)
            if math.isfinite(val) and val<best:best=val;best_state=copy.deepcopy(m.state_dict());selected_phase='LBFGS'
            history.append(dict(polish_outer=k+1,validation=val if math.isfinite(val) else None))
            unchanged=unchanged+1 if all(torch.equal(v,before[key]) for key,v in m.state_dict().items()) else 0
            if unchanged>=3:polish_status='unchanged_parameters';break
    m.load_state_dict(best_state)
    before_ls=best
    if polish:
        # Give the trained dictionaries the same rank-revealing linear solve
        # as QUILL; retain the previous model if validation does not improve.
        from scipy.linalg import lstsq
        with torch.no_grad():
            features=torch.cat([m.features(p) for p in x.split(512)]).numpy()
        design=np.column_stack((features,np.ones(len(x))))
        if np.isfinite(design).all():
            coef,_,rank,_=lstsq(design,y.numpy(),cond=1e-14,lapack_driver='gelsd')
            with torch.no_grad():
                m.head.weight.copy_(torch.from_numpy(coef[:-1].T))
                m.head.bias.copy_(torch.from_numpy(coef[-1]))
            val=score(m,xv,yv,wv,amp,offset)
            torch.save(dict(model=m.state_dict(),config=c,amplitude=amp,offset=offset,rank=rank),folder/'readout_ls.pt')
            if math.isfinite(val) and val<best:
                best=val;best_state=copy.deepcopy(m.state_dict());selected_phase='readout_LS'
    m.load_state_dict(best_state)
    torch.save(dict(model=best_state,config=c,amplitude=amp,offset=offset,phase=selected_phase),folder/'selected.pt')
    xx,ww=quadrature(128);yy=target(c['target'],xx)
    test=score(m,xx,yy,ww,amp,offset)
    row={**c,'steps':step,'training_samples':len(x),'validation':best,'adam_validation':adam_best,
         'relative_l2':test,'selected_phase':selected_phase,'best_adam_step':best_step,'validation_before_readout_ls':before_ls,
         'parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
         'status':status,'polish_status':polish_status,'polish_closures':calls,
         'seconds':time.monotonic()-start,'run_directory':str(folder.relative_to(ROOT))}
    save(summary,row);save(folder/'history.json',history)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.smoke:
        for method,variant in [('Adam','pytorch'),('NPU','native')]:
            c=dict(target='airy',method=method,variant=variant,N=64,width=WIDTHS[0],seed=0,lr=.001)
            print(run_one(c,1000,'smoke',polish=False),flush=True)
if __name__=='__main__':main()
