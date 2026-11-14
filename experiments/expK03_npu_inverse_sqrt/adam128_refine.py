"""Controlled 128-neuron Adam conditioning/initialization refinements."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.models.centered_tanh import CenteredTanhMLP
from experiments.expK03_npu_inverse_sqrt.run import OUT,save_json,halo_nodes,evaluation_grid,relative_l2,predict
from experiments.expK02_inverse_sqrt_extension.construction import Extension

DEST=OUT/'adam128_refine'


class ConditionedMLP(CenteredTanhMLP):
    def __init__(self,config):
        super().__init__(128,eps=config['epsilon'],seed=config['seed'])
        self.output_scale=config['output_scale']
        with torch.no_grad():
            if config['initializer']=='random_multiscale':
                rng=np.random.default_rng(config['seed'])
                e=Extension(eps=config['epsilon'])
                local=112;global_count=16
                lo,hi=e.eps/2,2*e.outer_scale
                dlog=math.log(hi/lo)/local
                centers=np.exp(math.log(lo)+(np.arange(local)+rng.random(local))*dlog)
                slopes=config['scale_gain']/(centers*dlog)*np.exp(rng.uniform(-.15,.15,local))
                # Broad features cover the full signed halo, in addition to local scales.
                halo=halo_nodes(e,256,32);a,b=float(halo.min()),float(halo.max())
                centers=np.r_[centers,rng.uniform(a,b,global_count)]
                slopes=np.r_[slopes,rng.uniform(.5,2,global_count)/(b-a)]
                weights=slopes*e.radius
                biases=weights+slopes*(e.eps-centers)
                self.net[0].weight.copy_(torch.from_numpy(weights[:,None]))
                self.net[0].bias.copy_(torch.from_numpy(biases))
            self.net[2].weight.div_(self.output_scale)
            self.net[2].bias.div_(self.output_scale)

    def forward(self,x):
        return self.output_scale*super().forward(x)

    def folded_state(self):
        state={k:v.detach().clone() for k,v in self.state_dict().items()}
        state['net.2.weight']*=self.output_scale
        state['net.2.bias']*=self.output_scale
        return state


def rate_at(step,config):
    if step<=config['decay_start']:return config['learning_rate']
    phase=(step-config['decay_start'])/(config['steps']-config['decay_start'])
    return config['final_learning_rate']+.5*(config['learning_rate']-config['final_learning_rate'])*(1+math.cos(math.pi*phase))


def configs():
    base={'epsilon':1e-4,'extension_order':16,'width':128,'seed':0,'data_seed':10000,
          'batch_size':512,'steps':100000,'minimum_selection_step':50000,'decay_start':50000,
          'final_learning_rate':1e-5,'log_every':500,'checkpoint_every':25000,
          'betas':[.9,.999],'adam_epsilon':1e-8,'halo_reference_N':256,'halo_reference_R':32,
          'halo_reference_width':4815,'dtype':'float64','training_mode':'matched_halo',
          'scale_gain':1.,'initializer':'pytorch','output_scale':1.,'learning_rate':.01}
    variants=[('standard_batch512_decay',{}),
              ('standard_output100_decay',{'output_scale':100.,'learning_rate':.003}),
              ('multiscale_gain1',{'initializer':'random_multiscale','learning_rate':.003}),
              ('multiscale_gain4',{'initializer':'random_multiscale','scale_gain':4.,'learning_rate':.003}),
              ('multiscale_gain1_output100',{'initializer':'random_multiscale','output_scale':100.,'learning_rate':.001}),
              ('multiscale_gain4_output100',{'initializer':'random_multiscale','scale_gain':4.,'output_scale':100.,'learning_rate':.001})]
    return [{**base,**changes,'name':name} for name,changes in variants]


def run_trial(config,destination):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if (destination/'summary.json').exists():
        assert json.loads((destination/'config.json').read_text())==config
        return json.loads((destination/'summary.json').read_text())
    save_json(destination/'config.json',config)
    start=time.monotonic();model=ConditionedMLP(config)
    e=Extension();nodes=halo_nodes(e,256,32);values=e.value(nodes)
    tx=torch.from_numpy(nodes[:,None]);ty=torch.from_numpy(values[:,None])
    validation=evaluation_grid(e,32)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=tuple(config['betas']),
                                eps=config['adam_epsilon'],foreach=False)
    rng=np.random.default_rng(config['data_seed'])
    history=[{'step':0,'validation_rel_l2':relative_l2(model,validation,e.eps)}]
    best=math.inf;chosen=None;losses=[]
    torch.save({'model':model.state_dict(),'config':config,'step':0},destination/'initial.pt')
    for step in range(1,config['steps']+1):
        indices=rng.integers(0,len(nodes),config['batch_size'])
        x,y=tx[indices],ty[indices]
        lr=rate_at(step,config)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        # A scalar normalization, with the same targets and relative sample weights.
        loss=((model(x)-y)/config['output_scale']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'{config["name"]} loss at {step}')
        loss.backward();optimizer.step()
        losses.append(float(loss.detach())*config['output_scale']**2)
        if step%config['log_every']==0:
            error=relative_l2(model,validation,e.eps)
            history.append({'step':step,'validation_rel_l2':error,'learning_rate':lr})
            if step>=config['minimum_selection_step'] and error<best:
                best=error
                chosen={'step':step,'model':copy.deepcopy(model.state_dict()),
                        'optimizer':copy.deepcopy(optimizer.state_dict()),
                        'rng_state':copy.deepcopy(rng.bit_generator.state),'config':config}
        if step%config['checkpoint_every']==0:
            torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'rng_state':rng.bit_generator.state,'config':config,'history':history,
                        'losses':losses,'best':chosen},destination/f'step{step}.pt')
            save_json(destination/'history.json',history)
            print(f'{config["name"]}: step {step}, validation {history[-1]["validation_rel_l2"]:.7g}, '
                  f'elapsed {time.monotonic()-start:.1f}s',flush=True)
    final=relative_l2(model,evaluation_grid(e,256),e.eps)
    assert chosen is not None
    torch.save(chosen,destination/'best_validation.pt')
    model.load_state_dict(chosen['model'])
    errors={str(n):relative_l2(model,evaluation_grid(e,n),e.eps) for n in [32,256,512]}
    folded=CenteredTanhMLP(128);folded.load_state_dict(model.folded_state())
    folded_error=relative_l2(folded,evaluation_grid(e,256),e.eps)
    torch.save({'model':model.folded_state(),'step':chosen['step'],'config':config,
                'note':'Output scale folded into the ordinary Linear readout.'},destination/'exported_mlp.pt')
    x=np.unique(np.r_[np.geomspace(e.eps,1,8193),np.linspace(e.eps,1,4097)])
    y=predict(model,torch.from_numpy(x[:,None])).numpy().ravel()
    np.savez(destination/'predictions.npz',x=x,target=e.value(x),prediction=y,halo_x=nodes,halo_target=values)
    row={'name':config['name'],'model':'Adam MLP','hidden_units':128,'parameters':385,
         'learning_rate':config['learning_rate'],'final_learning_rate':config['final_learning_rate'],
         'steps':config['steps'],'minimum_selection_step':config['minimum_selection_step'],
         'batch_size':config['batch_size'],'initializer':config['initializer'],
         'scale_gain':config['scale_gain'],'output_scale':config['output_scale'],
         'best_trained_validation_step':chosen['step'],'best_trained_validation_rel_l2':errors['32'],
         'best_trained_validation_test_rel_l2':errors['256'],'test_quadrature':errors,
         'exported_mlp_relative_l2':folded_error,'final_rel_l2':final,'seconds':time.monotonic()-start,
         'training_mode':'matched_halo','run_directory':str(destination.relative_to(ROOT))}
    save_json(destination/'summary.json',row)
    return row


def main():
    DEST.mkdir(parents=True,exist_ok=True);trials=configs()
    files=[Path(__file__),ROOT/'src/models/centered_tanh.py',Path(__file__).with_name('run.py'),
           ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py']
    specification={'trials':trials,'selection':'Minimum validation relative L2 over steps 50k-100k; test quadrature not used to select.',
                   'unchanged':['C17 target','N=256/R=32 halo-center inputs','relative sample weighting','physical dx evaluation'],
                   'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path=DEST/'specification.json'
    if path.exists():assert json.loads(path.read_text())==specification
    else:save_json(path,specification)
    rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(run_trial,c,DEST/c['name']):c for c in trials}
        for future in as_completed(pending):
            config=pending[future]
            try:
                row=future.result();rows.append(row)
                print(f'COMPLETE {row["name"]}: {row["best_trained_validation_test_rel_l2"]:.9g}',flush=True)
            except Exception as exc:failures.append({'name':config['name'],'error':repr(exc)})
            save_json(DEST/'summary.json',rows);save_json(DEST/'status.json',{'completed':len(rows),'failures':failures,'planned':len(trials)})
    if failures:raise RuntimeError(failures)
    selected=min(rows,key=lambda r:r['best_trained_validation_rel_l2'])
    save_json(DEST/'selected_reference.json',selected)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
