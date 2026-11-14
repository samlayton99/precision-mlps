"""Conventional Adam baseline only: default/Xavier initialization, no feature priors."""
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

DEST=OUT/'adam_standard'
ALLOWED_INITIALIZERS=('pytorch','xavier_uniform_tanh')


def build_model(config):
    assert config['initializer'] in ALLOWED_INITIALIZERS
    model=CenteredTanhMLP(config['width'],eps=config['epsilon'],seed=config['seed'])
    if config['initializer']=='xavier_uniform_tanh':
        generator=torch.Generator().manual_seed(config['seed'])
        torch.nn.init.xavier_uniform_(model.net[0].weight,gain=torch.nn.init.calculate_gain('tanh'),generator=generator)
        torch.nn.init.xavier_uniform_(model.net[2].weight,gain=1.,generator=generator)
        torch.nn.init.zeros_(model.net[0].bias);torch.nn.init.zeros_(model.net[2].bias)
    return model


def rate_at(step,c):
    if step<=c['decay_start']:return c['learning_rate']
    phase=(step-c['decay_start'])/(c['steps']-c['decay_start'])
    return c['final_learning_rate']+(c['learning_rate']-c['final_learning_rate'])*.5*(1+math.cos(math.pi*phase))


def configs():
    base={'width':128,'epsilon':1e-4,'extension_order':16,'seed':0,'data_seed':10000,
          'steps':100000,'minimum_selection_step':50000,'decay_start':50000,
          'final_learning_rate':1e-5,'log_every':500,'checkpoint_every':25000,
          'betas':[.9,.999],'adam_epsilon':1e-8,'halo_reference_N':256,'halo_reference_R':32,
          'halo_reference_width':4815,'dtype':'float64','output_scale':1.,
          'training_mode':'matched_halo','loss':'unweighted MSE on unchanged physical targets'}
    return [{**base,'initializer':init,'learning_rate':lr,'batch_size':batch,
             'name':f'{init}_lr{lr:.0e}_batch{batch}'}
            for init in ALLOWED_INITIALIZERS for lr in [.001,.01,.03] for batch in [100,1024]]


def run_trial(c,directory):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        assert json.loads((directory/'config.json').read_text())==c
        return json.loads((directory/'summary.json').read_text())
    save_json(directory/'config.json',c)
    start=time.monotonic();model=build_model(c)
    e=Extension();x=halo_nodes(e,256,32);y=e.value(x)
    tx=torch.from_numpy(x[:,None]);ty=torch.from_numpy(y[:,None]);grid=evaluation_grid(e,32)
    reference=np.load(OUT/'adam_mlp_1k/matched_halo_data.npz')
    assert np.array_equal(x,reference['x']) and np.array_equal(y,reference['y'])
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']),eps=c['adam_epsilon'],foreach=False)
    rng=np.random.default_rng(c['data_seed']);best=math.inf;selected=None;losses=[]
    history=[{'step':0,'validation_rel_l2':relative_l2(model,grid,e.eps)}]
    torch.save({'model':model.state_dict(),'config':c,'step':0},directory/'initial.pt')
    for step in range(1,c['steps']+1):
        indices=rng.integers(0,len(x),c['batch_size'])
        for group in optimizer.param_groups:group['lr']=rate_at(step,c)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(tx[indices])-ty[indices]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'{c["name"]}, step {step}')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if step%c['log_every']==0:
            error=relative_l2(model,grid,e.eps)
            history.append({'step':step,'validation_rel_l2':error,'learning_rate':rate_at(step,c)})
            if step>=c['minimum_selection_step'] and error<best:
                best=error;selected={'step':step,'model':copy.deepcopy(model.state_dict()),
                    'optimizer':copy.deepcopy(optimizer.state_dict()),'rng_state':copy.deepcopy(rng.bit_generator.state),'config':c}
        if step%c['checkpoint_every']==0:
            torch.save({'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'rng_state':rng.bit_generator.state,'history':history,'losses':losses,'config':c,'best':selected},directory/f'step{step}.pt')
            save_json(directory/'history.json',history)
            print(f'{c["name"]}: step={step}, validation={history[-1]["validation_rel_l2"]:.7g}, elapsed={time.monotonic()-start:.1f}s',flush=True)
    final=relative_l2(model,evaluation_grid(e,256),e.eps)
    assert selected is not None
    torch.save(selected,directory/'best_validation.pt');model.load_state_dict(selected['model'])
    errors={str(n):relative_l2(model,evaluation_grid(e,n),e.eps) for n in [32,256,512]}
    dense=np.unique(np.r_[np.geomspace(e.eps,1,8193),np.linspace(e.eps,1,4097)])
    prediction=predict(model,torch.from_numpy(dense[:,None])).numpy().ravel()
    np.savez(directory/'predictions.npz',x=dense,target=e.value(dense),prediction=prediction,halo_x=x,halo_target=y)
    row={'name':c['name'],'model':'Adam MLP','hidden_units':c['width'],'parameters':sum(p.numel() for p in model.parameters()),
         'initializer':c['initializer'],'output_scale':1.,'steps':c['steps'],'minimum_selection_step':c['minimum_selection_step'],
         'learning_rate':c['learning_rate'],'final_learning_rate':c['final_learning_rate'],'batch_size':c['batch_size'],
         'best_trained_validation_step':selected['step'],'best_trained_validation_rel_l2':errors['32'],
         'best_trained_validation_test_rel_l2':errors['256'],'test_quadrature':errors,'final_rel_l2':final,
         'training_mode':'matched_halo','seconds':time.monotonic()-start,'run_directory':str(directory.relative_to(ROOT))}
    save_json(directory/'summary.json',row)
    return row


def previous_eligible():
    # Existing 128-neuron runs with ordinary initialization and unscaled output.
    tuning=json.loads((OUT/'adam_tuning/long_runs/summary.json').read_text())
    refinements=json.loads((OUT/'adam128_refine/summary.json').read_text())
    rows=[{**r,'initializer':'pytorch','output_scale':1.,'source_family':'constant_rate_search',
           'name':f'prior_pytorch_lr{r["learning_rate"]:g}'} for r in tuning if r['hidden_units']==128]
    rows += [{**r,'source_family':'earlier_standard_decay'} for r in refinements
             if r['initializer']=='pytorch' and r['output_scale']==1.]
    return rows


def main():
    DEST.mkdir(parents=True,exist_ok=True);trials=configs()
    sources=[Path(__file__),ROOT/'src/models/centered_tanh.py',Path(__file__).with_name('run.py'),
             ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py']
    spec={'trials':trials,'allowed_initializers':ALLOWED_INITIALIZERS,
          'excluded':['random multiscale initializer','log input features','domain-dependent neuron centers or slopes','output gain reparameterization'],
          'selection':'Minimum validation relative L2 in steps 50k-100k; independent test quadrature for reporting.',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    spec=json.loads(json.dumps(spec))
    path=DEST/'specification.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:save_json(path,spec)
    old=previous_eligible();save_json(DEST/'previous_eligible.json',old)
    rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_trial,c,DEST/c['name']):c for c in trials}
        for future in as_completed(futures):
            c=futures[future]
            try:
                row=future.result();rows.append(row)
                print(f'COMPLETE {row["name"]}: {row["best_trained_validation_test_rel_l2"]:.9g}',flush=True)
            except Exception as exc:failures.append({'name':c['name'],'error':repr(exc)})
            save_json(DEST/'summary.json',rows);save_json(DEST/'status.json',{'completed':len(rows),'planned':len(trials),'failures':failures})
    if failures:raise RuntimeError(failures)
    selected=min(rows+old,key=lambda r:r['best_trained_validation_rel_l2'])
    assert selected['initializer'] in ALLOWED_INITIALIZERS and selected['hidden_units']>=64
    save_json(DEST/'selected_reference.json',selected)
    print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
