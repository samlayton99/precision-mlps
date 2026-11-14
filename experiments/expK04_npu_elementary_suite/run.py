"""Small illustrative NPU screen plus one fresh-seed check per target."""
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
from experiments.expK04_npu_elementary_suite.problem import A,B,TARGETS,POWER_TARGETS,grid,score,sample,target
from experiments.expK04_npu_elementary_suite.models import MODELS,build
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK04_npu_elementary_suite'
ACTIVE_TARGETS=['inv_sqrt','root_plus_linear','exp_over_root','log_over_root']
SCREEN_MODELS=['shallow_6','shallow_32','stacked_8']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identifier(c):
    return f'{c["target"]}__{c["model"]}__{c["sampling"]}__lr{c["learning_rate"]:.0e}__seed{c["seed"]}'


def run_one(c,directory,model_factory=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        assert json.loads((directory/'config.json').read_text())==c
        return json.loads((directory/'summary.json').read_text())
    save(directory/'config.json',c);start=time.monotonic();m=(model_factory or build)(c['model'],c['seed'])
    validation=grid(c['target'],64);training=grid(c['target'],32)
    test=grid(c['target'],256);norm=float((test[2]*test[1].square()).sum())
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=c['learning_rate'],foreach=False)
    rng=np.random.default_rng(10000+c['seed'])
    best=score(m,validation);best_step=0;best_state=copy.deepcopy(m.state_dict());history=[]
    adam_status='completed';first_success=None
    torch.save({'model':m.state_dict(),'config':c},directory/'initial.pt')
    for step in range(1,c['adam_steps']+1):
        phase=max(0,(step-c['adam_steps']/2)/(c['adam_steps']/2))
        lr=c['final_learning_rate']+.5*(c['learning_rate']-c['final_learning_rate'])*(1+math.cos(math.pi*phase))
        for group in optimizer.param_groups:group['lr']=lr
        x,y,w=sample(c['target'],rng,c['batch_size'],c['sampling'],norm)
        optimizer.zero_grad(set_to_none=True);loss=(w*(m(x)-y).square()).mean()
        if not torch.isfinite(loss):adam_status=f'nonfinite_loss_at_{step}';break
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad):
            adam_status=f'nonfinite_gradient_at_{step}';break
        optimizer.step()
        if step%c['log_every']==0:
            value=score(m,validation)
            if math.isfinite(value):
                history.append({'phase':'adam','step':step,'validation_rel_l2':value})
                if value<best:best=value;best_step=step;best_state=copy.deepcopy(m.state_dict())
                if value<1e-12 and first_success is None:first_success=step
    adam_final=score(m,validation)
    if not math.isfinite(adam_final):adam_final=None
    torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'rng_state':rng.bit_generator.state,
                'step':step,'config':c,'status':adam_status},directory/'adam_final.pt')
    m.load_state_dict(best_state)
    adam_test=score(m,test)
    torch.save({'model':best_state,'step':best_step,'config':c},directory/'adam_best.pt')
    adam_best_validation=best
    # Conventional deterministic full-batch polish, reported separately from Adam.
    lbfgs=torch.optim.LBFGS([p for p in m.parameters() if p.requires_grad],lr=1.,max_iter=5,max_eval=12,
                           history_size=50,tolerance_grad=1e-14,tolerance_change=0.,line_search_fn='strong_wolfe')
    x,y,w=training;train_norm=(w*y.square()).sum();closure_calls=0;polish_status='completed';best_phase='adam';best_polish_step=0
    unchanged=0
    def closure():
        nonlocal closure_calls
        closure_calls+=1;lbfgs.zero_grad(set_to_none=True)
        loss=(w*(m(x)-y).square()).sum()/train_norm
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite LBFGS loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad):
            raise FloatingPointError('nonfinite LBFGS gradient')
        return loss
    for outer in range(1,c['lbfgs_outer_steps']+1):
        before=copy.deepcopy(m.state_dict())
        try:lbfgs.step(closure)
        except (FloatingPointError,RuntimeError) as exc:
            polish_status=str(exc);m.load_state_dict(before);break
        value=score(m,validation)
        if not math.isfinite(value):polish_status='nonfinite_validation';m.load_state_dict(before);break
        history.append({'phase':'lbfgs','step':outer,'validation_rel_l2':value})
        if value<best:
            best=value;best_state=copy.deepcopy(m.state_dict());best_phase='lbfgs';best_polish_step=outer
        unchanged=unchanged+1 if all(torch.equal(v,before[k]) for k,v in m.state_dict().items()) else 0
        if unchanged>=3:polish_status='unchanged_parameters';break
    torch.save({'model':m.state_dict(),'optimizer':lbfgs.state_dict(),'config':c,'status':polish_status},directory/'lbfgs_final.pt')
    m.load_state_dict(best_state)
    errors={str(n):score(m,grid(c['target'],n)) for n in [64,256,512]}
    torch.save({'model':best_state,'config':c,'selected_phase':best_phase,'adam_step':best_step,
                'lbfgs_outer_step':best_polish_step},directory/'selected.pt')
    dense=np.unique(np.r_[np.geomspace(A,B,4097),np.linspace(A,B,2049)])
    with torch.no_grad():pred=torch.cat([m(z) for z in torch.from_numpy(dense[:,None]).split(512)]).numpy().ravel()
    truth=target(c['target'],dense)
    np.savez(directory/'predictions.npz',x=dense,target=truth,prediction=pred)
    row={**c,'parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
         'adam_status':adam_status,'adam_best_validation':adam_best_validation,'adam_best_test':adam_test,
         'adam_best_step':best_step,'adam_first_logged_success':first_success,'adam_final_validation':adam_final,
         'lbfgs_status':polish_status,'lbfgs_closure_calls':closure_calls,'selected_phase':best_phase,
         'best_validation':best,'test_relative_l2':errors['256'],'quadrature_errors':errors,
         'maximum_absolute_error_on_dense_grid':float(np.max(np.abs(pred-truth))),
         'seconds':time.monotonic()-start,'run_directory':str(directory.relative_to(ROOT))}
    save(directory/'history.json',history);save(directory/'summary.json',row)
    return row


def batch(jobs,folder,workers=4):
    folder.mkdir(parents=True,exist_ok=True);rows=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_one,c,folder/identifier(c)):c for c in jobs}
        for future in as_completed(pending):
            c=pending[future]
            try:
                row=future.result();rows.append(row)
                print(f'{folder.name} {identifier(c)}: Adam={row["adam_best_test"]:.3g}, polished={row["test_relative_l2"]:.3g}',flush=True)
            except Exception as exc:
                failure={**c,'error':repr(exc)};failures.append(failure);save(folder/identifier(c)/'failure.json',failure)
            save(folder/'summary.json',rows);save(folder/'status.json',{'completed':len(rows),'planned':len(jobs),'failures':failures})
    if failures:raise RuntimeError(f'{len(failures)} failed jobs; see {folder}/status.json')
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base={'seed':0,'sampling':'log_importance','adam_steps':20000,'batch_size':100,
          'final_learning_rate':1e-6,'log_every':100,'lbfgs_outer_steps':100}
    jobs=[{**base,'target':name,'model':model,'learning_rate':lr}
          for name in ACTIVE_TARGETS for model in SCREEN_MODELS for lr in [.001,.01]]
    controls=[{**base,'target':name,'model':model,'learning_rate':.001,'sampling':'uniform'}
              for name in ['inv_sqrt'] for model in MODELS[:3]]
    sources=list(Path(__file__).parent.glob('*.py'))+[ROOT/'src/models/neural_power_unit.py']
    spec={'domain':[A,B],'targets':{n:TARGETS[n] for n in ACTIVE_TARGETS},'models':SCREEN_MODELS,'screen_runs':len(jobs),'uniform_control_runs':len(controls),
          'base_config':base,'rates':[.001,.01],'repeat_seeds':[1],
          'selection':'Best order-64 validation checkpoint; select architecture/rate on seed 0; repeat that exact recipe at seed 1. Small illustrative screen, not a success-rate estimate.',
          'loss':'Physical dx relative L2 squared; importance-weighted log sampling or uniform sampling; no halo/extension targets.',
          'training_input':'Raw physical x, no engineered features or target-specific initial exponents.',
          'polish':'Start from Adam validation-best state; up to 500 ordinary full-batch LBFGS iterations on order-32 quadrature. Report separately.',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    spec=json.loads(json.dumps(spec));path=OUT/'specification.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:save(path,spec)
    rows=batch(jobs,OUT/'screen')
    choices={name:min((r for r in rows if r['target']==name),key=lambda r:r['best_validation']) for name in ACTIVE_TARGETS}
    save(OUT/'selected_recipes.json',choices)
    repeats=[{key:value for key,value in {**choices[name],'seed':seed}.items() if key in jobs[0]}
             for name in ACTIVE_TARGETS for seed in [1]]
    batch(repeats,OUT/'repeats');batch(controls,OUT/'uniform_controls')
    print('Completed all screening, fresh-seed repeats, and native-power uniform controls.',flush=True)


if __name__=='__main__':main()
