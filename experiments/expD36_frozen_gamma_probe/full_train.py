"""Full-sweep workers with ordinary updates and validation-only selection."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from . import core, full_core as f, full_kernels as kernels, full_screen as screen, train


def safe_json(value):
    if isinstance(value,dict):
        return {key:safe_json(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def load_banks(root,n,name,gammas):
    arrays=dict(np.load(root/f'common/N{n}/arrays.npz'))
    banks=[]
    for gamma in gammas:
        folder=root/'dictionaries'/screen.dictionary_id(n,name,gamma)
        meta=json.loads((folder/'meta.json').read_text())
        assert meta['complete']
        j=np.load(folder/'J.npy')
        assert core.array_hash(j)==meta['matrix_hash']
        banks.append(dict(meta=meta,J=j,spectrum=dict(np.load(folder/'spectrum.npz')),
                          U=np.load(folder/'U.npy',mmap_mode='r')))
    return arrays,banks


def column_targets(arrays,columns,split):
    return np.column_stack([f.polynomial_probe(arrays['x_'+split],c['degree'],len(arrays['x_train']))
        if 'degree' in c else arrays['y_'+split][:,c['target_index']] for c in columns])


def run_batch(root,cfg,n,name,gammas,tag,optimizer,columns,rates,epsilons,frontier,deadline,
              initial=None,state=None):
    folder=root/'training'/tag; folder.mkdir(parents=True,exist_ok=True)
    arrays,banks=load_banks(root,n,name,gammas)
    j_host=np.stack([b['J'] for b in banks]); B,m,d=j_host.shape; C=len(columns)
    y_host=np.broadcast_to(column_targets(arrays,columns,'train'),(B,m,C)).copy()
    scales=np.array([b['meta']['scales'] for b in banks]); neighbor=banks[0]['meta']['neighbor']
    if initial is None:
        initial=np.zeros((B,d,C))
    if state is None:
        state=kernels.initialize(initial,len(cfg['tolerances']))
    else:
        state=jax.tree.map(jnp.asarray,state)
    evaluations=[]
    if (folder/'state.npz').exists():
        state={key:jnp.asarray(value) for key,value in dict(np.load(folder/'state.npz')).items()}
        evaluations=[row for row in json.loads((folder/'evaluations.json').read_text())
                     if row['step']<=int(state['count'])]
    case=dict(tag=tag,n=n,map=name,gammas=gammas,optimizer=optimizer,columns=columns,
        rates=np.asarray(rates).tolist(),epsilons=np.asarray(epsilons).tolist(),
        matrix_hashes=[b['meta']['matrix_hash'] for b in banks],initial_hash=core.array_hash(initial),
        trace_columns=kernels.TRACE,trace_convention='state n before update n+1; endpoint separately evaluated',
        source_commit=os.environ.get('PROBE_SOURCE_COMMIT','local'))
    if (folder/'case.json').exists():
        old=json.loads((folder/'case.json').read_text())
        for key in ['columns','rates','epsilons','matrix_hashes','initial_hash']:
            assert old[key]==case[key],f'Incompatible resume: {tag} {key}'
    core.write_json(folder/'case.json',case)
    if not (folder/'initial.npz').exists():
        core.save_arrays(folder/'initial.npz',theta=initial)
    at=int(state['count']); begin=time.monotonic()
    j=jnp.asarray(j_host); y=jnp.asarray(y_host)
    device_args=(j,y,jnp.asarray(scales),jnp.asarray(rates),jnp.asarray(epsilons),jnp.array(cfg['tolerances']))
    kernel=kernels.make_chunk(optimizer,neighbor,cfg['chunk_size'],cfg['adam_pilot_steps'])
    correction=y_host-j_host@initial
    spectral=[]
    if optimizer=='gd':
        for bi,b in enumerate(banks):
            loading=b['U'].T@correction[bi]
            perp=correction[bi]-b['U']@loading
            spectral.append((loading,np.sum(perp**2,axis=0)))

    def evaluate():
        nonlocal state
        theta=np.asarray(state['theta']); failed=np.asarray(state['failed'])
        errors=np.linalg.norm(j_host@theta-y_host,axis=1)/np.linalg.norm(y_host,axis=1)
        hits=np.asarray(state['hits']).copy()
        hits=np.where((hits<0)&(errors[:,:,None]<=np.array(cfg['tolerances']))&(failed[:,:,None]==0),at,hits)
        state=dict(state,hits=jnp.asarray(hits))
        row=dict(step=at,train=errors.tolist(),failed=failed.tolist(),spectral=[],norms=[])
        for split in ['validation','evaluation']:
            grid=arrays['x_'+split]; target=column_targets(arrays,columns,split); values=[]
            for bi,gamma in enumerate(gammas):
                squared=np.zeros(C)
                for start in range(0,len(grid),4096):
                    xx=grid[start:start+4096]
                    a=core.design(xx,arrays['centers'],gamma)*np.sqrt(len(xx)/len(grid))
                    je=f.design_from_physical(a,scales[bi],neighbor)
                    residual=je@theta[bi]-target[start:start+len(xx)]
                    squared+=np.sum(residual**2,axis=0)
                values.append((np.sqrt(squared)/np.linalg.norm(target,axis=0)).tolist())
            row[split]=values
        for bi,b in enumerate(banks):
            physical=f.decode(theta[bi],scales[bi],neighbor)
            row['norms'].append({label:{norm:np.linalg.norm(v,ord=order,axis=0).tolist()
                for norm,order in [('l1',1),('l2',2),('linf',np.inf)]}
                for label,v in [('native',theta[bi]),('physical',physical)]})
            if optimizer=='gd':
                loading,floor=spectral[bi]; s=b['spectrum']['singular']; norm=np.linalg.norm(y_host[bi],axis=0)
                predicted=np.array([core.spectral_error(at,s,loading[:,ci,None],floor[ci:ci+1],
                    norm[ci:ci+1],rates[bi,ci])[0] for ci in range(C)])
                valid=failed[bi]==0
                np.testing.assert_allclose(errors[bi,valid],predicted[valid],rtol=2e-7,atol=1e-11,
                    err_msg=f'{tag} gamma {gammas[bi]} at {at}')
                row['spectral'].append(predicted.tolist())
        evaluations.append(safe_json(row))
        core.save_arrays(folder/f'checkpoint_{at:06d}.npz',**jax.tree.map(np.asarray,state))
        core.write_json(folder/'evaluations.json',evaluations)
        print(f'TRAIN {tag} step={at} max_error={np.nanmax(errors):.5g} failed={np.count_nonzero(failed)}',flush=True)

    if at>=frontier:
        return dict(np.load(folder/'state.npz')),evaluations
    if not evaluations:
        evaluate()
    while at<frontier and time.monotonic()<deadline:
        previous=at
        state,trace=kernel(state,*device_args); jax.block_until_ready(state)
        at=int(state['count'])
        # Each chunk is durable and is never concatenated/recompressed repeatedly.
        core.save_arrays(folder/f'trace_{previous:06d}_{at:06d}.npz',trace=np.asarray(trace))
        if at in cfg['checkpoints'] or at==frontier:
            evaluate()
        if at%5000==0 or at==frontier or time.monotonic()>=deadline:
            core.save_arrays(folder/'state.npz',**jax.tree.map(np.asarray,state))
            core.write_json(folder/'progress.json',dict(step=at,frontier=frontier,complete=at>=frontier,
                seconds=time.monotonic()-begin,failed=np.asarray(state['failed']).tolist()))
    if at<frontier:
        raise TimeoutError(f'{tag} stopped at {at} before its deadline')
    return jax.tree.map(np.asarray,state),evaluations


def plain_columns(cfg,targets):
    return [dict(target=target,target_index=cfg['targets'].index(target),initialization='zero') for target in targets]


def primary(root,cfg,n,name,gammas,targets,deadline):
    prefix=f'N{n}_{name}'; columns=plain_columns(cfg,targets); B=len(gammas); C=len(columns)
    _,banks=load_banks(root,n,name,gammas)
    rates=np.broadcast_to(np.array([.5/b['meta']['L'] for b in banks])[:,None],(B,C)).copy()
    run_batch(root,cfg,n,name,gammas,prefix+'_gd','gd',columns,rates,np.zeros((B,C)),cfg['training_steps'],deadline)
    recipes=list(itertools.product(cfg['adam_rates'],cfg['adam_epsilons']))
    trial_columns=[dict(c,initial_rate=rate,epsilon=eps) for c in columns for rate,eps in recipes]
    trial_rates=np.broadcast_to([c['initial_rate'] for c in trial_columns],(B,len(trial_columns))).copy()
    trial_eps=np.broadcast_to([c['epsilon'] for c in trial_columns],(B,len(trial_columns))).copy()
    state,rows=run_batch(root,cfg,n,name,gammas,prefix+'_adam_pilot','adam',trial_columns,
        trial_rates,trial_eps,cfg['adam_pilot_steps'],deadline)
    selected_rows=[r for r in rows if r['step'] in cfg['validation_steps']]
    assert sorted(r['step'] for r in selected_rows)==sorted(cfg['validation_steps'])
    scores=np.median(np.array([r['validation'] for r in selected_rows],dtype=float),axis=0)
    scores=np.where((state['failed']==0)&np.isfinite(scores),scores,np.inf)
    chosen=np.stack([np.argmin(scores[:,ti*len(recipes):(ti+1)*len(recipes)],axis=1)+ti*len(recipes) for ti in range(C)],axis=1)
    common=recipes.index((cfg['common_adam']['rate'],cfg['common_adam']['epsilon']))
    common_idx=np.broadcast_to(np.arange(C)*len(recipes)+common,(B,C))
    indices=np.concatenate([chosen,common_idx],axis=1)
    selected_state={}
    for key,value in state.items():
        if key in ['theta','mu','nu']:
            selected_state[key]=np.take_along_axis(value,indices[:,None,:],axis=2)
        elif key=='hits':
            selected_state[key]=np.take_along_axis(value,indices[:,:,None],axis=1)
        elif key=='failed':
            selected_state[key]=np.take_along_axis(value,indices,axis=1)
        else:
            selected_state[key]=value
    selected_rates=np.take_along_axis(trial_rates,indices,axis=1)
    selected_eps=np.take_along_axis(trial_eps,indices,axis=1)
    selection=dict(indices=indices.tolist(),rates=selected_rates.tolist(),epsilons=selected_eps.tolist(),
        scores=safe_json(scores.tolist()),eligible=np.isfinite(np.take_along_axis(scores,chosen,axis=1)).tolist(),
        boundary=(np.isin(selected_rates[:,:C],[min(cfg['adam_rates']),max(cfg['adam_rates'])])).tolist())
    core.write_json(root/'training'/f'{prefix}_selection.json',selection)
    final_columns=[dict(c,view=view) for view in ['selected','common'] for c in columns]
    run_batch(root,cfg,n,name,gammas,prefix+'_adam_continue','adam',final_columns,selected_rates,selected_eps,
        cfg['training_steps'],deadline,state=selected_state)


def robustness(root,cfg,name,deadline):
    n=cfg['n']; gammas=cfg['robust_gammas']; g=core.geometry(n); scale,neighbor=f.map_spec(g,name)
    columns=[]; init=[]
    for family,seed,target in itertools.product(cfg['initializations'],cfg['seeds'],cfg['robust_targets']):
        columns.append(dict(target=target,target_index=cfg['targets'].index(target),initialization=family,seed=seed))
        init.append(f.encode(f.initial_physical(g,family,seed)[:,None],scale,neighbor)[:,0])
    theta=np.broadcast_to(np.stack(init,axis=1),(len(gammas),g.width+1,len(columns))).copy()
    _,banks=load_banks(root,n,name,gammas)
    gd_rates=np.broadcast_to(np.array([.5/b['meta']['L'] for b in banks])[:,None],(len(gammas),len(columns))).copy()
    prefix=f'N{n}_{name}_initialization'
    run_batch(root,cfg,n,name,gammas,prefix+'_gd','gd',columns,gd_rates,np.zeros_like(gd_rates),
              cfg['training_steps'],deadline,initial=theta)
    chosen=json.loads((root/'training'/f'N{n}_{name}_selection.json').read_text())
    rates=np.array([[chosen['rates'][cfg['gammas'].index(gamma)][c['target_index']] for c in columns] for gamma in gammas])
    eps=np.array([[chosen['epsilons'][cfg['gammas'].index(gamma)][c['target_index']] for c in columns] for gamma in gammas])
    run_batch(root,cfg,n,name,gammas,prefix+'_adam','adam',columns,rates,eps,cfg['training_steps'],deadline,initial=theta)


def controls(root,cfg,name,stage,deadline):
    n=cfg['n']; gammas=cfg['robust_gammas']; B=len(gammas)
    _,banks=load_banks(root,n,name,gammas)
    if stage=='polynomial':
        columns=[dict(target=f'polynomial_{d}',degree=d,initialization='zero',chi=.5)
                 for d in cfg['polynomial_degrees']]
    elif stage=='rate':
        columns=[dict(c,chi=chi) for c in plain_columns(cfg,cfg['robust_targets'])
                 for chi in cfg['gd_rate_controls']]
    else:
        columns=[dict(c,chi=.5) for c in plain_columns(cfg,['sine_mix_2_6_10'])]
    rates=np.array([[c['chi']/bank['meta']['L'] for c in columns] for bank in banks])
    prefix=f'N{n}_{name}_{stage}'
    run_batch(root,cfg,n,name,gammas,prefix+'_gd','gd',columns,rates,np.zeros_like(rates),
              cfg['training_steps'],deadline)
    if stage=='coordinate':
        shape=(B,len(columns)); recipe=cfg['common_adam']
        run_batch(root,cfg,n,name,gammas,prefix+'_adam','adam',columns,np.full(shape,recipe['rate']),
                  np.full(shape,recipe['epsilon']),cfg['training_steps'],deadline)


def joint(root,cfg,n,deadline):
    folder=root/'joint'/f'N{n}'; folder.mkdir(parents=True,exist_ok=True)
    g=core.geometry(n); arrays=dict(np.load(root/f'common/N{n}/arrays.npz'))
    x=jnp.asarray(arrays['x_train']); y=f.target(x,'sine_mix_2_6_10',jnp)
    state=kernels.joint_initial(g.width,cfg['seeds']); rows=[]
    if (folder/'state.npz').exists():
        saved=dict(np.load(folder/'state.npz'))
        state={key:{p:jnp.asarray(saved[key+'_'+p]) for p in ['hidden','bias']} for key in ['params','mu','nu']}
        state.update(count=jnp.asarray(saved['count']),failed=jnp.asarray(saved['failed']))
        rows=json.loads((folder/'evaluations.json').read_text())
    kernel=kernels.make_joint_chunk(cfg['chunk_size'])
    predict=jax.jit(jax.vmap(kernels.joint_predict,in_axes=(0,None)))
    at=int(state['count'])
    def save():
        record=dict(step=at,n=n,width=g.width,failed=np.asarray(state['failed']).tolist())
        for label in ['train','validation','evaluation']:
            xx=jnp.asarray(arrays['x_'+label]); yy=f.target(xx,'sine_mix_2_6_10',jnp)
            values=np.asarray(predict(state['params'],xx))
            record[label]=(np.linalg.norm(values-np.asarray(yy),axis=1)/float(jnp.linalg.norm(yy))).tolist()
        slopes=np.asarray(state['params']['hidden'])[:,0]
        record['slope_quantiles']=np.quantile(np.abs(slopes),[.1,.5,.9,1],axis=1).T.tolist()
        rows.append(safe_json(record)); core.write_json(folder/'evaluations.json',rows)
        flat={key+'_'+p:np.asarray(value) for key in ['params','mu','nu'] for p,value in state[key].items()}
        flat.update(count=np.asarray(state['count']),failed=np.asarray(state['failed']))
        core.save_arrays(folder/'state.npz',**flat); core.save_arrays(folder/f'checkpoint_{at:06d}.npz',**flat)
        print(f'JOINT N={n} step={at}',flush=True)
    if not rows:
        save()
    while at<cfg['joint_steps'] and time.monotonic()<deadline:
        state,loss=kernel(state,x,y,cfg['joint_rate'],cfg['joint_epsilon']); jax.block_until_ready(state)
        old=at; at=int(state['count'])
        core.save_arrays(folder/f'loss_{old:06d}_{at:06d}.npz',loss=np.asarray(loss))
        if at in [1000,5000,10000,cfg['joint_steps']]:
            save()
    if at<cfg['joint_steps']:
        save(); raise TimeoutError('Joint baseline reached worker deadline')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--map',default='raw'); parser.add_argument('--n',type=int,default=512)
    parser.add_argument('--stage',choices=['primary','width','initialization','joint','rate','coordinate','polynomial'],required=True)
    parser.add_argument('--seconds',type=float,default=3000); parser.add_argument('--require-gpu',action='store_true')
    args=parser.parse_args(); cfg=json.loads((args.root/'manifest.json').read_text())['config']
    if args.require_gpu:
        train.verify_gpu(args.root,args.stage+'_'+args.map+str(args.n))
    deadline=time.monotonic()+args.seconds
    if args.stage=='primary':
        primary(args.root,cfg,cfg['n'],args.map,cfg['gammas'],cfg['targets'],deadline)
    elif args.stage=='width':
        primary(args.root,cfg,args.n,args.map,[1,4,args.n/8],cfg['robust_targets'],deadline)
    elif args.stage=='initialization':
        robustness(args.root,cfg,args.map,deadline)
    elif args.stage=='joint':
        joint(args.root,cfg,args.n,deadline)
    else:
        controls(args.root,cfg,args.map,args.stage,deadline)


if __name__=='__main__':
    main()
