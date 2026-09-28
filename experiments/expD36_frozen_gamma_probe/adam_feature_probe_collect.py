"""Select joint recipes, freeze their snapshots, and combine matched readout assays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .adam_feature_probe_prepare import geometry, save_input
    from .adam_feature_probe_analyze import envelope, setup, save
except ImportError:
    from adam_feature_probe_prepare import geometry, save_input
    from adam_feature_probe_analyze import envelope, setup, save


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def relative_error(p,x,y,width):
    a,b,c=p[:-1].reshape(3,width)
    return float(np.linalg.norm(np.tanh(x[:,None]*a+b)@c+p[-1]-y)/np.linalg.norm(y))


def prepare_learned(base,joint_run,output):
    import matplotlib.pyplot as plt
    manifest=json.loads((base/'manifest.json').read_text())
    _,info=geometry(manifest['width']); width=info['width']; h=info['spacing']
    assert width==512
    for k,v in info.items(): assert manifest[k]==v
    data=np.load(base/'input.npz')
    metadata=json.loads((joint_run/'metadata.json').read_text())
    assert metadata['width']==width
    recipes=metadata['recipes']
    state=np.load(joint_run/'state.npz'); params=state['p']
    errors=np.load(joint_run/'relative_error.npy',mmap_mode='r')
    rms=np.load(joint_run/'slope_rms.npy',mmap_mode='r')
    checkpoints=np.load(joint_run/'parameter_checkpoints.npy',mmap_mode='r')
    steps=np.load(joint_run/'checkpoint_steps.npy')
    horizon=len(errors)-1
    assert int(state['count'])==horizon and steps[-1]==horizon
    assert params.shape==(5,len(recipes),3*width+1)
    assert errors.shape==rms.shape==(horizon+1,5,len(recipes))
    np.testing.assert_array_equal(checkpoints[-1],params)
    scores=np.zeros((5,len(recipes))); endpoints=[]; max_gap=0.
    for seed in range(5):
        for ri,recipe in enumerate(recipes):
            p=params[seed,ri]
            train=relative_error(p,data['train_x'],data['target'],width)
            validation=relative_error(p,data['validation_x'],data['validation_target'],width)
            evaluation=relative_error(p,data['eval_x'],data['eval_target'],width)
            np.testing.assert_allclose(train,errors[-1,seed,ri],rtol=0,atol=1e-12)
            max_gap=max(max_gap,abs(train-errors[-1,seed,ri]))
            scores[seed,ri]=validation
            endpoints.append(dict(seed=seed,recipe_index=ri,**recipe,train_error=train,
                validation_error=validation,eval_error=evaluation,
                lambda_rms=float(np.sqrt(np.mean(p[:width]**2))*h)))
    # All candidates receive the same full budget. A single recipe is fixed
    # for each seed; its earlier snapshots are selected retrospectively.
    assert np.all(np.any(np.isfinite(scores),axis=1))
    choices=np.argmin(np.where(np.isfinite(scores),scores,np.inf),axis=1)
    for row in endpoints: row['selected']=row['recipe_index']==int(choices[row['seed']])
    wanted=[20000,100000,horizon]
    assert len(set(wanted))==3 and all(s in steps for s in wanted)
    aa=[];bb=[];rows=[]
    for step in wanted:
        k=int(np.flatnonzero(steps==step)[0])
        for seed,ri in enumerate(choices):
            p=checkpoints[k,seed,ri]
            a,b,c=p[:-1].reshape(3,width)
            actual=relative_error(p,data['train_x'],data['target'],width)
            np.testing.assert_allclose(actual,errors[step,seed,ri],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean(a*a)),rms[step,seed,ri],rtol=1e-14,atol=1e-14)
            max_gap=max(max_gap,abs(actual-errors[step,seed,ri]))
            aa.append(a);bb.append(b)
            rows.append(dict(name=f'learned_seed{seed}_step{step}',family='learned',seed=seed,
                gamma=None,snapshot_step=step,source_error=actual,
                lambda_rms=float(np.sqrt(np.mean(a*a))*h),joint_recipe_index=int(ri),
                joint_schedule=recipes[ri]['schedule'],joint_learning_rate=recipes[ri]['learning_rate']))
    save_input(output,aa,bb,rows,info,dict(joint_recipe_selection='One final-validation-selected joint recipe per seed, across schedules and learning rates; earlier snapshots of that selected trajectory are retrospective.',joint_source=str(joint_run),joint_input_sha256=metadata['input_sha256']))
    summary=dict(width=width,horizon=horizon,selected_recipe_indices=choices.tolist(),
        selection='Minimize final 4096-grid raw relative L2 error per seed; same recipe for all snapshots; 8192-grid errors are independent evaluation.',
        retrospective_snapshot_selection=True,endpoints=endpoints,max_reconstruction_gap=float(max_gap),
        source_metadata=metadata,source_state_sha256=digest(joint_run/'state.npz'))
    (output/'joint_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'legend.fontsize':9})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors=plt.get_cmap('tab10')(np.arange(5))
    for col,mode in enumerate(['cosine','constant','selected']):
        for seed,color in enumerate(colors):
            ri=int(choices[seed]) if mode=='selected' else next(i for i,r in enumerate(recipes) if r['schedule']==mode and np.isclose(r['learning_rate'],.002))
            label=f'Seed {seed}'
            if mode=='selected': label+=f" ({recipes[ri]['schedule']}, {recipes[ri]['learning_rate']:g})"
            envelope(axes[0,col],errors[:,seed,ri],color,label)
            envelope(axes[1,col],rms[:,seed,ri]*h,color)
        for row in range(2):
            setup(axes[row,col],horizon,'Relative output error' if row==0 else 'RMS slope × reference spacing (λ)')
            axes[row,col].set_xlabel('Joint-training updates (thousands)')
        axes[0,col].set_title('Validation-selected joint recipes' if mode=='selected' else f'{mode.title()}, shared initial rate 0.002')
        axes[0,col].legend()
        axes[1,col].set_title('Acquired slope scale (all 512 neurons)')
    fig.suptitle('Fresh joint training: output error and slope scale\nEvery hidden layer has exactly 512 neurons, including the construction halo budget',fontsize=15)
    save(fig,output/'joint_loss_and_scale.png')
    print(json.dumps(dict(learned_dictionaries=len(rows),selected_recipe_indices=choices.tolist(),output=str(output))))


def concatenate_npy(paths,output,axis=1,chunk=2048):
    arrays=[np.load(p,mmap_mode='r') for p in paths]
    first=arrays[0]; shape=list(first.shape);shape[axis]=sum(a.shape[axis] for a in arrays)
    for a in arrays:
        assert a.dtype==first.dtype and a.ndim==first.ndim
        assert all(a.shape[j]==first.shape[j] for j in range(a.ndim) if j!=axis)
    result=np.lib.format.open_memmap(output,mode='w+',dtype=first.dtype,shape=tuple(shape))
    offset=0
    for a in arrays:
        for begin in range(0,a.shape[0],chunk):
            src=[slice(None)]*a.ndim;src[0]=slice(begin,min(begin+chunk,a.shape[0]))
            dest=src.copy();dest[axis]=slice(offset,offset+a.shape[axis])
            result[tuple(dest)]=a[tuple(src)]
        offset+=a.shape[axis]
    result.flush()


def merge(base,learned,base_run,learned_run,output):
    manifests=[json.loads((p/'manifest.json').read_text()) for p in [base,learned]]
    for k in ['width','interior_intervals','interior_centers','halo_each_side','spacing','target_formula','target_normalizer','feature_layout']:
        assert manifests[0][k]==manifests[1][k],k
    assert manifests[0]['width']==512
    sources=[np.load(p/'input.npz') for p in [base,learned]]
    independent=['target','train_x','validation_x','eval_x','validation_target','eval_target']
    arrays={k:sources[0][k] for k in independent}
    for k in independent: np.testing.assert_array_equal(sources[0][k],sources[1][k])
    for k in ['a','b','features']:arrays[k]=np.concatenate([s[k] for s in sources],axis=0)
    assert arrays['features'].shape[-1]==513 and arrays['a'].shape[1]==512
    rows=manifests[0]['geometries']+manifests[1]['geometries']
    assert len(rows)==len(arrays['features']) and len({g['name'] for g in rows})==len(rows)
    output.mkdir(parents=True,exist_ok=True);run=output/'run';run.mkdir(exist_ok=True)
    np.savez_compressed(output/'input.npz',**arrays)
    del arrays
    manifest=dict(manifests[0]);manifest.update(geometries=rows,input_sha256=digest(output/'input.npz'),merged_sources=[str(base),str(learned)],joint_recipe_selection=manifests[1]['joint_recipe_selection'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    metas=[json.loads((p/'metadata.json').read_text()) for p in [base_run,learned_run]]
    for meta,source_manifest in zip(metas,manifests):
        assert meta['input_sha256']==source_manifest['input_sha256']
    assert metas[0]['recipes']==metas[1]['recipes'] and metas[0]['config']==metas[1]['config']
    for name in ['relative_error.npy','readout_checkpoints.npy']:
        concatenate_npy([p/name for p in [base_run,learned_run]],run/name,axis=1,
                        chunk=2048 if name=='relative_error.npy' else 4)
    steps=[np.load(p/'checkpoint_steps.npy') for p in [base_run,learned_run]]
    np.testing.assert_array_equal(*steps);np.save(run/'checkpoint_steps.npy',steps[0])
    states=[np.load(p/'state.npz') for p in [base_run,learned_run]]
    assert set(states[0].files)==set(states[1].files)
    joined={}
    for k in states[0].files:
        if k=='count':
            np.testing.assert_array_equal(states[0][k],states[1][k]);joined[k]=states[0][k]
        else:joined[k]=np.concatenate([s[k] for s in states],axis=0)
    np.savez(run/'state.npz',**joined)
    meta=dict(metas[0]);meta.update(input_sha256=manifest['input_sha256'],merged_run_sources=[str(base_run),str(learned_run)],merged_geometry_count=len(rows),feature_shape=[len(rows),manifest['samples'],manifest['width']+1])
    (run/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    endpoint=np.load(run/'relative_error.npy',mmap_mode='r')[-1]
    for i,p in enumerate([base_run,learned_run]):
        start=0 if i==0 else len(manifests[0]['geometries']);stop=start+len(manifests[i]['geometries'])
        np.testing.assert_array_equal(endpoint[start:stop],np.load(p/'relative_error.npy',mmap_mode='r')[-1])
    print(json.dumps(dict(merged_geometries=len(rows),output=str(output))))


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='adam_collect_test_') as tmp:
        p=Path(tmp);a=np.arange(7*2*3).reshape(7,2,3);b=np.arange(7*4*3).reshape(7,4,3)+1000
        np.save(p/'a.npy',a);np.save(p/'b.npy',b)
        concatenate_npy([p/'a.npy',p/'b.npy'],p/'c.npy',chunk=2)
        np.testing.assert_array_equal(np.load(p/'c.npy'),np.concatenate([a,b],axis=1))
        c=np.arange(5*2*3*9).reshape(5,2,3,9);d=np.arange(5*4*3*9).reshape(5,4,3,9)
        np.save(p/'a.npy',c);np.save(p/'b.npy',d)
        concatenate_npy([p/'a.npy',p/'b.npy'],p/'c.npy',chunk=2)
        np.testing.assert_array_equal(np.load(p/'c.npy'),np.concatenate([c,d],axis=1))
        _,info=geometry(512)
        for label,gcount in [('base',2),('learned',3)]:
            source=p/label;source.mkdir();run=source/'run';run.mkdir()
            shared=dict(target=np.ones(4),train_x=np.arange(4.),validation_x=np.arange(6.),
                        eval_x=np.arange(8.),validation_target=np.ones(6),eval_target=np.ones(8))
            np.savez(source/'input.npz',**shared,a=np.ones((gcount,512)),b=np.zeros((gcount,512)),features=np.ones((gcount,4,513)))
            manifest=dict(info,samples=4,target_formula='fixture',target_normalizer=1.,feature_layout='fixture',
                geometries=[dict(name=f'{label}_{i}') for i in range(gcount)],
                input_sha256=digest(source/'input.npz'),joint_recipe_selection='fixture')
            (source/'manifest.json').write_text(json.dumps(manifest))
            meta=dict(recipes=[dict(schedule='constant',learning_rate=.002)],config=dict(horizon=6),input_sha256=manifest['input_sha256'])
            (run/'metadata.json').write_text(json.dumps(meta))
            np.save(run/'relative_error.npy',np.ones((7,gcount,1)))
            np.save(run/'readout_checkpoints.npy',np.zeros((2,gcount,1,513)))
            np.save(run/'checkpoint_steps.npy',np.array([0,6]))
            np.savez(run/'state.npz',w=np.zeros((gcount,513,1)),m=np.zeros((gcount,513,1)),v=np.zeros((gcount,513,1)),count=6)
        merge(p/'base',p/'learned',p/'base/run',p/'learned/run',p/'combined')
        assert np.load(p/'combined/input.npz')['features'].shape==(5,4,513)
        assert np.load(p/'combined/run/relative_error.npy').shape==(7,5,1)
        assert json.loads((p/'combined/run/metadata.json').read_text())['feature_shape']==[5,4,513]
    print('Chunked trace and parameter-checkpoint merge checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    for name in ['prepare-learned','merge','self-test']:modes.add_argument('--'+name,action='store_true')
    for name in ['base','joint-run','learned','base-run','learned-run','output']:p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.self_test:self_test()
    elif args.prepare_learned:
        if not all([args.base,args.joint_run,args.output]):p.error('--prepare-learned requires --base --joint-run --output')
        prepare_learned(args.base,args.joint_run,args.output)
    else:
        if not all([args.base,args.learned,args.base_run,args.learned_run,args.output]):p.error('--merge requires --base --learned --base-run --learned-run --output')
        merge(args.base,args.learned,args.base_run,args.learned_run,args.output)
