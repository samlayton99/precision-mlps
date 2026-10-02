"""Offline physical-gamma and original-group diagnostics; no test predictions.

  python -m experiments.expD40_bimodal_gamma.diagnostics initial
  python -m experiments.expD40_bimodal_gamma.diagnostics saturation
  python -m experiments.expD40_bimodal_gamma.diagnostics pilot --ablate
  python -m experiments.expD40_bimodal_gamma.diagnostics final --ablate

Final mode requires the selected seed-0 20k checkpoints; pilot mode uses their
selected 10k pilot counterparts and writes separately suffixed artifacts.
Neither selects an initializer or modifies parameters used in training. Optional ablations replace
one last-hidden-layer group's activations by their fitting-data means without
refitting the readout; they measure distribution-shift sensitivity.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precision_d40_matplotlib')
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expD40_bimodal_gamma import run as campaign

OUT = campaign.OUT
FIGS = OUT / 'figures'
COLORS = {'low':'tab:blue', 'high':'tab:orange', 'unmixed':'#777777'}
NAMES = {'low':'Originally low', 'high':'Originally high',
         'unmixed':'Layer initialized without a mixture'}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})


def save_plot(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS/name, dpi=155, bbox_inches='tight')
    plt.close(fig)


def validate_sources(cfg):
    for path, expected in cfg['d40_source_sha256'].items():
        assert campaign.digest(ROOT/path) == expected, ('Training source changed',path)


def fitting_data(task, cfg, validation=False):
    arrays, metadata = campaign.base.load_data(task,cfg)
    splits = ['train','val'] if validation else ['train']
    # The common loader defines the split. Held-out arrays never reach any
    # prediction, geometry diagnostic, or ablation in this module.
    inputs = {k:torch.from_numpy(arrays[k][0]) for k in splits}
    targets = {k:arrays[k][1] for k in splits}
    return inputs, targets, metadata


def groups(layer_info):
    high = np.asarray(layer_info['high_mask'],dtype=bool)
    if layer_info['changed']:
        return {'low':~high, 'high':high}
    return {'unmixed':np.ones(len(high),dtype=bool)}


def quantiles(values):
    return dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(values,[0,.25,.5,.75,1]))))


@torch.no_grad()
def verify_reference(model, info, reference, cfg):
    assert torch.equal(model.fc3.weight,reference.fc3.weight)
    assert torch.equal(model.fc3.bias,reference.fc3.bias)
    for index,name in enumerate(['fc1','fc2']):
        layer, original = getattr(model,name), getattr(reference,name)
        gamma, ref_gamma = layer.weight.norm(dim=1), original.weight.norm(dim=1)
        torch.testing.assert_close(layer.weight/gamma[:,None],original.weight/ref_gamma[:,None],rtol=0,atol=1e-13)
        torch.testing.assert_close(-layer.bias/gamma,-original.bias/ref_gamma,rtol=1e-12,atol=1e-12)
        record = info['layers'][index]
        assert record['high_count'] == len(gamma)//2
        if record['changed']:
            mask=torch.tensor(record['high_mask'])
            assert int(mask.sum())==256
            g=record['reference_gamma']
            lo,hi=[cfg['d40_design'][key] for key in ['low_multiplier','high_multiplier']]
            torch.testing.assert_close(gamma[~mask],torch.full_like(gamma[~mask],lo*g),rtol=1e-13,atol=1e-13)
            torch.testing.assert_close(gamma[mask],torch.full_like(gamma[mask],hi*g),rtol=1e-13,atol=1e-13)
        else:
            assert torch.equal(layer.weight,original.weight)
            assert torch.equal(layer.bias,original.bias)


@torch.no_grad()
def layer_records(model, info, fitting_x):
    x=fitting_x[:2048]
    rows=[]
    for index,name in enumerate(['fc1','fc2']):
        layer=getattr(model,name)
        w=layer.weight.detach().numpy().copy()
        gamma=np.linalg.norm(w,axis=1)
        assert np.all(np.isfinite(gamma)) and np.all(gamma>0)
        z=layer(x)
        x=torch.tanh(z)
        activations=x.numpy()
        preactivation=z.numpy()
        row=dict(name=name,changed=info['layers'][index]['changed'],
                 reference_gamma=info['layers'][index]['reference_gamma'],
                 high_mask=info['layers'][index]['high_mask'],
                 gamma=gamma.tolist(),gamma_summary=quantiles(gamma),
                 centers=(-layer.bias.detach().numpy()/gamma).tolist(),groups={})
        for group,mask in groups(info['layers'][index]).items():
            h=activations[:,mask]
            row['groups'][group]=dict(count=int(mask.sum()),gamma=quantiles(gamma[mask]),
                saturation_fraction=float(np.mean(np.abs(h)>.99)),
                mean_tanh_derivative=float(np.mean(1-h*h)),
                activation_rms=float(np.sqrt(np.mean(h*h))),
                preactivation_rms=float(np.sqrt(np.mean(preactivation[:,mask]**2))),
                median_activation_std=float(np.median(np.std(h,axis=0))))
        rows.append(row)
    return rows


def histogram_bounds(rows):
    values=np.concatenate([np.asarray(r['gamma']) for r in rows])
    lower=10**(np.floor(np.log10(values.min())*2)/2)
    upper=10**(np.ceil(np.log10(values.max())*2)/2)
    if lower==upper:
        lower/=2;upper*=2
    return np.geomspace(lower,upper,65)


def histogram(ax, row, bins, reference=None):
    gamma=np.asarray(row['gamma'])
    width=len(gamma)
    for group,mask in groups(row).items():
        ax.hist(gamma[mask],bins=bins,weights=np.full(mask.sum(),100/width),
                color=COLORS[group],alpha=.46,label=NAMES[group])
    if reference is not None and row['changed']:
        ref=np.asarray(reference['gamma'])
        ax.hist(ref,bins=bins,weights=np.full(len(ref),100/len(ref)),
                histtype='step',color='#777777',lw=1.2,label='Paired centered reference')
    ax.set(xscale='log',xlim=(bins[0],bins[-1]),ylim=(0,105),
           xlabel=r'Physical row norm $\gamma=\|w_i\|_2$',ylabel='All layer rows (%)')
    ax.grid(axis='y',alpha=.15)


def legends(include_reference=False):
    entries=[Line2D([],[],color=COLORS[k],lw=7,alpha=.55,label=NAMES[k]) for k in ['low','high','unmixed']]
    if include_reference:
        entries.append(Line2D([],[],color='#777777',lw=1.2,label='Paired centered reference'))
    return entries


def initial():
    cfg=campaign.config();validate_sources(cfg)
    spec=cfg['d40_design']
    result=dict(source_sha256=cfg['d40_source_sha256'],diagnostic_script_sha256=campaign.digest(Path(__file__)),
                seed=0,width=512,probe='First min(2048,n_fit) fitting rows; no validation/test predictions.',
                group_definition='Low/high labels are the original assignment, only in layers initialized as mixtures.',rows=[])
    for task in spec['tasks']:
        inputs,targets,metadata=fitting_data(task,cfg)
        reference,ref_info=campaign.make_model(metadata['d_in'],512,0,'centered',inputs['train'],cfg)
        models={'centered':(reference,ref_info)}
        for arm in ['mix_both','mix_last']:
            model,info=campaign.make_model(metadata['d_in'],512,0,arm,inputs['train'],cfg)
            verify_reference(model,info,reference,cfg)
            models[arm]=(model,info)
        for arm,(model,info) in models.items():
            record=dict(task=task,scheme=arm,data=metadata,
                        pairing_verified=True,layers=layer_records(model,info,inputs['train']))
            # Match any pilot already written, using its fit split only. This
            # check neither requires nor waits for the training run to finish.
            paths=list((OUT/'data/pilot').glob(f'{task}_{arm}_w512_*_seed0_steps10000.json'))
            if paths:
                assert len(paths)==1
                saved=json.loads(paths[0].read_text())
                assert saved['identity']['config']==cfg and saved['data']==metadata
                pred,_=campaign.base.predictions(model,inputs)
                error=campaign.base.mse(pred['train'],targets['train'])
                np.testing.assert_allclose(error,saved['trace'][0]['learned']['train'],rtol=1e-10,atol=1e-12)
                record['initial_fit_mse']=error
                record['initial_fit_trace_verified']=True
            result['rows'].append(record)
        print('INITIAL',task,'reference + both/last mixtures verified',flush=True)
    bins=histogram_bounds([layer for r in result['rows'] for layer in r['layers']])
    result['histogram_edges']=bins.tolist()
    campaign.base.save_json(OUT/'gamma_diagnostics_initial.json',result)
    lookup={(r['task'],r['scheme']):r for r in result['rows']}
    for arm in ['mix_both','mix_last']:
        fig,axes=plt.subplots(4,2,figsize=(12,13),sharex=True,sharey=True)
        for i,task in enumerate(spec['tasks']):
            for j in [0,1]:
                row=lookup[task,arm]['layers'][j]
                histogram(axes[i,j],row,bins,lookup[task,'centered']['layers'][j])
                title=f"{campaign.base.TITLES[task]} · layer {j+1}"
                if not row['changed']:title+=' (unmixed)'
                axes[i,j].set_title(title+'\nWidth 512')
        fig.suptitle('Actual initial gamma distributions · '+('both layers mixed' if arm=='mix_both' else 'layer 2 mixed')+' · seed 0\n0.1g + 3g (30:1 separation); g = paired layer median',y=1.014,fontsize=14)
        fig.legend(handles=legends(True),loc='upper center',bbox_to_anchor=(.5,.973),ncol=2,frameon=False)
        fig.text(.5,.008,'Common logarithmic gamma bins and limits; heights are percentages of all 512 rows.\n'
                 'Low/high are initial assignment labels. Gray layers have no low/high mixture.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.895,bottom=.065,hspace=.46,wspace=.16)
        save_plot(fig,f'gamma_initial_{arm}.png')


def selected_path(task, arm, cfg, phase='compare'):
    assert phase in ['pilot','compare']
    recipes=json.loads((campaign.BASE_OUT/'selected_recipe.json').read_text())
    lr=recipes[task]['lr']
    steps=cfg['d40_design']['screen_steps' if phase=='pilot' else 'confirmation_steps']
    assert steps==(10000 if phase=='pilot' else 20000)
    root=OUT/'followup' if phase=='pilot' and arm.startswith('mild_') else OUT
    return root/'data'/phase/f'{task}_{arm}_w512_lr{lr:g}_seed0_steps{steps}.json'


def initial_saturation():
    """Plot the saved initialization probe; no data loading or model evaluation."""
    source=OUT/'gamma_diagnostics_initial.json'
    result=json.loads(source.read_text())
    rows={(r['task'],r['scheme']):r for r in result['rows']}
    tasks=list(dict.fromkeys(r['task'] for r in result['rows']))
    fig,axes=plt.subplots(2,2,figsize=(11,6.6),sharex=True,sharey=True)
    summary=[]
    for column,arm in enumerate(['mix_both','mix_last']):
        for layer in [0,1]:
            ax=axes[layer,column]
            for index,task in enumerate(tasks):
                row=rows[task,arm]['layers'][layer]
                for group,record in row['groups'].items():
                    offset={'low':-.10,'high':.10,'unmixed':0}[group]
                    percent=100*record['saturation_fraction']
                    ax.scatter(index+offset,percent,color=COLORS[group],s=50,zorder=3,clip_on=False)
                    summary.append(dict(task=task,scheme=arm,layer=layer+1,
                                        group=group,saturation_percent=percent))
            title=('Both layers mixed' if column==0 else 'Layer 2 mixed')+f' · layer {layer+1}'
            if column==1 and layer==0:title+=' (unmixed)'
            ax.set(title=title,ylim=(0,100),xlim=(-.45,len(tasks)-.55),
                   xticks=range(len(tasks)),xticklabels=[campaign.base.TITLES[t] for t in tasks],
                   yticks=[0,25,50,75,100],ylabel='Saturated activations (%)')
            ax.tick_params(axis='x',rotation=15,labelsize=9)
            ax.grid(axis='y',alpha=.18)
    fig.suptitle('Initial saturation by original gamma group\n0.1g + 3g mixture · two width-512 tanh layers · seed 0',y=1.055)
    fig.legend(handles=legends(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.005,'Saturated: |tanh(z)| > 0.99. First min(2,048, n_fit) fitting inputs; percentages are within each group.\n'
             'This describes activation ranges; it does not establish whether saturation causes learning differences.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.80,bottom=.12,hspace=.32,wspace=.16)
    save_plot(fig,'group_initial_saturation.png')
    campaign.base.save_json(OUT/'gamma_diagnostics_saturation.json',dict(
        source_file=str(source.relative_to(ROOT)),source_sha256=campaign.digest(source),
        diagnostic_script_sha256=campaign.digest(Path(__file__)),
        plot='figures/group_initial_saturation.png',rows=summary))


def final(ablate=False):
    """Strict 20k confirmation diagnostics; never fall back to pilot checkpoints."""
    movement('compare',ablate=ablate)


def pilot(ablate=False):
    """Selected 10k pilot diagnostics; no confirmation files are read or written."""
    movement('pilot',ablate=ablate)


def movement(phase,ablate=False):
    assert phase in ['pilot','compare']
    steps=10000 if phase=='pilot' else 20000
    from experiments.expD40_bimodal_gamma import followup
    selection_path=OUT/'selection.json'
    selection=json.loads(selection_path.read_text())
    selection_hash=campaign.digest(selection_path)
    original_cfg=campaign.config();validate_sources(original_cfg)
    mild_cfg=followup.config();validate_sources(mild_cfg)
    assert selection['source_sha256']==original_cfg['d40_source_sha256']
    assert selection['followup_sha256']==mild_cfg['d40_followup_sha256']
    assert selection['followup_plan_sha256']==mild_cfg['d40_followup_plan_sha256']
    for path,expected in selection['screen_sha256'].items():
        assert campaign.digest(ROOT/path)==expected,('Locked screen changed',path)
    result=dict(source_sha256=original_cfg['d40_source_sha256'],
                followup_sha256=mild_cfg['d40_followup_sha256'],
                followup_plan_sha256=mild_cfg['d40_followup_plan_sha256'],
                diagnostic_script_sha256=campaign.digest(Path(__file__)),
                selection_sha256=selection_hash,seed=0,steps=steps,phase=phase,
                selection_caveat='Pilot mixtures were chosen on these validation data; this movement audit makes no new training choices.',
                probe='First min(2048,n_fit) fitting rows for saturation; all fit/validation rows for prediction verification.',
                caveat='Original groups remain fixed labels. Layer-2 directions use hidden-neuron coordinates. No final bank lambda is assigned.',
                movement_caveat='Rotation, norm ratios, and relative displacement depend on the initial norm; compare absolute displacement separately. None alone measures learning or causal importance.',
                ablation_caveat='Train-mean replacement without refitting measures distribution-shift sensitivity, not causal importance.',rows=[])
    for task in original_cfg['d40_design']['tasks']:
        arm=selection['choices'][task]['mixture']
        assert arm in ['mix_both','mix_last','mild_mix_both','mild_mix_last']
        assert [task,arm] in selection['confirmation_pairs']
        expected_cfg=mild_cfg if arm.startswith('mild_') else original_cfg
        path=selected_path(task,arm,expected_cfg,phase)
        saved=json.loads(path.read_text());identity=saved['identity']
        cfg=identity['config']
        assert cfg==expected_cfg,'Saved configuration differs from the locked initializer'
        assert saved['complete'] and saved['trace'][0]['step']==0 and saved['trace'][-1]['step']==steps
        if phase=='pilot':
            assert all(set(q['learned'])=={'train','val'} for q in saved['trace'])
        recipes=json.loads((campaign.BASE_OUT/'selected_recipe.json').read_text())
        expected=dict(task=task,scheme=arm,seed=0,steps=steps,width=512,phase=phase,
                      lr=recipes[task]['lr'],config=cfg)
        if task in cfg.get('task_data_protocols',{}):
            expected['data_protocol']=cfg['task_data_protocols'][task]
        assert campaign.base.same_identity(identity,expected)
        assert campaign.base.valid_data_identity(identity,cfg)
        inputs,targets,metadata=fitting_data(task,cfg,validation=True)
        assert metadata==saved['data']
        reference,_=followup.make_model(metadata['d_in'],512,0,'centered',inputs['train'],cfg)
        model,info=followup.make_model(metadata['d_in'],512,0,arm,inputs['train'],cfg)
        verify_reference(model,info,reference,cfg)
        initial_weights={name:getattr(model,name).weight.detach().numpy().copy() for name in ['fc1','fc2']}
        initial_rows=layer_records(model,info,inputs['train'])
        pred,_=campaign.base.predictions(model,inputs)
        initial_errors={k:campaign.base.mse(pred[k],targets[k]) for k in inputs}
        for split,error in initial_errors.items():
            np.testing.assert_allclose(error,saved['trace'][0]['learned'][split],rtol=1e-10,atol=1e-12)
        checkpoint_path=path.with_suffix('.pt')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert campaign.base.same_identity(identity,checkpoint['identity'])
        model.load_state_dict(checkpoint['model'])
        pred,features=campaign.base.predictions(model,inputs,with_features=ablate)
        final_errors={k:campaign.base.mse(pred[k],targets[k]) for k in inputs}
        for split,error in final_errors.items():
            np.testing.assert_allclose(error,saved['trace'][-1]['learned'][split],rtol=1e-10,atol=1e-12)
        final_rows=layer_records(model,info,inputs['train'])
        record=dict(task=task,scheme=arm,identity=identity,data=metadata,
                    low_multiplier=cfg['d40_design']['low_multiplier'],
                    high_multiplier=cfg['d40_design']['high_multiplier'],
                    source_file=str(path.relative_to(ROOT)),source_sha256=campaign.digest(path),
                    checkpoint_sha256=campaign.digest(checkpoint_path),
                    initial_errors=initial_errors,final_errors=final_errors,
                    trace_predictions_verified=True,initial=initial_rows,final=final_rows,layers=[])
        for i,name in enumerate(['fc1','fc2']):
            before=initial_weights[name];after=getattr(model,name).weight.detach().numpy()
            gb=np.linalg.norm(before,axis=1);ga=np.linalg.norm(after,axis=1)
            cosine=np.clip(np.sum((before/gb[:,None])*(after/ga[:,None]),axis=1),-1,1)
            ratios=ga/gb
            displacement=np.linalg.norm(after-before,axis=1)
            relative_displacement=displacement/gb
            np.testing.assert_allclose(displacement**2,ga**2+gb**2-2*ga*gb*cosine,rtol=1e-10,atol=1e-12)
            row=dict(name=name,changed=info['layers'][i]['changed'],high_mask=info['layers'][i]['high_mask'],
                     row_cosine=cosine.tolist(),gamma_ratio=ratios.tolist(),
                     row_l2_displacement=displacement.tolist(),
                     relative_row_l2_displacement=relative_displacement.tolist(),groups={})
            for group,mask in groups(row).items():
                row['groups'][group]=dict(count=int(mask.sum()),gamma_ratio=quantiles(ratios[mask]),
                    row_cosine=quantiles(cosine[mask]),
                    row_l2_displacement=quantiles(displacement[mask]),
                    relative_row_l2_displacement=quantiles(relative_displacement[mask]),
                    initial_saturation_fraction=initial_rows[i]['groups'][group]['saturation_fraction'],
                    final_saturation_fraction=final_rows[i]['groups'][group]['saturation_fraction'])
            record['layers'].append(row)
        if ablate:
            weights=model.fc3.weight.detach().numpy().T
            mean=features['train'].mean(0)
            record['ablation']={}
            for group,mask in groups(info['layers'][1]).items():
                errors={}
                for split,h in features.items():
                    replaced=pred[split]-(h[:,mask]-mean[mask])@weights[mask]
                    errors[split]=campaign.base.mse(replaced,targets[split])
                record['ablation'][group]=dict(replacement='Per-neuron fitting-data mean',
                    mse=errors,mse_ratio={k:errors[k]/final_errors[k] for k in errors})
            # Mean replacement must preserve the fitting-data mean prediction.
            all_replaced=pred['train']-(features['train']-mean)@weights
            np.testing.assert_allclose(all_replaced.mean(0),pred['train'].mean(0),rtol=1e-10,atol=1e-12)
        result['rows'].append(record)
        print(phase.upper(),task,arm,steps,'initial/endpoint train+val predictions and groups verified',flush=True)
    assert campaign.digest(selection_path)==selection_hash,'Selection changed during diagnostics'
    bins=histogram_bounds([layer for r in result['rows'] for stage in ['initial','final'] for layer in r[stage]])
    result['histogram_edges']=bins.tolist()
    campaign.base.save_json(OUT/('gamma_diagnostics_pilot.json' if phase=='pilot' else 'gamma_diagnostics_final.json'),result)
    final_plots(result,bins)


def final_plots(result,bins):
    is_pilot=result['phase']=='pilot'
    file_suffix='_pilot' if is_pilot else ''
    step_label=f"{result['steps']:,} steps"+(' · pilot' if is_pilot else '')
    for record in result['rows']:
        task,arm=record['task'],record['scheme']
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for i in [0,1]:
            for j,stage in enumerate(['initial','final']):
                histogram(axes[i,j],record[stage][i],bins)
                suffix=' (unmixed)' if not record[stage][i]['changed'] else ''
                axes[i,j].set_title(f'Layer {i+1}{suffix} · '+('initialization' if stage=='initial' else step_label))
        modes=f"{record['low_multiplier']:g}g + {record['high_multiplier']:g}g"
        fig.suptitle(f"{campaign.base.TITLES[task]} · {arm.replace('_',' ')} · physical gamma movement\n{modes} initially · width 512 + 512 · seed 0",y=1.015)
        fig.legend(handles=legends(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False,fontsize=9)
        fig.text(.5,.012,'Colors retain original group labels; they do not classify final neurons by their current norm.\n'
                 'All tasks share physical gamma bins and limits. Each panel contains every row.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.82,bottom=.10,hspace=.38,wspace=.18)
        save_plot(fig,f'gamma_movement_{task}{file_suffix}.png')
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
        for i,row in enumerate(record['layers']):
            values=np.asarray(row['row_cosine'])
            for group,mask in groups(row).items():
                axes[i].hist(values[mask],bins=np.linspace(-1,1,81),weights=np.full(mask.sum(),100/len(mask)),
                    color=COLORS[group],alpha=.46,label=NAMES[group])
            axes[i].set(title=f'Layer {i+1}',xlim=(-1,1),ylim=(0,105),xlabel='Cosine with own initial row',ylabel='All layer rows (%)')
            axes[i].grid(axis='y',alpha=.15)
        fig.suptitle(f"{campaign.base.TITLES[task]} · direction retention by original group\n{arm.replace('_',' ')} · width 512 + 512 · {step_label} · seed 0",y=1.1)
        fig.legend(handles=legends(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False,fontsize=9)
        fig.subplots_adjust(top=.78,bottom=.14,wspace=.16)
        save_plot(fig,f'gamma_direction_cosines_{task}{file_suffix}.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    labels=[campaign.base.TITLES[r['task']] for r in result['rows']]
    for j,record in enumerate(result['rows']):
        for i,row in enumerate(record['layers']):
            for group,summary in row['groups'].items():
                offset={'low':-.1,'high':.1,'unmixed':0}[group]
                for column,metric in [(0,'gamma_ratio'),(1,'row_cosine')]:
                    q=summary[metric];median=q['median']
                    axes[i,column].errorbar(j+offset,median,yerr=[[median-q['q25']],[q['q75']-median]],
                        marker='o',color=COLORS[group],capsize=3,linestyle='none')
    for i in [0,1]:
        axes[i,0].set(yscale='log',ylabel='Endpoint / initial row norm',title=f'Layer {i+1}: median and middle 50%')
        axes[i,0].axhline(1,color='#999999',ls=':',lw=1)
        axes[i,1].set(ylim=(-1,1),ylabel='Cosine with own initial row',title=f'Layer {i+1}: median and middle 50%')
        for ax in axes[i]:
            ax.set_xticks(range(len(labels)),labels,rotation=15,ha='right')
            ax.grid(axis='y',alpha=.15)
    limits=[ax.get_ylim() for ax in axes[:,0]]
    for ax in axes[:,0]:ax.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
    fig.suptitle(f'Original-group row norms and direction retention\nLocked task-specific mixtures · {step_label} · seed 0',y=1.045)
    fig.legend(handles=legends(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.085,hspace=.38,wspace=.2)
    save_plot(fig,f'group_movement_summary{file_suffix}.png')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey='col')
    for j,record in enumerate(result['rows']):
        for i,row in enumerate(record['layers']):
            for group,summary in row['groups'].items():
                offset={'low':-.1,'high':.1,'unmixed':0}[group]
                for column,metric in enumerate(['row_l2_displacement','relative_row_l2_displacement']):
                    q=summary[metric];median=q['median']
                    axes[i,column].errorbar(j+offset,median,yerr=[[median-q['q25']],[q['q75']-median]],
                        marker='o',color=COLORS[group],capsize=3,linestyle='none')
    for i in [0,1]:
        for column in [0,1]:
            axes[i,column].set(yscale='log',title=f'Layer {i+1} · '+('absolute' if column==0 else 'relative to initial norm'),
                ylabel=(r'$\|w_{end}-w_0\|_2$' if column==0 else r'$\|w_{end}-w_0\|_2 / \|w_0\|_2$'))
            axes[i,column].set_xticks(range(len(labels)),labels,rotation=15,ha='right')
            axes[i,column].grid(axis='y',which='both',alpha=.15)
    fig.suptitle(f'Absolute and relative weight-row displacement\n{step_label} · seed 0 · median and middle 50% of original groups',y=1.055)
    fig.legend(handles=legends(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.008,'Common logarithmic limits within each column. These are weight-row changes; biases are excluded.\n'
             'Large relative movement or rotation need not mean large absolute movement or greater learning.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.79,bottom=.16,hspace=.35,wspace=.20)
    save_plot(fig,f'group_displacement{file_suffix}.png')
    if all('ablation' in r for r in result['rows']):
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
        for ax,split in zip(axes,['train','val']):
            for j,record in enumerate(result['rows']):
                for group,offset in [('low',-.12),('high',.12)]:
                    ax.scatter(j+offset,record['ablation'][group]['mse_ratio'][split],color=COLORS[group],s=40)
            ax.set(yscale='log',title='Fitting' if split=='train' else 'Validation',ylabel='Mean-replacement MSE / actual MSE',
                   xticks=range(len(labels)),xticklabels=labels)
            ax.tick_params(axis='x',rotation=18)
            ax.axhline(1,color='#777777',ls=':');ax.grid(axis='y',which='both',alpha=.15)
        fig.suptitle('Last hidden layer: sensitivity to replacing a group by its fitting-data mean\n'+step_label+' · seed 0',y=1.15)
        fig.legend(handles=legends()[:2],loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
        fig.text(.5,.008,'No readout refit. This changes the representation distribution and is not a causal-importance estimate.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.18,wspace=.15)
        save_plot(fig,f'group_mean_replacement_sensitivity{file_suffix}.png')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['initial','saturation','pilot','final'])
    p.add_argument('--ablate',action='store_true',help='Pilot/final mode: optional hidden2 group mean-replacement diagnostic')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if args.mode=='initial':
        assert not args.ablate,'Mean-replacement diagnostics require final checkpoints'
        initial()
    elif args.mode=='saturation':
        assert not args.ablate,'Mean-replacement diagnostics require final checkpoints'
        initial_saturation()
    elif args.mode=='pilot':
        pilot(ablate=args.ablate)
    else:
        final(ablate=args.ablate)
