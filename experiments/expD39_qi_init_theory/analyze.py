"""Complete-campaign validation selection, plots and confirmation summaries."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precision_d39_matplotlib')
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from pathlib import Path
import yaml
from experiments.expD39_qi_init_theory.run import base, config, OUT, BASE_OUT, HERE, ROOT

LABELS = {'standard':'Standard', 'qi':'Original QI', 'spacing':'Exact spacing',
          'balanced':'Balanced banks', 'centered':'Centered banks', 'collar':'25% collar',
          'common':'Common gamma', 'lambda05':r'$\lambda=0.5$', 'lambda10':r'$\lambda=1$',
          'directions64':'64 directions', 'first_only':'QI in layer 1', 'last_only':'QI in layer 2',
          'soft24':'24 directions · soft', 'sharp64':'64 directions · sharp'}
FIGS = OUT / 'figures'
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design():
    return yaml.safe_load((HERE / 'config.yaml').read_text())


def validated_screen():
    cfg, spec = config(), design()
    manifest = json.loads((OUT/'variant_manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert file_hash(ROOT/name) == digest, f'Source changed since screening: {name}'
    recipes = json.loads((BASE_OUT / 'selected_recipe.json').read_text())
    rows = {}
    for task in spec['screen_tasks']:
        for arm in spec['screen_variants']:
            seed, steps, lr = spec['screen_seed'], spec['screen_steps'], recipes[task]['lr']
            path = OUT / 'data/pilot' / f'{task}_{arm}_w512_lr{lr:g}_seed{seed}_steps{steps}.json'
            r = json.loads(path.read_text())
            expected = dict(task=task, scheme=arm, lr=lr, seed=seed, steps=steps,
                            width=512, phase='pilot', config=cfg)
            if task in cfg.get('task_data_protocols', {}):
                expected['data_protocol'] = cfg['task_data_protocols'][task]
            assert base.same_identity(r['identity'], expected), path
            assert r.get('complete') and r['trace'][-1]['step'] == steps, path
            assert all(set(q['learned']) == {'train','val'} for q in r['trace']), 'Screen exposed test scores'
            expected_steps = sorted(set([0,1,10,30,100,300,1000] + list(range(2000, steps+1,1000)) + [steps]))
            assert [q['step'] for q in r['trace']] == expected_steps
            rows[task, arm] = (r, path)
    return rows


def select():
    rows = validated_screen()
    spec = design()
    from experiments.expD39_qi_init_theory.followup import collect, EXTRA
    rows.update(collect())
    arms=spec['screen_variants']+list(EXTRA)
    scores = {}
    for arm in arms:
        scores[arm] = {'task_ratios':{}}
        for task in spec['screen_tasks']:
            error = lambda a: min(q['learned']['val'] for q in rows[task,a][0]['trace'])
            scores[arm]['task_ratios'][task] = error(arm)/error('qi')
        scores[arm]['geomean_ratio'] = float(np.exp(np.mean(np.log(list(scores[arm]['task_ratios'].values())))))
    candidate = min((a for a in scores if a != 'qi'), key=lambda a:scores[a]['geomean_ratio'])
    task_candidates={t:min(arms,key=lambda a:scores[a]['task_ratios'][t]) for t in spec['screen_tasks']}
    confirm_pairs={(t,candidate) for t in spec['confirmation_tasks']}
    confirm_pairs.update((t,a) for t,a in task_candidates.items() if a!='qi')
    confirm_pairs.update((t,'spacing') for t in spec['screen_tasks'])
    hashes = {str(p.relative_to(ROOT)):file_hash(p) for p in
              [HERE/'initialization.py', HERE/'run.py', HERE/'config.yaml', HERE/'analyze.py',
               ROOT/'experiments/expF04_qi_init_real_data/model.py',
               ROOT/'experiments/expD38_init_readout_baseline/run.py',
               BASE_OUT/'selected_recipe.json', OUT/'STATUS.md', HERE/'followup.py',OUT/'followup_plan.md',
               HERE/'confirm_campaign.py']}
    result = dict(candidate=candidate, confirm_variants=[candidate], scores=scores,
                  beats_original_on_screen=scores[candidate]['geomean_ratio']<1,
                  task_candidates=task_candidates,confirm_pairs=sorted(confirm_pairs),
                  screen_files={str(p.relative_to(ROOT)):file_hash(p) for _,p in rows.values()},
                  provenance=hashes, selected_using='four-task validation geomean only',
                  confirmation_steps=20000, confirmation_seeds=[0,1,2])
    path = OUT/'selection.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['candidate'] == candidate and old['screen_files'] == result['screen_files'], 'Selection is locked'
        return old
    base.save_json(path, result)
    print(json.dumps(dict(candidate=candidate, scores=scores), indent=2))
    return result


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS/name, dpi=160, bbox_inches='tight')
    plt.close(fig)


def capped_line(ax, x, y, style, *, color, label, bottom):
    """Keep common log limits without silently hiding out-of-range samples."""
    x, y = np.asarray(x), np.asarray(y)
    ax.plot(x, np.clip(y, bottom, 1), style, color=color, label=label)
    for mask, bound, marker in [(y > 1, 1, '^'), (y < bottom, bottom, 'v')]:
        ax.scatter(x[mask], np.full(mask.sum(), bound), marker=marker,
                   color=color, s=28, clip_on=False, zorder=5)


def axis_note(fig, extra=''):
    fig.text(.5, .012, 'Step 0 omitted on log x; triangles mark MSE beyond the shared limits.' + extra,
             ha='center', fontsize=9)


def screen_plots():
    selected = json.loads((OUT/'selection.json').read_text())
    spec, rows = design(), validated_screen()
    from experiments.expD39_qi_init_theory.followup import collect
    rows.update(collect())
    arms = list(selected['scores'])
    values = np.array([[selected['scores'][a]['task_ratios'][t] for t in spec['screen_tasks']] +
                       [selected['scores'][a]['geomean_ratio']] for a in arms])
    fig, ax = plt.subplots(figsize=(12.5,7.7))
    img = ax.imshow(np.log2(values), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(arms)):
        for j in range(5):
            label=f'{values[i,j]:.3f}' if j==4 else f'{values[i,j]:.2f}'
            ax.text(j,i,label,ha='center',va='center',color='white' if abs(np.log2(values[i,j]))>.8 else 'black')
    ax.set_xticks(range(5), [base.TITLES[t] for t in spec['screen_tasks']]+['Geometric mean'])
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
    ax.set_title('Validation MSE / original QI · lower is better\nTwo tanh hidden layers · width 512 · 10k steps · seed 0', pad=18)
    cb = fig.colorbar(img, ax=ax, ticks=[-1,0,1], shrink=.7,extend='max')
    cb.ax.set_yticklabels(['0.5×','1×','2×'])
    save(fig,'validation_screen.png')
    groups = [('corrections',['qi','spacing','balanced','centered','collar','common']),
              ('alternatives',['centered','lambda05','lambda10','directions64','first_only','last_only']),
              ('direction_slope',['centered','directions64','soft24','sharp64'])]
    for name, variants in groups:
        fig, axes = plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
        colors = plt.get_cmap('tab10').colors
        for ax,task in zip(axes.flat,spec['screen_tasks']):
            for arm,color in zip(variants,colors):
                trace = [q for q in rows[task,arm][0]['trace'] if q['step']>0]
                capped_line(ax,[q['step'] for q in trace],[q['learned']['val'] for q in trace],'-',
                            color=color,label=LABELS[arm],bottom=.005)
            ax.set(xscale='log', yscale='log', xlim=(1,10000), ylim=(.005,1),
                   title=base.TITLES[task]+'\nWidth 512',xlabel='Gradient step',ylabel='Validation MSE')
            ax.grid(alpha=.2,which='both')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
        axis_note(fig)
        fig.subplots_adjust(top=.85,hspace=.36,wspace=.16)
        save(fig,f'screen_{name}_trajectories.png')


def diagnostic_plots():
    record = json.loads((OUT/'geometry.json').read_text())
    rows = {(r['task'],r['scheme']):r for r in record['rows']}
    fig, axes = plt.subplots(2,2,figsize=(11,8),sharey='col')
    for layer in range(2):
        for arm,color in [('qi','tab:orange'),('centered','tab:green'),('common','tab:purple')]:
            row = rows['airfoil',arm]['layers'][layer]
            for ax,key,bins in [(axes[layer,0],'gamma',np.linspace(0,6,51)),
                                (axes[layer,1],'actual_lambda',np.arange(.2395,.311,.001))]:
                values = row[key]
                ax.hist(values,bins=bins,weights=np.ones(len(values))*100/len(values),alpha=.4,color=color,label=LABELS[arm])
        axes[layer,0].set(title=f'Layer {layer+1}: row norms',xlabel=r'$\gamma=\|w_i\|_2$',ylabel='Rows (%)',xlim=(0,6))
        axes[layer,1].set(title=f'Layer {layer+1}: slope × actual spacing',xlabel=r'$\gamma\,\Delta c$',ylabel='Within-bank gaps (%)',xlim=(.24,.31),ylim=(0,105))
    fig.suptitle('Airfoil initialization: raw scales and the QI invariant\nWidth 512',y=1.05)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=3)
    fig.subplots_adjust(top=.84,hspace=.4,wspace=.25)
    save(fig,'gamma_and_spacing.png')
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    variants=['standard','qi','common','lambda05','lambda10','directions64']
    colors=['tab:blue','tab:orange','tab:purple','tab:red','tab:brown','tab:green']
    for ax,task in zip(axes.flat,design()['screen_tasks']):
        for arm,color in zip(variants,colors):
            singular=np.array(rows[task,arm]['layers'][1]['feature_spectrum']['singular_values'])
            ax.plot(np.arange(1,len(singular)+1),singular/singular[0],color=color,label=LABELS[arm])
        ax.set(xscale='log',yscale='log',xlim=(1,512),ylim=(1e-16,1),
               title=base.TITLES[task]+'\nWidth 512',xlabel='Singular-value index',ylabel=r'$\sigma_j/\sigma_1$')
        ax.grid(which='both',alpha=.15)
    fig.suptitle('Initial second-hidden-layer features · centered · fitting inputs only',y=1.025)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.985),ncol=3)
    fig.subplots_adjust(top=.83,hspace=.34,wspace=.18)
    save(fig,'initial_feature_spectra.png')
    analytic = json.loads((OUT/'analytic_checks.json').read_text())
    fig, axes = plt.subplots(1,3,figsize=(13,4.7),sharey=True)
    for ax,target in zip(axes,['sin_pi','sin_8pi','runge']):
        for lam,color in [(.25,'tab:blue'),(.5,'tab:orange'),(1.,'tab:green')]:
            for halo,style in [(0,'--'),(16,'-')]:
                rr = [r for r in analytic['rows'] if r['target']==target and r['lam']==lam and r['halo']==halo]
                ax.plot([r['intervals'] for r in rr],[r['relative_l2'] for r in rr],style+'o',color=color)
        ax.set(yscale='log',xscale='log',xticks=[16,32,64],xticklabels=['16','32','64'],
               ylim=(1e-15,1),title=target.replace('_',' '),xlabel='Interior intervals N',ylabel='Relative L2 error')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2,which='both')
    handles = [Line2D([],[],color=c,label=f'λ={l}') for l,c in [(.25,'tab:blue'),(.5,'tab:orange'),(1,'tab:green')]]
    handles += [Line2D([],[],color='black',ls=s,label=label) for s,label in [('--','No halo: W=N+1'),('-','16 per side: W=N+33')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=5)
    fig.suptitle('Analytic scalar checks · frozen geometry, solved readout · cutoff 10⁻¹²',y=1.08)
    fig.subplots_adjust(top=.77,wspace=.15)
    save(fig,'analytic_geometry.png')


def confirmation():
    selection = json.loads((OUT/'selection.json').read_text())
    for name,digest in selection['provenance'].items():
        if name.startswith('experiments/') and not name.endswith('/analyze.py'):
            assert file_hash(ROOT/name)==digest, f'Training source changed after selection: {name}'
    candidate, spec = selection['candidate'], design()
    all_runs, summary = {}, []
    recipes = json.loads((BASE_OUT/'selected_recipe.json').read_text())
    for task in spec['confirmation_tasks']:
        extra_arms=[a for t,a in selection['confirm_pairs'] if t==task and a!=candidate]
        for arm in ['standard','qi',candidate,*extra_arms]:
            root = BASE_OUT if arm in ('standard','qi') else OUT
            paths = sorted((root/'data/compare').glob(f'{task}_{arm}_w512_*_steps20000.json'))
            runs = [json.loads(p.read_text()) for p in paths]
            runs = [r for r in runs if r.get('complete') and r['identity']['seed'] in [0,1,2] and base.valid_data_identity(r['identity'],config())]
            assert len(runs)==3 and {r['identity']['seed'] for r in runs}=={0,1,2}, (task,arm,len(runs))
            for run in runs:
                cfg = base.config() if arm in ('standard','qi') else config()
                if arm in ('soft24','sharp64'):
                    from experiments.expD39_qi_init_theory.followup import config as followup_config
                    cfg=followup_config()
                expected = dict(task=task,scheme=arm,lr=recipes[task]['lr'],seed=run['identity']['seed'],
                                steps=20000,width=512,phase='compare',config=cfg)
                if task in cfg.get('task_data_protocols', {}):
                    expected['data_protocol']=cfg['task_data_protocols'][task]
                assert base.same_identity(run['identity'],expected), (task,arm,'recipe mismatch')
                assert run['trace'][-1]['step']==20000
                expected_steps=sorted(set([0,1,10,30,100,300,1000]+list(range(2000,20001,1000))))
                assert [q['step'] for q in run['trace']]==expected_steps
                assert [q['step'] for q in run['trace'] if 'solved' in q]==[0,10,100,300,1000,3000,5000,10000,15000,20000]
                for row in run['trace']:
                    if 'solved' in row:
                        assert row['solved']['train']<=row['learned']['train']+1e-8
            runs.sort(key=lambda r:r['identity']['seed'])
            if arm!='standard':
                assert all(a['data']==b['data'] for a,b in zip(runs,all_runs[task,'standard'])), (task,'data changed')
            all_runs[task,arm] = runs
            selected_rows = [min(r['trace'],key=lambda q:q['learned']['val']) for r in runs]
            summary.append(dict(task=task,arm=arm,
                final={k:float(np.mean([r['trace'][-1]['learned'][k] for r in runs])) for k in ['train','val','test']},
                selected={k:float(np.mean([q['learned'][k] for q in selected_rows])) for k in ['train','val','test']},
                selected_steps=[q['step'] for q in selected_rows],
                selected_test_per_seed=[q['learned']['test'] for q in selected_rows],
                per_seed_final_test=[r['trace'][-1]['learned']['test'] for r in runs]))
    base.save_json(OUT/'confirmation_summary.json',dict(candidate=candidate,rows=summary))
    # Validation-chosen task-specific recipes were also locked before test data.
    # Airfoil's selected recipe is the existing original QI reference.
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    for ax,task in zip(axes.flat,spec['screen_tasks']):
        selected=selection['task_candidates'][task]
        arms=list(dict.fromkeys(['standard','qi',selected]))
        for arm in arms:
            color={'standard':'tab:blue','qi':'tab:orange'}.get(arm,'tab:green')
            runs=all_runs[task,arm]
            traces=[[q for q in r['trace'] if q['step']>0] for r in runs]
            steps=[q['step'] for q in traces[0]]
            values=np.array([[q['learned']['test'] for q in tr] for tr in traces])
            capped_line(ax,steps,values.mean(0),'-',color=color,label=LABELS[arm],bottom=.005)
            ax.fill_between(steps,values.min(0),values.max(0),color=color,alpha=.1)
        ax.axhline(all_runs[task,'standard'][0]['affine']['test'],ls=':',color='gray')
        ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(.005,1),xlabel='Gradient step',ylabel='Test MSE',
               title=base.TITLES[task]+' · '+LABELS[selected]+'\nWidth 512')
        ax.grid(which='both',alpha=.17)
    handles=[Line2D([],[],color=c,label=l) for c,l in [('tab:blue','Standard'),('tab:orange','Original QI'),('tab:green','Task-specific candidate')]]
    handles+=[Line2D([],[],color='gray',ls=':',label='Input linear regression')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=4)
    fig.suptitle('Task-specific choices locked on validation · 3 seeds · mean and observed range',y=1.025)
    axis_note(fig)
    fig.subplots_adjust(top=.86,hspace=.36,wspace=.16)
    save(fig,'task_specific_confirmation.png')
    # Isolated spacing bug: compare mean endpoints and validation-selected
    # checkpoints separately, so a clean formula is not mistaken for a gain.
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    task_labels=[base.TITLES[t] for t in spec['screen_tasks']]
    table={(r['task'],r['arm']):r for r in summary}
    for ax,kind in zip(axes,['final','selected']):
        for i,task in enumerate(spec['screen_tasks']):
            key='per_seed_final_test' if kind=='final' else 'selected_test_per_seed'
            ratios=np.array(table[task,'spacing'][key])/np.array(table[task,'qi'][key])
            ax.scatter(np.full(3,i)+np.array([-.12,0,.12]),ratios,color='tab:purple',alpha=.7)
            meanratio=table[task,'spacing'][kind]['test']/table[task,'qi'][kind]['test']
            ax.plot(i,meanratio,'k_',ms=17,mew=2)
        ax.axhline(1,color='gray',ls=':')
        ax.set(xticks=range(4),xticklabels=task_labels,title='Final 20k step' if kind=='final' else 'Validation-selected step',ylabel='Spacing fix / original QI · test MSE')
        ax.tick_params(axis='x',rotation=15,labelsize=9)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Isolating the off-by-one correction · width 512 · dots: paired seeds; bars: ratio of means',y=1.05)
    fig.tight_layout()
    save(fig,'spacing_control.png')
    colors = {'standard':'tab:blue','qi':'tab:orange',candidate:'tab:green'}
    for split in ['val','test']:
        fig,axes = plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,spec['confirmation_tasks']):
            for arm in ['standard','qi',candidate]:
                runs = all_runs[task,arm]
                steps = np.array([q['step'] for q in runs[0]['trace']])
                keep = steps>0
                values = np.array([[q['learned'][split] for q in r['trace']] for r in runs])
                capped_line(ax,steps[keep],values[:,keep].mean(0),'-',
                            color=colors[arm],label=LABELS[arm],bottom=.003)
                ax.fill_between(steps[keep],values[:,keep].min(0),values[:,keep].max(0),color=colors[arm],alpha=.10)
            ax.axhline(all_runs[task,'standard'][0]['affine'][split],ls=':',color='gray',label='Input linear regression')
            ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(.003,1),xlabel='Gradient step',ylabel=f'{"Validation" if split=="val" else "Test"} MSE',title=base.TITLES[task]+'\nWidth 512')
            ax.grid(which='both',alpha=.17)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=4)
        fig.suptitle('Two tanh hidden layers · 3 seeds · mean and observed range',y=1.02)
        axis_note(fig)
        fig.subplots_adjust(top=.86,hspace=.36,wspace=.16)
        save(fig,f'confirmation_{split}.png')
    # Requested learned versus solved readouts: selected candidate vs standard.
    for split in ['train','test']:
        fig,axes = plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,spec['confirmation_tasks']):
            for arm,color in [('standard','tab:blue'),(candidate,'tab:orange')]:
                for kind,style in [('learned','-'),('solved',':')]:
                    traces = [[q for q in r['trace'] if kind in q and q['step']>0] for r in all_runs[task,arm]]
                    steps = [q['step'] for q in traces[0]]
                    values = np.array([[q[kind][split] for q in tr] for tr in traces])
                    capped_line(ax,steps,values.mean(0),style,color=color,
                                label=f'{LABELS[arm]} · {kind}',bottom=1e-4 if split=='train' else .003)
            ax.axhline(all_runs[task,'standard'][0]['affine'][split],ls=':',color='gray',label='Input linear regression')
            ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(1e-4 if split=='train' else .003,1),
                   xlabel='Gradient step',ylabel=f'{split.title()} MSE',title=base.TITLES[task]+'\nWidth 512')
            ax.grid(which='both',alpha=.17)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=3)
        fig.suptitle('Locked QI candidate vs standard · trained and diagnostic LS readouts · 3 seeds',y=1.02)
        axis_note(fig)
        fig.subplots_adjust(top=.84,hspace=.36,wspace=.16)
        save(fig,f'confirmation_readouts_{split}.png')
    policy={t:selection['task_candidates'].get(t,'qi') for t in spec['confirmation_tasks']}
    policy_summary={t:{'recipe':policy[t],
                        'selection_basis':'screen validation' if t in selection['task_candidates'] else 'unscreened; original QI retained',
                        'standard':table[t,'standard'],
                        'original_qi':table[t,'qi'],
                        'selected_qi':table[t,policy[t]]} for t in policy}
    base.save_json(OUT/'task_policy_summary.json',policy_summary)
    for split in ['train','test']:
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,spec['confirmation_tasks']):
            for arm,color in [('standard','tab:blue'),(policy[task],'tab:orange')]:
                for kind,style in [('learned','-'),('solved',':')]:
                    traces=[[q for q in r['trace'] if kind in q and q['step']>0] for r in all_runs[task,arm]]
                    steps=[q['step'] for q in traces[0]]
                    values=np.array([[q[kind][split] for q in tr] for tr in traces])
                    label=('Standard' if arm=='standard' else 'QI recipe')+' · '+kind
                    capped_line(ax,steps,values.mean(0),style,color=color,label=label,
                                bottom=1e-4 if split=='train' else .003)
            ax.axhline(all_runs[task,'standard'][0]['affine'][split],ls=':',color='gray',label='Input linear regression')
            ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(1e-4 if split=='train' else .003,1),
                   xlabel='Gradient step',ylabel=f'{split.title()} MSE',
                   title=base.TITLES[task]+' · '+LABELS[policy[task]]+'\nWidth 512')
            ax.grid(which='both',alpha=.17)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=3)
        fig.suptitle('Task-specific QI recipes · 3 seeds · trained and diagnostic LS readouts',y=1.025)
        axis_note(fig,' Bike/Pol retain the unscreened original QI recipe.')
        fig.subplots_adjust(top=.84,hspace=.4,wspace=.16)
        save(fig,f'task_specific_readouts_{split}.png')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['select','screen','diagnostics','confirm'])
    a=p.parse_args()
    {'select':select,'screen':screen_plots,'diagnostics':diagnostic_plots,'confirm':confirmation}[a.mode]()
