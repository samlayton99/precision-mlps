"""Validation-only selection and paired confirmation for the gamma mixture."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precision_d40_matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.expD40_bimodal_gamma.run import base, config, design, digest, ROOT, OUT, BASE_OUT, D39_OUT

COLORS = dict(centered='tab:blue', low='tab:cyan', mid='tab:green', high='tab:red', rms='tab:purple', mix='tab:orange')
LABELS = dict(centered='Centered QI reference', low='All low (0.1g)', mid='All middle (g)',
              high='All high (3g)', rms='Single scale · matched weight norm', mix='Bimodal (0.1g + 3g)')
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})


def save(fig, name):
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'figures'/name,dpi=155,bbox_inches='tight')
    plt.close(fig)


def path_for(task, arm, seed, phase='pilot', root=OUT):
    if phase=='pilot' and root==OUT and arm.startswith('mild_'):root=OUT/'followup'
    lr=json.loads((BASE_OUT/'selected_recipe.json').read_text())[task]['lr']
    steps=design()['screen_steps' if phase=='pilot' else 'confirmation_steps']
    return root/'data'/phase/f'{task}_{arm}_w512_lr{lr:g}_seed{seed}_steps{steps}.json'


def expected_identity(task,arm,seed,phase):
    cfg=config()
    if arm.startswith('mild_'):
        from experiments.expD40_bimodal_gamma.followup import config as mild_config
        cfg=mild_config()
    lr=json.loads((BASE_OUT/'selected_recipe.json').read_text())[task]['lr']
    result=dict(task=task,scheme=arm,seed=seed,phase=phase,width=512,
                steps=design()['screen_steps' if phase=='pilot' else 'confirmation_steps'],lr=lr,config=cfg)
    if task in cfg.get('task_data_protocols',{}):result['data_protocol']=cfg['task_data_protocols'][task]
    return result


def read_new(task,arm,seed,phase):
    p=path_for(task,arm,seed,phase)
    r=json.loads(p.read_text())
    assert r['complete'] and p.with_suffix('.pt').exists(),p
    assert base.same_identity(r['identity'],expected_identity(task,arm,seed,phase)),p
    steps=r['identity']['steps']
    assert [q['step'] for q in r['trace']]==sorted(set([0,1,10,30,100,300,1000]+list(range(2000,steps+1,1000))+[steps]))
    if phase=='pilot':
        assert all(set(q['learned'])=={'train','val'} for q in r['trace']), 'Test exposure during screen'
    else:
        assert [q['step'] for q in r['trace'] if 'solved' in q]==[0,10,100,300,1000,3000,5000,10000,15000,20000]
        for q in r['trace']:
            if 'solved' in q: assert q['solved']['train']<=q['learned']['train']+1e-8
    return r,p


def collect_screen(include_mild=True):
    rows={}
    for task in design()['tasks']:
        for scope in design()['scopes']:
            for shape in design()['shapes']:
                arm=f'{shape}_{scope}'
                rows[task,arm]=read_new(task,arm,0,'pilot')
        from experiments.expD40_bimodal_gamma.followup import ARMS
        if include_mild:
            for arm in ARMS:rows[task,arm]=read_new(task,arm,0,'pilot')
        p=path_for(task,'centered',0,root=D39_OUT)
        r=json.loads(p.read_text())
        assert r['complete'] and r['trace'][-1]['step']==10000
        from experiments.expD39_qi_init_theory.run import config as reference_config
        expected=expected_identity(task,'centered',0,'pilot')
        expected['config']=reference_config()
        assert base.same_identity(r['identity'],expected),p
        assert [q['step'] for q in r['trace']]==[0,1,10,30,100,300,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        assert r['identity']['config']['training_engine_sha256']==digest(ROOT/'experiments/expD38_init_readout_baseline/run.py')
        assert all(set(q['learned'])=={'train','val'} for q in r['trace'])
        for key,value in base.recipe_config(base.config()).items():
            assert r['identity']['config'][key]==value,(task,key)
        assert r['data']==rows[task,'mix_both'][0]['data']
        assert r['identity']['config']['initializer_sha256']==digest(ROOT/'experiments/expD39_qi_init_theory/initialization.py')
        rows[task,'centered']=(r,p)
        assert all(q[0]['data']==r['data'] for (t,a),q in rows.items() if t==task)
    return rows


def select():
    rows=collect_screen()
    choices={}
    pairs=set()
    for task in design()['tasks']:
        score=lambda arm:min(q['learned']['val'] for q in rows[task,arm][0]['trace'])
        mixture=min(['mix_both','mix_last','mild_mix_both','mild_mix_last'],key=score)
        scope=mixture.split('_')[-1]
        rms=('mild_' if mixture.startswith('mild_') else '')+f'rms_{scope}'
        uniform=min([f'{s}_{scope}' for s in ['low','mid','high','rms']]+[f'mild_rms_{scope}'],key=score)
        choices[task]=dict(mixture=mixture,uniform=uniform,rms=rms,
             best_val={a:score(a) for t,a in rows if t==task},
             mixture_ratio_to_best_uniform=score(mixture)/score(uniform),
             mixture_ratio_to_centered=score(mixture)/score('centered'))
        pairs.update((task,a) for a in [mixture,uniform,rms,'centered'])
    from experiments.expD40_bimodal_gamma.followup import config as mild_config
    selection=dict(choices=choices,confirmation_pairs=sorted(pairs),
        source_sha256=config()['d40_source_sha256'],
        followup_sha256=mild_config()['d40_followup_sha256'],
        followup_plan_sha256=mild_config()['d40_followup_plan_sha256'],
        screen_sha256={str(p.relative_to(ROOT)):digest(p) for r,p in rows.values()},
        selected_using='minimum validation MSE only; mixed and homogeneous errors at common checkpoint schedule')
    path=OUT/'selection.json'
    if path.exists():
        assert json.loads(path.read_text())==selection,'Selection is locked'
    else:base.save_json(path,selection)
    print(json.dumps(choices,indent=2))
    return selection


def line(ax,x,y,*,color,label,style='-',bottom=1e-4):
    x,y=np.asarray(x),np.asarray(y)
    ax.plot(x,np.clip(y,bottom,1),style,color=color,label=label)
    for mask,value,marker in [(y>1,1,'^'),(y<bottom,bottom,'v')]:
        ax.scatter(x[mask],np.full(mask.sum(),value),color=color,marker=marker,s=20,clip_on=False,zorder=4)


def screen_plots(include_mild=True):
    rows=collect_screen(include_mild)
    shapes=design()['shapes']+(['mild_rms','mild_mix'] if include_mild else [])
    arms=['centered']+[f'{s}_{scope}' for scope in ['both','last'] for s in shapes]
    tasks=design()['tasks']
    ratios=[]
    for arm in arms:
        ratios.append([min(q['learned']['val'] for q in rows[t,arm][0]['trace'])/
                       min(q['learned']['val'] for q in rows[t,'centered'][0]['trace']) for t in tasks])
    values=np.array(ratios)
    fig,ax=plt.subplots(figsize=(11,9))
    im=ax.imshow(np.log2(values),aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
    for i in range(len(arms)):
        for j in range(len(tasks)):
            ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if abs(np.log2(values[i,j]))>.8 else 'black')
    extra_labels={'mild_mix':'Bimodal (0.1g + g)','mild_rms':'Matched norm for mild mixture'}
    labels=[LABELS['centered']]+[(LABELS|extra_labels)[s]+' · '+('both layers' if scope=='both' else 'layer 2 only')
             for scope in ['both','last'] for s in shapes]
    ax.set(xticks=range(4),xticklabels=[base.TITLES[t] for t in tasks],yticks=range(len(arms)),yticklabels=labels,
           title='Best validation MSE / centered QI reference · lower is better\nWidth 512 · two tanh layers · 10k steps · seed 0')
    cb=fig.colorbar(im,ax=ax,ticks=[-1,0,1],shrink=.7,extend='both')
    cb.ax.set_yticklabels(['0.5×','1×','2×'])
    save(fig,'screen_validation.png' if include_mild else 'screen_validation_strong.png')
    for scope in ['both','last']:
        fig,axes=plt.subplots(4,2,figsize=(12,15),sharex=True,sharey=True)
        for i,task in enumerate(tasks):
            for j,split in enumerate(['train','val']):
                ax=axes[i,j]
                for shape in ['centered',*design()['shapes']]:
                    arm=shape if shape=='centered' else f'{shape}_{scope}'
                    trace=[q for q in rows[task,arm][0]['trace'] if q['step']>0]
                    line(ax,[q['step'] for q in trace],[q['learned'][split] for q in trace],
                         color=COLORS[shape],label=LABELS[shape])
                ax.set(xscale='log',yscale='log',xlim=(1,10000),ylim=(1e-4,1),
                       title=base.TITLES[task]+' · '+('Fitting' if split=='train' else 'Validation')+'\nWidth 512',
                       xlabel='Gradient step',ylabel='MSE')
                ax.grid(which='both',alpha=.15)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=3)
        fig.suptitle('Gamma distribution in '+('both hidden layers' if scope=='both' else 'hidden layer 2 only')+' · validation-only screen',y=1.025)
        fig.text(.5,.012,'Step 0 omitted; triangles mark values outside the common limits. All curves are actual trained readouts.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.91,bottom=.065,hspace=.5,wspace=.15)
        save(fig,f'screen_{scope}_trajectories.png')
    for scope in (['both','last'] if include_mild else []):
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,tasks):
            variants=[('centered','tab:blue','Centered reference'),('mid_'+scope,'tab:green','All middle (g)'),
                      ('mix_'+scope,'tab:red','Bimodal: 0.1g + 3g'),('mild_mix_'+scope,'tab:orange','Bimodal: 0.1g + g'),
                      ('mild_rms_'+scope,'tab:purple','Matched norm: mild mixture')]
            for arm,color,label in variants:
                tr=[q for q in rows[task,arm][0]['trace'] if q['step']>0]
                line(ax,[q['step'] for q in tr],[q['learned']['val'] for q in tr],color=color,label=label,bottom=.005)
            ax.set(xscale='log',yscale='log',xlim=(1,10000),ylim=(.005,1),title=base.TITLES[task]+'\nWidth 512',
                   xlabel='Gradient step',ylabel='Validation MSE')
            ax.grid(which='both',alpha=.15)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
        fig.suptitle('Strong versus mild gamma separation · '+('both layers' if scope=='both' else 'layer 2 only'),y=1.04)
        fig.text(.5,.012,'Step 0 omitted; triangles mark off-scale MSE. One screening seed.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.84,hspace=.35,wspace=.18)
        save(fig,f'screen_mode_ranges_{scope}.png')


def historical(task,arm,root=BASE_OUT):
    runs=[]
    for seed in [0,1,2]:
        p=path_for(task,arm,seed,'compare',root=root)
        r=json.loads(p.read_text())
        assert r['complete'] and r['trace'][-1]['step']==20000
        for key,value in base.recipe_config(base.config()).items():
            assert r['identity']['config'][key]==value,(task,arm,key)
        assert r['identity']['seed']==seed and r['identity']['width']==512
        assert base.valid_data_identity(r['identity'],base.config())
        assert r['identity']['lr']==expected_identity(task,arm,seed,'compare')['lr']
        runs.append(r)
    return runs


def confirmation():
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['source_sha256']==config()['d40_source_sha256']
    from experiments.expD40_bimodal_gamma.followup import config as mild_config
    assert selection['followup_sha256']==mild_config()['d40_followup_sha256']
    assert selection['followup_plan_sha256']==mild_config()['d40_followup_plan_sha256']
    for name,sha in selection['screen_sha256'].items():assert digest(ROOT/name)==sha
    all_runs={}
    for task,arm in selection['confirmation_pairs']:
        all_runs[task,arm]=[read_new(task,arm,s,'compare')[0] for s in [0,1,2]]
    for task in design()['tasks']:
        for arm in ['standard','qi']:
            all_runs[task,arm]=historical(task,arm)
        oldarm={'airfoil':'qi','kin8nm':'soft24','sarcos':'last_only','superconductivity':'spacing'}[task]
        if oldarm!='qi':all_runs[task,oldarm]=historical(task,oldarm,D39_OUT)
        for (t,arm),runs in all_runs.items():
            if t==task:
                assert all(a['data']==b['data'] for a,b in zip(runs,all_runs[task,'standard']))
    summaries={}
    for (task,arm),runs in all_runs.items():
        record={}
        for kind in ['final','selected']:
            chosen=[r['trace'][-1] if kind=='final' else min(r['trace'],key=lambda q:q['learned']['val']) for r in runs]
            record[kind]=dict(steps=[q['step'] for q in chosen],
                mean={s:float(np.mean([q['learned'][s] for q in chosen])) for s in ['train','val','test']},
                per_seed={s:[q['learned'][s] for q in chosen] for s in ['train','val','test']})
        record['final_solved']={s:float(np.mean([r['trace'][-1]['solved'][s] for r in runs])) for s in ['train','val','test']}
        record['initial_frozen_ridge']={s:float(np.mean([r['frozen_ridge'][s] for r in runs])) for s in ['train','val','test']}
        record['input_affine']={s:float(np.mean([r['affine'][s] for r in runs])) for s in ['train','val','test']}
        summaries.setdefault(task,{})[arm]=record
    ratios={}
    for task,choice in selection['choices'].items():
        mixture=summaries[task][choice['mixture']]
        ratios[task]={}
        for arm in dict.fromkeys(['centered',choice['uniform'],choice['rms'],'standard','qi']):
            ratios[task][arm]={kind:{s:mixture[kind]['mean'][s]/summaries[task][arm][kind]['mean'][s]
                  for s in ['train','val','test']} for kind in ['final','selected']}
    base.save_json(OUT/'confirmation_summary.json',dict(choices=selection['choices'],rows=summaries,mixture_control_ratios=ratios))
    for split in ['train','val','test']:
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,design()['tasks']):
            choice=selection['choices'][task]
            arms=[('standard','tab:blue','Standard'),('centered','gray','Centered QI reference'),
                  (choice['rms'],'tab:purple','Matched weight norm'),
                  (choice['uniform'],'tab:green','Validation-chosen single scale'),
                  (choice['mixture'],'tab:orange','Bimodal')]
            seen=set()
            for arm,color,label in arms:
                if arm in seen:continue
                seen.add(arm)
                runs=all_runs[task,arm]
                traces=[[q for q in r['trace'] if q['step']>0] for r in runs]
                x=[q['step'] for q in traces[0]]
                y=np.array([[q['learned'][split] for q in tr] for tr in traces])
                line(ax,x,y.mean(0),color=color,label=label,bottom=1e-4 if split=='train' else .003)
                ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.08)
            scope='both layers' if choice['mixture'].endswith('_both') else 'layer 2 only'
            single_key=choice['uniform'].removesuffix('_both').removesuffix('_last')
            single={'low':'0.1g','mid':'g','high':'3g','rms':'2.123g','mild_rms':'0.711g'}[single_key]
            mode='0.1g + g' if choice['mixture'].startswith('mild_') else '0.1g + 3g'
            ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(1e-4 if split=='train' else .003,1),
                   title=base.TITLES[task]+' · '+scope+' · '+mode+'\nWidth 512 · selected single scale: '+single,
                   xlabel='Gradient step',ylabel={'train':'Fitting','val':'Validation','test':'Test'}[split]+' MSE')
            ax.grid(which='both',alpha=.15)
        handles=[Line2D([],[],color=c,label=l) for c,l in [('tab:blue','Standard'),('gray','Centered QI reference'),
            ('tab:purple','Matched weight norm'),('tab:green','Validation-chosen single scale'),('tab:orange','Bimodal')]]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3)
        fig.suptitle('Bimodal initialization · 3 paired seeds · actual trained readout',y=1.03)
        fig.text(.5,.01,'Shading: observed seed range. Step 0 omitted; triangles mark off-scale MSE. Green omitted when the selected single scale is the purple RMS control.',ha='center',fontsize=8)
        fig.subplots_adjust(top=.83,bottom=.10,hspace=.42,wspace=.18)
        save(fig,f'confirmation_{split}.png')
    # Same-checkpoint fitting and validation comparisons answer the user directly.
    for kind in ['final','selected']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,split in zip(axes,['train','val']):
            for i,task in enumerate(design()['tasks']):
                choice=selection['choices'][task]
                for offset,arm,color in [(-.15,'centered','gray'),(0,choice['rms'],'tab:purple'),(.15,choice['uniform'],'tab:green')]:
                    if color=='tab:green' and choice['uniform']==choice['rms']:continue
                    a=np.array(summaries[task][choice['mixture']][kind]['per_seed'][split])
                    b=np.array(summaries[task][arm][kind]['per_seed'][split])
                    ax.scatter(i+offset+np.array([-.035,0,.035]),a/b,color=color,s=25,alpha=.65)
                    ax.plot(i+offset,a.mean()/b.mean(),'_',ms=13,mew=2,color=color)
            ax.axhline(1,color='black',ls=':',lw=1)
            ax.set(yscale='log',xticks=range(4),xticklabels=[base.TITLES[t] for t in design()['tasks']],
                   title=('Fitting' if split=='train' else 'Validation')+' MSE ratio',ylabel='Bimodal / control · lower is better')
            ax.tick_params(axis='x',rotation=15,labelsize=9)
            ax.grid(axis='y',which='both',alpha=.2)
        handles=[Line2D([],[],color=c,marker='o',ls='',label=l) for c,l in [('gray','Centered QI'),('tab:purple','Matched weight norm'),('tab:green','Selected single scale')]]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=3)
        fig.suptitle(('Final 20k models' if kind=='final' else 'Each run at its validation-selected checkpoint')+' · width 512',y=1.14)
        fig.text(.5,-.035,'Same checkpoint per run. Dots: paired seeds; bars: ratios of means. Green omitted when identical to purple.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.77,wspace=.14)
        save(fig,f'paired_fit_validation_{kind}.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,split in zip(axes,['val','test']):
        for i,task in enumerate(design()['tasks']):
            choice=selection['choices'][task]
            a=np.array(summaries[task][choice['mixture']]['selected']['per_seed'][split])
            b=np.array(summaries[task][choice['rms']]['selected']['per_seed'][split])
            ax.scatter(i+np.array([-.06,0,.06]),a/b,color='tab:orange',s=35,alpha=.75)
            ax.plot(i,a.mean()/b.mean(),'_',ms=20,mew=3,color='tab:orange')
        ax.axhline(1,color='gray',ls=':',lw=1.5)
        ax.set(ylim=(.85,1.2),yticks=np.arange(.85,1.201,.05),xticks=range(4),
               xticklabels=[base.TITLES[t] for t in design()['tasks']],
               title=('Validation' if split=='val' else 'Test')+' MSE ratio',
               ylabel='Bimodal / matched single scale')
        ax.tick_params(axis='x',rotation=15,labelsize=9)
        ax.grid(axis='y',alpha=.2)
    handles=[Line2D([],[],color='tab:orange',marker='o',ls='',label='Paired seeds'),
             Line2D([],[],color='tab:orange',marker='_',ms=15,ls='',label='Ratio of mean errors'),
             Line2D([],[],color='gray',ls=':',label='Equal error')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=3)
    fig.suptitle('Bimodal versus a single scale with the same initial squared weight norm · width 512',y=1.14)
    fig.text(.5,-.035,'Each run uses its validation-selected checkpoint. Above 1 means the mixture has higher error. Same layer scope and training recipe.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.77,wspace=.14)
    save(fig,'matched_scale_generalization.png')
    for split in ['train','val','test']:
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
        for ax,task in zip(axes.flat,design()['tasks']):
            mixture=selection['choices'][task]['mixture']
            for arm,color,name in [('standard','tab:blue','Standard'),(mixture,'tab:orange','Bimodal')]:
                for field,style in [('learned','-'),('solved',':')]:
                    traces=[[q for q in r['trace'] if field in q and q['step']>0] for r in all_runs[task,arm]]
                    x=[q['step'] for q in traces[0]]
                    y=np.array([[q[field][split] for q in tr] for tr in traces])
                    line(ax,x,y.mean(0),color=color,label=name+' · '+field,style=style,bottom=1e-4 if split=='train' else .003)
            ax.axhline(all_runs[task,'standard'][0]['affine'][split],color='gray',ls=':',label='Input linear regression')
            ax.set(xscale='log',yscale='log',xlim=(1,20000),ylim=(1e-4 if split=='train' else .003,1),
                   title=base.TITLES[task]+'\nWidth 512',xlabel='Gradient step',ylabel={'train':'Fitting','val':'Validation','test':'Test'}[split]+' MSE')
            ax.grid(which='both',alpha=.15)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=3)
        fig.suptitle('Standard vs validation-chosen bimodal · trained and diagnostic LS heads · 3 seeds',y=1.04)
        fig.text(.5,.012,'Step 0 omitted; triangles mark MSE outside shared limits. LS never changes the trained model.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.83,hspace=.35,wspace=.18)
        save(fig,f'readouts_{split}.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,split in zip(axes,['val','test']):
        for i,task in enumerate(design()['tasks']):
            mixture=selection['choices'][task]['mixture']
            runs=all_runs[task,mixture]
            chosen=[min(r['trace'],key=lambda q:q['learned']['val']) for r in runs]
            fitted=np.array([q['learned'][split] for q in chosen])
            for offset,field,color in [(-.1,'affine','gray'),(.1,'frozen_ridge','tab:purple')]:
                control=np.array([r[field][split] for r in runs])
                ax.scatter(i+offset+np.array([-.025,0,.025]),fitted/control,color=color,s=25,alpha=.65)
                ax.plot(i+offset,fitted.mean()/control.mean(),'_',ms=13,mew=2,color=color)
        ax.axhline(1,color='black',ls=':',lw=1)
        ax.set(yscale='log',xticks=range(4),xticklabels=[base.TITLES[t] for t in design()['tasks']],
               title=('Validation' if split=='val' else 'Test')+' MSE ratio',ylabel='Trained bimodal MLP / baseline')
        ax.tick_params(axis='x',rotation=15,labelsize=9)
        ax.grid(axis='y',which='both',alpha=.2)
    handles=[Line2D([],[],color=c,marker='o',ls='',label=l) for c,l in [('gray','Input linear regression'),('tab:purple','Ridge on frozen initial features')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=2)
    fig.suptitle('Does training improve on the initial representation? · width 512',y=1.14)
    fig.text(.5,-.035,'Each trained run uses its validation-selected checkpoint. Dots: paired seeds; bars: ratios of means. Lower is better.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.77,wspace=.14)
    save(fig,'feature_learning.png')
    print(json.dumps(ratios,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['select','screen','screen_strong','confirm'])
    args=parser.parse_args()
    {'select':select,'screen':screen_plots,'screen_strong':lambda:screen_plots(False),'confirm':confirmation}[args.mode]()
