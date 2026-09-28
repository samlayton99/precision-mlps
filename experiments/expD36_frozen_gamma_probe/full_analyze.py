"""Curated full-sweep evidence and publication figures; never generates prose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from . import core, full_screen as screen, full_train as train

COLORS=['#245c9f','#d65f28','#438c67','#8d5c9d','#ab8c25']
LABELS=dict(raw='Raw',collective='Collective',individual='Individual',
    collective_neighbor='Collective + neighbor',individual_neighbor='Individual + neighbor')
TARGETS=dict(sine_mix_2_6_10='Sine mixture',exp_sin_3pi='exp(sin(3πx))',runge_25='Runge',
    quadratic='Quadratic',sine_2pi='Single sine')


def read(path):
    return json.loads(path.read_text())


def save(fig,root,name):
    folder=root/'figures'; folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(folder/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def final(root,tag):
    row=read(root/'training'/tag/'evaluations.json')[-1]
    assert row['step']==read(root/'manifest.json')['config']['training_steps'],f'Incomplete plot input: {tag}'
    return row


def precision_value(root,n,name,optimizer,ti=0):
    tag=f'N{n}_{name}_'+('gd' if optimizer=='gd' else 'adam_continue')
    row=final(root,tag); values=np.array(row['evaluation'],dtype=float)[:,ti]
    return np.where(np.array(row['failed'])[:,ti]==0,values,np.nan)


def finish_axis(ax,xlabel='Frozen slope γ',ylabel=None):
    ax.set_xscale('log'); ax.set_xlabel(xlabel)
    if xlabel=='Frozen slope γ':
        limits=ax.get_xlim(); ticks=[x for x in [1,4,16,64] if limits[0]<=x<=limits[1]]
        ax.set_xticks(ticks,[str(x) for x in ticks]); ax.xaxis.set_minor_formatter(NullFormatter())
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True,which='major',alpha=.2)
    ax.spines[['top','right']].set_visible(False)


def banner(root,cfg,name):
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.2),layout='constrained')
    ax=axes[0]; widths=[]; slopes=[]; initial=[]
    for n in cfg['widths']:
        rows=read(root/'joint'/f'N{n}'/'evaluations.json')
        assert rows[-1]['step']==cfg['joint_steps'] and not any(rows[-1]['failed'])
        widths.append(rows[-1]['width'])
        slopes.append(np.array(rows[-1]['slope_quantiles'])[:,1])
        initial.append(np.array(rows[0]['slope_quantiles'])[:,1])
    slopes=np.array(slopes); initial=np.array(initial)
    ax.plot(widths,np.median(slopes,axis=1),'o-',color=COLORS[0],label='Trained median |a|')
    ax.fill_between(widths,slopes.min(axis=1),slopes.max(axis=1),color=COLORS[0],alpha=.16,label='Range across 5 seeds')
    ax.plot(widths,np.median(initial,axis=1),':',color='#777777',label='Xavier initial median')
    ax.plot(widths,np.array(cfg['widths'])/8,'--',color='#222222',label='Reference γ = N/8')
    ax.set_yscale('log'); finish_axis(ax,'Hidden width W','Physical slope magnitude')
    ax.set_title('(a) End-to-end Adam, 20k updates'); ax.legend(fontsize=7.5,loc='best')
    ax=axes[1]; gamma=np.array(cfg['gammas'])
    for optimizer,color,label in [('gd',COLORS[0],'GD, η = 0.5/L'),('adam',COLORS[1],'Validation-selected Adam')]:
        values=precision_value(root,cfg['n'],name,optimizer)
        ax.plot(gamma,values,'o-',color=color,label=label)
    ax.axvline(cfg['n']/8,color='#555555',ls=':',lw=.8,label='Reference γ = N/8')
    ax.set_yscale('log'); finish_axis(ax,ylabel='Independent relative L2 error')
    ax.set_title(f'(b) {LABELS[name]}, 200k updates'); ax.legend(fontsize=7.5)
    ax=axes[2]; first=np.load(root/'training'/f'N{cfg["n"]}_{name}_gd/hitting_audit.npz')['first'][:,0,0]
    reached=first>=0; ceiling=1e11; budget=cfg['training_steps']
    ax.scatter(gamma[reached],first[reached],color=COLORS[0],s=28,label='Executed first hit',zorder=5)
    ax.scatter(gamma[~reached],np.full(np.count_nonzero(~reached),budget),marker='^',facecolors='none',edgecolors=COLORS[0],s=42,label='Not reached by 200k')
    ax.axhline(budget,color=COLORS[0],lw=.8,alpha=.35)
    for kind,color,label in [('analytic',COLORS[2],'Analytic C2 lower bound'),('directional',COLORS[1],'Directional C2 estimate')]:
        values=[]; unresolved=[]
        for g in gamma:
            folder=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,g)
            c=next(c for c in read(folder/'certificates.json') if c['target']==cfg['targets'][0] and c['epsilon']==.01 and c['kind']==kind)
            valid=kind=='analytic' or c.get('resolution')=='fp64_estimate'
            values.append(min(ceiling,c['bound'] if c['bound'] is not None else 10**min(c['log10_bound'],np.log10(ceiling))) if c['log10_bound'] is not None and valid else np.nan)
            unresolved.append(not valid)
        values=np.array(values)
        ax.plot(gamma,values,'s--',ms=3.5,color=color,label=label)
        clipped=values>=ceiling
        ax.scatter(gamma[clipped],values[clipped],marker='^',color=color,s=45)
        if any(unresolved):
            text=', '.join(str(int(g)) for g in gamma[np.array(unresolved)])
            ax.text(.98,.025,'FP64 directional unresolved: γ = '+text,ha='right',transform=ax.transAxes,fontsize=7,color='#555555')
    high=[r for r in read(root/'validation/full_precision.json')['comparisons']
          if r['map']==name and r['target']==cfg['targets'][0] and r['gamma']==1]
    if high:
        ax.scatter([1],[ceiling],marker='*',color=COLORS[1],s=85,zorder=6)
        ax.annotate('120-digit estimate: 10³³ steps',xy=(1,ceiling),xytext=(1.25,1e10),fontsize=7)
    ax.set_yscale('log'); ax.set_ylim(.2,3e11)
    finish_axis(ax,ylabel='Steps to 1% training error')
    ax.set_title('(c) Executed hits and necessary times'); ax.legend(fontsize=7,loc='best')
    fig.suptitle(f'Frozen-gamma theorem evaluation · sine mixture · N={cfg["n"]}',fontsize=12)
    save(fig,root,'banner_'+name)


def primary_precision(root,cfg):
    fig,axes=plt.subplots(2,5,figsize=(17,6.2),layout='constrained')
    for ti,target in enumerate(cfg['targets']):
        for oi,optimizer in enumerate(['gd','adam']):
            ax=axes[oi,ti]
            for color,name in zip(COLORS,cfg['maps']):
                ax.plot(cfg['gammas'],precision_value(root,cfg['n'],name,optimizer,ti),'o-',ms=3,color=color,label=LABELS[name])
            ax.set_yscale('log'); finish_axis(ax,ylabel='Independent relative L2 error' if ti==0 else None)
            ax.set_title(TARGETS[target]+(' · GD' if oi==0 else ' · selected Adam'))
    axes[1,0].legend(fontsize=7)
    fig.suptitle('All targets and readout maps at 200k updates; zero start',fontsize=12)
    save(fig,root,'primary_precision')


def access(root,cfg):
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ri,name in enumerate(cfg['width_maps']):
        folder=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,4)
        a=np.load(folder/'access.npz'); degree=np.arange(len(a['E'])); L=read(folder/'meta.json')['L']
        ax=axes[ri,0]
        for ti,target in enumerate(cfg['targets']):
            ax.semilogy(degree,np.where(a['E'][:,ti]>0,a['E'][:,ti],np.nan),color=COLORS[ti],label=TARGETS[target])
        ax.set_ylim(1e-13,2); ax.set_title('Target tail Eₖ'); ax.set_ylabel(LABELS[name]); ax.legend(fontsize=7)
        ax.text(.03,.04,'Quadratic: Eₖ = 0 for k ≥ 2',transform=ax.transAxes,fontsize=7)
        ax=axes[ri,1]
        floor=a['noise'][0,0]*a['E_measured'][0,0]**2
        resolved=a['mu'][:,0]>a['noise'][:,0]
        for label,value,style,color in [('Analytic Bₖ/L',np.exp(a['log_B_used'])/L,'-',COLORS[2]),
            ('Frobenius/L',np.where(a['frobenius']>floor,a['frobenius']/L,np.nan),':','#777777'),
            ('Sampled bₖ/L',np.where(a['b']>floor,a['b']/L,np.nan),'o',COLORS[0]),
            ('Directional μₖ/L',np.where(resolved,a['mu'][:,0]/L,np.nan),'-',COLORS[1])]:
            ax.semilogy(degree,value,style,color=color,ms=3,label=label)
        ax.set_title('Sine mixture access, γ = 4'); ax.legend(fontsize=7); ax.set_ylim(1e-35,1e3)
        ax.text(.03,.04,'Unresolved FP64 tails omitted',transform=ax.transAxes,fontsize=7)
        ax=axes[ri,2]
        for color,gamma in zip(COLORS,[1,4,16,64]):
            bank=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,gamma)
            z=np.load(bank/'access.npz'); curvature=read(bank/'meta.json')['L']
            resolved=z['mu'][:,0]>z['noise'][:,0]
            ax.semilogy(np.arange(len(resolved)),np.where(resolved,z['mu'][:,0]/curvature,np.nan),color=color,label=f'γ = {gamma}')
        ax.set_title('Resolved directional access μₖ/L'); ax.legend(fontsize=7)
        for ax in axes[ri]:
            ax.set_xlabel('Polynomial cutoff k'); ax.set_xlim(0,128); ax.grid(alpha=.2)
    save(fig,root,'access_and_tails')


def initialization(root,cfg):
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for oi,optimizer in enumerate(['gd','adam']):
        for ti,target in enumerate(cfg['robust_targets']):
            ax=axes[oi,ti]
            for fi,family in enumerate(cfg['initializations']):
                med=[]; low=[]; high=[]
                for name in cfg['maps']:
                    tag=f'N{cfg["n"]}_{name}_initialization_{optimizer}'
                    case=read(root/'training'/tag/'case.json'); values=np.array(final(root,tag)['evaluation'])
                    indices=[i for i,c in enumerate(case['columns']) if c['target']==target and c['initialization']==family]
                    gain=np.log10(values[0,indices]/values[-1,indices])
                    med.append(np.median(gain)); low.append(np.min(gain)); high.append(np.max(gain))
                x=np.arange(len(cfg['maps']))+(fi-.5)*.18
                ax.errorbar(x,med,yerr=[np.array(med)-low,np.array(high)-med],fmt='o',capsize=3,
                    color=COLORS[fi],label=family.replace('_',' '))
            ax.axhline(0,color='#555555',lw=.7); ax.set_xticks(np.arange(len(cfg['maps'])),[LABELS[x].replace(' + ','\n+ ') for x in cfg['maps']],fontsize=7)
            ax.set_title(TARGETS[target]+' · '+optimizer.upper()); ax.set_ylabel('Digits gained from γ = 4 to 64')
            ax.grid(axis='y',alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Paired initialization sensitivity: median and range over five seeds',fontsize=12)
    save(fig,root,'initialization_gains')


def widths(root,cfg):
    references=read(root/'reference/measurements.json')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ri,name in enumerate(cfg['width_maps']):
        for ti,target in enumerate(cfg['robust_targets']):
            ax=axes[ri,ti]
            for gi,label in enumerate(['γ = 1','γ = 4','γ = N/8']):
                for optimizer,style in [('gd','--'),('adam','-')]:
                    values=[]
                    for n in cfg['widths']:
                        target_index=cfg['targets'].index(target) if n==cfg['n'] else ti
                        gamma_index=cfg['gammas'].index([1,4,n/8][gi]) if n==cfg['n'] else gi
                        values.append(precision_value(root,n,name,optimizer,target_index)[gamma_index])
                    ax.loglog(cfg['widths'],values,'o'+style,color=COLORS[gi],ms=3,label=label+' · '+optimizer.upper())
            reference=[next(r['relative_error_extended'] for r in references if r['n']==n and r['target']==target and r['digits']==80) for n in cfg['widths']]
            ax.loglog(cfg['widths'],reference,':',color='#222222',label='QI recovery reference')
            finish_axis(ax,'Core resolution N','Independent relative L2 error')
            ax.set_title(LABELS[name]+' · '+TARGETS[target]); ax.legend(fontsize=6.5)
    save(fig,root,'width_precision')


def damping(root,cfg):
    rows=read(root/'diagnostics/damping.json'); fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ti,target in enumerate(cfg['robust_targets']):
        for pi,probe in enumerate(['full_target','common_tail','shared_GD20k_residual']):
            ax=axes[ti,pi]
            for mi,name in enumerate(cfg['width_maps']):
                for gi,gamma in enumerate(cfg['robust_gammas']):
                    selected=[r for r in rows if r['target']==target and r['probe']==probe and r['map']==name and r['gamma']==gamma]
                    x=np.array([r['rho'] for r in selected]); y=np.array([r['remaining'] for r in selected])
                    resolved=np.array([r['status']=='fp64_estimate' for r in selected])
                    ax.loglog(x,np.where(resolved,y,np.nan),'-' if mi==0 else '--',color=COLORS[gi],label=LABELS[name]+f', γ={gamma}')
                    if (~resolved).any():
                        ax.scatter(x[~resolved],y[~resolved],marker='x',color=COLORS[gi],s=12)
            ax.set_title(TARGETS[target]+' · '+probe.replace('_',' ')); ax.set_xlabel('Relative damping ρ = ζ/L')
            ax.set_ylabel('Remaining residual fraction'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6)
    save(fig,root,'damping_recovery')


def forecasts(root,cfg):
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,name in zip(axes,cfg['width_maps']):
        for cutoff,style in zip(cfg['cutoffs'],[':', '--','-']):
            values=[]
            for gamma in cfg['gammas']:
                folder=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,gamma)
                row=next(c for c in read(folder/'capacity.json') if c['target']==cfg['targets'][0] and c['cutoff']==cutoff)
                prediction=row['predictions'][0]
                values.append(10**prediction['log10_steps'] if prediction['log10_steps'] is not None else np.nan)
            ax.loglog(cfg['gammas'],values,'o'+style,label=f'SVD cutoff {cutoff:g}')
        ax.axhline(cfg['training_steps'],color='#444444',lw=1,label='Executed budget')
        ax.set_xlim(min(cfg['gammas'])*.95,max(cfg['gammas'])*1.05)
        ax.set_title(LABELS[name]); finish_axis(ax,ylabel='Predicted GD steps to 1% error'); ax.legend(fontsize=8)
    fig.suptitle('Unexecuted spectral forecasts and cutoff sensitivity · sine mixture',fontsize=12)
    save(fig,root,'spectral_forecasts')


def tightness(root,cfg):
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ri,name in enumerate(cfg['width_maps']):
        folder=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,4)
        a=np.load(folder/'access.npz'); k=np.arange(len(a['E']))
        resolved=(a['mu'][:,0]>a['noise'][:,0])&np.isfinite(a['b'])
        ax=axes[ri,0]
        for values,label in [(np.exp(a['log_B_used'])/a['frobenius'],'Envelope / Frobenius'),
                              (a['frobenius']/a['b'],'Frobenius / subspace'),
                              (a['b']/a['mu'][:,0],'Subspace / directional')]:
            ax.semilogy(k[resolved],values[resolved],'o-',ms=3,label=label)
        ax.set_title(LABELS[name]+' · γ = 4'); ax.set_ylabel('Access slack factor'); ax.legend(fontsize=7)
        ax=axes[ri,1]
        for gamma,color in zip([4,16,64],COLORS):
            bank=root/'dictionaries'/screen.dictionary_id(cfg['n'],name,gamma)
            z=np.load(bank/'access.npz'); L=read(bank/'meta.json')['L']; degree=np.arange(len(z['E']))
            numerator=np.maximum(z['E'][:,0]-.01,0)**2
            factor=L/np.log(2)*np.log(100)/.99**2
            good=(z['mu'][:,0]>z['noise'][:,0])&(numerator>0)
            ax.semilogy(degree[good],factor*numerator[good]/z['mu'][good,0],color=color,label=f'γ = {gamma}')
        ax.set_title('Directional C2 before integer ceiling'); ax.set_ylabel('Necessary updates'); ax.legend(fontsize=7)
        ax=axes[ri,2]; rows=read(root/'training'/f'N{cfg["n"]}_{name}_gd/evaluations.json')
        for gamma,color in zip([4,16,64],COLORS):
            bi=cfg['gammas'].index(gamma); steps=[r['step']/1000 for r in rows]
            ax.semilogy(steps,[r['train'][bi][0] for r in rows],'o',ms=4,color=color,label=f'GD γ = {gamma}')
            ax.semilogy(steps,[r['spectral'][bi][0] for r in rows],'-',lw=1,color=color)
        ax.set_title('GD markers and spectral lines'); ax.set_xlabel('Executed updates (thousands)'); ax.set_ylabel('Relative training error'); ax.legend(fontsize=7)
        for ax in axes[ri,:2]:
            ax.set_xlabel('Polynomial cutoff k'); ax.set_xlim(0,40)
        for ax in axes[ri]:
            ax.grid(alpha=.2)
    save(fig,root,'tightness_and_gd_checks')


def norm_budgets(root,cfg):
    rows=read(root/'diagnostics/ridge.json'); fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ti,target in enumerate(cfg['robust_targets']):
        for mi,name in enumerate(cfg['width_maps']):
            ax=axes[ti,mi]
            for gi,gamma in enumerate(cfg['robust_gammas']):
                selected=[r for r in rows if r['target']==target and r['map']==name and r['gamma']==gamma]
                good=np.array([r['status']=='fp64_estimate' for r in selected])
                x=np.array([r['native_norm'] for r in selected]); y=np.array([r['error'] for r in selected])
                ax.loglog(x,np.where(good,y,np.nan),'o-',ms=3,color=COLORS[gi],label=f'γ = {gamma}')
                if (~good).any():
                    ax.scatter(x[~good],y[~good],marker='x',s=12,color=COLORS[gi])
            ax.set_title(TARGETS[target]+' · '+LABELS[name]); finish_axis(ax,'Native coefficient L2 norm','Relative fitted error')
            ax.legend(fontsize=8)
    fig.suptitle('Detached ridge paths: coordinate-dependent coefficient budgets',fontsize=12)
    save(fig,root,'coefficient_budgets')


def polynomial_and_rate(root,cfg):
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ax,name in zip(axes[:2],cfg['width_maps']):
        tag=f'N{cfg["n"]}_{name}_polynomial_gd'; values=np.array(final(root,tag)['evaluation'])
        for di,degree in enumerate(cfg['polynomial_degrees']):
            ax.loglog(cfg['robust_gammas'],values[:,di],'o-',color=COLORS[di],label=f'Degree {degree}')
        ax.set_title('Unit polynomial probes · '+LABELS[name]); finish_axis(ax,ylabel='Independent relative L2 error')
        ax.legend(fontsize=7)
    ax=axes[2]; folder=root/'training'/f'N{cfg["n"]}_raw_rate_gd'
    case=read(folder/'case.json'); values=np.array(final(root,folder.name)['evaluation'])
    for ci,column in enumerate(case['columns']):
        ax.loglog(cfg['robust_gammas'],values[:,ci],'o-',color=COLORS[ci],label=TARGETS[column['target']]+f', χ={column["chi"]}')
    ax.set_title('Raw GD learning-rate controls'); finish_axis(ax,ylabel='Independent relative L2 error'); ax.legend(fontsize=7)
    save(fig,root,'polynomial_and_rate_controls')


def coordinate_controls(root,cfg):
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,optimizer in zip(axes,['gd','adam']):
        for ci,name in enumerate(cfg['coordinate_controls']):
            row=final(root,f'N{cfg["n"]}_{name}_coordinate_{optimizer}')
            ax.loglog(cfg['robust_gammas'],np.array(row['evaluation'])[:,0],'o-',color=COLORS[ci],label=name.replace('_',' '))
        for ci,name in enumerate(['raw','collective']):
            values=precision_value(root,cfg['n'],name,optimizer,0 if optimizer=='gd' else len(cfg['targets']))
            ax.loglog(cfg['robust_gammas'],values[[cfg['gammas'].index(g) for g in cfg['robust_gammas']]],'s--',color=COLORS[ci+3],label=LABELS[name])
        ax.set_title('Sine mixture · '+('GD' if optimizer=='gd' else 'common-recipe Adam'))
        finish_axis(ax,ylabel='Independent relative L2 error'); ax.legend(fontsize=7)
    save(fig,root,'coordinate_controls')


def damping_bounds(root,cfg):
    rows=read(root/'diagnostics/damping.json'); fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ti,target in enumerate(cfg['robust_targets']):
        for mi,name in enumerate(cfg['width_maps']):
            ax=axes[ti,mi]
            for gi,gamma in enumerate(cfg['robust_gammas']):
                selected=[r for r in rows if r['target']==target and r['probe']=='common_tail' and r['map']==name and r['gamma']==gamma]
                x=[r['rho'] for r in selected]
                for key,style,label in [('remaining','-','Measured'),('measured_lower','--','Directional lower'),('analytic_lower',':','Analytic lower')]:
                    values=np.array([r[key] for r in selected]); resolved=np.array([r['status']=='fp64_estimate' for r in selected])
                    ax.loglog(x,np.where(resolved,values,np.nan) if key=='remaining' else values,style,color=COLORS[gi],label=f'γ={gamma}, '+label)
                    if key=='remaining':
                        ax.scatter(np.array(x)[~resolved],values[~resolved],marker='x',color=COLORS[gi],s=12)
            ax.set_title(TARGETS[target]+' · '+LABELS[name]); finish_axis(ax,'Relative damping ρ = ζ/L','Remaining tail fraction')
            ax.legend(fontsize=6); ax.set_ylim(1e-14,2)
    save(fig,root,'damping_lower_bounds')


def summarize(root,cfg):
    rows=[]; certificates=[]
    for path in sorted((root/'training').glob('*/case.json')):
        folder=path.parent; case=read(path); endpoint=read(folder/'evaluations.json')[-1]
        expected=cfg['adam_pilot_steps'] if folder.name.endswith('_adam_pilot') else cfg['training_steps']
        assert endpoint['step']==expected,f'Incomplete summary input: {folder.name}'
        audit=np.load(folder/'hitting_audit.npz')
        for bi,gamma in enumerate(case['gammas']):
            for ci,column in enumerate(case['columns']):
                rows.append(dict(run=case['tag'],n=case['n'],map=case['map'],gamma=gamma,optimizer=case['optimizer'],
                    **column,steps=endpoint['step'],train_error=endpoint['train'][bi][ci],
                    rate=case['rates'][bi][ci],optimizer_epsilon=case['epsilons'][bi][ci],
                    evaluation_error=endpoint['evaluation'][bi][ci],validation_error=endpoint['validation'][bi][ci],
                    failed_at=int(endpoint['failed'][bi][ci]),first_hits=audit['first'][bi,ci].tolist(),
                    sustained_hits=audit['sustained'][bi,ci].tolist(),
                    norms={basis:{key:values[ci] for key,values in values.items()} for basis,values in endpoint['norms'][bi].items()}))
        diagnostics=root/'diagnostics'/case['tag']/'certificates.json'
        if diagnostics.exists():
            certificates.extend(read(diagnostics))
    joint={str(n):read(root/'joint'/f'N{n}'/'evaluations.json')[-1] for n in cfg['widths']}
    for row in joint.values():
        assert row['step']==cfg['joint_steps']
    selections={path.stem:read(path) for path in sorted((root/'training').glob('*_selection.json'))}
    independent=sum(not r['run'].endswith('_adam_continue') for r in rows)+sum(len(r['failed']) for r in joint.values())
    result=dict(config=cfg,training=rows,certificates=certificates,complete=True,joint=joint,selections=selections,
        independent_optimizer_trajectories=independent,
        failed_cells=sum(r['failed_at']>0 for r in rows),
        precision=read(root/'validation/full_precision.json'),reference=read(root/'reference/measurements.json'))
    core.write_json(root/'summary.json',train.safe_json(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root; manifest=read(root/'manifest.json'); cfg=manifest['config']
    required=['worker_0.json','worker_1.json','full_precision.json','reference_complete.json','diagnostics_complete.json']
    assert manifest['complete'] and all(read(root/'validation'/p)['complete'] for p in required),'Full campaign is incomplete'
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.frameon':False,'svg.fonttype':'none'})
    summarize(root,cfg)
    for name in cfg['width_maps']:
        banner(root,cfg,name)
    for function in [primary_precision,access,initialization,widths,damping,forecasts,norm_budgets,
                     polynomial_and_rate,coordinate_controls,damping_bounds,tightness]:
        function(root,cfg)


if __name__=='__main__':
    main()
