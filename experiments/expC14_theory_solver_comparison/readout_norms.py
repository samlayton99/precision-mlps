"""Native readout L1 versus hidden width: three cutoff/scaling choices.

All three use FP64 DGEJSV and the same pairwise projection/export. Reuse
the two saved DGEJSV scaled runs; compute the unscaled DGEJSV arm anew.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.affine_grid import training
from experiments.expC14_theory_solver_comparison.jacobi_probe import jacobi_svd
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare

ARMS = ['scaled_theorem', 'scaled_fixed', 'unscaled_fixed']


def draw(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'pdf.fonttype':42,'ps.fonttype':42})
    labels = [r'$e^x$',r'$\sin(4\pi x)$',r'$\sin(24\pi x)$','Chirp','Mixture','Wave packet']
    arms=ARMS
    legend_labels=[r'$D$ + theorem cutoff',r'$D$ + fixed cutoff','No $D$ + fixed cutoff']
    styles = [('#D55E00',(0,(2,2)),'s'),('#0072B2',(0,(5,2)),'^'),('#222222','-','o')]
    if all('scaled_sharp' in row for row in rows):
        arms=['scaled_theorem','scaled_sharp','scaled_fixed','unscaled_fixed']
        styles=[('#D55E00',(0,(2,2)),'s'),('#009E73',(0,(5,1,1,1)),'D'),
                ('#0072B2',(0,(5,2)),'^'),('#222222','-','o')]
        legend_labels=[r'$D$ + original bound',r'$D$ + sharper bound',
                       r'$D$ + fixed cutoff','No $D$ + fixed cutoff']
    fig, axes = plt.subplots(2,3,figsize=(10.8,6.4),sharex=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.12,top=.84,wspace=.25,hspace=.32)
    for ax, name, label in zip(axes.flat,r.NAMES,labels):
        rr=sorted([x for x in rows if x['target']==name],key=lambda x:x['W'])
        for arm,(color,ls,marker) in zip(arms,styles):
            ax.plot([x['W'] for x in rr],[x[arm]['l1'] for x in rr],
                    color=color,ls=ls,marker=marker,lw=1.8,ms=3.6,
                    mfc='white' if arm!='unscaled_fixed' else color,mew=.9)
        ax.set_title(label,fontsize=13,pad=8)
        ax.set_yscale('symlog',linthresh=1,linscale=.5)
        maximum=max(x[a]['l1'] for x in rr for a in arms)
        ax.set_ylim(0,maximum*1.2)
        last=int(np.floor(np.log10(maximum*1.2)))
        if last<1:
            ticks=[t for t in [0,1,2,5,10] if t<=maximum*1.2]
        else:
            step=max(1,int(np.ceil(last/4)))
            ticks=([0,1] if last<5 else [0])+[10.**e for e in range(step,last+1,step)]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:
            f'{v:g}' if v<10 else rf'$10^{{{int(round(np.log10(v)))}}}$'))
        ax.minorticks_off()
        ax.set_xlim(65,870)
        ax.set_xticks([87,301,577,847])
        ax.grid(axis='y',alpha=.16)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',labelsize=10)
    fig.legend(handles=[Line2D([],[],color=c,ls=ls,marker=m,ms=4,lw=1.8,
        mfc='white' if a!='unscaled_fixed' else c,label=label)
        for a,(c,ls,m),label in zip(arms,styles,legend_labels)],
        loc='upper center',bbox_to_anchor=(.535,.99),frameon=False,
        ncol=2 if len(arms)==4 else 3,handlelength=3)
    fig.supxlabel(r'Hidden width $W$',fontsize=14,y=.025)
    fig.supylabel(r'Readout $\ell^1$ norm',fontsize=14,x=.012)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'readout_l1.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--theorem',type=Path,required=True)
    parser.add_argument('--scaled-fixed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load((args.theorem/'config.yaml').read_text())
    fixed_cfg=yaml.safe_load((args.scaled_fixed/'config.yaml').read_text())
    assert fixed_cfg['scaled_driver']=='gejsv'
    for key in ['N','lambda','a_H','delta','Delta','sampling_density','feature_parameterization']:
        assert cfg[key]==fixed_cfg[key],key
    out=args.output;out.mkdir(exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(dict(geometry=cfg,
        driver_all_arms='gejsv',fixed_relative_cutoff=2**-52,
        plotted_quantity='sum of absolute native output coefficients, including bias',
        x_axis='W=N+2R+1 hidden neurons; excludes output bias',
        theorem_source=str(args.theorem.resolve()),scaled_fixed_source=str(args.scaled_fixed.resolve())),sort_keys=False))
    snap=out/'source_snapshot';snap.mkdir()
    hashes={}
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,snap/f.name);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    provenance=dict(sources=hashes,inputs={},python=sys.version,packages={k:importlib.metadata.version(k)
        for k in ['numpy','scipy','gmpy2','matplotlib','PyYAML','threadpoolctl']})
    rows=[]
    with threadpool_limits(limits=cfg['threads']):
        for N in cfg['N']:
            params=prepare(N,cfg)
            A,y=training(N,params['R'],cfg['sampling_density'])
            U,s,Vt,meta=jacobi_svd(A/math.sqrt(len(A)),joba=0,jobv=1)
            yn=y/math.sqrt(len(A));tau=2**-52*s[0];keep=s>tau
            cs=[]
            for j in range(6):
                p=r.projection(U,yn[:,j]);z=np.zeros_like(p)
                z[keep]=p[keep]/s[keep]
                cs.append(r.export(Vt,z,np.ones(len(z))))
            raw=np.column_stack(cs)
            tf=args.theorem/f'coefficients_N{N}.npz'
            sf=args.scaled_fixed/f'coefficients_N{N}.npz'
            tj=args.theorem/f'N{N}.json';sj=args.scaled_fixed/f'N{N}.json'
            for f in [tf,sf,tj,sj]:provenance['inputs'][str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
            with np.load(tf) as t, np.load(sf) as f:
                assert np.array_equal(params['D'],t['D']) and np.array_equal(t['D'],f['D'])
                coefficients=dict(scaled_theorem=t['theory'],scaled_fixed=f['scaled'],unscaled_fixed=raw)
            tr=json.loads(tj.read_text());sr=json.loads(sj.read_text())
            for j,name in enumerate(r.NAMES):
                assert tr[j]['target']==sr[j]['target']==name
                row=dict(N=N,W=len(params['centers']),target=name)
                for arm in ARMS:
                    c=coefficients[arm][:,j]
                    row[arm]=dict(l1=float(np.sum(abs(c))),
                        neuron_weights_l1=float(np.sum(abs(c[:-1]))),absolute_bias=float(abs(c[-1])))
                row['scaled_theorem'].update(rank=tr[j]['theory']['rank'],absolute_cutoff=tr[j]['theory']['tau'])
                row['scaled_fixed'].update(rank=sr[j]['scaled_rank'],absolute_cutoff=sr[j]['scaled_absolute_cutoff'],relative_cutoff=2**-52)
                row['unscaled_fixed'].update(rank=int(keep.sum()),absolute_cutoff=float(tau),relative_cutoff=2**-52)
                rows.append(row)
            np.savez_compressed(out/f'coefficients_N{N}.npz',**coefficients,D=params['D'],unscaled_singular=s)
            print(f'N={N}, W={len(params["centers"])}: '+', '.join(f'{name} '+ '/'.join(
                f'{rows[-6+j][a]["l1"]:.3g}' for a in ARMS) for j,name in enumerate(r.NAMES)),flush=True)
    (out/'norms.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    draw(out,rows)


if __name__=='__main__':main()
