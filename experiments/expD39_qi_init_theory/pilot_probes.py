"""Posthoc initial/final readout probes on the 52 locked validation pilots.

No test predictions, training, parameter writes, or candidate selection. One
centered SVD per geometry supplies the fixed-cutoff LS probe and, initially,
the same validation-selected frozen-ridge control used by D38.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/precision_d39_matplotlib')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expD39_qi_init_theory import run as campaign
from experiments.expD39_qi_init_theory.followup import EXTRA

OUT = campaign.OUT
RESULT = OUT / 'pilot_readout_probes.json'
FIGURE = OUT / 'figures/pilot_readout_probes.png'
LABELS = {'qi':'Original QI', 'spacing':'Spacing', 'balanced':'Balanced',
          'centered':'Centered', 'collar':'Collar', 'common':'Common gamma',
          'lambda05':'24 dirs / λ .5', 'lambda10':'24 dirs / λ 1',
          'directions64':'64 dirs / soft', 'first_only':'First only',
          'last_only':'Last only', 'soft24':'24 dirs / soft',
          'sharp64':'64 dirs / sharp'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_features(features, targets, rcond, alphas=None):
    """Equivalent to D38's centered solve/ridge grid, sharing a single SVD."""
    hm = features['train'].mean(0)
    ym = targets['train'].mean(0)
    centered = features['train'] - hm
    u, singular, vt = linalg.svd(centered, full_matrices=False, lapack_driver='gesdd')
    rhs = u.T @ (targets['train'] - ym)
    keep = singular > singular[0] * rcond

    def score(alpha):
        if alpha == 0:
            factor = np.zeros_like(singular)
            factor[keep] = 1 / singular[keep]
        else:
            factor = singular / (singular**2 + len(centered) * alpha)
        coef = vt.T @ (factor[:, None] * rhs)
        # Evaluate stored coefficients in feature coordinates, rather than
        # claiming the exact-arithmetic projection residual as computed MSE.
        errors = {k:campaign.base.mse((h-hm) @ coef + ym, targets[k])
                  for k,h in features.items()}
        return dict(alpha=float(alpha), **errors,
                    coef_norm=float(np.linalg.norm(coef)),
                    intercept=float((ym - hm @ coef).item()))

    least_squares = score(0.)
    least_squares.update(rank=int(keep.sum()), singular_max=float(singular[0]),
                         singular_min=float(singular[-1]),
                         retained_condition=float(singular[0]/singular[keep][-1]),
                         participation_rank=float(np.sum(singular**2)**2/np.sum(singular**4)),
                         rcond=float(rcond), solver='scipy.linalg.svd:gesdd')
    result = {'ls':least_squares}
    if alphas is not None:
        grid = [least_squares if alpha == 0 else score(alpha) for alpha in alphas]
        result['frozen_ridge_grid'] = grid
        result['frozen_ridge'] = min(grid, key=lambda row:row['val'])
    return result


def checks():
    """A numerical equivalence check against the reused D38 reference helpers."""
    rng = np.random.default_rng(312)
    features = {k:rng.normal(size=(n,13)) for k,n in [('train',70),('val',31)]}
    targets = {k:rng.normal(size=(len(h),1)) for k,h in features.items()}
    alphas = [0.,1e-6,.01,1.]
    probes = solve_features(features, targets, 1e-12, alphas)
    w,b,info = campaign.base.affine_fit(features['train'], targets['train'], 1e-12)
    for split in features:
        np.testing.assert_allclose(probes['ls'][split],
            campaign.base.mse(features[split]@w+b,targets[split]), rtol=1e-12,atol=1e-14)
    assert probes['ls']['rank'] == info['rank']
    best, grid = campaign.base.frozen_ridge(features, targets, alphas, 1e-12)
    assert probes['frozen_ridge']['alpha'] == best['alpha']
    for actual, expected in zip(probes['frozen_ridge_grid'],grid):
        for split in features:
            np.testing.assert_allclose(actual[split],expected[split],rtol=1e-12,atol=1e-14)


def audit():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    campaign.VARIANTS.update(EXTRA)
    checks()
    selection_path = OUT / 'selection.json'
    selection = json.loads(selection_path.read_text())
    selection_hash = digest(selection_path)
    script_hash = digest(Path(__file__))
    core_sources = ['experiments/expD39_qi_init_theory/initialization.py',
                    'experiments/expD39_qi_init_theory/run.py',
                    'experiments/expD39_qi_init_theory/followup.py',
                    'experiments/expD38_init_readout_baseline/run.py',
                    'experiments/expF04_qi_init_real_data/model.py']
    for source in core_sources:
        assert digest(ROOT/source) == selection['provenance'][source], source
    files = selection['screen_files']
    assert len(files) == 52
    result = dict(selection_sha256=selection_hash, script_sha256=script_hash,
                  purpose='Observational initial/final fit and validation probes; no test predictions or selection changes.',
                  fixed_rcond=1e-12, ridge_selection='initial features only; minimum validation MSE over D38 grid',
                  numerical_equivalence_check='Passed against D38 LS/ridge helpers on a deterministic full-rank problem.',
                  complete=False, rows=[])
    if RESULT.exists():
        old = json.loads(RESULT.read_text())
        assert old['selection_sha256'] == selection_hash and old['script_sha256'] == script_hash
        result = old
    done = {r['source_file']:r for r in result['rows']}
    loaded_tasks = {}
    start = time.perf_counter()
    for source, expected_hash in files.items():
        path = ROOT/source
        assert digest(path) == expected_hash, source
        if source in done:
            assert done[source]['checkpoint_sha256'] == digest(path.with_suffix('.pt'))
            continue
        saved = json.loads(path.read_text())
        identity = saved['identity']
        task, arm, cfg = identity['task'], identity['scheme'], identity['config']
        assert saved['complete'] and identity['steps'] == 10000 and identity['seed'] == 0
        assert saved['trace'][-1]['step'] == 10000
        assert all(set(q['learned']) == {'train','val'} for q in saved['trace'])
        assert cfg['least_squares_rcond'] == 1e-12
        if task not in loaded_tasks:
            arrays, metadata = campaign.base.load_data(task,cfg)
            # The shared loader defines all splits. Discard held-out arrays:
            # every prediction and solve below receives train/val only.
            arrays = {k:arrays[k] for k in ['train','val']}
            inputs = {k:torch.from_numpy(v[0]) for k,v in arrays.items()}
            targets = {k:v[1] for k,v in arrays.items()}
            loaded_tasks[task] = metadata,inputs,targets
        metadata, inputs, targets = loaded_tasks[task]
        assert metadata == saved['data'], (task,arm,'data mismatch')
        checkpoint_path = path.with_suffix('.pt')
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        assert campaign.base.same_identity(checkpoint['identity'],identity)
        model,_ = campaign.make_model(metadata['d_in'],identity['width'],identity['seed'],arm,inputs['train'],cfg)
        record = dict(task=task,scheme=arm,identity=identity,source_file=source,
                      source_sha256=expected_hash,checkpoint_sha256=digest(checkpoint_path),
                      data=metadata,affine=saved['affine'])
        for stage, trace_row in [('initial',saved['trace'][0]),('final',saved['trace'][-1])]:
            if stage == 'final':
                model.load_state_dict(checkpoint['model'])
            predictions,features = campaign.base.predictions(model,inputs,with_features=True)
            actual = {k:campaign.base.mse(predictions[k],targets[k]) for k in inputs}
            for split in inputs:
                np.testing.assert_allclose(actual[split],trace_row['learned'][split],rtol=1e-10,atol=1e-12,
                    err_msg=f'{task}/{arm}/{stage}/{split} does not reconstruct')
            probe = solve_features(features,targets,cfg['least_squares_rcond'],
                                   cfg['feature_ridge_alphas'] if stage=='initial' else None)
            assert probe['ls']['train'] <= actual['train']+1e-8, (task,arm,stage,'LS residual')
            probe.update(trained=actual,step=trace_row['step'],verified_against_trace=True)
            record[stage] = probe
        record['feature_fit_ratio_final_to_initial_ls'] = record['final']['ls']['train']/record['initial']['ls']['train']
        record['readout_fit_ratio_trained_to_final_ls'] = record['final']['trained']['train']/record['final']['ls']['train']
        result['rows'].append(record)
        result['elapsed_seconds_this_invocation'] = time.perf_counter()-start
        campaign.base.save_json(RESULT,result)
        print(f"PROBED {len(result['rows'])}/52 {task} {arm} initial/final train+val",flush=True)
    assert len(result['rows']) == 52
    assert digest(selection_path) == selection_hash, 'Selection changed during observational audit'
    result['complete'] = True
    campaign.base.save_json(RESULT,result)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    record = json.loads(RESULT.read_text())
    assert record['complete'] and len(record['rows']) == 52
    selection = json.loads((OUT/'selection.json').read_text())
    assert digest(OUT/'selection.json') == record['selection_sha256']
    arms = list(selection['scores'])
    tasks = ['airfoil','kin8nm','sarcos','superconductivity']
    rows = {(r['task'],r['scheme']):r for r in record['rows']}
    curves = [('Initial features · LS', '#3476ad', ':', 'o', lambda r:r['initial']['ls']),
              ('Initial features · validation-selected ridge', '#4a9b69', '--', 's', lambda r:r['initial']['frozen_ridge']),
              ('Final features · LS', '#91569e', ':', 'D', lambda r:r['final']['ls']),
              ('Final trained readout', '#dd7b1f', '-', 'o', lambda r:r['final']['trained'])]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(4,2,figsize=(18,16),sharex=True,sharey=True)
    plotted = [getter(rows[t,a])[s] for t in tasks for a in arms for s in ['train','val']
               for _,_,_,_,getter in curves]
    bottom = 10.**np.floor(np.log10(min(plotted)))
    top = 1.
    for i,task in enumerate(tasks):
        for j,split in enumerate(['train','val']):
            ax=axes[i,j]
            for label,color,style,marker,getter in curves:
                values=np.array([getter(rows[task,a])[split] for a in arms])
                visible=np.where(values<=top,values,np.nan)
                ax.plot(np.arange(len(arms)),visible,linestyle=style,marker=marker,color=color,lw=1.5,ms=4,label=label)
                above=values>top
                ax.scatter(np.flatnonzero(above),np.full(above.sum(),.87*top),marker='^',s=33,color=color,zorder=5)
            ax.axhline(rows[task,arms[0]]['affine'][split],color='#777777',ls='--',lw=.8,alpha=.7)
            ax.set(yscale='log',ylim=(bottom,top),xlim=(-.5,len(arms)-.5),
                   ylabel='MSE (standardized target)',
                   title=f"{campaign.base.TITLES[task]} · {'Fit' if split=='train' else 'Validation'}\nWidth 512 + 512")
            ax.grid(axis='y',which='both',alpha=.16)
            ax.set_xticks(np.arange(len(arms)),[LABELS[a] for a in arms],rotation=48,ha='right')
    handles=[Line2D([],[],color=c,ls=s,marker=m,label=l) for l,c,s,m,_ in curves]
    handles.append(Line2D([],[],color='#777777',ls='--',label='Input linear regression'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.975),ncol=3,frameon=False)
    fig.suptitle('Initial and learned feature quality versus the trained readout\n52 validation pilots · seed 0 · 10,000 steps · observational solves only',y=.999,fontsize=16)
    fig.text(.5,.012,'Shared logarithmic MSE axes. Upward triangles mark values above 1; complete values are saved in JSON.\n'
             'Ridge alpha is selected on this validation set, so its validation score is optimistic. No test predictions were made.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.90,bottom=.12,hspace=.33,wspace=.17)
    FIGURE.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURE,dpi=155,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['run','plot','check'],default='run',nargs='?')
    mode=p.parse_args().mode
    if mode=='check':
        checks()
        print('Numerical LS/ridge equivalence check passed.')
    elif mode=='plot':
        plot()
    else:
        audit()
        plot()
