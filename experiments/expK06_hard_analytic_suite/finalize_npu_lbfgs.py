"""Validate and prepare updated NPU curves, preserving the Adam-only artifacts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.expK06_hard_analytic_suite.common import OUT,WIDTHS,save,quadrature
from experiments.expK06_hard_analytic_suite.targets import MAIN,target
from experiments.expK06_hard_analytic_suite.verify import load_row
from experiments.expK06_hard_analytic_suite.train import prediction,data
from experiments.expK06_hard_analytic_suite.integration import accurate_score
from experiments.expK06_hard_analytic_suite.integration import zero_crossings

DEST=OUT/'npu_adam_lbfgs'

def main():
    torch.set_num_threads(1)
    new=json.loads((DEST/'summary.json').read_text());assert len(new)==24
    old=json.loads((OUT/'verified_summary.json').read_text())
    checks=[]
    for name in MAIN:
        assert sorted(r['width'] for r in new if r['target']==name)==WIDTHS
    for r in new:
        assert r['method']=='NPU' and r['variant']!='real_native' and not r['readout_refit']
        assert r['steps']==50000 and 0<=r['lbfgs_iterations']<=1000
        assert r['validation']<=r['original_adam_validation']
        source=ROOT/r['source_run_directory']/'selected.pt'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_checkpoint_sha256']
        m,amp,offset=load_row(r);x,y,a,o=data(r);assert amp==a and offset==o
        order=max(map(int,r['independent_refined_errors']))
        error=accurate_score(r['target'],m,amp,offset,order)
        assert np.isclose(error,r['relative_l2'],rtol=1e-12,atol=1e-15)
        checks.append(dict(target=r['target'],width=r['width'],reloaded_error=error,
                           quadrature_stable=r['quadrature_stable'],source_hash_matches=True,
                           training_grid_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()))
    assert all(r['quadrature_stable'] for r in checks),'Inspect quadrature before publishing'
    rows=[r for r in old if r['method']!='NPU']+new
    # User-approved alternative after the pointwise audit, not a claim that this
    # checkpoint minimizes L2. Preserve all six measured scaling points.
    selection_note=('For each function, the displayed NPU fit is the saved Adam-to-L-BFGS model '
                    'with the smallest sampled maximum absolute residual on the pointwise audit grid '
                    '(4097 uniform points plus exact input zeros and logarithmic probes around them). '
                    'This user-requested post-audit selection replaces minimum-L2 selection. '
                    'Independent L2 errors are reported for the selected models. Adam and QUILL '
                    'selection and all six measured scaling points are unchanged.')
    audit=json.loads((OUT/'npu_spike_probe/pointwise_candidate_audit.json').read_text())
    alternatives={name:min((r for r in audit if r['target']==name),key=lambda r:r['max_sampled_absolute_residual']) for name in MAIN}
    pointwise_checks=[]
    selected=[]
    for name in MAIN:
        for method in ['QUILL','Adam','NPU']:
            r=min((r for r in rows if r['target']==name and r['method']==method and r['width'] in WIDTHS),key=lambda r:r['validation'])
            if method=='NPU':
                diagnostic=alternatives[name]
                r=next(a for a in new if a['target']==name and a['width']==diagnostic['width'])
                r={**r,'display_selection':selection_note,'pointwise_audit':diagnostic}
                r=dict(r);m,amp,offset=load_row(r)
                x=np.linspace(-1,1,8193);true=target(name,x);yp=prediction(m,x,amp,offset)
                xq,wq=quadrature(128);rms=np.sqrt(np.dot(wq,target(name,xq)**2)/2)
                folder=DEST/'predictions';folder.mkdir(exist_ok=True)
                p=folder/f'{name}__NPU.npz'
                np.savez(p,x=x,truth=true,prediction=yp,residual=yp-true,normalized_residual=(yp-true)/rms,rms=rms)
                r['predictions']=str(p.relative_to(ROOT))
                # Additional denser/random probes verify the selected fit, but do
                # not change the audit-grid selection. This is not a sup-norm proof.
                probes=[np.linspace(-1,1,65537),np.random.default_rng(9031).uniform(-1,1,8192)]
                for root in zero_crossings(m):
                    delta=np.geomspace(3e-18,.1,233)
                    probes.extend([np.array([root]),root-delta,root+delta])
                probe_x=np.unique(np.clip(np.concatenate(probes),-1,1))
                diff=prediction(m,probe_x,amp,offset)-target(name,probe_x)
                check=dict(target=name,width=r['width'],variant=r['variant'],
                           validation_max_absolute=diagnostic['max_sampled_absolute_residual'],
                           independent_max_sampled_absolute=float(np.max(abs(diff))),points=len(probe_x))
                pointwise_checks.append(check);r['pointwise_check']=check
            else:
                r=next(a for a in json.loads((OUT/'selected_fits.json').read_text()) if a['target']==name and a['method']==method)
            selected.append(r)
    prior=DEST/'prior_adam_only';prior.mkdir(exist_ok=True)
    for name in ['hard_functions_3x4.png','hard_functions_3x4.pdf','figure_provenance.json']:
        if not (prior/name).exists():shutil.copy2(OUT/name,prior/name)
    save(DEST/'verified_summary.json',rows);save(DEST/'selected_fits.json',selected)
    save(DEST/'verification.json',dict(models=checks,no_readout_refitting=True,no_realnpu=True,
         adam_and_quill_results_unchanged=True,display_selection=selection_note,
         scaling_data_unchanged=True,pointwise_checks=pointwise_checks))
    save(DEST/'display_selection.json',dict(description=selection_note,alternatives=alternatives,
         original_l2_winners={name:min((r for r in audit if r['target']==name),key=lambda r:r['validation_l2']) for name in MAIN}))
    labels={
        'airy':r'\(\mathord{\operatorname{Ai}(-30x)\;\text{(Airy)}}\)',
        'kepler':r'\(\mathord{y-0.9\sin y=\pi x\;\text{(Kepler inversion)}}\)',
        'elliptic_k':r'\(\mathord{K\!\left(\frac{0.99}{1+9x^2}\right)\;\text{(elliptic)}}\)',
        'chirped_division':r'\(\mathord{\frac{e^{-2x^2}\cos(30x+12x^2)}{1+100(x-0.2)^2}}\)',
    }
    lines=['**Table 1. High-precision approximation of difficult functions**','',
           '| Function on [-1,1] | Adam-trained MLP | Gated NPU | QUILL |',
           '|:---|---:|---:|---:|']
    for name in MAIN:
        vals={r['method']:r['relative_l2'] for r in selected if r['target']==name}
        lines.append(f"| {labels[name]} | {vals['Adam']:.1e} | {vals['NPU']:.1e} | **{vals['QUILL']:.1e}** |")
    lines+=['','Independent test relative L2 errors across six hidden widths. '
            'MLP: 50,000 Adam steps. NPU: 50,000 Adam steps followed by up to 1,000 full-batch L-BFGS iterations; '
            'no readout refitting or RealNPU. All methods use Float64 and identical sample locations and function '
            'values at each width. Selected widths may differ between methods. '
            'Displayed NPU models are selected by the smallest sampled maximum absolute residual across '
            'the six widths, including probes near input zero crossings; their independent relative L2 '
            'errors are reported here. The Airy model uses the composed gated NPU variant. '
            'Adam and QUILL models retain validation-L2 selection.']
    (DEST/'table1.md').write_text('\n'.join(lines)+'\n')
    save(DEST/'selected_comparison.json',[dict(target=r['target'],method=r['method'],width=r['width'],
         relative_l2=r['relative_l2'],validation=r['validation'],selected_phase=r.get('selected_phase','QUILL')) for r in selected])
    for r in selected:
        print(r['target'],r['method'],r['width'],r['relative_l2'])

if __name__=='__main__':main()
