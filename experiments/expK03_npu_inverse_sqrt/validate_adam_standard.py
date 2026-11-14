"""Verify the conventional-initialization baseline and document excluded runs."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.adam_standard import DEST,ALLOWED_INITIALIZERS,build_model
from experiments.expK03_npu_inverse_sqrt.run import OUT,save_json,evaluation_grid,relative_l2
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    torch.set_num_threads(1)
    rows=json.loads((DEST/'summary.json').read_text())
    old=json.loads((DEST/'previous_eligible.json').read_text())
    selected=json.loads((DEST/'selected_reference.json').read_text())
    assert len(rows)==12 and 'pending_new_standard_trials' not in selected
    e=Extension();grids={n:evaluation_grid(e,n) for n in [32,256,512]};checks=[]
    original=np.load(OUT/'adam_mlp_1k/matched_halo_data.npz')
    for row in rows+old:
        directory=ROOT/row['run_directory'];c=json.loads((directory/'config.json').read_text())
        c['initializer']=c.get('initializer','pytorch')
        assert c['initializer'] in ALLOWED_INITIALIZERS and c.get('output_scale',1)==1
        assert c['width']==128 and c['steps']==100000
        legacy=row.get('source_family')=='constant_rate_search'
        prefix='Adam MLP_matched_halo_'
        initial_path=directory/(prefix+'initial.pt' if legacy else 'initial.pt')
        initial=torch.load(initial_path,weights_only=False)
        expected=build_model(c)
        assert all(torch.equal(v,initial['model'][k]) for k,v in expected.state_dict().items())
        max_weight=float(initial['model']['net.0.weight'].abs().max())
        assert max_weight<=1
        # Exact independence of initialization from epsilon/domain information.
        other=build_model({**c,'epsilon':.01})
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in expected.state_dict().items())
        path=directory/(prefix+'best_trained_validation.pt' if legacy else 'best_validation.pt')
        checkpoint=torch.load(path,weights_only=False);assert checkpoint['step']>=50000
        expected.load_state_dict(checkpoint['model'])
        errors={str(n):relative_l2(expected,g,e.eps) for n,g in grids.items()}
        np.testing.assert_allclose(errors['32'],row['best_trained_validation_rel_l2'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(errors['256'],row['best_trained_validation_test_rel_l2'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(errors['256'],errors['512'],rtol=1e-10,atol=1e-14)
        history=json.loads((directory/(prefix+'history.json' if legacy else 'history.json')).read_text())
        assert history[-1]['step']==100000
        best=min((r for r in history if r['step']>=50000),key=lambda r:r['validation_rel_l2'])
        assert best['step']==checkpoint['step']
        predictions=np.load(directory/(prefix+'predictions.npz' if legacy else 'predictions.npz'))
        assert np.array_equal(predictions['halo_x'],original['x']) and np.array_equal(predictions['halo_target'],original['y'])
        checks.append({'name':row['name'],'initializer':c['initializer'],'selected_step':checkpoint['step'],
                       'initial_max_abs_hidden_weight':max_weight,'errors':errors,
                       'exact_standard_initialization':True,'domain_independent_initialization':True,'unchanged_halo':True})
    assert min(rows+old,key=lambda r:r['best_trained_validation_rel_l2'])==selected
    save_json(DEST/'validation.json',{'checks':checks,'all_12_new_runs_completed_100k':True,
              'excluded_all_multiscale_and_output_scaled_runs':True,'selection_validation_only':True})
    fields=['name','initializer','hidden_units','batch_size','learning_rate','final_learning_rate',
            'best_trained_validation_step','best_trained_validation_test_rel_l2','final_rel_l2']
    with (DEST/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows+old)
    lines=['# Conventional Adam baseline: replacement for the rejected multiscale initialization','',
           '**The random-multiscale result is excluded from this baseline and from the current plot.** '
           'It is preserved only as an archived exploratory result.','',
           f'Selected: **{selected["name"]}**, 128 neurons / 385 parameters, '
           f'checkpoint **{selected["best_trained_validation_step"]:,}**, relative L2 '
           f'**{selected["best_trained_validation_test_rel_l2"]:.9g}**.','',
           '## Allowed techniques','',
           '- Unmodified PyTorch Linear initialization, or library Xavier uniform (tanh gain 5/3 in the hidden layer, gain 1 in the linear readout, zero biases). Initial hidden weights are at most 1 in magnitude and independent of epsilon and the target function.',
           '- Ordinary 128-neuron single-hidden-layer tanh MLP. The same fixed affine input centering as before; no log inputs, engineered feature scales, center placement or activation reparameterization.',
           '- Raw unweighted MSE on exactly the same C17 halo-center labels. No output gain or target rescaling.',
           '- Adam rates 0.001, 0.01, 0.03; batches 100 or 1024; rate held for 50k updates then cosine-decayed to 1e-5 by 100k. One fixed seed 0, float64, no clipping or weight decay.','',
           'This makes 12 new 100k-step runs. Four earlier eligible standard 128-neuron runs '
           '(three constant rates and one batch-512 cosine run) remain candidates; the selection '
           'does not throw away a better allowed result. All selected checkpoints must have '
           'at least 50k updates. Only order-32 validation scores select a model; independent '
           'order-256/512 quadrature determines the reported test error.','',
           '| Trial | Init | Batch | Peak LR | Selected step | Test relative L2 |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows+old:
        lines.append(f'| {r["name"]} | {r["initializer"]} | {r.get("batch_size",100)} | '
                     f'{r["learning_rate"]:g} | {r["best_trained_validation_step"]:,} | '
                     f'{r["best_trained_validation_test_rel_l2"]:.9g} |')
    lines+=['','## Verification and scope','',
            'All 16 eligible candidates were reloaded. Initial tensors match the corresponding '
            'standard initializer exactly and do not change when epsilon is changed. Every '
            'new run completed 100k updates. Checkpoint selection, independent test quadrature, '
            'and bitwise equality of the halo labels were checked. Fourteen implementation tests '
            'passed, including standard-library Xavier equivalence and domain independence.','',
            'The target, halo-center dataset (N=256, R=32), and physical dx relative-L2 metric '
            'on [0.0001,1] are unchanged. Training still uses unweighted halo-sample MSE, whose '
            'domain/weights differ from the evaluation metric. This is a conventional Adam '
            'baseline under the stated protocol, not a lower bound on all possible training. '
            'NPU/RealNPU curves and their earlier budgets are unchanged.','',
            'See specification.json, summary.csv, selected_reference.json, validation.json, '
            'and all per-trial configs, initial models, optimizer checkpoints, histories and predictions.']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    print('Verified',len(checks),'eligible candidates. Selected:',selected['name'],selected['best_trained_validation_test_rel_l2'])


if __name__=='__main__':main()
