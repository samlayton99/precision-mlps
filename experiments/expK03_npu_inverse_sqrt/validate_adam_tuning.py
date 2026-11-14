"""Audit all nine long Adam runs and the validation-only selection process."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,make_model,evaluation_grid,relative_l2,save_json
from experiments.expK03_npu_inverse_sqrt.adam_tuning import select_widths,RATES
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    torch.set_num_threads(1);output=OUT/'adam_tuning'
    pilots=json.loads((output/'pilots/summary.json').read_text())
    finals=json.loads((output/'long_runs/summary.json').read_text())
    selection=json.loads((output/'width_selection.json').read_text())
    selected=json.loads((output/'selected_reference.json').read_text())
    assert len(pilots)==24 and len(finals)==9
    widths,_=select_widths(pilots);assert widths==selection['selected_widths']
    assert {(r['hidden_units'],r['learning_rate']) for r in finals}=={(h,lr) for h in widths for lr in RATES}
    e=Extension();grids={n:evaluation_grid(e,n) for n in [32,256,512]};checks=[]
    for row in finals:
        folder=ROOT/row['run_directory'];stem='Adam MLP_matched_halo'
        checkpoint=torch.load(folder/f'{stem}_best_trained_validation.pt',weights_only=False)
        c=checkpoint['config'];assert c['steps']==100000 and c['minimum_selection_step']==50000
        assert checkpoint['step']>=50000
        model=make_model(c,'Adam MLP');model.load_state_dict(checkpoint['model'])
        assert sum(p.numel() for p in model.parameters())==3*c['width']+1
        errors={str(n):relative_l2(model,g,e.eps) for n,g in grids.items()}
        np.testing.assert_allclose(errors['32'],row['best_trained_validation_rel_l2'],rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(errors['256'],row['best_trained_validation_test_rel_l2'],rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(errors['256'],errors['512'],rtol=1e-10,atol=1e-14)
        history=json.loads((folder/f'{stem}_history.json').read_text())
        assert history[-1]['step']==100000
        chosen=min((r for r in history if r['step']>=50000),key=lambda r:r['validation_rel_l2'])
        assert chosen['step']==checkpoint['step']==row['best_trained_validation_step']
        pilot=next(r for r in pilots if r['hidden_units']==c['width'] and r['learning_rate']==c['learning_rate'])
        p=torch.load(ROOT/pilot['run_directory']/f'{stem}_step10000.pt',weights_only=False)
        q=torch.load(folder/f'{stem}_step10000.pt',weights_only=False)
        assert all(torch.equal(v,q['model'][k]) for k,v in p['model'].items())
        assert p['losses']==q['losses']
        pred=np.load(folder/f'{stem}_predictions.npz')
        target=np.load(output/'matched_halo_data.npz')
        assert np.array_equal(pred['halo_x'],target['x']) and np.array_equal(pred['halo_target'],target['y'])
        checks.append({'width':c['width'],'learning_rate':c['learning_rate'],'steps':100000,
                       'selected_step':checkpoint['step'],'errors':errors,
                       'pilot_prefix_bitwise_identical':True,'halo_data_bitwise_identical':True})
        print(f'Checked h={c["width"]} lr={c["learning_rate"]:g}, step={checkpoint["step"]}, error={errors["256"]:.8g}',flush=True)
    expected=min(finals,key=lambda r:(r['best_trained_validation_rel_l2'],r['hidden_units'],r['learning_rate']))
    assert expected==selected
    save_json(output/'validation.json',{'checks':checks,'all_nine_completed_100k':True,
              'selection_uses_validation_only':True,'selected_checkpoint_step':selected['best_trained_validation_step']})
    fields=['hidden_units','parameters','learning_rate','steps','minimum_selection_step',
            'best_trained_validation_step','best_trained_validation_rel_l2',
            'best_trained_validation_test_rel_l2','final_rel_l2']
    for name,rows in [('pilot_table',pilots),('long_run_table',finals)]:
        with (output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Adam baseline: width/rate pilots and nine 100k runs','',
           f'Selected baseline: **width {selected["hidden_units"]}, lr={selected["learning_rate"]:g}, '
           f'checkpoint {selected["best_trained_validation_step"]:,}**, test relative L2 '
           f'**{selected["best_trained_validation_test_rel_l2"]:.9g}**. '
           'Selection uses validation scores, never the reported test quadrature.','',
           'Pilot widths: 8, 16, 32, 64, 128, 256, 512, 1,000. Each is trained for 10k '
           'steps at learning rates 1e-2, 1e-3, 1e-4 (24 pilots). Rank each width by its '
           'lowest validation error in steps 5k–10k over those three rates.','',
           f'The resulting candidate widths are **{widths}**. All nine width/rate combinations '
           'are trained from initialization to 100,000 updates. The plotted checkpoint is '
           'selected only from steps 50k–100k, so it is not a short-run or initialization result.','',
           '| Width | LR | Selected step | Test relative L2 | Final relative L2 |',
           '|---:|---:|---:|---:|---:|']
    for r in finals:
        lines.append(f'| {r["hidden_units"]} | {r["learning_rate"]:g} | {r["best_trained_validation_step"]:,} | '
                     f'{r["best_trained_validation_test_rel_l2"]:.8g} | {r["final_rel_l2"]:.8g} |')
    lines+=['','## Shared protocol','',
            'Ordinary centered tanh MLP, all weights/biases trained, PyTorch default initialization, '
            'seed 0, float64. The C17 target, N=256/R=32 halo-center dataset (4,815 nodes), '
            'data seed 10000, batch 100, Adam betas=(0.9,0.999), epsilon=1e-8, and unweighted '
            'MSE are unchanged from the previous Adam reference. No warmup, LR decay or target '
            'rescaling. Evaluation is physical dx relative L2 on [1e-4,1].','',
            'The halo training loss and original-interval evaluation metric have different '
            'domains/weights. This is the specified halo-fitting protocol, not an assertion '
            'of globally optimal Adam training. NPU/RealNPU curves retain their earlier 20k '
            'budgets; the Adam baseline now receives this larger tuning/training budget.','',
            '## Validation and saved artifacts','',
            'Every selected checkpoint was reloaded and scored with independent quadrature '
            'orders 256 and 512. Each long run completed all 100k steps. Its first 10k updates '
            'match the corresponding pilot model and loss history bitwise. Target arrays '
            'match the saved C17 halo data bitwise. The eleven implementation tests include '
            'selection-window enforcement and validation-only width selection.','',
            'See `specification.json`, `width_selection.json`, `pilot_table.csv`, '
            '`long_run_table.csv`, `selected_reference.json`, `validation.json`, and '
            'per-run configs, model/optimizer checkpoints, histories and predictions. '
            'All previous 20k Adam references and both NPU sweeps are preserved.']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(output/'results.md')


if __name__=='__main__':main()
