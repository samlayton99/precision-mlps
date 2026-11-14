"""Validate exact per-width halo matching and summarize the completed sweep."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,QI,make_model,halo_nodes,evaluation_grid,relative_l2,save_json
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    torch.set_num_threads(1);output=OUT/'matched_construction_widths'
    rows=json.loads((output/'summary.json').read_text());qi=json.loads((QI/'scaling_1k_5k.json').read_text())
    spec=json.loads((output/'sweep_specification.json').read_text())
    assert len(rows)==2*len(qi)
    e=Extension();grids={n:evaluation_grid(e,n) for n in [32,256,512]};checks=[]
    for name in ['NPU','RealNPU']:
        assert sorted(r['hidden_units'] for r in rows if r['model']==name)==[r['width'] for r in qi]
    for row in rows:
        point=next(r for r in qi if r['width']==row['hidden_units'])
        expected_x=halo_nodes(e,point['N'],point['halo']);expected_y=e.value(expected_x)
        saved=np.load(ROOT/row['halo_dataset'])
        assert np.array_equal(saved['x'],expected_x) and np.array_equal(saved['y'],expected_y)
        assert hashlib.sha256(expected_x.tobytes()+expected_y.tobytes()).hexdigest()==row['halo_data_sha256']
        folder=ROOT/row['run_directory'];stem=f'{row["model"]}_matched_halo'
        predicted=np.load(folder/f'{stem}_predictions.npz')
        assert np.array_equal(predicted['halo_x'],expected_x) and np.array_equal(predicted['halo_target'],expected_y)
        checkpoint=torch.load(folder/f'{stem}_best_trained_validation.pt',weights_only=False)
        c=checkpoint['config']
        for key in ['learning_rate','batch_size','seed','data_seed','steps','betas','adam_epsilon','dtype']:
            assert c[key]==spec['base_config'][key]
        assert c['width']==point['width'] and c['halo_reference_N']==point['N'] and c['halo_reference_R']==point['halo']
        model=make_model(c,row['model']);model.load_state_dict(checkpoint['model'])
        expected_parameters=(3 if row['model']=='NPU' else 2)*point['width']+1
        assert sum(p.numel() for p in model.parameters())==expected_parameters==row['parameters']
        errors={str(n):relative_l2(model,grid,e.eps) for n,grid in grids.items()}
        np.testing.assert_allclose(errors['256'],row['best_trained_validation_test_rel_l2'],rtol=2e-13,atol=1e-15)
        np.testing.assert_allclose(errors['256'],errors['512'],rtol=1e-9,atol=1e-14)
        history=json.loads((folder/f'{stem}_history.json').read_text())
        chosen=min((r for r in history if r['step']>0),key=lambda r:r['validation_rel_l2'])
        assert checkpoint['step']==chosen['step']==row['best_trained_validation_step']
        checks.append({'model':row['model'],'width':point['width'],'N':point['N'],
                       'parameters':expected_parameters,'step':checkpoint['step'],'errors':errors,
                       'halo_inputs_and_targets_bitwise_equal':True})
        print(f'Checked {row["model"]} width={point["width"]} error={errors["256"]:.8g}',flush=True)
    save_json(output/'validation.json',{'checks':checks,'all_44_points_match':True})
    fields=['model','hidden_units','parameters','construction_N','construction_halo',
            'best_trained_validation_step','best_trained_validation_test_rel_l2','final_rel_l2']
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lookup={(r['model'],r['width']):r for r in checks}
    best={name:min((r for r in rows if r['model']==name),key=lambda r:lookup[(name,r['hidden_units'])]['errors']['32'])
          for name in ['NPU','RealNPU']}
    save_json(output/'findings.json',{'best_by_validation':best,'points':len(rows),'new_runs':42,'reused_runs':2})
    lines=['# Identical-width NPU and QUILL scaling','',
           'NPU and RealNPU use all 22 hidden widths from the saved QUILL construction. '
           'At every point their training nodes and C17 target values match that construction\'s '
           'own mathematical halo-center grid exactly. One fixed seed, Adam lr=0.001, batch 100, 20k updates, float64; '
           'no extra LR/seed search.','']
    for name,r in best.items():
        lines.append(f'- Best {name}: {r["best_trained_validation_test_rel_l2"]:.8g} relative L2 '
                     f'({100*r["best_trained_validation_test_rel_l2"]:.5g}%), width {r["hidden_units"]:,}, '
                     f'{r["parameters"]:,} parameters, selected step {r["best_trained_validation_step"]:,}.')
    lines+=['','| Width | QUILL relative L2 | NPU best trained relative L2 | RealNPU best trained relative L2 |',
            '|---:|---:|---:|---:|']
    for point in qi:
        values={name:next(r for r in rows if r['model']==name and r['hidden_units']==point['width']) for name in ['NPU','RealNPU']}
        lines.append(f'| {point["width"]:,} | {point["relative_l2"]:.7g} | '
                     f'{values["NPU"]["best_trained_validation_test_rel_l2"]:.7g} | '
                     f'{values["RealNPU"]["best_trained_validation_test_rel_l2"]:.7g} |')
    lines+=['','## What matches','',
            'The same `Extension(eps=1e-4, order=16)` supplies targets; score on [1e-4,1] with physical '
            'dx relative L2. For QUILL interval count N and halo R=32, each block uses physical nodes '
            'x[j,n]=4*s[j]*n/N, n=-32,...,N+32. NPU/RealNPU train on values at exactly those nodes, '
            'with the same sample-index RNG per model/width. The physical halo changes with N, '
            'exactly as in the original construction sweep. All points use the unchanged C17 function.','',
            'Training minimizes unweighted minibatch MSE over the halo-center indices. '
            'The reported physical-domain error has a different domain and weighting; '
            'training is not directly minimizing that quadrature-weighted evaluation loss. '
            'The results therefore describe this specified halo-fitting protocol, not an '
            'architecture-only approximation lower bound.','',
            'The grid above contains the network centers and R=32 halo centers on each side. '
            'QI also uses Kc=160 extra derivative-stencil padding locations; all those extra '
            'prototype derivative values are exactly zero, outside its compact support. '
            'They are not added as extra constant-tail value observations to the training set. '
            'Bitwise checks compare baseline arrays generated from the shared mathematical center grid, '
            'not distinct floating-point evaluation paths inside the original mpmath convolution.','',
            'For each trained model, select the minimum logged order-32 validation error after at least '
            'one update; logs are every 250 steps. Independent orders 256 and 512 verify the test score. '
            'Every checkpoint, parameter count, selection step and per-width target array was checked.','',
            'The 4,815-unit runs are reused from the verified previous sweep because their grid/target '
            'and full training recipe are identical. Forty-two new runs supply the other points. '
            'The source data include widths 990 and 5,055 to bracket the displayed 1k–5k window.','',
            '## Interpretation','',
            'Width is matched by explicit user choice. NPU and QUILL each have 3h+1 parameters; '
            'RealNPU has 2h+1. They have different operations and information access: QUILL constructs '
            'weights from analytic derivatives and an anchor, whereas NPU/RealNPU fit function values. '
            'This is one seed and a fixed optimization budget, not a guarantee of best possible NPU accuracy. '
            'The dotted Adam baseline uses standard PyTorch or Xavier initialization and '
            'ordinary Adam training at width 128; multiscale initialization is explicitly '
            'excluded. It is not an Adam width sweep on this plot. The fixed N=256/R=32 '
            'halo-center training dataset is retained. See ../adam_standard/results.md '
            'and validation.json.','',
            'Raw data and checkpoints are preserved. The figure is unsmoothed; white centers mark '
            'the actual sampled widths. See `summary.csv`, `validation.json`, `halo_datasets.json`, '
            '`sweep_specification.json`, and `figure_provenance.json`.']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name:{'width':r['hidden_units'],'error':r['best_trained_validation_test_rel_l2']} for name,r in best.items()},indent=2))


if __name__=='__main__':main()
