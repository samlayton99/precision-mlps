"""Reload and independently score every capacity-sweep checkpoint."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,make_model,evaluation_grid,relative_l2,save_json
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    torch.set_num_threads(1)
    output=OUT/'width_sweep'
    rows=json.loads((output/'summary.json').read_text())
    specification=json.loads((output/'sweep_specification.json').read_text())
    assert len(rows)==sum(map(len,specification['widths'].values()))
    e=Extension();grids={n:evaluation_grid(e,n) for n in [32,256,512]}
    checks=[]
    for row in rows:
        folder=ROOT/row['run_directory'];stem=f'{row["model"]}_matched_halo'
        reused=row['reused_six_unit_run']
        suffix='best_validation' if reused else 'best_trained_validation'
        checkpoint=torch.load(folder/f'{stem}_{suffix}.pt',weights_only=False)
        c=checkpoint['config'];model=make_model(c,row['model']);model.load_state_dict(checkpoint['model'])
        expected=(3 if row['model']=='NPU' else 2)*row['hidden_units']+1
        assert sum(p.numel() for p in model.parameters())==expected==row['parameters']
        assert c['seed']==0 and c['data_seed']==10000 and c['learning_rate']==.001 and c['steps']==20000
        assert c['halo_reference_N']==256 and c['halo_reference_R']==32
        errors={str(n):relative_l2(model,grid,e.eps) for n,grid in grids.items()}
        np.testing.assert_allclose(errors['256'],row['best_trained_validation_test_rel_l2'],rtol=2e-13,atol=1e-15)
        np.testing.assert_allclose(errors['256'],errors['512'],rtol=1e-10,atol=1e-14)
        history=json.loads((folder/f'{stem}_history.json').read_text())
        best=min((r for r in history if r['step']>0),key=lambda r:r['validation_rel_l2'])
        assert checkpoint['step']==best['step']==row['best_trained_validation_step']
        checks.append({'model':row['model'],'width':row['hidden_units'],'parameters':expected,
                       'selected_step':checkpoint['step'],'errors':errors,'reload_verified':True,
                       'selected_gate_raw':model.power.gate.detach().tolist(),
                       'selected_max_abs_real_weight':float(model.power.real.detach().abs().max()),
                       'selected_max_abs_imag_weight':float(model.power.imag.detach().abs().max())
                           if model.power.imag is not None else None})
    source=np.load(OUT/'matched_halo_data.npz');sweep=np.load(output/'matched_halo_data.npz')
    assert np.array_equal(source['x'],sweep['x']) and np.array_equal(source['y'],sweep['y'])
    save_json(output/'validation.json',{'same_target_nodes_bitwise':True,'checks':checks})
    print(f'Validated all {len(checks)} checkpoints, parameter counts, selection steps and quadrature.')


if __name__=='__main__':main()
