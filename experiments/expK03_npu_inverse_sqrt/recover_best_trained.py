"""Recover a positive-step validation optimum and verify exact final replay."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,make_model,evaluation_grid,relative_l2,save_json
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=OUT/'adam_mlp';stem='Adam MLP_matched_halo'
    config=json.loads((output/'config.json').read_text())
    history=json.loads((output/f'{stem}_history.json').read_text())
    chosen=min((r for r in history if r['step']>0),key=lambda r:r['validation_rel_l2'])
    checkpoint_step=max(s for s in range(config['checkpoint_every'],chosen['step']+1,
                                        config['checkpoint_every']))
    checkpoint=torch.load(output/f'{stem}_step{checkpoint_step}.pt',weights_only=False)
    model=make_model(config,'Adam MLP');model.load_state_dict(checkpoint['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],
                                betas=tuple(config['betas']),eps=config['adam_epsilon'],foreach=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=checkpoint['rng_state']
    data=np.load(output/'matched_halo_data.npz');x,y=data['x'],data['y']
    best_state=None
    if checkpoint_step==chosen['step']:
        best_state={k:v.clone() for k,v in model.state_dict().items()}
    for step in range(checkpoint_step+1,config['steps']+1):
        indices=rng.integers(0,len(x),config['batch_size'])
        inputs=torch.from_numpy(x[indices,None]);targets=torch.from_numpy(y[indices,None])
        optimizer.zero_grad(set_to_none=True)
        loss=(model(inputs)-targets).square().mean();loss.backward();optimizer.step()
        if step==chosen['step']:
            best_state={k:v.clone() for k,v in model.state_dict().items()}
    final=torch.load(output/f'{stem}_step{config["steps"]}.pt',weights_only=False)
    assert all(torch.equal(v,final['model'][k]) for k,v in model.state_dict().items())
    assert best_state is not None
    model.load_state_dict(best_state)
    extension=Extension(eps=config['epsilon'],order=config['extension_order'])
    errors={str(o):relative_l2(model,evaluation_grid(extension,o),extension.eps) for o in [32,256,512]}
    assert abs(errors['32']-chosen['validation_rel_l2'])<1e-14
    torch.save({'model':best_state,'step':chosen['step'],'config':config},
               output/f'{stem}_best_trained_validation.pt')
    row=json.loads((output/f'{stem}_summary.json').read_text())
    row.update(best_trained_validation_step=chosen['step'],
               best_trained_validation_test_rel_l2=errors['256'],
               best_trained_validation_test_quadrature=errors,
               replay_final_parameters_bitwise_identical=True)
    save_json(output/f'{stem}_summary.json',row);save_json(output/'summary.json',[row])
    save_json(output/'recovery_validation.json',{'selected_step':chosen['step'],
              'replayed_from_step':checkpoint_step,'final_parameters_bitwise_identical':True,
              'test_errors':errors,'selection':'best logged validation checkpoint with step > 0'})
    print(json.dumps(row,indent=2))


if __name__=='__main__':main()
