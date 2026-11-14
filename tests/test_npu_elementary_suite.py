import math
import numpy as np
import torch
from experiments.expK04_npu_elementary_suite.problem import A,B,grid,score,target
from experiments.expK04_npu_elementary_suite.models import build,MODELS
from experiments.expK04_npu_elementary_suite.run import ACTIVE_TARGETS,run_one


def test_general_physical_l2_and_importance_weight_identity():
    for name in ACTIVE_TARGETS:
        data=grid(name,64);x,y,dx=data
        class RelativePerturbation(torch.nn.Module):
            def forward(self,x):return 1.01*target(name,x)
        np.testing.assert_allclose(score(RelativePerturbation(),data),.01,atol=2e-16)
        norm=(dx*y.square()).sum()
        log_probability_weights=dx/(x*math.log(B/A))
        importance_weights=x*math.log(B/A)/norm
        error=.01*y
        np.testing.assert_allclose((log_probability_weights*importance_weights*error.square()).sum(),1e-4,rtol=1e-13)
        assert torch.isfinite(y).all()


def test_native_power_representation():
    model=build('fixed_gate_zero',0)
    with torch.no_grad():model.real.fill_(-.5)
    assert score(model,grid('inv_sqrt',256))<1e-12


def test_stacked_architecture_can_represent_algebraic_composition():
    model=build('stacked_8',0)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.first.gate.fill_(1)
        model.first.real[0,0]=.5;model.first.real[1,0]=1
        model.mix[0,0]=1;model.mix[0,1]=1
        model.second.gate[0]=1;model.second.real[0,0]=-1
        model.readout[0,0]=1
    assert score(model,grid('root_plus_linear',256))<1e-11


def test_initial_models_finite_and_trainable_without_target_specific_initialization():
    x=torch.logspace(-4,0,100,dtype=torch.float64)[:,None]
    for name in MODELS:
        m=build(name,0);y=m(x)
        assert y.shape==x.shape and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


def test_small_training_smoke(tmp_path):
    c={'target':'inv_sqrt','model':'fixed_gate_zero','seed':0,'learning_rate':.001,
       'sampling':'log_importance','adam_steps':20,'batch_size':100,'final_learning_rate':1e-6,
       'log_every':5,'lbfgs_outer_steps':2}
    # Runner stores repo-relative paths; use a repo-local temporary output.
    from pathlib import Path
    import tempfile
    from experiments.expK04_npu_elementary_suite.run import OUT
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smoke_',dir=OUT) as directory:
        row=run_one(c,Path(directory))
        assert np.isfinite(row['test_relative_l2'])
        assert row['test_relative_l2']<=row['adam_best_test']+1e-12


def test_compact_stack_expresses_composition_without_oracle_training():
    from experiments.expK04_npu_elementary_suite.compact_check import CompactStack
    model=CompactStack(0)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==5
    with torch.no_grad():
        model.first.real[:,0]=torch.tensor([.5,1.],dtype=torch.float64)
        model.mix.fill_(1.);model.last.real.fill_(-1.)
    assert score(model,grid('root_plus_linear',256))<1e-11
