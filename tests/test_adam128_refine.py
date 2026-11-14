import numpy as np
import torch
from experiments.expK03_npu_inverse_sqrt.adam128_refine import ConditionedMLP,configs,rate_at
from src.models.centered_tanh import CenteredTanhMLP


def test_output_conditioning_preserves_initial_physical_function():
    c=configs()[1];m=ConditionedMLP(c);baseline=CenteredTanhMLP(128)
    x=torch.linspace(-.4096,3.6864,500,dtype=torch.float64)[:,None]
    torch.testing.assert_close(m(x),baseline(x),rtol=1e-13,atol=1e-15)


def test_multiscale_model_remains_exportable_128_neuron_mlp():
    x=torch.from_numpy(np.r_[np.geomspace(1e-4,1,500),[-.4096,0.,3.6864]][:,None])
    for c in configs():
        m=ConditionedMLP(c);exported=CenteredTanhMLP(128);exported.load_state_dict(m.folded_state())
        assert sum(p.numel() for p in m.parameters())==385
        torch.testing.assert_close(m(x),exported(x),rtol=1e-12,atol=1e-14)
        assert torch.isfinite(m(x)).all()
    assert ConditionedMLP(configs()[2]).net[0].weight.detach().abs().max()>1e4


def test_decay_schedule_endpoints_and_monotonicity():
    for c in configs():
        assert rate_at(1,c)==c['learning_rate']
        assert rate_at(50000,c)==c['learning_rate']
        assert rate_at(100000,c)==c['final_learning_rate']
        values=[rate_at(t,c) for t in range(50000,100001,500)]
        assert np.all(np.diff(values)<=0)
