import math
import torch
from experiments.expK03_npu_inverse_sqrt.adam_standard import build_model,configs,rate_at
from src.models.centered_tanh import CenteredTanhMLP


def test_default_is_unmodified_pytorch_initialization():
    c=configs()[0];a=build_model(c);b=CenteredTanhMLP(128)
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)


def test_xavier_matches_standard_library_and_is_domain_independent():
    c=next(c for c in configs() if c['initializer']=='xavier_uniform_tanh')
    model=build_model(c);other=build_model({**c,'epsilon':.01})
    g=torch.Generator().manual_seed(0)
    hidden=torch.empty_like(model.net[0].weight);head=torch.empty_like(model.net[2].weight)
    torch.nn.init.xavier_uniform_(hidden,gain=5/3,generator=g)
    torch.nn.init.xavier_uniform_(head,gain=1.,generator=g)
    torch.testing.assert_close(model.net[0].weight,hidden,rtol=0,atol=0)
    torch.testing.assert_close(model.net[2].weight,head,rtol=0,atol=0)
    assert torch.count_nonzero(model.net[0].bias)==0 and torch.count_nonzero(model.net[2].bias)==0
    for key,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
    assert model.net[0].weight.detach().abs().max()<=math.sqrt(6/129)*5/3


def test_all_trials_have_only_standard_initialization_and_regular_schedules():
    trials=configs();assert len(trials)==12
    for c in trials:
        m=build_model(c)
        assert sum(p.numel() for p in m.parameters())==385
        assert c['output_scale']==1 and c['width']==128
        assert m.net[0].weight.detach().abs().max()<=1
        assert rate_at(1,c)==c['learning_rate']
        assert rate_at(100000,c)==c['final_learning_rate']
