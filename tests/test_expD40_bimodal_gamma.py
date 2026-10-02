import pytest
import torch
from experiments.expD40_bimodal_gamma.initialization import make_model
from experiments.expD40_bimodal_gamma.run import config


def models(scope):
    x = torch.randn(300, 5, generator=torch.Generator().manual_seed(37), dtype=torch.float64)
    cfg = config()
    return x, {shape:make_model(5, 512, 0, 'centered' if shape=='centered' else f'{shape}_{scope}', x, cfg)
               for shape in ['centered', 'low', 'mid', 'high', 'rms', 'mix']}


@pytest.mark.parametrize('scope', ['both', 'last'])
def test_actual_modes_paired_geometry_and_readout(scope):
    x, mm = models(scope)
    ref = mm['centered'][0]
    for shape, (model, info) in mm.items():
        assert torch.equal(model.fc3.weight, ref.fc3.weight)
        assert torch.equal(model.fc3.bias, ref.fc3.bias)
        for idx, name in enumerate(['fc1', 'fc2']):
            a, b = getattr(model, name), getattr(ref, name)
            ga, gb = a.weight.norm(dim=1), b.weight.norm(dim=1)
            assert torch.allclose(a.weight / ga[:,None], b.weight / gb[:,None], atol=1e-14)
            assert torch.allclose(-a.bias / ga, -b.bias / gb, atol=1e-14)
            if scope=='last' and idx==0:
                assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
            elif shape=='mix':
                g = info['layers'][idx]['reference_gamma']
                high = torch.tensor(info['layers'][idx]['high_mask'])
                assert high.sum()==256
                assert torch.allclose(ga[high], torch.full_like(ga[high], 3*g), atol=1e-13)
                assert torch.allclose(ga[~high], torch.full_like(ga[~high], .1*g), atol=1e-13)
        pred = model(x)
        pred.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('scope', ['both', 'last'])
def test_rms_control_matches_squared_weight_norm_and_reproduction(scope):
    x, mm = models(scope)
    mixture, info = mm['mix']
    rms = mm['rms'][0]
    for name in ['fc1', 'fc2']:
        assert torch.allclose(getattr(mixture,name).weight.square().sum(),
                              getattr(rms,name).weight.square().sum(), rtol=1e-13,atol=1e-13)
    again, again_info = make_model(5,512,0,f'mix_{scope}',x,config())
    assert info == again_info
    assert all(torch.equal(value, again.state_dict()[key]) for key,value in mixture.state_dict().items())
