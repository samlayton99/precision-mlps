import pytest
import torch
from experiments.expD39_qi_init_theory.initialization import initialize_layer, VARIANTS
from experiments.expD39_qi_init_theory.run import make_model, config
from experiments.expF04_qi_init_real_data.model import qi_ridge_init_layer_


def sample():
    return torch.randn(400, 5, generator=torch.Generator().manual_seed(19), dtype=torch.float64)


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_actual_spacing_and_bank_directions(variant):
    x = sample()
    layer = torch.nn.Linear(5, 512).double()
    report = initialize_layer(layer, x, generator=torch.Generator().manual_seed(11), **VARIANTS[variant])
    assert sum(b['size'] for b in report['banks']) == 512
    for b in report['banks']:
        sl = slice(b['start'], b['start'] + b['size'])
        w, bias = layer.weight[sl].detach(), layer.bias[sl].detach()
        gamma = w.norm(dim=1)
        c = -bias / gamma
        assert torch.allclose(w, w[:1].expand_as(w), atol=1e-14)
        assert torch.allclose(gamma[:-1] * c.diff(), torch.full_like(c.diff(), report['lam']), atol=1e-12)
    if variant == 'common':
        assert layer.weight.norm(dim=1).std().item() < 1e-14
    if variant != 'spacing':
        assert max(b['size'] for b in report['banks']) - min(b['size'] for b in report['banks']) <= 1


def test_spacing_control_changes_only_gamma_for_fixed_input():
    x = sample()
    old, new = [torch.nn.Linear(5, 512).double() for _ in range(2)]
    qi_ridge_init_layer_(old, x, centers_per_dir=22, generator=torch.Generator().manual_seed(11))
    info = initialize_layer(new, x, balanced=False, generator=torch.Generator().manual_seed(11))
    for b in info['banks']:
        sl = slice(b['start'], b['start'] + b['size'])
        ratio = (b['size'] - 1) / b['size']
        assert torch.allclose(new.weight[sl], old.weight[sl] * ratio, atol=1e-14)
        assert torch.allclose(new.bias[sl], old.bias[sl] * ratio, atol=1e-14)


def test_centered_banks_are_translation_equivariant():
    x = sample()
    shift = x.new_tensor([3, -2, 1, 4, -.5])
    a, b = [torch.nn.Linear(5, 512).double() for _ in range(2)]
    initialize_layer(a, x, centered=True, generator=torch.Generator().manual_seed(11))
    initialize_layer(b, x + shift, centered=True, generator=torch.Generator().manual_seed(11))
    assert torch.allclose(a(x), b(x + shift), atol=1e-12)


def test_initializers_preserve_paired_readout_and_single_layer_controls():
    x = sample()
    cfg = config()
    standard, _ = make_model(5, 512, 0, 'standard', x, cfg)
    for variant in [*VARIANTS, 'qi', 'first_only', 'last_only']:
        model, _ = make_model(5, 512, 0, variant, x, cfg)
        assert torch.equal(model.fc3.weight, standard.fc3.weight)
        assert torch.equal(model.fc3.bias, standard.fc3.bias)
        if variant == 'first_only':
            assert torch.equal(model.fc2.weight, standard.fc2.weight)
        if variant == 'last_only':
            assert torch.equal(model.fc1.weight, standard.fc1.weight)


def test_direction_slope_controls_match_their_normalized_scales():
    from experiments.expD39_qi_init_theory.followup import EXTRA
    for arm, kwargs in EXTRA.items():
        layer=torch.nn.Linear(5,512).double()
        info=initialize_layer(layer,sample(),generator=torch.Generator().manual_seed(11),**kwargs)
        slopes={round(b['gamma']*(b['hi']-b['lo'])/2,10) for b in info['banks']}
        assert slopes==({.875,.91875} if arm=='soft24' else {2.5})
