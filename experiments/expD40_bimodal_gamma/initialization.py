"""Literal bimodal row norms on a common, paired QI reference geometry."""
from __future__ import annotations
import math
import torch
from experiments.expD39_qi_init_theory.run import make_model as reference_model


def high_mask(banks, width, seed):
    """Alternate along centers, balance odd banks, consume no factory RNG."""
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(width, dtype=torch.bool)
    odd = [i for i, b in enumerate(banks) if b['size'] % 2]
    assert len(odd) % 2 == 0, 'Exact half allocation requires even width'
    ceil_banks = {odd[i] for i in torch.randperm(len(odd), generator=gen)[:len(odd)//2].tolist()}
    for i, bank in enumerate(banks):
        n, start = bank['size'], bank['start']
        phase = (0 if i in ceil_banks else 1) if n % 2 else int(torch.randint(2, (), generator=gen))
        mask[start:start+n] = torch.arange(n) % 2 == phase
    assert int(mask.sum()) == width // 2
    return mask


@torch.no_grad()
def rescale_rows(layer, norms):
    factor = norms.to(layer.weight) / layer.weight.norm(dim=1)
    layer.weight.mul_(factor[:, None])
    layer.bias.mul_(factor)


@torch.no_grad()
def make_model(d, width, seed, scheme, train_x, cfg):
    model, reference = reference_model(d, width, seed, 'centered', train_x, cfg)
    settings = cfg['d40_design']
    assert settings['high_fraction'] == .5, 'This experiment implements an exact half/half mixture'
    low, high = settings['low_multiplier'], settings['high_multiplier']
    if scheme == 'centered':
        scope, shape = 'reference', 'reference'
    else:
        shape, scope = scheme.split('_')
        assert shape in settings['shapes'] and scope in settings['scopes']
    info = {'scheme': scheme, 'reference_geometry': 'D39 centered', 'layers': []}
    x = train_x[:2048]
    for index, (name, key) in enumerate([('fc1', 'first'), ('fc2', 'second')]):
        layer = getattr(model, name)
        norms = layer.weight.norm(dim=1)
        reference_gamma = float(norms.median())
        mask = high_mask(reference[key]['banks'], width, 70000 + 2 * seed + index)
        changed = scope == 'both' or (scope == 'last' and index == 1)
        multiplier = {'low': low, 'mid': 1., 'high': high,
                      'rms': math.sqrt((low**2 + high**2) / 2)}
        if changed:
            target = (torch.where(mask, torch.full_like(norms, high), torch.full_like(norms, low)) if shape == 'mix'
                      else torch.full_like(norms, multiplier[shape])) * reference_gamma
            rescale_rows(layer, target)
        z = layer(x)
        x = torch.tanh(z)
        gamma = layer.weight.norm(dim=1)
        info['layers'].append(dict(name=name, changed=changed, reference_gamma=reference_gamma,
            initial_gamma=gamma.tolist(), high_mask=mask.tolist(), high_count=int(mask.sum()),
            centers=(-layer.bias/gamma).tolist(),
            weight_squared_norm=float(layer.weight.square().sum()),
            bias_squared_norm=float(layer.bias.square().sum()),
            preactivation_rms=float(z.square().mean().sqrt()),
            activation_rms=float(x.square().mean().sqrt()),
            saturation_fraction=float((x.abs() > .99).double().mean()),
            mean_tanh_derivative=float((1-x.square()).mean())))
    info['initial_output_rms'] = float(model.fc3(x).square().mean().sqrt())
    return model, info
