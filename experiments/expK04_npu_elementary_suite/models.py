"""Preset NPU architectures; no target-dependent exponents or feature transforms."""
import math
import torch
from src.models.neural_power_unit import NeuralPowerUnit,NPURegressor

MODELS=['fixed_gate_zero','fixed_gate_random','learned_gate','shallow_6','shallow_32','stacked_8']


class StackedNPU(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed)
        self.first=NeuralPowerUnit(1,8,generator=g)
        self.mix=torch.nn.Parameter(torch.empty(4,8,dtype=torch.float64))
        torch.nn.init.uniform_(self.mix,-math.sqrt(6/12),math.sqrt(6/12),generator=g)
        self.second=NeuralPowerUnit(4,4,generator=g)
        self.readout=torch.nn.Parameter(torch.empty(1,4,dtype=torch.float64))
        torch.nn.init.uniform_(self.readout,-math.sqrt(6/5),math.sqrt(6/5),generator=g)

    def forward(self,x):
        return self.second(self.first(x)@self.mix.T)@self.readout.T


def build(name,seed):
    if name.startswith('shallow_'):return NPURegressor(int(name.split('_')[1]),seed=seed)
    if name=='stacked_8':return StackedNPU(seed)
    if name not in MODELS:raise ValueError(name)
    m=NeuralPowerUnit(1,1,complex_weights=False,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        if name=='fixed_gate_zero':m.real.zero_()
        if name.startswith('fixed_gate'):m.gate.fill_(1.)
    if name.startswith('fixed_gate'):m.gate.requires_grad_(False)
    return m
