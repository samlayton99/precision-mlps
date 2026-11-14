"""Torch port of Heim et al. (2020), GatedNPUX/GatedNPU + NAU.

Reference: nmheim/NeuralPowerUnits, submission/src/{npu,nau}.jl.
The reference NAU is an unconstrained linear map without a bias. Gates are
clipped during forward evaluation, not projected after optimizer steps.
"""
import math
import torch
from torch import nn


class NeuralPowerUnit(nn.Module):
    def __init__(self, inputs, outputs, *, complex_weights=True, dtype=torch.float64,
                 generator=None):
        super().__init__()
        self.real = nn.Parameter(torch.empty(outputs, inputs, dtype=dtype))
        nn.init.uniform_(self.real, -math.sqrt(6 / (inputs + outputs)),
                         math.sqrt(6 / (inputs + outputs)), generator=generator)
        self.imag = nn.Parameter(torch.zeros_like(self.real)) if complex_weights else None
        self.gate = nn.Parameter(torch.full((inputs,), .5, dtype=dtype))

    def forward(self, x):
        gate = self.gate.clamp(0, 1)
        radius = gate * (x.abs() + torch.finfo(x.dtype).eps) + (1 - gate)
        angle = (-x.sign()).clamp_min(0) * math.pi * gate
        logr = radius.log()
        magnitude = logr @ self.real.T
        phase = angle @ self.real.T
        if self.imag is not None:
            magnitude = magnitude - angle @ self.imag.T
            phase = phase + logr @ self.imag.T
        return magnitude.exp() * phase.cos()


class NPURegressor(nn.Module):
    """Paper's NPU(1, h) -> NAU(h, 1), adapted to the scalar target."""
    def __init__(self, width=6, *, complex_weights=True, seed=0, dtype=torch.float64):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.power = NeuralPowerUnit(1, width, complex_weights=complex_weights,
                                     dtype=dtype, generator=generator)
        self.readout = nn.Parameter(torch.empty(1, width, dtype=dtype))
        nn.init.uniform_(self.readout, -math.sqrt(6 / (width + 1)),
                         math.sqrt(6 / (width + 1)), generator=generator)

    def forward(self, x):
        return self.power(x) @ self.readout.T
