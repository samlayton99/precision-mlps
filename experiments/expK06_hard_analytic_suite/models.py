"""Conventional random tanh initialization and documented NPU options."""
import math
import torch
from src.models.neural_power_unit import NeuralPowerUnit

class Tanh(torch.nn.Module):
    def __init__(self,width,variant,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden=torch.nn.Linear(1,width,dtype=torch.float64)
            self.head=torch.nn.Linear(width,1,dtype=torch.float64)
            if variant=='xavier':
                torch.nn.init.xavier_uniform_(self.hidden.weight,gain=5/3)
                torch.nn.init.zeros_(self.hidden.bias)
                torch.nn.init.xavier_uniform_(self.head.weight)
                torch.nn.init.zeros_(self.head.bias)
    def features(self,x):return self.hidden(x).tanh()
    def forward(self,x):return self.head(self.features(x))

class Power(torch.nn.Module):
    def __init__(self,width,variant,seed):
        super().__init__();self.variant=variant
        g=torch.Generator().manual_seed(seed)
        self.power=NeuralPowerUnit(2 if variant=='composed' else 1,width,generator=g)
        if variant=='composed':
            self.mix=torch.nn.Parameter(torch.empty(2,2,dtype=torch.float64))
            torch.nn.init.uniform_(self.mix,-math.sqrt(6/4),math.sqrt(6/4),generator=g)
        if variant=='positive':
            # Zero imaginary weights on positive inputs have zero gradient forever.
            # Random complex weights give NPU its full oscillatory branch.
            with torch.no_grad():
                self.power.gate.fill_(1.)
                self.power.gate.requires_grad_(False)
                self.power.imag.uniform_(-math.sqrt(6/(width+1)),math.sqrt(6/(width+1)),generator=g)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+991)
            self.head=torch.nn.Linear(width,1,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.uniform_(-math.sqrt(6/(width+1)),math.sqrt(6/(width+1)),generator=g)
            self.head.bias.zero_()
    def features(self,x):
        if self.variant=='positive':x=1.5+.5*x
        elif self.variant=='composed':x=torch.cat((x,torch.ones_like(x)),1)@self.mix.T
        return self.power(x)
    def forward(self,x):return self.head(self.features(x))

def build(method,width,variant,seed):
    return (Tanh if method=='Adam' else Power)(width,variant,seed)
