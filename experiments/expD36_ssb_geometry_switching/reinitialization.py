"""One matched parameter-reset event, separate from metric interventions."""
import numpy as np
from experiments.expD35_optimization_exploration import core


def replace(z, config):
    g=core.old.geometry(config['n'])
    c,gamma=map(np.array,core.physical(z,g,config['coordinates']))
    mask=np.asarray(config['reset_mask'],dtype=int)
    rng=np.random.default_rng(88001+config['seed'])
    slope=np.abs(rng.normal(size=g.width))*5/3*np.sqrt(2/(g.width+1))
    weight=rng.normal(size=g.width)*g.alpha[1:]
    mode=config['reinitialization']
    if mode in ('none','state_only'):return z
    if mode not in ('zero_physical','scaled_physical','scaled_bandwidth'):raise ValueError(mode)
    c[1+mask]=0. if mode=='zero_physical' else weight[mask]
    gamma[mask]=slope[mask]/(g.h if mode=='scaled_bandwidth' else 1.)
    return core.encode(c,gamma,g,config['coordinates'])
