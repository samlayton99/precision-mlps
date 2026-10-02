"""Replay the unchanged D28 PyTorch trainer for a full 20k baseline check."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.expD28_loss_gradient_decomposition import run as previous
from .analyze import load_snapshots
from .run import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    cfg=previous.config()|dict(resolution=128,halo=24,seed=0,n_train=1024,
        steps=20000,diagnostic_snapshots=20001,learning_rate=.002)
    steps,snap=load_snapshots(args.bundle/'p0',20000)
    result=[]
    with np.load(args.bundle/'p0/state_000020000.npz') as endpoint:
        for i,target in enumerate(('sine','runge')):
            old=previous.train(target,'xavier',cfg)
            z=np.stack((old['a'],old['b'],old['v'][:,:-1]),axis=1)
            errors=[np.max(abs(z[steps]-snap['z'][i])),np.max(abs(old['v'][steps,-1]-snap['d'][i])),
                np.max(abs(z[-1]-endpoint['z'][i])),abs(old['v'][-1,-1]-endpoint['d'][i])]
            np.testing.assert_allclose(z[steps],snap['z'][i],atol=2e-11,rtol=2e-10)
            np.testing.assert_allclose(z[-1],endpoint['z'][i],atol=2e-11,rtol=2e-10)
            np.testing.assert_allclose(old['v'][-1,-1],endpoint['d'][i],atol=2e-11,rtol=2e-10)
            result.append(dict(target=target,updates=20000,checked_states=len(steps)+1,
                maximum_parameter_absolute_error=float(max(errors)),passed=True))
            print(json.dumps(result[-1]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(check='unchanged D28 PyTorch versus batched JAX',results=result))


if __name__=='__main__':main()
