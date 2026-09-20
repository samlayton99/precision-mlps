"""Fill only missing cosine trajectories for the matched J/J* comparison."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD33_current_readout_split import run as current
from experiments.expD31_split_adam import long_run as prior_long

OUTPUT=current.RESULTS/'j_vs_jstar/data/trajectories'
MUS=[100,250,500,1000]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',required=True);args=parser.parse_args()
    cfg=prior_long.config('cosine')|dict(mu_values=MUS)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg['threads'])
    with threadpool_limits(cfg['threads']):
        for mu in MUS:
            for method in ('Jstar','J'):
                reused=prior_long.case_path(args.target,mu,'cosine')
                if method=='Jstar' and reused.exists():
                    print(f'REUSE {method}/{args.target}/mu={mu}',flush=True);continue
                path=OUTPUT/f'{method}__{args.target}__{mu}.npz'
                if path.exists():continue
                start=time.perf_counter()
                c=(current.train(args.target,mu,cfg) if method=='J' else
                   current.prior.train(args.target,'xavier',mu,cfg))
                c['seconds']=np.array(time.perf_counter()-start)
                current.prior.gd.save(c,cfg,path)
                print(f'COMPLETE {method}/{args.target}/mu={mu}: {c["status"]}, step={c["step"][-1]}',flush=True)


if __name__=='__main__':main()
