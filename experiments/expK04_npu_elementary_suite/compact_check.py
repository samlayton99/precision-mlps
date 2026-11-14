"""Small generic composition control: two powers, learned sum, outer power."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.models.neural_power_unit import NeuralPowerUnit
from experiments.expK04_npu_elementary_suite.run import OUT,ACTIVE_TARGETS,run_one,identifier,save


class CompactStack(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed)
        self.first=NeuralPowerUnit(1,2,complex_weights=False,generator=g)
        self.mix=torch.nn.Parameter(torch.empty(1,2,dtype=torch.float64))
        torch.nn.init.uniform_(self.mix,-math.sqrt(2),math.sqrt(2),generator=g)
        self.last=NeuralPowerUnit(1,1,complex_weights=False,generator=g)
        with torch.no_grad():self.first.gate.fill_(1);self.last.gate.fill_(1)
        self.first.gate.requires_grad_(False);self.last.gate.requires_grad_(False)

    def forward(self,x):return self.last(self.first(x)@self.mix.T)


def factory(name,seed):
    assert name=='compact_stack'
    return CompactStack(seed)


def main():
    output=OUT/'compact_check';output.mkdir(parents=True,exist_ok=True)
    base={'seed':0,'sampling':'log_importance','adam_steps':20000,'batch_size':100,
          'final_learning_rate':1e-6,'log_every':100,'lbfgs_outer_steps':100,'model':'compact_stack'}
    jobs=[{**base,'target':name,'learning_rate':lr} for name in ACTIVE_TARGETS for lr in [.001,.01]]
    save(output/'specification.json',{'jobs':jobs,'purpose':'Follow-up to check a small composition-friendly architecture before interpreting failures.',
         'architecture':'(a*x^p+b*x^q)^r with five learned parameters, standard random initialization, gates fixed at 1.',
         'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('run.py')]}})
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(run_one,c,output/identifier(c),factory):c for c in jobs}
        for future in as_completed(pending):
            row=future.result();rows.append(row);save(output/'summary.json',rows)
            print(row['target'],row['learning_rate'],'Adam',row['adam_best_test'],'polished',row['test_relative_l2'],flush=True)


if __name__=='__main__':main()
