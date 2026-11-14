"""Re-rank existing NPU pilots and validate full runs with resolved integrals.

Training samples, gradients, optimizer states and update rules are unchanged.
The refinement repairs quadrature under-resolution observed in a pilot at x=0.
"""
import copy,json,hashlib,math
from pathlib import Path
import torch
from experiments.expK06_hard_analytic_suite import train
from experiments.expK06_hard_analytic_suite.common import ROOT,OUT,save
from experiments.expK06_hard_analytic_suite.models import build
from experiments.expK06_hard_analytic_suite.integration import accurate_score


def refine_pilot(r):
    if r['method']!='NPU':return r
    torch.set_num_threads(1)
    folder=OUT/'refined_pilots'/train.identifier(r);path=folder/'summary.json'
    if path.exists():return json.loads(path.read_text())
    original=ROOT/r['run_directory']/'last.pt';state=torch.load(original,weights_only=False)
    m=build(r['method'],r['width'],r['variant'],r['seed'])
    m.load_state_dict(state['best_state'])
    best=accurate_score(r['target'],m,state['amplitude'],state['offset'])
    m.load_state_dict(state['model'])
    final=accurate_score(r['target'],m,state['amplitude'],state['offset'])
    if math.isfinite(final) and final<best:
        best=final;state['best_state']=copy.deepcopy(state['model']);state['best_step']=state['step']
    state['best']=best
    folder.mkdir(parents=True,exist_ok=True);torch.save(state,folder/'last.pt')
    refined={**r,'original_validation':r['validation'],'validation':best,
             'best_adam_step':state['best_step'],'original_run_directory':r['run_directory'],
             'run_directory':str(folder.relative_to(ROOT)),
             'original_checkpoint_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
    save(path,refined);return refined


def full_run(c,steps,stage='full',resume=None,polish=True):
    original=train.score
    def resolved_score(m,x,y,w,amp,offset):
        if hasattr(m,'power'):return accurate_score(c['target'],m,amp,offset,order=len(x)//64)
        return original(m,x,y,w,amp,offset)
    train.score=resolved_score
    try:return train.run_one(c,steps,stage,resume,polish)
    finally:train.score=original
