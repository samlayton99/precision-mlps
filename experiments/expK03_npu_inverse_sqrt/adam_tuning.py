"""10k width/rate pilots, followed by nine validation-selected 100k Adam runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,save_json,halo_nodes
from experiments.expK03_npu_inverse_sqrt.width_sweep import run_one
from experiments.expK02_inverse_sqrt_extension.construction import Extension

WIDTHS=[8,16,32,64,128,256,512,1000]
RATES=[.01,.001,.0001]


def select_widths(rows,count=3):
    rankings=[]
    for width in sorted({r['hidden_units'] for r in rows}):
        candidates=[r for r in rows if r['hidden_units']==width]
        chosen=min(candidates,key=lambda r:(r['best_trained_validation_rel_l2'],r['learning_rate']))
        rankings.append({'width':width,'validation_rel_l2':chosen['best_trained_validation_rel_l2'],
                         'learning_rate':chosen['learning_rate'],
                         'selected_step':chosen['best_trained_validation_step']})
    rankings.sort(key=lambda r:(r['validation_rel_l2'],r['width']))
    return sorted(r['width'] for r in rankings[:count]),rankings


def stage(base,widths,steps,minimum_step,folder,workers):
    folder.mkdir(parents=True,exist_ok=True);jobs=[];rows=[];failures=[]
    for width in widths:
        for lr in RATES:
            config={**base,'width':width,'learning_rate':lr,'steps':steps,
                    'minimum_selection_step':minimum_step,
                    'checkpoint_every':5000 if steps==10000 else 10000,
                    'architecture':f'Linear(1,{width}), Tanh, Linear({width},1); fixed QUILL affine input centering',
                    'selection':f'Minimum order-32 validation error at logged steps {minimum_step} through {steps}; evaluation with independent order-256/512 quadrature.'}
            jobs.append((config,'Adam MLP',folder/f'h{width}_lr{lr:.0e}'))
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,*job):job for job in jobs}
        for future in as_completed(futures):
            config,_,directory=futures[future]
            try:
                row=future.result();rows.append(row)
                print(f'COMPLETE {folder.name}: h={row["hidden_units"]} lr={row["learning_rate"]:g} '
                      f'validation={row["best_trained_validation_rel_l2"]:.8g} '
                      f'step={row["best_trained_validation_step"]}',flush=True)
            except Exception as exc:
                failure={'width':config['width'],'learning_rate':config['learning_rate'],'error':repr(exc)}
                failures.append(failure);save_json(directory/'failure.json',failure)
            save_json(folder/'summary.json',sorted(rows,key=lambda r:(r['hidden_units'],-r['learning_rate'])))
            save_json(folder/'status.json',{'completed':len(rows),'planned':len(jobs),'failures':failures,
                                         'elapsed_seconds':time.monotonic()-start})
    if failures:raise RuntimeError(f'Failed {len(failures)} runs in {folder}; see status.json')
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=OUT/'adam_tuning')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(__file__).with_name('config_adam_1k.json').read_text())
    base['recipe_note']='Fixed C17 target and N=256/R=32 halo-center dataset; one seed; width/rate tuning only. No warmup, LR decay, target rescaling, or QI initialization.'
    sources=[Path(__file__),Path(__file__).with_name('run.py'),Path(__file__).with_name('width_sweep.py'),
             ROOT/'src/models/centered_tanh.py',ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py']
    spec={'pilot_widths':WIDTHS,'learning_rates':RATES,'pilot_steps':10000,'pilot_minimum_selection_step':5000,
          'candidate_count':3,'final_steps':100000,'final_minimum_selection_step':50000,'final_runs':9,
          'width_selection':'Rank each width by its minimum validation error over 5k-10k, across all three pilot learning rates. Select the three best distinct widths; break ties by smaller width.',
          'final_selection':'Minimum validation error across nine long runs over 50k-100k; test quadrature is never used for model selection.',
          'base_config':base,'workers':args.workers,
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          'torch':torch.__version__,'numpy':np.__version__}
    path=output/'specification.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:save_json(path,spec)
    e=Extension();x=halo_nodes(e,base['halo_reference_N'],base['halo_reference_R']);y=e.value(x)
    previous=np.load(OUT/'adam_mlp_1k/matched_halo_data.npz')
    assert np.array_equal(x,previous['x']) and np.array_equal(y,previous['y'])
    np.savez(output/'matched_halo_data.npz',x=x,y=y)
    pilots=stage(base,WIDTHS,10000,5000,output/'pilots',args.workers)
    widths,ranking=select_widths(pilots)
    save_json(output/'width_selection.json',{'selected_widths':widths,'ranking':ranking,'selection_metric':'validation only'})
    print(f'SELECTED WIDTHS: {widths}',flush=True)
    finals=stage(base,widths,100000,50000,output/'long_runs',args.workers)
    selected=min(finals,key=lambda r:(r['best_trained_validation_rel_l2'],r['hidden_units'],r['learning_rate']))
    save_json(output/'selected_reference.json',selected)
    print(f'SELECTED REFERENCE: h={selected["hidden_units"]}, lr={selected["learning_rate"]:g}, '
          f'step={selected["best_trained_validation_step"]}, '
          f'test_rel_L2={selected["best_trained_validation_test_rel_l2"]:.9g}',flush=True)


if __name__=='__main__':main()
