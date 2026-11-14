"""Single-seed NPU/RealNPU capacity sweep on the fixed shared C17 halo."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.expK03_npu_inverse_sqrt.run import OUT, train, save_json, halo_nodes
from experiments.expK02_inverse_sqrt_extension.construction import Extension

WIDTHS={'NPU':[6,32,128,512,2048,4815],
        'RealNPU':[6,32,128,512,2048,4815,7222]}


def run_one(config, name, destination):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    existing=destination/f'{name}_matched_halo_summary.json'
    if existing.exists():
        assert json.loads((destination/'config.json').read_text())==config
        row=json.loads(existing.read_text())
    else:
        save_json(destination/'config.json',config)
        row=train(config,name,'matched_halo',destination)
    return {**row,'run_directory':str(destination.relative_to(ROOT)), 'reused_six_unit_run':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=OUT/'width_sweep')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    base=json.loads((OUT/'config.json').read_text())
    base['training_modes']=['matched_halo']
    base['selection']='Best positive-step order-32 validation checkpoint, scored with order-256/512 quadrature; final iterate also saved.'
    extension=Extension(eps=base['epsilon'],order=base['extension_order'])
    nodes=halo_nodes(extension,base['halo_reference_N'],base['halo_reference_R'])
    values=extension.value(nodes)
    previous=np.load(OUT/'matched_halo_data.npz')
    assert np.array_equal(nodes,previous['x']) and np.array_equal(values,previous['y'])
    np.savez(output/'matched_halo_data.npz',x=nodes,y=values)
    sources=[Path(__file__),Path(__file__).with_name('run.py'),ROOT/'src/models/neural_power_unit.py',
             ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py']
    specification={'widths':WIDTHS,'base_config':base,'workers':args.workers,
                   'parameter_counts':{'NPU':'3*h+1','RealNPU':'2*h+1','QUILL':'3*h+1'},
                   'target_parameter_budget':14446,'one_seed':True,
                   'halo_reference_width':4815,'source_sha256':{
                       str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   'data_sha256':hashlib.sha256(nodes.tobytes()+values.tobytes()).hexdigest(),
                   'torch':torch.__version__,'numpy':np.__version__}
    spec_path=output/'sweep_specification.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text())==specification, 'Use a new output for a changed protocol.'
    else:save_json(spec_path,specification)
    rows=[];jobs=[];failures=[]
    old=json.loads((OUT/'summary.json').read_text())
    for name,widths in WIDTHS.items():
        for width in widths:
            if width==6:
                row=next(r for r in old if r['model']==name and r['training_mode']=='matched_halo')
                assert row['best_validation_step']>0
                rows.append({**row,'best_trained_validation_step':row['best_validation_step'],
                             'best_trained_validation_test_rel_l2':row['best_validation_test_rel_l2'],
                             'run_directory':str(OUT.relative_to(ROOT)),'reused_six_unit_run':True})
                continue
            config={**base,'models':[name],'width':width}
            jobs.append((config,name,output/f'{name}_h{width}'))
    save_json(output/'summary.json',rows)
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_one,*job):job for job in jobs}
        for future in as_completed(pending):
            config,name,destination=pending[future]
            try:
                row=future.result();rows.append(row)
                print(f'COMPLETE {name} width={config["width"]} '
                      f'best={row["best_trained_validation_test_rel_l2"]:.8g} '
                      f'final={row["final_rel_l2"]:.8g}',flush=True)
            except Exception as exc:
                failure={'model':name,'width':config['width'],'error':repr(exc),
                         'directory':str(destination.relative_to(ROOT))}
                failures.append(failure)
                save_json(destination/'failure.json',failure)
                print(f'FAILED {failure}',flush=True)
            save_json(output/'summary.json',sorted(rows,key=lambda r:(r['model'],r['parameters'])))
            save_json(output/'status.json',{'completed':len(rows),'planned':sum(map(len,WIDTHS.values())),
                                          'failures':failures,'seconds':time.monotonic()-start})
    if failures:raise RuntimeError(f'{len(failures)} failed runs; see status.json')
    print(f'All {len(rows)} sweep points saved in {output}',flush=True)


if __name__=='__main__':main()
