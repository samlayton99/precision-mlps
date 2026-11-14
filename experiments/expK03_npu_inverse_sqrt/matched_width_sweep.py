"""NPU and RealNPU at every saved QUILL hidden width and its own halo grid."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK03_npu_inverse_sqrt.run import OUT,QI,save_json,halo_nodes
from experiments.expK03_npu_inverse_sqrt.width_sweep import run_one
from experiments.expK02_inverse_sqrt_extension.construction import Extension


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=OUT/'matched_construction_widths')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    source=QI/'scaling_1k_5k.json';construction=json.loads(source.read_text())
    prior=json.loads((OUT/'width_sweep/summary.json').read_text())
    base=json.loads((Path(__file__).with_name('config.json')).read_text())
    base['training_modes']=['matched_halo']
    base['selection']='Best positive-step validation checkpoint; final iterate also saved.'
    base['recipe_note']='At each hidden width, use the exact QUILL N/R/15-block physical sampling locations and unchanged C17 target. One seed, 20k Adam updates, lr=0.001.'
    files=[Path(__file__),Path(__file__).with_name('run.py'),Path(__file__).with_name('width_sweep.py'),
           ROOT/'src/models/neural_power_unit.py',ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py']
    specification={'construction_points':[{'N':r['N'],'width':r['width'],'halo':r['halo']} for r in construction],
                   'models':['NPU','RealNPU'],'base_config':base,'workers':args.workers,
                   'x_axis':'identical hidden-layer widths','per_width_halo_matching':True,
                   'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   'construction_data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                   'torch':torch.__version__,'numpy':np.__version__}
    specpath=output/'sweep_specification.json'
    if specpath.exists():assert json.loads(specpath.read_text())==specification
    else:save_json(specpath,specification)
    extension=Extension();rows=[];jobs=[];datasets=[]
    for point in construction:
        n,width,halo=point['N'],point['width'],point['halo']
        nodes=halo_nodes(extension,n,halo);target=extension.value(nodes)
        assert len(nodes)==width
        data_path=output/'datasets'/f'width{width}.npz';data_path.parent.mkdir(exist_ok=True)
        np.savez(data_path,x=nodes,y=target)
        info={'construction_N':n,'construction_width':width,'construction_halo':halo,
              'halo_range':[float(nodes.min()),float(nodes.max())],
              'halo_data_sha256':hashlib.sha256(nodes.tobytes()+target.tobytes()).hexdigest(),
              'halo_dataset':str(data_path.relative_to(ROOT))}
        datasets.append(info)
        for name in ['NPU','RealNPU']:
            config={**base,'models':[name],'width':width,'halo_reference_N':n,
                    'halo_reference_R':halo,'halo_reference_width':width}
            if width==4815:
                previous=next(r for r in prior if r['model']==name and r['hidden_units']==width)
                rows.append({**previous,**info,'reused_previous_run':True})
            else:
                jobs.append((config,name,output/f'{name}_h{width}',info))
    save_json(output/'halo_datasets.json',datasets)
    save_json(output/'summary.json',rows)
    start=time.monotonic();failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_one,c,name,d):(c,name,d,info) for c,name,d,info in jobs}
        for future in as_completed(pending):
            config,name,directory,info=pending[future]
            try:
                row={**future.result(),**info,'reused_previous_run':False};rows.append(row)
                print(f'COMPLETE {name} width={config["width"]} best={row["best_trained_validation_test_rel_l2"]:.8g}',flush=True)
            except Exception as exc:
                failure={'model':name,'width':config['width'],'error':repr(exc)}
                failures.append(failure);save_json(directory/'failure.json',failure)
                print(f'FAILED {failure}',flush=True)
            save_json(output/'summary.json',sorted(rows,key=lambda r:(r['model'],r['hidden_units'])))
            save_json(output/'status.json',{'completed':len(rows),'planned':2*len(construction),
                       'failures':failures,'seconds':time.monotonic()-start})
    if failures:raise RuntimeError(f'{len(failures)} failed runs; see status.json')
    print(f'All {len(rows)} matched-width points saved in {output}',flush=True)


if __name__=='__main__':main()
