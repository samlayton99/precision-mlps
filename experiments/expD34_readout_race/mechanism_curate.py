"""Losslessly package scalar evidence and selected full D34 diagnostic states."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate(root,archives):
    output=root/'curated'; output.mkdir(exist_ok=True)
    audit=[]
    for path in sorted(root.glob('*_metrics/metrics.csv'))+sorted(root.glob('*_metrics/movement_windows.csv')):
        destination=output/path.parent.name/(path.name+'.gz'); destination.parent.mkdir(exist_ok=True)
        raw=path.read_bytes()
        with gzip.GzipFile(filename=str(destination),mode='wb',mtime=0) as f: f.write(raw)
        assert gzip.decompress(destination.read_bytes())==raw
        audit.append(dict(source=str(path),artifact=str(destination.relative_to(root)),sha256=digest(destination),
                          uncompressed_sha256=digest(path),lossless=True))
    for path in sorted(root.glob('*_metrics/modal_forces.npz')):
        f=dict(np.load(path)); destination=output/path.parent.name/path.name
        # Keep all sampled times for degrees 2..9; full modal tables remain reproducible.
        np.savez_compressed(destination,row=f['row'],mode_labels=np.arange(2,10),
            **{key:f[key][:,:8] for key in ('norm','along_effective','outward')})
        audit.append(dict(source=str(path),artifact=str(destination.relative_to(root)),sha256=digest(destination),mode_labels=list(range(2,10))))
    for folder in sorted(root.glob('*_geometry')):
        destination=output/folder.name; destination.mkdir(exist_ok=True)
        for path in folder.glob('*'):
            if path.is_file():
                shutil.copy2(path,destination/path.name)
                assert digest(path)==digest(destination/path.name)
                audit.append(dict(source=str(path),artifact=str((destination/path.name).relative_to(root)),sha256=digest(path),lossless=True))
    for path in archives:
        f=dict(np.load(path)); steps=f['steps']; name=path.parent.name
        selected={0,len(steps)-1}
        selected.update(int(i) for i,s in enumerate(steps) if s in (0,2000,20000,100000,600000,int(steps[0])+1))
        metrics=root/(name+'_metrics')/'metrics.csv'
        if name=='signal_recovery': metrics=root/'primary_metrics/metrics.csv'
        if name=='missing': metrics=root/'missing_metrics/metrics.csv'
        if metrics.exists():
            with metrics.open() as stream: rows=list(csv.DictReader(stream))
            cases=json.loads(str(f['cases']))
            for case in cases:
                if isinstance(case,list): case=dict(seed=case[0],target=case[1])
                rr=[r for r in rows if int(r['seed'])==case['seed'] and r['target']==case['target']
                    and r['arm']==case.get('arm','joint') and int(r['step'])>=20000]
                if rr:
                    peak=max(rr,key=lambda r:float(r['full_slope_norm']))
                    selected.add(int(np.flatnonzero(steps==int(peak['step']))[0]))
        chosen=np.array(sorted(selected)); count=len(json.loads(str(f['cases'])))
        packed={k:(v[:,chosen] if v.ndim>=2 and v.shape[:2]==(count,len(steps)) else v) for k,v in f.items()}
        packed['steps']=steps[chosen]
        destination=output/'states'/name; destination.mkdir(parents=True,exist_ok=True)
        archive=destination/'compact_states.npz'; np.savez_compressed(archive,**packed)
        with np.load(archive) as saved:
            for k,v in packed.items(): np.testing.assert_array_equal(saved[k],v)
        for extra in ('manifest.json','status.json','replay_verification.json'):
            if (path.parent/extra).exists(): shutil.copy2(path.parent/extra,destination/extra)
        for environment in path.parent.glob('environment_*.json'):
            shutil.copy2(environment,destination/environment.name)
        audit.append(dict(source=str(path),source_sha256=digest(path),artifact=str(archive.relative_to(root)),sha256=digest(archive),
            original_states=len(steps),retained_states=len(chosen),selected_steps=steps[chosen].tolist(),
            selection='fixed milestones, first step, endpoint, and each case sampled maximum raw slope force at or after 20k'))
    (output/'artifact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--archives',type=Path,nargs='+',required=True)
    args=parser.parse_args(); curate(args.root,args.archives)


if __name__=='__main__': main()
