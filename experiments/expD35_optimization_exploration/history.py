"""Losslessly consolidate this campaign's immutable history files after quota errors."""
import argparse
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile


def arrays(folder,pattern):
    """Read current or archived NPZ evidence; current files take precedence."""
    import numpy as np
    locations={}
    for path in sorted(folder.glob('history_*.zip')):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if fnmatch.fnmatch(name,pattern):locations[name]=path
    for path in folder.glob(pattern):locations[path.name]=None
    for name,path in sorted(locations.items()):
        if path is None:raw=(folder/name).read_bytes()
        else:
            with zipfile.ZipFile(path) as archive:raw=archive.read(name)
        with np.load(io.BytesIO(raw)) as data:yield name,{k:data[k] for k in data.files}


def evaluations(folder):
    index=folder/'evaluation_history.json'
    rows={r['step']:r for r in json.loads(index.read_text())} if index.exists() else {}
    for path in folder.glob('evaluation_*.json'):
        if path.name=='evaluation_history.json':continue
        row=json.loads(path.read_text());rows[row['step']]=row
    return [rows[s] for s in sorted(rows)]


def consolidate(folder):
    paths=[]
    for pattern in ('trace_*.npz','ssb_trace_*.npz','snapshot_*.npz','evaluation_*.json','agreement_*.npz'):
        paths.extend(p for p in folder.glob(pattern) if p.name!='evaluation_history.json')
    if not paths:return 0
    latest_snapshot=max(folder.glob('snapshot_*.npz'),default=None)
    if paths==[latest_snapshot] and (folder/'evaluation_history.json').exists():return 0
    index=evaluations(folder)
    previous=[p.name for p in folder.glob('history_*.zip')]
    offloaded=folder/'offloaded_archives.json'
    if offloaded.exists():previous.extend(json.loads(offloaded.read_text()))
    sequence=1+max((int(Path(name).stem.split('_')[1]) for name in previous),default=-1)
    archive=folder/f'history_{sequence:04d}.zip';temporary=archive.with_suffix('.tmpzip')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as stream:
        for path in paths:stream.write(path,path.name)
        stream.writestr('sha256.json',json.dumps(hashes,sort_keys=True))
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    with zipfile.ZipFile(temporary) as stream:
        for name,digest in hashes.items():
            if hashlib.sha256(stream.read(name)).hexdigest()!=digest:raise OSError(f'Archive verification failed: {name}')
    temporary.replace(archive)
    index_tmp=folder/'evaluation_history.tmp'
    with index_tmp.open('w') as stream:
        json.dump(index,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    index_tmp.replace(folder/'evaluation_history.json')
    removed=0
    for path in paths:
        if path==latest_snapshot:continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[path.name]:raise OSError('History changed during consolidation')
        path.unlink();removed+=1
    return removed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();removed=0
    for i,folder in enumerate(sorted(args.root.iterdir())):
        if folder.is_dir() and (folder/'case.json').exists():removed+=consolidate(folder)
        if i%50==0:print(json.dumps(dict(folders=i,files_consolidated=removed)),flush=True)
    print(json.dumps(dict(files_consolidated=removed)),flush=True)


if __name__=='__main__':main()
