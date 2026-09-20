"""Recover campaign storage by losslessly compressing completed affine archives."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import zipfile
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    args=p.parse_args();records=[]
    for path in sorted(args.root.glob('core_N128_s*/p1/*.npz')):
        if '.compressed.' in path.name: continue
        with zipfile.ZipFile(path) as archive:
            if all(v.compress_type==zipfile.ZIP_DEFLATED for v in archive.infolist()): continue
        with np.load(path) as data: arrays={k:data[k] for k in data.files}
        before=path.stat().st_size;temporary=path.with_suffix('.compressed.npz')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        with np.load(temporary) as data:
            assert set(data.files)==set(arrays)
            for key,value in arrays.items():
                restored=data[key]
                assert restored.dtype==value.dtype and restored.shape==value.shape
                assert restored.tobytes()==value.tobytes(),(path,key)
        after=temporary.stat().st_size
        assert after>0
        temporary.replace(path)
        records.append(dict(path=str(path),before=before,after=after,all_array_bytes_equal=True))
        if len(records)%100==0:
            args.ledger.write_text(json.dumps(records))
            print(json.dumps(dict(files=len(records),bytes_recovered=sum(r['before']-r['after'] for r in records))),flush=True)
    args.ledger.write_text(json.dumps(records))
    print(json.dumps(dict(files=len(records),bytes_recovered=sum(r['before']-r['after'] for r in records),complete=True)),flush=True)


if __name__=='__main__':main()
