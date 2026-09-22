"""Stop dependent stages on partial, nonfinite, or unresolved source bundles."""
import argparse
import json
from pathlib import Path
import numpy as np


def check(root,stage):
    paths=([root/'probes'/f'seed{s}_{t}' for s in range(3) for t in (100000,600000)]
        if stage=='discovery' else [root/'confirm'/f'primary_{s}' for s in range(20,25)])
    for folder in paths:
        status=json.loads((folder/'status.json').read_text())
        if not status['complete'] or np.any(status['failed']) or np.any(status.get('unresolved_steps',status.get('unresolved',0))):
            raise ValueError(f'Incomplete or invalid bundle: {folder}: {status}')
        for key in ('motion_identity','channel_identity','identity_max'):
            if np.max(np.abs(status.get(key,0)))>1e-9:raise ValueError(f'Failed {key}: {folder}')
        if stage=='confirm':
            for step in (100000,600000):
                with np.load(folder/f'forecast_{step}.npz') as forecast:
                    if int(forecast['issued_step'])!=step:raise ValueError(f'Forecast issuance mismatch: {folder}')
    print(json.dumps(dict(stage=stage,bundles=len(paths),passed=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('discovery','confirm'),required=True)
    args=p.parse_args();check(args.root,args.stage)
