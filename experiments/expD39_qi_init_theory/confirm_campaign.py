"""Execute only the candidates locked before held-out confirmation."""
import argparse
import json
import hashlib
from experiments.expD39_qi_init_theory import run as parent
from experiments.expD39_qi_init_theory.followup import EXTRA, config as followup_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    selection=json.loads((parent.OUT/'selection.json').read_text())
    for name,digest in selection['provenance'].items():
        if name.startswith('experiments/') and not name.endswith('/analyze.py'):
            assert hashlib.sha256((parent.ROOT/name).read_bytes()).hexdigest()==digest,name
    parent.VARIANTS.update(EXTRA)
    parent.base.make_model=parent.make_model
    parent.base.OUT=parent.OUT
    recipes=json.loads((parent.BASE_OUT/'selected_recipe.json').read_text())
    jobs=[]
    for task,arm in selection['confirm_pairs']:
        for seed in [0,1,2]:
            cfg=followup_config() if arm in EXTRA else parent.config()
            jobs.append((task,arm,recipes[task]['lr'],seed,20000,'compare',cfg,512))
    for i,job in enumerate(jobs):
        if i%a.shards==a.shard:
            parent.base.run_one(job)


if __name__=='__main__':
    main()
