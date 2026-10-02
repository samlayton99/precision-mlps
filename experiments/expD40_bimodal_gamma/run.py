"""Matched D40 runs through the unchanged D38 training engine."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from experiments.expD38_init_readout_baseline import run as base
from experiments.expD40_bimodal_gamma.initialization import make_model

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/checkpoint_D_optimizers/expD40_bimodal_gamma'
BASE_OUT = ROOT/'results/checkpoint_D_optimizers/expD38_init_readout_baseline'
D39_OUT = ROOT/'results/checkpoint_D_optimizers/expD39_qi_init_theory'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def config():
    cfg = base.config()
    cfg['d40_design'] = design()
    sources = [HERE/'initialization.py', HERE/'run.py', HERE/'config.yaml',
        ROOT/'experiments/expD39_qi_init_theory/initialization.py',
        ROOT/'experiments/expD39_qi_init_theory/run.py',
        ROOT/'experiments/expD38_init_readout_baseline/run.py',
        ROOT/'experiments/expF04_qi_init_real_data/model.py']
    cfg['d40_source_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['screen', 'confirm', 'smoke'])
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    cfg, spec = config(), design()
    recipes = json.loads((BASE_OUT/'selected_recipe.json').read_text())
    if args.phase == 'confirm':
        selection = json.loads((OUT/'selection.json').read_text())
        assert selection['source_sha256'] == cfg['d40_source_sha256']
        for path, expected in selection['screen_sha256'].items():
            assert digest(ROOT/path) == expected
        pairs = selection['confirmation_pairs']
        seeds, steps, phase = spec['confirmation_seeds'], spec['confirmation_steps'], 'compare'
    else:
        arms = [f'{shape}_{scope}' for scope in spec['scopes'] for shape in spec['shapes']]
        pairs = [(task, arm) for task in spec['tasks'] for arm in arms]
        seeds, steps, phase = [spec['screen_seed']], spec['screen_steps'], 'pilot'
        if args.phase == 'smoke':
            pairs, steps = [('airfoil', 'mix_both'), ('airfoil', 'mix_last')], 30
    base.make_model = make_model
    base.OUT = OUT if args.phase != 'smoke' else OUT/'smoke'
    jobs = [(t, a, recipes[t]['lr'], seed, steps, phase, cfg, spec['width'])
            for t, a in pairs for seed in seeds]
    for i, job in enumerate(jobs):
        if i % args.shards == args.shard:
            base.run_one(job)


if __name__ == '__main__':
    main()
