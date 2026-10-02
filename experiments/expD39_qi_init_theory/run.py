"""Use D38's unchanged training/evaluation engine with versioned initializers."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expD38_init_readout_baseline import run as base
from experiments.expD39_qi_init_theory.initialization import initialize_layer, VARIANTS

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/checkpoint_D_optimizers/expD39_qi_init_theory'
BASE_OUT = base.OUT
BASE_MAKE = base.make_model


def make_model(d, width, seed, scheme, train_x, cfg):
    if scheme in ('qi', 'standard'):
        return BASE_MAKE(d, width, seed, scheme, train_x, cfg)
    model, _ = BASE_MAKE(d, width, seed, 'standard', train_x, cfg)
    gen = torch.Generator().manual_seed(10000 + seed)
    info = {}
    kwargs = VARIANTS['centered' if scheme in ('first_only', 'last_only') else scheme]
    original_first = {k: v.clone() for k, v in model.fc1.state_dict().items()}
    info['first'] = initialize_layer(model.fc1, train_x, generator=gen, **kwargs)
    if scheme == 'last_only':
        # Consume the same first-bank RNG, then calibrate layer 2 on the actual
        # standard first layer. This preserves second-bank direction pairing.
        model.fc1.load_state_dict(original_first)
        info['first'] = {'scheme': 'standard'}
    with torch.no_grad():
        hidden = model.hidden1(train_x[:4096])
    if scheme != 'first_only':
        info['second'] = initialize_layer(model.fc2, hidden, generator=gen, **kwargs)
    else:
        info['second'] = {'scheme': 'standard'}
    return model, info


def config():
    cfg = base.config()
    cfg['d39_version'] = 1
    cfg['initializer_sha256'] = hashlib.sha256((HERE / 'initialization.py').read_bytes()).hexdigest()
    cfg['training_engine_sha256'] = hashlib.sha256((ROOT / 'experiments/expD38_init_readout_baseline/run.py').read_bytes()).hexdigest()
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['screen', 'confirm'])
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--variants', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int)
    args = p.parse_args()
    design = yaml.safe_load((HERE / 'config.yaml').read_text())
    cfg = config()
    recipes = json.loads((BASE_OUT / 'selected_recipe.json').read_text())
    if args.phase == 'screen':
        tasks = args.tasks or design['screen_tasks']
        variants = args.variants or design['screen_variants']
        seeds = args.seeds or [design['screen_seed']]
        steps, phase = design['screen_steps'], 'pilot'
    else:
        tasks = args.tasks or design['confirmation_tasks']
        variants = args.variants or json.loads((OUT / 'selection.json').read_text())['confirm_variants']
        seeds = args.seeds or design['confirmation_seeds']
        steps, phase = design['confirmation_steps'], 'compare'
    # Process-local dependency injection: no changes to historical D38 source,
    # data, model checkpoints or initialization implementation.
    base.OUT = OUT
    base.make_model = make_model
    jobs = [(t, v, recipes[t]['lr'], s, steps, phase, cfg, 512)
            for t in tasks for s in seeds for v in variants]
    for i, job in enumerate(jobs):
        if i % args.shards == args.shard:
            base.run_one(job)


if __name__ == '__main__':
    main()
