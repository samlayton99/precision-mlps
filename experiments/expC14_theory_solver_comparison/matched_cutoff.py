"""Matched-cutoff comparisons on the same N64--768 geometry.

Both arms use relative cutoff 2**-52. Default preserves the previous drivers;
--scaled-driver gelsd uses the identical solver on A versus A D, with the
same labels and no asymmetric sample normalization. Neither variant is the
original A.65 recovery algorithm.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison import affine_grid
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare
from experiments.expC14_theory_solver_comparison.jacobi_probe import jacobi_svd


def run_one(N, cfg, out, previous, ye, yv):
    started = time.perf_counter()
    params = prepare(N, cfg)
    A, y = affine_grid.training(N, params['R'], cfg['sampling_density'])
    rc = cfg['actual_relative_cutoff']
    assert rc == cfg['scaled_relative_cutoff'] == 2**-52
    raw, raw_ranks, raw_s = [], [], []
    for j in range(6):
        c, _, rank, singular = lstsq(A, y[:, j], cond=rc,
                                    lapack_driver='gelsd', check_finite=False)
        raw.append(c); raw_ranks.append(int(rank)); raw_s.append(singular)
    actual = np.column_stack(raw)
    scaled = []
    if cfg['scaled_driver'] == 'gelsd':
        F = A * params['D']
        for j in range(6):
            a, _, rank, sj = lstsq(F, y[:, j], cond=rc,
                                  lapack_driver='gelsd', check_finite=False)
            if j:
                assert np.array_equal(sj, singular) and rank == scaled_rank
            singular = sj; scaled_rank = int(rank)
            scaled.append(params['D'] * a)
        driver_info = dict(driver='gelsd', common_relative_cutoff=rc,
                           matrix='A D', labels='y', sample_normalization='none in either arm')
    else:
        F = A * params['D'] / math.sqrt(len(A))
        yn = y / math.sqrt(len(A))
        U, singular, Vt, driver_info = jacobi_svd(F, joba=0, jobv=1)
        keep = singular > rc * singular[0]
        scaled_rank = int(keep.sum())
        for j in range(6):
            p = r.projection(U, yn[:, j])
            z = np.zeros_like(p)
            z[keep] = p[keep] / singular[keep]
            scaled.append(r.export(Vt, z, params['D']))
        del U, Vt
    tau = rc * singular[0]
    scaled = np.column_stack(scaled)
    del A, F
    coefficients = np.column_stack((actual, scaled))
    errors, maxima = affine_grid.evaluate(N, params['R'], coefficients, np.tile(ye, (1, 2)))
    validation, _ = affine_grid.evaluate(N, params['R'], coefficients, np.tile(yv, (1, 2)))
    old = np.load(previous / f'coefficients_N{N}.npz')
    assert np.array_equal(actual, old['actual']), 'Unscaled arm changed unexpectedly'
    assert np.array_equal(params['D'], old['D']), 'Reference scaling changed'
    if cfg['scaled_driver'] == 'gejsv':
        assert np.array_equal(singular, old['singular']), 'Scaled factorization changed'
    old_rows = {x['target']: x for x in json.loads((previous / f'N{N}.json').read_text())}
    rows = []
    for j, name in enumerate(r.NAMES):
        rows.append(dict(N=N, W=len(params['centers']), R=params['R'], target=name,
            actual_error=float(errors[j]), scaled_error=float(errors[j+6]),
            actual_validation_error=float(validation[j]), scaled_validation_error=float(validation[j+6]),
            actual_linf=float(maxima[j]), scaled_linf=float(maxima[j+6]),
            actual_relative_cutoff=rc, scaled_relative_cutoff=rc,
            actual_absolute_cutoff=float(rc*raw_s[j][0]), scaled_absolute_cutoff=float(tau),
            actual_rank=raw_ranks[j], scaled_rank=scaled_rank,
            previous_theory_absolute_cutoff=old_rows[name]['theory']['tau'],
            previous_theory_error=old_rows[name]['theory_error'],
            geometry_checks=params['checks'], actual_unchanged=True,
            same_solver=cfg['scaled_driver']=='gelsd',
            scaled_singular_values_unchanged=True if cfg['scaled_driver']=='gejsv' else None,
            uses_original_theorem_cutoff=False, scaled_driver_info=driver_info,
            actual_l1=float(np.sum(abs(actual[:,j]))), scaled_l1=float(np.sum(abs(scaled[:,j])))))
    np.savez_compressed(out/f'coefficients_N{N}.npz', actual=actual, scaled=scaled,
        D=params['D'], singular=singular, centers=params['centers'], gamma=params['slope'],
        hidden_biases=old['hidden_biases'])
    (out/f'N{N}.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print(f'N={N} ({time.perf_counter()-started:.1f}s): '+', '.join(
        f"{v['target']} {v['actual_error']:.2e}/{v['scaled_error']:.2e}" for v in rows), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scaled-driver', choices=['gejsv','gelsd'], default='gejsv')
    args = parser.parse_args()
    old_cfg = yaml.safe_load((args.previous/'config.yaml').read_text())
    shared = ['N','targets','lambda','a_H','delta','Delta','sampling_density','sampling_degree',
              'ellipse_rho','eval_points','validation_points','threads','feature_parameterization',
              'extended_scalar_bounds','actual_relative_cutoff','input_error_route']
    cfg = {k: old_cfg[k] for k in shared}
    cfg.update(experiment='expC14_matched_relative_cutoff_N64_768', scaled_relative_cutoff=2**-52,
        comparison_arm='scaled', plot_auto_ymax=True, actual_driver='gelsd', scaled_driver=args.scaled_driver,
        interpretation='Matched relative cutoff; NOT original A.65 theoretical recovery.',
        changed_from_previous='Only scaled cutoff; same geometry, data, drivers, projection and export.',
        previous_output=str(args.previous.resolve()))
    if args.scaled_driver == 'gelsd':
        cfg.update(experiment='expC14_same_solver_same_cutoff_diagonal_ablation_N64_768',
            changed_from_previous='Matched DGELSD and relative cutoff in both arms; compare A versus A D on identical y.',
            between_arm_difference='Diagonal column scaling and corresponding coefficient map-back only.',
            sample_normalization='None in either arm')
    out=args.output; out.mkdir(exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    hashes={}
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,snapshot/f.name); hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','gmpy2','mpmath','matplotlib','PyYAML','threadpoolctl']}
    (out/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,packages=packages,
        python=sys.version,previous=str(args.previous.resolve()),fresh_run=True),indent=2)+'\n')
    xe=-1+2*(np.arange(cfg['eval_points'])+.5)/cfg['eval_points']
    xv=-1+2*(np.arange(cfg['validation_points'])+.5)/cfg['validation_points']
    ye,yv=r.target_values(xe),r.target_values(xv)
    rows=[]
    with threadpool_limits(limits=cfg['threads']):
        for N in cfg['N']: rows.extend(run_one(N,cfg,out,args.previous,ye,yv))
    summary=dict(pairs=len(rows),relative_cutoff=2**-52,all_unscaled_coefficients_unchanged=True,
        actual_driver='gelsd', scaled_driver=cfg['scaled_driver'],
        all_scaled_singular_values_unchanged=True if cfg['scaled_driver']=='gejsv' else None,
        max_scaled_over_unscaled=max(x['scaled_error']/x['actual_error'] for x in rows),
        max_unscaled_over_scaled=max(x['actual_error']/x['scaled_error'] for x in rows),
        max_validation_relative_change=max(abs(x[f'{a}_validation_error']/x[f'{a}_error']-1)
            for x in rows for a in ['actual','scaled']),
        max_errors_N_ge_256={a:max(x[f'{a}_error'] for x in rows if x['N']>=256)
            for a in ['actual','scaled']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
