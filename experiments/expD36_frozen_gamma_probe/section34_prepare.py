"""Prepare the matched width-512 long-horizon Section 3.4 experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adam_feature_probe_prepare import geometry, grid, initial_parameters, target


def prepare(output, target_name, horizon):
    centers, info = geometry(512)
    x, vx, ex = grid(2048), grid(4096), grid(8192)
    fn = target if target_name == 'mixed_sines' else lambda z: z*z
    normalizer = float(np.sqrt(np.mean(fn(x)**2)))
    y, vy, ey = [fn(z)/normalizer for z in (x, vx, ex)]
    bandwidths = np.array([.03125, .0625, .09375, .125, .25, .5, 1.])
    gammas = bandwidths/info['spacing']
    a = np.broadcast_to(gammas[:, None], (len(gammas), 512)).copy()
    b = -gammas[:, None]*centers
    features = np.concatenate((np.tanh(x[None,:,None]*a[:,None,:]+b[:,None,:]),
                               np.ones((len(gammas),len(x),1))), axis=-1)
    base = output/'base'
    base.mkdir(parents=True, exist_ok=True)
    if (base/'input.npz').exists():
        raise FileExistsError('Use a fresh experiment directory')
    np.savez_compressed(base/'input.npz', features=features, target=y, a=a, b=b,
                        train_x=x, validation_x=vx, eval_x=ex,
                        validation_target=vy, eval_target=ey)
    parameters = np.array([initial_parameters(512,info['interior_intervals'],s) for s in range(5)])
    np.savez_compressed(base/'joint_input.npz', x=x, target=y, initial_parameters=parameters)
    rows = [dict(name=f'uniform_lambda{lam:g}', family='uniform', gamma=float(gamma),
                 lambda_rms=float(lam), seed=None, snapshot_step=None, source_error=None)
            for lam,gamma in zip(bandwidths,gammas)]
    manifest = dict(**info, geometries=rows, samples=len(x), validation_samples=len(vx),
                    eval_samples=len(ex), target_name=target_name, target_normalizer=normalizer,
                    target_formula='sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)' if target_name=='mixed_sines' else 'x**2',
                    input_sha256=hashlib.sha256((base/'input.npz').read_bytes()).hexdigest(),
                    feature_layout='raw tanh columns followed by output bias',
                    parameter_initialization='zero readout for frozen features; five paired affine Xavier joint initializations',
                    seed_ids=list(range(5)),
                    evaluation_role='8192 midpoint resolution check, previously inspected; not an untouched test set')
    (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    common = dict(horizon=horizon, epsilon=1e-8, chunk_size=100,
                  schedules=['constant','cosine'])
    for name, rates in [('adam',[.0002,.002,.02,.05,.1]),
                        ('gd',[.0002,.002,.02,.05,.1,.2,.5,1.])]:
        config = dict(**common, optimizer=name, learning_rates=rates)
        (output/f'joint_{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    frozen = dict(**common, learning_rates=[1e-5,1e-4,.001,.002,.01,.1])
    (output/'frozen_adam.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(output=str(output), horizon=horizon, target=target_name,
                         feature_shape=features.shape, **info)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target',choices=['mixed_sines','quadratic'],default='mixed_sines')
    parser.add_argument('--horizon',type=int,default=2_000_000)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error('--horizon must be positive')
    prepare(args.output,args.target,args.horizon)
