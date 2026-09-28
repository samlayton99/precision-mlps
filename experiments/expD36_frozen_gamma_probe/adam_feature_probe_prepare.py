"""Prepare a fixed total-width Adam comparison, counting every halo feature."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def grid(n):
    return -1 + (np.arange(n) + .5) * 2 / n


def target(x):
    return np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(14*np.pi*x)


def geometry(width):
    if width < 2 or width & (width-1):
        raise ValueError('Total hidden width must be a power of two')
    choices = [(n, int(np.ceil(np.sqrt(n)))) for n in range(1, width)]
    exact = [(n, r) for n, r in choices if n + 1 + 2*r == width]
    if not exact:
        raise ValueError('No exact symmetric allocation for this width and halo rule')
    n, radius = exact[-1]
    h = 2/n
    centers = -1 + h*np.arange(-radius, n+radius+1)
    assert len(centers) == width
    np.testing.assert_allclose(centers, -centers[::-1], rtol=0, atol=5e-16)
    return centers, dict(width=width, interior_intervals=n, interior_centers=n+1,
                         halo_each_side=radius, spacing=h,
                         halo_rule='ceil(sqrt(interior_intervals)); inverted within total width')


def initial_parameters(width, n, seed):
    bound = np.sqrt(6/(width+1))
    rng = np.random.default_rng([seed, n])
    a, b = rng.uniform(-bound, bound, (2, width))
    c = np.random.default_rng([seed, n, 24]).uniform(-bound, bound, width)
    return np.r_[a, b, c, 0.]


def save_input(output, a, b, rows, geometry_info, extra):
    x, vx, ex = grid(2048), grid(4096), grid(8192)
    scale = np.sqrt(np.mean(target(x)**2))
    y = target(x)/scale
    a, b = np.array(a), np.array(b)
    features = np.concatenate((np.tanh(x[None,:,None]*a[:,None,:]+b[:,None,:]),
                               np.ones((len(a),len(x),1))), axis=-1)
    assert features.shape[-1] == geometry_info['width']+1
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'input.npz', features=features, target=y, a=a, b=b,
                        train_x=x, validation_x=vx, eval_x=ex,
                        validation_target=target(vx)/scale, eval_target=target(ex)/scale)
    manifest = dict(geometries=rows, **geometry_info, samples=2048,
                    validation_samples=4096, eval_samples=8192, target_normalizer=float(scale),
                    target_formula='sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)',
                    input_sha256=hashlib.sha256((output/'input.npz').read_bytes()).hexdigest(),
                    feature_layout='raw tanh columns followed by output bias',
                    parameter_initialization='zero readout, output bias, and Adam moments', **extra)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(shape=features.shape, **geometry_info)))


def prepare(output):
    centers, info = geometry(512)
    h = info['spacing']
    aa, bb, rows = [], [], []
    for bandwidth in (.03125,.0625,.09375,.125,.25,.5,1.):
        gamma = bandwidth/h
        aa.append(np.full(512,gamma)); bb.append(-gamma*centers)
        rows.append(dict(name=f'uniform_lambda{bandwidth:g}',family='uniform',
                         gamma=gamma,lambda_rms=bandwidth,seed=None,snapshot_step=None,source_error=None))
    pp = np.array([initial_parameters(512,info['interior_intervals'],s) for s in range(5)])
    x=grid(2048); y=target(x)/np.sqrt(np.mean(target(x)**2))
    for seed,p in enumerate(pp):
        a,b,c=p[:-1].reshape(3,-1)
        error=np.linalg.norm(np.tanh(x[:,None]*a+b)@c+p[-1]-y)/np.linalg.norm(y)
        aa.append(a);bb.append(b)
        rows.append(dict(name=f'learned_seed{seed}_step0',family='learned',gamma=None,
                         lambda_rms=float(np.sqrt(np.mean(a*a))*h),seed=seed,
                         snapshot_step=0,source_error=float(error)))
    save_input(output,aa,bb,rows,info,dict(initialization='D24 affine Xavier convention, resolution replaced by budget-derived intervals'))
    np.savez_compressed(output/'joint_input.npz',x=x,target=y,initial_parameters=pp)


def self_test():
    for width in (32,64,128,256,512,1024):
        centers,info=geometry(width)
        assert len(centers)==width
        assert info['interior_centers']+2*info['halo_each_side']==width
        assert np.count_nonzero((centers>=-1-1e-14)&(centers<=1+1e-14))==info['interior_centers']
    _,info=geometry(512)
    assert (info['interior_intervals'],info['halo_each_side'])==(467,22)
    try: geometry(559)
    except ValueError: pass
    else: raise AssertionError('Non-power-of-two width accepted')
    assert initial_parameters(512,467,0).shape==(1537,)
    print('Total-width allocation and initialization checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    if args.output:prepare(args.output)
