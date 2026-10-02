"""Train-input geometry and independent analytic checks; no test-set tuning."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import numpy as np
from scipy import linalg
import torch
import yaml
from experiments.expD39_qi_init_theory.run import base, config, make_model, OUT, HERE
from src.construction.qi_mpmath import construct_qi, evaluate_qi


def spectral(a):
    s = linalg.svdvals(a)
    return dict(singular_values=s.tolist(), rank_1e12=int(np.sum(s > s[0] * 1e-12)),
                participation_rank=float(np.sum(s*s)**2 / np.sum(s**4)))


def geometry():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    design = yaml.safe_load((HERE / 'config.yaml').read_text())
    rows = []
    for task in design['screen_tasks']:
        arrays, metadata = base.load_data(task, config())
        train = torch.from_numpy(arrays['train'][0])
        for scheme in ['standard', *design['screen_variants']]:
            model, info = make_model(metadata['d_in'], 512, 0, scheme, train, config())
            x = train[:2048]
            layers = []
            with torch.no_grad():
                for layer, name in [(model.fc1, 'first'), (model.fc2, 'second')]:
                    w = layer.weight.numpy()
                    gamma = np.linalg.norm(w, axis=1)
                    a = layer(x)
                    h = torch.tanh(a).numpy()
                    centered_h = h - h.mean(0)
                    # Fraction of feature energy unexplained by an affine map
                    # from this layer's inputs: checks nonlinear activation use.
                    fit_w, fit_b, _ = base.affine_fit(x.numpy(), h)
                    residual = h - x.numpy() @ fit_w - fit_b
                    row = dict(gamma=gamma.tolist(), weight_spectrum=spectral(w),
                               feature_spectrum=spectral(centered_h),
                               saturated_fraction=float(np.mean(np.abs(h) > .99)),
                               mean_tanh_derivative=float(np.mean(1-h*h)),
                               nonlinear_feature_energy_fraction=float(np.sum(residual**2)/np.sum(centered_h**2)))
                    if scheme == 'qi':
                        sizes = [22]*23 + [6]
                    elif info.get(name, {}).get('banks'):
                        sizes = [b['size'] for b in info[name]['banks']]
                    else:
                        sizes = []
                    start, lambdas = 0, []
                    for size in sizes:
                        g = gamma[start:start+size]
                        c = -layer.bias.detach().numpy()[start:start+size] / g
                        lambdas.extend((g[:-1] * np.diff(c)).tolist())
                        start += size
                    row['actual_lambda'] = lambdas
                    layers.append(row)
                    x = torch.from_numpy(h)
            rows.append(dict(task=task, scheme=scheme, layers=layers, init=info))
            print('geometry', task, scheme, flush=True)
    base.save_json(OUT / 'geometry.json', dict(config=config(), rows=rows))


def analytic():
    x = np.linspace(-1, 1, 2049)
    xe = np.linspace(-1, 1, 4098)[1::2]
    targets = {'sin_pi': lambda z: np.sin(np.pi*z),
               'sin_8pi': lambda z: np.sin(8*np.pi*z),
               'runge': lambda z: 1/(1+25*z*z)}
    rows = []
    for name, fn in targets.items():
        y, ye = fn(x)[:, None], fn(xe)[:, None]
        for n in [16, 32, 64]:
            for halo in [0, 16]:
                for lam in [.25, .5, 1.]:
                    h = 2/n
                    centers = -1 + np.arange(-halo, n+halo+1)*h
                    gamma = lam/h
                    features = np.tanh(gamma*(x[:, None]-centers))
                    eval_features = np.tanh(gamma*(xe[:, None]-centers))
                    w, b, diag = base.affine_fit(features, y, 1e-12)
                    e = (eval_features@w+b-ye).ravel()
                    rows.append(dict(target=name, intervals=n, halo=halo, width=len(centers),
                                     lam=lam, gamma=gamma, actual_lambda=gamma*h,
                                     relative_l2=float(np.linalg.norm(e)/np.linalg.norm(ye)),
                                     linf=float(np.max(np.abs(e))),
                                     boundary_linf=float(np.max(np.abs(e[np.abs(xe)>.9]))),
                                     **diag))
    # Exercise the real construction, including derivative convolution, not just
    # an LS fit with a similar-looking bank. This is an fp64 reference check.
    qi = construct_qi(lambda z: np.sin(np.pi*z), lambda z: np.pi*np.cos(np.pi*z),
                      N=64, precision='fp64')
    err = float(np.max(np.abs(evaluate_qi(qi, xe)-np.sin(np.pi*xe))))
    assert err < 1e-10, err
    base.save_json(OUT / 'analytic_checks.json', dict(rows=rows,
                   construction=dict(N=qi.N, halo=qi.halo, width=len(qi.centers),
                                     lam=qi.lambda_val, linf=err, precision='fp64')))
    print('reference construction Linf', err)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['geometry', 'analytic'])
    a = p.parse_args()
    {'geometry': geometry, 'analytic': analytic}[a.kind]()
