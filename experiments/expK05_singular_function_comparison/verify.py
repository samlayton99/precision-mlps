"""Independent evaluation, ordinary Torch export, and saved-reference checks."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expK05_singular_function_comparison.run import OUT, save
from experiments.expK05_singular_function_comparison.construction import (
    TARGETS, TargetExtension, Network, build, metrics, derivative, quadrature,
)
from experiments.expK04_npu_elementary_suite.problem import target, grid, score
from experiments.expK04_npu_elementary_suite.models import build as build_npu


def main():
    torch.set_num_threads(1)
    references = json.loads((ROOT/'results/checkpoint_K_practical_recipes/expK04_npu_elementary_suite/final_primary_recipes.json').read_text())
    checks = {}
    for name in TARGETS:
        e = TargetExtension(name=name)
        folder = OUT/name
        net = Network.load(folder/'N256_R32.npz')
        info = dict(N=256, width=net.width, parameters=3*net.width+1,
                    quadrature={str(q):metrics(net, name, q) for q in [64, 128, 256, 512]})
        assert max(info['quadrature'].values()) < 1e-12
        x = np.unique(np.r_[np.geomspace(e.eps, 1, 32769), np.linspace(e.eps, 1, 8193),
                            np.exp(np.random.default_rng(7341).uniform(np.log(e.eps), 0, 4096))])
        truth, pred = target(name, x), net.predict(x)
        relative = (pred-truth)/truth
        info.update(dense_points=len(x), maximum_sampled_relative=float(abs(relative).max()),
                    maximum_sampled_absolute=float(abs(pred-truth).max()),
                    max_hidden_weight=float(abs(net.weight).max()))
        np.savez(folder/'selected_predictions.npz', x=x, truth=truth, prediction=pred, relative_error=relative)
        model = torch.nn.Sequential(torch.nn.Linear(1, net.width, dtype=torch.float64), torch.nn.Tanh(),
                                    torch.nn.Linear(net.width, 1, dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(net.weight[:,None]))
            model[0].bias.copy_(torch.from_numpy(net.hidden_bias))
            model[2].weight.copy_(torch.from_numpy(net.readout[None,:]))
            model[2].bias.fill_(net.output_bias)
            xq, logw = quadrature(e.eps, 512)
            z = torch.from_numpy(e.centered(xq)[:,None])
            yp = torch.cat([model(part) for part in z.split(256)]).numpy().ravel()
        yq = target(name, xq)
        info['torch_relative_l2'] = float(np.sqrt(np.dot(logw*xq, (yp-yq)**2)/np.dot(logw*xq, yq**2)))
        assert info['torch_relative_l2'] < 1e-12
        torch.save(dict(model=model.state_dict(), target=name, epsilon=e.eps,
                        input_normalization='z=(x-epsilon)/radius-1', radius=e.radius), folder/'selected_torch.pt')
        xx = np.geomspace(e.eps, 1, 4097)
        total = sum(s**-1.5*e.block_derivative(xx/(2*s)-1, s)/2 for s in e.scales)
        info['derivative_partition_max_relative'] = float(np.max(abs((total-derivative(name, xx))/derivative(name, xx))))
        assert info['derivative_partition_max_relative'] < 5e-15
        # More halo must preserve the result's precision regime.
        wider = build(e, 256, OUT/'cache', halo=64)
        wider.save(folder/'halo_check_N256_R64.npz')
        info['halo64_relative_l2'] = metrics(wider, name, 512)
        assert info['halo64_relative_l2'] < 1e-12
        # Evaluate the saved float64 coefficients with 60-decimal arithmetic.
        import mpmath as mp
        with mp.workdps(60):
            weights, biases, readouts = [list(map(mp.mpf, a)) for a in (net.weight, net.hidden_bias, net.readout)]
            errors = []
            for xx in np.geomspace(e.eps, 1, 25):
                xm = mp.mpf(xx)
                zz = (xm-mp.mpf(e.eps))/mp.mpf(e.radius)-1
                args = [w*zz+b for w,b in zip(weights,biases)]
                prediction = mp.mpf(net.output_bias)+mp.fsum(
                    v*(mp.sign(a) if abs(a)>45 else mp.tanh(a)) for v,a in zip(readouts,args))
                truth = 1/(mp.sqrt(xm)+xm) if name=='root_plus_linear' else mp.exp(-xm)/mp.sqrt(xm)
                errors.append(float(abs((prediction-truth)/truth)))
        info['high_precision_saved_network'] = dict(decimal_digits=60, points=25,
            maximum_relative=max(errors), tanh_saturation_absolute_bound=2*math.exp(-90)*float(abs(net.readout).sum()))
        assert max(errors) < 1e-12
        ref = references[name]
        npu = build_npu(ref['model'], ref['seed'])
        npu.load_state_dict(torch.load(ROOT/ref['run_directory']/'selected.pt', weights_only=False)['model'])
        measured = {str(q):score(npu, grid(name, q)) for q in [256, 512]}
        np.testing.assert_allclose(list(measured.values()), ref['test_relative_l2'], rtol=1e-8)
        info['npu_reference'] = dict(run_directory=ref['run_directory'], hidden_width=32,
                                    parameters=ref['parameters'], seed=0, adam_steps=ref['adam_steps'],
                                    learning_rate=ref['learning_rate'], polish='up to 500 L-BFGS iterations',
                                    adam_error=ref['adam_best_test'], quadrature=measured,
                                    training_domain=[.0001, 1.], halo_training=False)
        checks[name] = info
        save(OUT/'verification.json', checks)
        print(name, json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
