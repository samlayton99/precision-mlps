"""Train the published NPU architecture on the saved QUILL C17 extension."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.neural_power_unit import NPURegressor
from src.models.centered_tanh import CenteredTanhMLP
from experiments.expK02_inverse_sqrt_extension.construction import Extension, quadrature

OUT = ROOT / 'results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt'
QI = ROOT / 'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def halo_nodes(extension, n, halo):
    # QI samples z_n=-1+2n/N, u=2+2z, then physical x=s_j*u.
    return np.concatenate([4 * s * np.arange(-halo, n + halo + 1) / n
                           for s in extension.scales])


def evaluation_grid(extension, order):
    x, weights = quadrature(extension.eps, order)
    target = extension.value(x)
    return (torch.from_numpy(x[:, None]), torch.from_numpy(target[:, None]),
            torch.from_numpy(weights[:, None]))


@torch.no_grad()
def predict(model, x, batch=512):
    return torch.cat([model(part) for part in x.split(batch)])


def make_model(config, model_name):
    if model_name == 'Adam MLP':
        return CenteredTanhMLP(config['width'], eps=config['epsilon'], seed=config['seed'])
    if model_name not in ['NPU', 'RealNPU']:
        raise ValueError(model_name)
    return NPURegressor(config['width'], complex_weights=model_name == 'NPU', seed=config['seed'])


@torch.no_grad()
def relative_l2(model, grid, eps):
    x, y, weights = grid
    error = (predict(model, x) - y) / y
    # f^2 dx=d(log x) on the scored interval; exactly the QUILL metric.
    return float(torch.sqrt((weights * error.square()).sum() / math.log(1 / eps)))


def train(config, model_name, mode, output):
    start = time.monotonic()
    extension = Extension(eps=config['epsilon'], order=config['extension_order'])
    model = make_model(config, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 betas=tuple(config['betas']), eps=config['adam_epsilon'],
                                 weight_decay=config['weight_decay'], foreach=False)
    rng = np.random.default_rng(config['data_seed'])
    nodes = halo_nodes(extension, config['halo_reference_N'], config['halo_reference_R'])
    values = extension.value(nodes)
    grid = evaluation_grid(extension, config['validation_quadrature_order'])
    history = [{'step': 0, 'validation_rel_l2': relative_l2(model, grid, extension.eps)}]
    best = history[0]['validation_rel_l2']
    best_step = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_trained = math.inf
    best_trained_step = None
    best_trained_state = None
    losses = []
    stem = f'{model_name}_{mode}'
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config}, output / f'{stem}_initial.pt')
    for step in range(1, config['steps'] + 1):
        if mode == 'matched_halo':
            indices = rng.integers(0, len(nodes), config['batch_size'])
            x, y = nodes[indices], values[indices]
        elif mode == 'interval_only':
            x = rng.uniform(extension.eps, 1., config['batch_size'])
            y = extension.value(x)
        else:
            raise ValueError(mode)
        inputs = torch.from_numpy(x[:, None])
        targets = torch.from_numpy(y[:, None])
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{stem} became nonfinite at step {step}')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'{stem} gradient became nonfinite at step {step}')
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % config['log_every'] == 0 or step == config['steps']:
            error = relative_l2(model, grid, extension.eps)
            history.append({'step': step, 'validation_rel_l2': error})
            if error < best:
                best, best_step = error, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step >= config.get('minimum_selection_step', 1) and error < best_trained:
                best_trained, best_trained_step = error, step
                best_trained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step % config['checkpoint_every'] == 0 or step == config['steps']:
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'rng_state': rng.bit_generator.state, 'step': step, 'config': config,
                          'model_name': model_name, 'training_mode': mode, 'history': history,
                          'losses': losses, 'best_state': best_state, 'best_step': best_step,
                          'best_trained_state': best_trained_state, 'best_trained_step': best_trained_step}
            torch.save(checkpoint, output / f'{stem}_step{step}.pt')
            save_json(output / f'{stem}_history.json', history)
            print(f'{stem} width={config["width"]} lr={config["learning_rate"]:g} step={step} rel_L2={history[-1]["validation_rel_l2"]:.6g} '
                  f'elapsed={time.monotonic()-start:.1f}s', flush=True)
    finals = {str(order): relative_l2(model, evaluation_grid(extension, order), extension.eps)
              for order in config['test_quadrature_orders']}
    dense = np.unique(np.r_[np.geomspace(extension.eps, 1., 8193),
                            np.linspace(extension.eps, 1., 4097)])
    with torch.no_grad():
        predicted = predict(model, torch.from_numpy(dense[:, None])).numpy().ravel()
        halo_prediction = predict(model, torch.from_numpy(nodes[:, None])).numpy().ravel()
    true = extension.value(dense)
    np.savez(output / f'{stem}_predictions.npz', x=dense, target=true, prediction=predicted,
             halo_x=nodes, halo_target=values, halo_prediction=halo_prediction)
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_test = relative_l2(model, evaluation_grid(extension, 256), extension.eps)
    torch.save({'model': best_state, 'step': best_step, 'config': config},
               output / f'{stem}_best_validation.pt')
    assert best_trained_state is not None and best_trained_step > 0
    model.load_state_dict(best_trained_state)
    best_trained_tests = {str(order): relative_l2(model, evaluation_grid(extension, order), extension.eps)
                          for order in config['test_quadrature_orders']}
    torch.save({'model': best_trained_state, 'step': best_trained_step, 'config': config},
               output / f'{stem}_best_trained_validation.pt')
    row = {'model': model_name, 'training_mode': mode, 'seed': config['seed'],
           'hidden_units': config['width'], 'parameters': sum(p.numel() for p in model.parameters()),
           'steps': config['steps'], 'learning_rate': config['learning_rate'],
           'final_rel_l2': finals['256'], 'test_quadrature': finals,
           'best_validation_step': best_step, 'best_validation_test_rel_l2': best_test,
           'best_trained_validation_step': best_trained_step,
           'best_trained_validation_rel_l2': best_trained,
           'minimum_selection_step': config.get('minimum_selection_step', 1),
           'best_trained_validation_test_rel_l2': best_trained_tests['256'],
           'best_trained_validation_test_quadrature': best_trained_tests,
           'max_sampled_relative_error': float(np.max(np.abs((predicted-true)/true))),
           'halo_node_relative_l2': float(np.linalg.norm(halo_prediction-values)/np.linalg.norm(values)),
           'gate_raw': final_state['power.gate'].tolist() if 'power.gate' in final_state else None,
           'seconds': time.monotonic()-start,
           'checkpoint': f'{stem}_step{config["steps"]}.pt'}
    save_json(output / f'{stem}_summary.json', row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for mode in config['training_modes']:
        for model_name in config['models']:
            path = args.output / f'{model_name}_{mode}_summary.json'
            if path.exists():
                raise FileExistsError(f'Preserving completed run: {path}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    extension = Extension(eps=config['epsilon'], order=config['extension_order'])
    nodes = halo_nodes(extension, config['halo_reference_N'], config['halo_reference_R'])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'matched_halo_data.npz', x=nodes, y=extension.value(nodes))
    save_json(args.output / 'config.json', config)
    source_paths = [Path(__file__), ROOT/'src/models/neural_power_unit.py', ROOT/'src/models/centered_tanh.py',
                    ROOT/'experiments/expK02_inverse_sqrt_extension/construction.py',
                    QI/'selected_config.json', QI/'scaling_1k_5k.json']
    save_json(args.output / 'provenance.json', {
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
        'halo_node_count': len(nodes), 'halo_physical_range': [float(nodes.min()),float(nodes.max())],
        'evaluation_domain': config['domain'], 'same_extension_implementation': True,
        'halo_reference_width': config['halo_reference_width'],
        'limitations': ['One seed, not a success-rate estimate.',
                       'QUILL uses analytic derivative samples; trained models use function values.',
                       f'Trained models have fixed width {config["width"]}, not a width sweep.',
                       'Halo matches the fixed 4815-neuron QUILL reference, not every sweep width.',
                       'All training and evaluation is PyTorch float64.']})
    rows = []
    for mode in config['training_modes']:
        for model_name in config['models']:
            path = args.output / f'{model_name}_{mode}_summary.json'
            if path.exists():
                raise FileExistsError(f'Preserving completed run: {path}')
            rows.append(train(config, model_name, mode, args.output))
            save_json(args.output / 'summary.json', rows)


if __name__ == '__main__':
    main()
