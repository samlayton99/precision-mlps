"""Project archived Adam checkpoints onto the frozen raw-readout kernel modes.

No training and no regeneration of the unavailable every-update trace audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

GAMMAS = [8, 12, 16, 64]
DEFAULT_ROOT = Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep')


def sustained_acquisition(errors, epsilon):
    """Contiguous full trace, including initial state and final endpoint."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError('A nonempty finite scalar trace is required')
    above = np.flatnonzero(errors > epsilon)
    below = np.flatnonzero(errors <= epsilon)
    last = int(above[-1]) if len(above) else -1
    sustained = last + 1 if last < len(errors)-1 else -1
    return dict(first=int(below[0]) if len(below) else -1, last_above=last,
                sustained=sustained, confirmation_updates=len(errors)-sustained if sustained >= 0 else 0)


def projected_energy(singular, vh, loading, theta, norm_sq):
    """Retained-mode energy, not a claim that omitted modes are null."""
    coefficients = singular[:, None] * (vh @ theta) - loading
    return coefficients**2 / norm_sq[None, :]


def analyze(root):
    root = Path(root)
    hashes = {}
    def record(path):
        path = Path(path)
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):
        return json.loads(record(root/path).read_text())
    def arrays(path):
        with np.load(record(root/path)) as z:
            return {k: z[k] for k in z.files}
    config = read('manifest.json')['config']
    assert config['n'] == 512 and config['samples_per_cell'] == 16
    assert config['training_steps'] == 200000
    common = arrays('common/N512/arrays.npz')
    y = common['y_train']; norm_sq = np.sum(y*y, axis=0)
    targets = config['targets']; nt = len(targets)
    assert y.shape == (8193, 5)
    base = 'training/N512_raw_adam_'
    cases = {stage: read(base+stage+'/case.json') for stage in ['pilot', 'continue']}
    evaluations = {stage: read(base+stage+'/evaluations.json') for stage in cases}
    selection = read('training/N512_raw_selection.json')
    audit = arrays(base+'continue/hitting_audit.npz')
    state = arrays(base+'continue/state.npz')
    assert int(state['count']) == 200000
    np.testing.assert_array_equal(audit['first'], state['hits'])
    assert cases['pilot']['gammas'] == cases['continue']['gammas']
    assert cases['pilot']['matrix_hashes'] == cases['continue']['matrix_hashes']
    for case in cases.values():
        assert case['map'] == 'raw' and case['n'] == 512 and case['optimizer'] == 'adam'
    # Load only actual saved coefficients; continuation includes the pilot endpoint.
    checkpoint = {}
    for stage in cases:
        for row in evaluations[stage]:
            step = row['step']
            checkpoint[stage, step] = arrays(base+stage+f'/checkpoint_{step:06d}.npz')
            assert int(checkpoint[stage, step]['count']) == step
    steps = sorted(set(row['step'] for rows in evaluations.values() for row in rows))
    out = dict(checkpoint_steps=np.array(steps))
    rows = []; diagnostics = []
    for gamma in GAMMAS:
        gi = cases['continue']['gammas'].index(gamma)
        folder = f'dictionaries/N512_raw_g{gamma}/'
        meta = read(folder+'meta.json'); spec = arrays(folder+'spectrum.npz')
        assert meta['neighbor'] is False and np.all(np.asarray(meta['scales']) == 1)
        j = np.column_stack((np.ones(len(y)), np.tanh(gamma*(common['x_train'][:, None]-common['centers']))))/np.sqrt(len(y))
        assert hashlib.sha256(np.ascontiguousarray(j).tobytes()).hexdigest() == meta['matrix_hash']
        s, vh, loading = spec['singular'], spec['Vh'], spec['loadings']
        np.testing.assert_allclose(norm_sq, spec['norm_y']**2, rtol=1e-13)
        order = np.argsort(s)
        prefix = f'g{gamma}_'
        out[prefix+'rho'] = (s[order]/s[0])**2
        initial = loading[order]**2/norm_sq
        out[prefix+'initial_energy'] = initial
        out[prefix+'initial_cumulative'] = np.cumsum(initial, axis=0)
        out[prefix+'initial_unresolved_energy'] = spec['floor_sq']/norm_sq
        max_error = 0.; min_remainder = 0.
        for view in ['common', 'selected']:
            ci = [next(i for i,c in enumerate(cases['continue']['columns']) if c['view']==view and c['target']==t) for t in targets]
            pi = np.asarray(selection['indices'])[gi, ci]
            for target, a, b in zip(targets, ci, pi):
                pc = cases['pilot']['columns'][b]
                assert pc['target'] == target and pc['initialization'] == 'zero'
                assert pc['initial_rate'] == cases['continue']['rates'][gi][a]
                assert pc['epsilon'] == cases['continue']['epsilons'][gi][a]
                if view == 'common':
                    assert pc['initial_rate'] == .001 and pc['epsilon'] == 1e-12
            for key in ['theta', 'mu', 'nu']:
                np.testing.assert_array_equal(checkpoint['pilot',50000][key][gi][:,pi], checkpoint['continue',50000][key][gi][:,ci])
            np.testing.assert_array_equal(checkpoint['pilot',50000]['failed'][gi,pi], checkpoint['continue',50000]['failed'][gi,ci])
            energies=[]; totals=[]; remainders=[]; tail_bounds=[]
            for step in steps:
                stage = 'pilot' if step <= 50000 else 'continue'
                indices = pi if stage == 'pilot' else ci
                theta = checkpoint[stage, step]['theta'][gi][:, indices]
                if step == 0:
                    assert not np.any(theta)
                actual = j@theta-y
                total = np.sum(actual**2, axis=0)/norm_sq
                ev = next(r for r in evaluations[stage] if r['step']==step)
                saved = np.asarray(ev['train'])[gi, indices]
                gap = float(np.max(np.abs(np.sqrt(total)-saved)))
                max_error = max(max_error, gap)
                np.testing.assert_allclose(np.sqrt(total), saved, rtol=2e-9, atol=2e-12)
                energy = projected_energy(s,vh,loading,theta,norm_sq)[order]
                remainder = total-np.sum(energy,axis=0)
                min_remainder = min(min_remainder,float(remainder.min()))
                # FP64 accounting diagnostic, not an interval certificate.
                if remainder.min() < -1e-10:
                    raise AssertionError('Resolved energy exceeds directly measured energy')
                omitted_theta = theta-vh.T@(vh@theta)
                bound = (np.sqrt(spec['floor_sq'])+1e-14*s[0]*np.linalg.norm(omitted_theta,axis=0))**2/norm_sq
                energies.append(energy); totals.append(total); remainders.append(remainder); tail_bounds.append(bound)
            out[prefix+view+'_residual_energy'] = np.array(energies)
            out[prefix+view+'_residual_cumulative'] = np.cumsum(energies,axis=1)
            out[prefix+view+'_total_energy'] = np.array(totals)
            out[prefix+view+'_unresolved_energy_signed'] = np.array(remainders)
            out[prefix+view+'_omitted_svd_tail_bound_fp64'] = np.array(tail_bounds)
            for ti,(target,c) in enumerate(zip(targets,ci)):
                first=int(audit['first'][gi,c,0]); last=int(audit['last_above'][gi,c,0]); hit=int(audit['sustained'][gi,c,0])
                assert hit == (last+1 if last < 200000 else -1)
                assert state['failed'][gi,c] == 0
                rows.append(dict(gamma=gamma,view=view,target=target,first_hit=first,last_above=last,
                    sustained_hit=hit,confirmation_updates=200001-hit if hit>=0 else 0,
                    final_error=float(np.sqrt(totals[-1][ti])),initial_rate=cases['continue']['rates'][gi][c],
                    epsilon=cases['continue']['epsilons'][gi][c],pilot_column=int(pi[ti]),continuation_column=int(c)))
        diagnostics.append(dict(gamma=gamma,retained_modes=len(s),relative_singular_cutoff=1e-14,
            maximum_checkpoint_error_gap=max_error,minimum_signed_unresolved_energy=min_remainder,
            matrix_hash=meta['matrix_hash'],largest_eigenvalue=float(s[0]**2),archived_gd_curvature=meta['L']))
    result=dict(gammas=GAMMAS,targets=targets,checkpoint_steps=steps,cases=rows,diagnostics=diagnostics,
        normalization='All energies divided by original target squared norm; rho = eigenvalue / largest eigenvalue.',
        sustained_definition='First update below 1% at every subsequent executed update through 200000; not infinite-horizon convergence. confirmation_updates counts confirming states inclusively, including the first and final state.',
        schedule='Adam beta1=.9 beta2=.999; initial LR until20k, cosine decay to0.001 times initial LR at50k, constant thereafter.',
        selection='Median validation residual at40k,42500,45000,47500,50k;5 initial rates x2 epsilons; continuation preserves moments.',
        limitations=['No GD law asserted for Adam.','Existing every-update hitting audit reused; raw traces absent locally.',
            'Saved spectrum truncated at relative singular value1e-14; omitted energy is unresolved, not nullspace.',
            'Signed unresolved energy includes subtraction roundoff; no interval certification.',
            'SVD tail bound uses saved FP64 factors and cutoff, not a rigorous numerical certificate.'],
        array_layout=dict(rho='ascending retained modes', initial_energy='mode,target', residual_energy='checkpoint,mode,target', total_and_unresolved_energy='checkpoint,target', cumulative='sum over ascending retained modes only'),
        source_sha256=hashes,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result,out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path); args=parser.parse_args()
    result,arrays=analyze(args.root)
    output=args.output or args.root/'refinements/gamma_optimizer_access'
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'projections.npz',**arrays)
    print(json.dumps(dict(cases=len(result['cases']),checkpoints=len(result['checkpoint_steps']),diagnostics=result['diagnostics'])))


if __name__ == '__main__':
    main()
