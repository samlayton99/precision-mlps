"""Retrospective signed effective-force reinforcement at archived states.

Reuse existing state provenance and the exact persistence kernel. Scalars are
instantaneous effective-flow derivatives, even at states sampled from GD.
They are neither future envelopes nor per-neuron trajectory predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import mechanism_persistence_kernel as kernel
from . import population_coverage as pc
from .mechanism_persistence import validate_backend


def observables(p, x, y):
    state = kernel.decomposition(p, x, y)
    F, J, jc, gram, basis = (state[k] for k in ('F', 'J', 'JC', 'gram', 'basis'))
    f2 = F@F
    second = kernel._second_output(p, x, F)
    second_fine = kernel._fine(second, basis)
    second_coarse = basis.T@second/len(x)
    fine_velocity = kernel._fine(J@F, basis)
    ell_generated = jnp.linalg.solve(gram, jc@(J.T@state['fH']/len(x)))
    ell_target = jnp.linalg.solve(gram, jc@(J.T@state['yH']/len(x)))
    terms = dict(residual_relaxation=-jnp.mean(fine_velocity**2),
                 generated_geometry=-jnp.mean(state['fH']*second_fine),
                 target_geometry=jnp.mean(state['yH']*second_fine),
                 generated_compensation=ell_generated@second_coarse,
                 target_compensation=-ell_target@second_coarse)
    geometry = -jnp.mean(state['eH']*second_fine)
    compensation = state['balance']@second_coarse
    rhs = terms['residual_relaxation']+geometry+compensation
    derivative = jax.jvp(lambda point: kernel.effective(point, x, y), (p,), (-F,))[1]
    direct = F@derivative
    eig = jnp.linalg.eigvalsh(gram)
    resolved = eig[0] > 64*jnp.finfo(p.dtype).eps*jnp.maximum(1., eig[-1])
    sigma = jnp.sqrt(jnp.maximum(0., eig[0]))
    r2 = jnp.sum(p[:-1].reshape(3, -1)**2, axis=0)
    M, B = jnp.sum(r2), jnp.sum(r2**3)**(1/6)
    width = len(r2)
    particle_p = p[:-1].reshape(3, -1)
    particle_F = F[:-1].reshape(3, -1)
    particle_derivative = derivative[:-1].reshape(3, -1)
    force_squared = jnp.sum(particle_F**2, axis=0)
    Y = jnp.sqrt(jnp.mean(state['eH']**2))
    HC = jnp.sqrt(2.)+4*jnp.sqrt(M)
    ratio = lambda value: jnp.where((f2 > 0)&resolved, value/jnp.where(f2 > 0, f2, 1.), jnp.nan)
    geometric_bound = 6*jnp.sqrt(2.)*Y*B**2
    compensation_bound = 3*Y*HC*B**3/jnp.where(sigma > 0, sigma, 1.)
    directional = ratio(Y*jnp.sqrt(jnp.mean(second_fine**2))
                        +jnp.linalg.norm(state['balance'])*jnp.linalg.norm(second_coarse))
    row = dict(resolved=resolved, F_norm=jnp.sqrt(f2), fine_norm=Y, B=B, M=M,
               coarse_sigma=sigma, half_norm_squared_dot=rhs,
               half_norm_squared_dot_jvp=direct, log_force_rate=ratio(rhs),
               log_force_rate_jvp=ratio(direct), geometry_total=geometry,
               compensation_total=compensation, geometry_rate=ratio(geometry),
               compensation_rate=ratio(compensation),
               curvature_rate=ratio(geometry+compensation),
               generated_rate=ratio(terms['generated_geometry']+terms['generated_compensation']),
               target_rate=ratio(terms['target_geometry']+terms['target_compensation']),
               directional_curvature_rate_bound=directional,
               structural_geometry_rate_bound=geometric_bound,
               structural_compensation_rate_bound=compensation_bound,
               structural_curvature_rate_bound=geometric_bound+compensation_bound,
               measured_ell_compensation_rate_bound=jnp.linalg.norm(state['balance'])*HC,
               structural_bound_minus_signed_rate=geometric_bound+compensation_bound-ratio(rhs),
               identity_absolute_error=jnp.abs(rhs-direct),
               identity_rate_error=jnp.abs(ratio(rhs-direct)),
               curvature_split_absolute_error=jnp.abs(sum(terms.values())-rhs),
               coarse_tangency_norm=jnp.linalg.norm(jc@F))
    row.update(terms)
    row.update({key+'_rate': ratio(value) for key, value in terms.items()})
    row['absolute_split_rate_sum'] = sum(jnp.abs(row[key+'_rate']) for key in terms)
    omega = ratio(width*jnp.sum(r2*force_squared))
    m4 = width*jnp.sum(r2**2)
    transport = ratio(-2*width*jnp.sum(jnp.sum(particle_p*particle_F, axis=0)*force_squared))
    def redistribution_from(force_derivative):
        blocks = force_derivative[:-1].reshape(3, -1)
        return (ratio(2*width*jnp.sum(r2*jnp.sum(particle_F*blocks, axis=0)))
                -2*omega*ratio(F@force_derivative))

    redistribution = redistribution_from(derivative)
    # The residual-only term in dot F is -Pi J_H^* J_H F. All remaining
    # terms include evolving geometry/projector compensation. These probes
    # follow effective flow -F, not the archived state's full-GD velocity.
    relax_gradient = J.T@fine_velocity/len(x)
    derivative_relaxation = -relax_gradient+jc.T@jnp.linalg.solve(gram, jc@relax_gradient)
    redistribution_relaxation = redistribution_from(derivative_relaxation)
    redistribution_other = redistribution_from(derivative-derivative_relaxation)
    participation = ratio(width*jnp.sum(force_squared**2))/jnp.where(f2 > 0, f2, 1.)
    participation_dot = (ratio(4*width*jnp.sum(force_squared*jnp.sum(particle_F*particle_derivative, axis=0)))
                         /jnp.where(f2 > 0, f2, 1.)-4*participation*ratio(direct))
    f = jnp.sqrt(f2)

    def normalized(value, denominator):
        return jnp.where(resolved&(denominator > 0), value/jnp.where(denominator > 0, denominator, 1.), jnp.nan)

    kappa_omega = normalized(redistribution, 2*f*omega)
    kappa_participation = normalized(participation_dot, 4*f*participation)
    coarse_direction_bound = jnp.sqrt(2.)+4*jnp.sqrt(omega/width)
    ell_m4_bound = 3*jnp.sqrt(2.)*Y*m4/jnp.where(eig[0] > 0, eig[0]*width, 1.)
    row.update(force_weighted_second_moment=omega, M4=m4,
               force_weighted_fourth_moment=ratio(width**2*jnp.sum(r2**2*force_squared)),
               hidden_force_energy_concentration=participation,
               hidden_force_energy_concentration_dot=participation_dot,
               omega_dot_transport=transport, omega_dot_redistribution=redistribution,
               omega_dot_effective=transport+redistribution,
               omega_dot_redistribution_relaxation=redistribution_relaxation,
               omega_dot_redistribution_geometry_compensation=redistribution_other,
               omega_redistribution_split_absolute_error=jnp.abs(redistribution-redistribution_relaxation-redistribution_other),
               kappaOmega_signed=kappa_omega, kappaOmega_positive=jnp.maximum(kappa_omega, 0.),
               kappaI_signed=kappa_participation, kappaI_positive=jnp.maximum(kappa_participation, 0.),
               transport_saturation=normalized(transport, 2*f*jnp.sqrt(omega*participation)),
               dlogOmega_pertravel=normalized(transport+redistribution, f*omega),
               sqrt_M4_dot_effective=-2*width*jnp.sum(r2*jnp.sum(particle_p*particle_F, axis=0))/jnp.sqrt(m4),
               sqrt_M4_speed_bound=2*jnp.sqrt(omega*f2),
               weighted_geometry_rate_bound=6*jnp.sqrt(2.)*Y*omega/width,
               weighted_compensation_rate_bound=ell_m4_bound*coarse_direction_bound,
               weighted_curvature_rate_bound=6*jnp.sqrt(2.)*Y*omega/width+ell_m4_bound*coarse_direction_bound,
               measured_ell_norm=jnp.linalg.norm(state['balance']), ell_M4_bound=ell_m4_bound,
               particle_force_energy_fraction=ratio(jnp.sum(force_squared)))
    return row


@lru_cache(maxsize=4)
def load_input(path):
    return pc.load(path)


@lru_cache(maxsize=4)
def load_snapshot(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in ('p', 'failed') if key in data}


def load_state(row):
    """Resolve the two archive layouts already used by population_output_audit."""
    source, index = Path(row['source']), int(row['index'])
    paths = [source]
    if row['role'] == 'static':
        pack, _ = load_input(str(source))
        p, x, y = pack['p'][index], pack['x'], pack['y'][index]
    elif row['role'] == 'trajectory':
        data = load_snapshot(str(source))
        if 'failed' in data and data['failed'][index]:
            raise ValueError('Archived branch is marked failed.')
        p = data['p'][index]
        if source.parent.name == 'snapshots':
            run = source.parent.parent
            if not run.name.endswith('_run'):
                raise ValueError('Expected existing natural-run snapshot layout.')
            inputs = run.with_name(run.name[:-4])/'inputs.npz'
            target_index = index
        else:
            manifest_path = source.parent/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            paths.append(manifest_path)
            inputs = Path(manifest['input'])
            target_index = manifest['cases'][index]['input_index']
        pack, _ = load_input(str(inputs))
        paths.append(inputs)
        x, y = pack['x'], pack['y'][target_index]
    else:
        raise ValueError('Unknown archive role.')
    if not all(np.all(np.isfinite(value)) for value in (p, x, y)):
        raise ValueError('Nonfinite archived state or data.')
    digest = hashlib.sha256(p.tobytes()+x.tobytes()+y.tobytes()).hexdigest()
    if digest != row['state_sha256']:
        raise ValueError('State/data hash differs from the source audit.')
    if abs(np.mean(x)) > 1e-13 or np.max(abs(x)) > 1+1e-14:
        raise ValueError('Expected centered inputs in [-1,1].')
    return p, x, y, paths


def audit(args):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true.')
    validate_backend(args.backend)
    with args.source.open(newline='') as stream:
        inputs = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=False)
    measure = jax.jit(observables)
    rows, sources, counts = [], {str(args.source): pc.digest(args.source)}, {}
    for index, original in enumerate(inputs):
        if args.role != 'all' and original['role'] != args.role:
            continue
        row = dict(original, reinforcement_source_row=index)
        try:
            p, x, y, paths = load_state(original)
        except (ValueError, OSError, KeyError, IndexError) as error:
            row.update(reinforcement_status='archive_load_failure', reinforcement_reason=str(error))
        else:
            for path in paths:
                if str(path) not in sources:
                    sources[str(path)] = pc.digest(path)
            values = {key: np.asarray(value).item() for key, value in measure(p, x, y).items()}
            row.update({'reinforcement_'+key: value for key, value in values.items()})
            status = ('unresolved_coarse_solve' if not values['resolved'] else
                      'stationary_effective_flow' if values['F_norm'] == 0 else 'finite')
            if status == 'finite' and not all(np.isfinite(value) for value in values.values()):
                status = 'nonfinite_diagnostic'
            row['reinforcement_status'] = status
            if original.get('F_norm'):
                archived = float(original['F_norm'])
                difference = abs(values['F_norm']-archived)
                row['reinforcement_archived_force_difference'] = difference
                row['reinforcement_archived_force_relative_difference'] = difference/archived if archived > 0 else None
        rows.append(row)
        status = row['reinforcement_status']
        counts[status] = counts.get(status, 0)+1
        if len(rows) % 25 == 0:
            pc.write_csv(args.output/'states.csv', rows)
            print(json.dumps(dict(rows=len(rows), statuses=counts)), flush=True)
    pc.write_csv(args.output/'states.csv', rows)
    manifest = dict(rows=len(rows), statuses=counts, sources=sources, backend=jax.default_backend(),
                    helper_sha256=pc.digest(__file__), kernel_sha256=pc.digest(kernel.__file__),
                    role='retrospective exact-state effective-flow scalar derivatives; no future certificate',
                    derivative_convention='half d||F||^2/dt and dlog||F||/dt along velocity -F',
                    scalar_scope='whole-parameter effective norm; not a signed slope-acquisition rate')
    (args.output/'manifest.json').write_text(json.dumps(pc.clean(manifest), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=('all', 'static', 'trajectory'), default='all')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    audit(parser.parse_args())
