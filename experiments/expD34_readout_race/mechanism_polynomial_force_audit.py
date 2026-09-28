"""Separate polynomial truncation from omitted tracking at the original forks."""
import argparse
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from . import effective_feedback as ef, transport
from .mechanism_polynomial import modal_setup, field
from .mechanism_splitting_baselines import matrices
from .mechanism_splitting_diagnostics import write_csv


def audit(root):
    rows = []
    for n in (128, 512, 1024):
        pp, x, yy, cases = ef.load_inputs(root/'width_inputs'/f'N{n}_fork20000.npz')
        q = transport.basis(x, 65)
        for p, y, case in zip(pp, yy, cases):
            w = (len(p)-1)//3
            g, T, _, e = matrices(p, x, y, np.ones(len(p)), q)
            exact = T@e
            for degree in (3, 5):
                transform, target = modal_setup(x, y, degree)
                force = np.asarray(field(jnp.asarray(p), jnp.asarray(transform), jnp.asarray(target), degree)[0])
                row = dict(case, degree=degree)
                for label, section in [('all', slice(None)), ('slope', slice(0, w))]:
                    norm = np.linalg.norm(exact[section])
                    error = np.linalg.norm((force-exact)[section])
                    tracking = np.linalg.norm((g-exact)[section])
                    row[label+'_exact_force_norm'] = float(norm)
                    row[label+'_polynomial_error_norm'] = float(error)
                    row[label+'_polynomial_relative_error'] = float(error/norm) if norm else np.nan
                    row[label+'_tracking_norm'] = float(tracking)
                    row[label+'_tracking_relative'] = float(tracking/norm) if norm else np.nan
                rows.append(row)
    write_csv(root/'polynomial/force_audit.csv', rows)
    for row in rows:
        if row['target'] == 'moment5':
            print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
