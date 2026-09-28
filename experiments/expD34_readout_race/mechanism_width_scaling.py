"""Reconstructed reproducibility check for the inline job-1247 scaling audit.

The original executed source is unavailable. This source independently reproduces
its scalar CSV from preserved inputs; fine-error change is a frozen-S forecast,
not an observed change of the trained network residual.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import effective_feedback as ef, transport
from .mechanism_splitting_baselines import matrices
from .mechanism_splitting_diagnostics import write_csv


def run(root):
    old = root/'diagnostics/width_scaling.csv'
    with old.open() as stream:
        reference = list(csv.DictReader(stream))
    rows = []
    for n in (128, 512, 1024):
        pp, x, yy, cases = ef.load_inputs(root/'width_inputs'/f'N{n}_fork20000.npz')
        q = transport.basis(x, 65)
        for p, y, case in zip(pp, yy, cases):
            w = (len(p)-1)//3; a, b, c = p[:-1].reshape(3, w)
            u = x[:, None]*a+b
            ex = np.exp(-2*abs(u)); der = 4*ex/(1+ex)**2
            J = np.concatenate((x[:, None]*der*c, der*c, np.tanh(u), np.ones((len(x), 1))), axis=1)
            JC = q[:, :2].T@J/len(x)
            _, T, JH, e = matrices(p, x, y, np.ones(len(p)), q)
            S = JH@T; F = T@e; ceig = np.linalg.eigvalsh(JC@JC.T)
            e_next = np.linalg.matrix_power(np.eye(len(e))-.002*S, 20000)@e
            row = dict(case, W=w, eta_horizon_lambda_max_S=float(40*np.linalg.eigvalsh(S)[-1]),
                fine_error_fraction_change=float(np.linalg.norm(e_next-e)/np.linalg.norm(e)),
                coarse_K_min=float(ceig[0]), coarse_K_max=float(ceig[-1]),
                W15_rms_F_a=float(w**1.5*np.sqrt(np.mean(F[:w]**2))),
                W15_max_F_a=float(w**1.5*abs(F[:w]).max()),
                W_fine_J_fro=float(w*np.linalg.norm(JH)), sqrtW_fine_J_fro=float(np.sqrt(w)*np.linalg.norm(JH)),
                fine_J_fro=float(np.linalg.norm(JH)), effective_lambda_rate_rms=float(2/n*np.sqrt(np.mean(F[:w]**2))))
            for label, block, section in zip(('a', 'b', 'c'), (a,b,c), (slice(0,w),slice(w,2*w),slice(2*w,3*w))):
                row['sqrtW_rms_'+label] = float(np.sqrt(w*np.mean(block**2)))
                row['sqrtW_max_'+label] = float(np.sqrt(w)*abs(block).max())
                row['sqrtW_fine_J_'+label+'_fro'] = float(np.sqrt(w)*np.linalg.norm(JH[:, section]))
            row['sqrtW_fine_J_d_fro'] = float(np.sqrt(w)*np.linalg.norm(JH[:, -1]))
            rows.append(row)
    lookup = {(r['target'], int(r['seed']), int(r['nref'])): r for r in reference}
    discrepancies = {}
    for row in rows:
        oldrow = lookup[(row['target'], row['seed'], row['nref'])]
        for key, value in row.items():
            if isinstance(value, (int, float)) and key in oldrow:
                difference = abs(value-float(oldrow[key]))
                discrepancies[key] = max(discrepancies.get(key, 0.), difference)
                np.testing.assert_allclose(value, float(oldrow[key]), rtol=1e-6, atol=5e-12, err_msg=f'{row["target"]}/{row["seed"]}/{row["nref"]}/{key}')
    write_csv(root/'diagnostics/width_scaling_reproduced.csv', rows)
    record = dict(original_source='inline source unavailable; original job1247 log preserved',
        reconstructed_source_sha256=ef.digest(__file__), reference_sha256=ef.digest(old),
        reproduced_sha256=ef.digest(root/'diagnostics/width_scaling_reproduced.csv'),
        max_absolute_discrepancy_by_column=discrepancies, cases=len(rows), degree=65,
        residual_change_meaning='norm(((I-0.002*S)^20000-I)e0)/norm(e0): frozen-S model forecast, not observed residual evolution')
    (root/'diagnostics/width_scaling_reproduction.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
