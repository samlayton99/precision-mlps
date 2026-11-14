"""Compare saved original/sharpened cutoff runs and both fixed-cutoff fits."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.readout_norms import draw


def main():
    base=r.OUT
    original=base/'option_c_review_64_768'
    sharp=base/'option_c_sharp_bound_64_768'
    prior=base/'readout_l1_three_methods_64_768'/'norms.json'
    out=base/'readout_l1_sharp_bound_64_768';out.mkdir(exist_ok=False)
    rows=json.loads(prior.read_text());cache={};inputs={str(prior):hashlib.sha256(prior.read_bytes()).hexdigest()}
    fitted=[]
    for N in sorted({x['N'] for x in rows}):
        coeff_path=sharp/f'coefficients_N{N}.npz';json_path=sharp/f'N{N}.json'
        with np.load(coeff_path) as ss,np.load(original/f'coefficients_N{N}.npz') as old:
            assert np.array_equal(ss['actual'],old['actual'])
            assert np.array_equal(ss['D'],old['D'])
            assert np.array_equal(ss['singular'],old['singular'])
            cache[N]=ss['theory'].copy()
        results=json.loads(json_path.read_text())
        for x in results:
            assert all(x['geometry_checks'].values())
            assert all(x['theory'][key] for key in ['cutoff_pass','transformed_norm_pass','sample_bound_pass','coefficient_bound_pass'])
            assert x['approximation_bound']['E_per_B_upper']<=x['approximation_bound']['original_E_per_B_upper']
        fitted.extend(results)
        for f in [coeff_path,json_path]:inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for row in rows:
        j=r.NAMES.index(row['target']);c=cache[row['N']][:,j]
        row['scaled_sharp']=dict(l1=float(np.sum(abs(c))),neuron_weights_l1=float(np.sum(abs(c[:-1]))),absolute_bias=float(abs(c[-1])))
    (out/'norms.json').write_text(json.dumps(rows,indent=2)+'\n')
    draw(out,rows)
    files=[Path(__file__),Path(__file__).with_name('readout_norms.py')]
    snap=out/'plot_source';snap.mkdir()
    for f in files:(snap/f.name).write_bytes(f.read_bytes())
    (out/'provenance.json').write_text(json.dumps(dict(inputs=inputs,
        plot_sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        methods='All four readout arms use FP64 DGEJSV; fixed relative cutoff=2**-52.',
        measure='L1 of native output weights including bias; W counts hidden neurons.'),indent=2)+'\n')
    summary=dict(pairs=len(fitted),all_recovery_checks_pass=True,
        all_unscaled_coefficients_and_scaled_singular_values_unchanged=True,
        max_sharp_error_N_ge_256=max(x['theory_error'] for x in fitted if x['N']>=256),
        max_validation_relative_change=max(abs(x['theory_validation_error']/x['theory_error']-1) for x in fitted))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
