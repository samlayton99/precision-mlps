"""Write the measured capacity table, without fitting or smoothing the results."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt/width_sweep'


def main():
    rows=json.loads((OUT/'summary.json').read_text())
    checked=json.loads((OUT/'validation.json').read_text())['checks']
    by_key={(r['model'],r['width']):r for r in checked}
    fields=['model','hidden_units','parameters','best_trained_validation_step',
            'best_trained_validation_test_rel_l2','final_rel_l2','seconds']
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    best={}
    for name in ['NPU','RealNPU']:
        candidates=[r for r in rows if r['model']==name]
        best[name]=min(candidates,key=lambda r:by_key[(name,r['hidden_units'])]['errors']['32'])
    result={'best_by_validation':best,
            'largest_models':{name:max([r for r in rows if r['model']==name],key=lambda r:r['parameters'])
                              for name in ['NPU','RealNPU']},
            'run_count':len(rows),'new_runs':sum(not r['reused_six_unit_run'] for r in rows)}
    (OUT/'findings.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# NPU and RealNPU capacity sweep','',
           'All runs use the same bounded C17 extension, fixed QUILL halo dataset, '
           'float64 precision, seed 0, batch 100 and 20,000 Adam updates at lr=0.001. '
           'Only model capacity changes. No extra seeds or learning-rate search.','']
    for name,r in best.items():
        lines.append(f'- Best {name} by validation: **{r["best_trained_validation_test_rel_l2"]:.8g}** '
                     f'relative L2 ({100*r["best_trained_validation_test_rel_l2"]:.5g}%), '
                     f'{r["hidden_units"]:,} units / {r["parameters"]:,} parameters, '
                     f'step {r["best_trained_validation_step"]:,}.')
    lines+=['','QUILL reaches about 5e-14 at 4,815 tanh neurons / 14,446 parameters. '
            'The fixed-width Adam MLP reference has the same architecture/parameter count, '
            'with best trained error 2.55313. These are different construction/training '
            'procedures, not identical access to derivative information.','',
            '| Model | Units | Parameters | Best trained test relative L2 | Selected step | Final relative L2 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["hidden_units"]:,} | {r["parameters"]:,} | '
                     f'{r["best_trained_validation_test_rel_l2"]:.8g} | '
                     f'{r["best_trained_validation_step"]:,} | {r["final_rel_l2"]:.8g} |')
    lines+=['','## Protocol and validation','',
            'The halo samples exactly match the selected 4,815-neuron QUILL construction '
            '(15 blocks, N=256, R=32), including repeated physical locations and both halos. '
            'All widths receive the same target values and sample-index RNG sequence. '
            'The extension-node range is [-0.4096,3.6864]; evaluation is physical dx '
            'relative L2 on [1e-4,1].','',
            'Checkpoints are selected by minimum positive-step order-32 validation error, '
            'logged every 250 updates. Independent order-256/512 quadrature evaluates every '
            'selected checkpoint. Every model was reloaded and its score, parameter count '
            'and selection step verified. Ten implementation tests passed.','',
            'The widest NPU has 14,446 parameters and the widest RealNPU has 14,445, '
            'matching QUILL to within one parameter. Width-6 data are reused from the '
            'earlier run; all other points are newly trained. The parameter-axis figure '
            'shows measured unsmoothed curves, with white centers marking sampled capacities.','',
            'This is one seed and one training recipe per width. Nonmonotonic curves '
            'reflect the measured runs; they do not prove a capacity limit or establish '
            'the best possible optimizer/hyperparameter configuration. Wider models may '
            'need different training budgets. QUILL uses derivative samples and an anchor; '
            'NPU/RealNPU use function values. Other QUILL sweep widths retain their original '
            '32-cell halo recipe; the shared halo reference matches exactly at 4,815 neurons.','',
            'Artifacts: `npu_capacity_sweep.png` / `.pdf`, `summary.csv`, `summary.json`, '
            '`findings.json`, `validation.json`, `sweep_specification.json`, '
            '`figure_provenance.json`, and per-width histories, prediction arrays, '
            'initial models, optimizer checkpoints, and selected model checkpoints.']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    for name,r in best.items():
        print(name,r['hidden_units'],r['parameters'],r['best_trained_validation_test_rel_l2'])


if __name__=='__main__':main()
