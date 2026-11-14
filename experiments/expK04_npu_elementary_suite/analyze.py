"""Validate the small NPU screen and write its illustrative results."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK04_npu_elementary_suite.run import OUT,ACTIVE_TARGETS,save
from experiments.expK04_npu_elementary_suite.models import build
from experiments.expK04_npu_elementary_suite.compact_check import factory
from experiments.expK04_npu_elementary_suite.problem import grid,score


def main():
    torch.set_num_threads(1)
    stages=['screen','repeats','uniform_controls','compact_check','selected_recipe_rechecks']
    rows=[{**r,'stage':stage} for stage in stages for r in json.loads((OUT/stage/'summary.json').read_text())]
    primary=json.loads((OUT/'final_primary_recipes.json').read_text())
    checks=[];flagged=[]
    for r in rows:
        folder=ROOT/r['run_directory'];checkpoint=torch.load(folder/'selected.pt',weights_only=False)
        m=(factory if r['model']=='compact_stack' else build)(r['model'],r['seed'])
        m.load_state_dict(checkpoint['model'])
        assert sum(p.numel() for p in m.parameters() if p.requires_grad)==r['parameters']
        errors={str(n):score(m,grid(r['target'],n)) for n in [256,512]}
        np.testing.assert_allclose(errors['256'],r['test_relative_l2'],rtol=1e-10,atol=2e-15)
        stable=abs(errors['256']-errors['512'])<=max(1e-12,1e-5*max(errors.values()))
        check={'target':r['target'],'model':r['model'],'seed':r['seed'],'learning_rate':r['learning_rate'],
               'sampling':r['sampling'],'quadrature_agrees':stable,'errors':errors,'run_directory':r['run_directory']}
        checks.append(check)
        if not stable:flagged.append(check)
        r['quadrature_agrees']=stable
    table=[]
    for name,chosen in primary.items():
        first=next(r for r in rows if r['run_directory']==chosen['run_directory'])
        repeat=next(r for r in rows if r['target']==name and r['seed']==1 and
                    all(r[k]==chosen[k] for k in ['model','learning_rate','sampling']))
        assert first['quadrature_agrees'] and repeat['quadrature_agrees']
        lowest=min((r for r in rows if r['target']==name and r['quadrature_agrees']),key=lambda r:r['test_relative_l2'])
        table.append({'target':name,'model':chosen['model'],'parameters':chosen['parameters'],
                      'learning_rate':chosen['learning_rate'],'sampling':chosen['sampling'],
                      'seed0_adam_error':first['adam_best_test'],'seed0_polished_error':first['test_relative_l2'],
                      'seed1_same_recipe_error':repeat['test_relative_l2'],
                      'lowest_observed_any_seed':lowest['test_relative_l2'],
                      'lowest_observed_model':lowest['model'],'lowest_observed_seed':lowest['seed']})
    save(OUT/'findings.json',{'illustrative_table':table,'total_runs':len(rows),'quadrature_flags':flagged,
                           'all_selected_results_and_repeats_verified':True})
    save(OUT/'validation.json',{'checks':checks,'note':'Some nonselected stacked fits fail quadrature stability; those are flagged, not used as precision evidence.'})
    fields=['target','model','seed','sampling','learning_rate','parameters','adam_best_test',
            'test_relative_l2','selected_phase','adam_status','lbfgs_status','quadrature_agrees','run_directory']
    with (OUT/'all_runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (OUT/'illustrative_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    names={'inv_sqrt':'1/sqrt(x)','root_plus_linear':'1/(sqrt(x)+x)',
           'exp_over_root':'exp(-x)/sqrt(x)','log_over_root':'-log(x)/sqrt(x)'}
    lines=['# Small NPU screen near the singularity','',
           'Four functions on **[0.0001,1]**, evaluated with physical-dx relative L2. '
           'No halo or continuation is imposed on NPU. This is an illustrative optimization '
           'screen, not a proof of an NPU approximation limit or a success-rate estimate.','',
           '| Function | Selected architecture | Seed 0, Adam | Seed 0, best after polish | Same recipe, seed 1 |',
           '|---|---|---:|---:|---:|']
    for r in table:
        lines.append(f'| {names[r["target"]]} | {r["model"]} ({r["parameters"]} parameters) | '
                     f'{r["seed0_adam_error"]:.6g} | {r["seed0_polished_error"]:.6g} | '
                     f'{r["seed1_same_recipe_error"]:.6g} |')
    lines+=['','The pure power reproduces near-machine precision. None of the three modified '
            'functions reached 1e-12 in this screen. Errors near 1e-4 are good ordinary '
            'approximations, but do not establish the high-precision behavior sought here. '
            'The log/root example is particularly initialization-sensitive. A separate '
            'shallow-six-unit repeat reached about 1.66e-4 on it, while the selected five-parameter '
            'compact architecture scored about 0.119 on its fresh-seed repeat. All runs are retained.','',
            '## Protocol','',
            '- Initial screen: shallow 6-unit and 32-unit NPU-to-linear models, plus NPU(1,8) → linear(8,4) → NPU(4,4) → linear(4,1). Standard Glorot real/readout weights, imaginary weights zero, learned gates initialized at 0.5.',
            '- Two Adam rates (0.001 and 0.01), 20k updates, batch 100. Hold the rate for 10k, then cosine decay to 1e-6. Physical inputs remain x; log coordinates are used for sampling only.',
            '- Log sampling uses the importance weight x*log(1/a)/integral(f^2 dx), so expected loss is exactly physical relative-L2 squared. It does not silently replace the objective with unweighted log-domain MSE.',
            '- Polish the validation-best Adam model with ordinary full-batch L-BFGS, up to 500 iterations, strong-Wolfe search, history 50. Train on order-32 composite log-panel quadrature, validate on order 64; report independent order-256/512 scores.',
            '- Three scalar inverse-root controls use uniform sampling: fixed gate with zero/random exponent, and learned gate with random exponent. Exponents are learned; the correct value is not used to initialize training.',
            '- A follow-up compact stack learns five parameters in a sum-of-two-powers followed by a power. It uses random Glorot weights and fixed input gates at 1. The architecture is generic and was tested on all four functions.',
            '- Recipe selection uses seed-0 validation scores. One fresh seed checks each selected recipe. The expanded initial plan was reduced at the user\'s request; the other target functions in problem.py were not run.','',
            'Positive-input shallow NPUs with zero imaginary initialization remain on their '
            'RealNPU branch. The larger stack can activate imaginary parameters on signed '
            'intermediate inputs. No target-specific feature engineering or function-value '
            'construction initializes any trained model.','',
            '## Interpreting the algebraic example','',
            '1/(sqrt(x)+x) is exactly representable (up to stabilization/rounding) by the '
            'small compositional NPU architecture. A test verifies this by assigning its '
            'known parameters; those values are NOT used in training. Its observed training '
            'error therefore demonstrates optimization difficulty under these budgets, '
            'not a representational impossibility.','',
            '## Checks and limitations','',
            'Six implementation tests passed: the general physical-L2 metric, importance '
            'weight identity, native power representation, compositional representation, '
            'finite initial gradients, and an end-to-end training smoke test. Every saved '
            'selected checkpoint was reloaded. The reported primary and fresh-seed results '
            'agree across quadrature orders 256/512. Some nonselected large stacked fits '
            'show quadrature sensitivity and are flagged in validation.json; their apparent '
            'low sampled losses are not accepted as high-precision solutions.','',
            'This modest search does not rule out better NPU results with other starts, '
            'architectures or optimizers. QUILL has not yet been run on the three new '
            'functions; no QUILL superiority result on them is claimed. The natural '
            'transcendental examples exp(-x)/sqrt(x) and -log(x)/sqrt(x) are useful next '
            'candidates for such a comparison.','',
            'The paper\'s published benchmarks are different: joint arithmetic/extrapolation '
            'tasks and Float32 in the released code. The tiny fixed-gate Float64 power '
            'control estimates one exponent, which explains its much better precision '
            'without contradicting the paper\'s aggregate errors.','',
            '## Saved artifacts','',
            'all_runs.csv, illustrative_results.csv, findings.json, validation.json, '
            'specification.json, original_source_manifest.json and per-run configs, '
            'initial/final/selected checkpoints, histories and dense predictions. '
            'The compact follow-up has its own specification.']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(table,indent=2));print('Quadrature-sensitive nonselected fits:',len(flagged))


if __name__=='__main__':main()
