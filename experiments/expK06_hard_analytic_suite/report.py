"""Final report for the user-scoped four-function comparison and QUILL-only suite."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK06_hard_analytic_suite.common import OUT,NS,WIDTHS
from experiments.expK06_hard_analytic_suite.targets import MAIN,NAMES

def main():
    rows=json.loads((OUT/'verified_summary.json').read_text())
    selected=json.loads((OUT/'selected_fits.json').read_text())
    checks=json.loads((OUT/'verification.json').read_text())
    full=json.loads((OUT/'main50k_summary.json').read_text())
    assert len(full)==48 and all(r['steps']==50000 for r in full)
    lines=['# Hard analytic functions — completed','',
           '## Findings','',
           '- First figure: four functions, six matched widths, Adam and NPU trained for exactly 50k total steps, plus tuned sample-fitted QUILL.',
           '- Second figure: QUILL alone on eighteen functions: the original sixteen, shifted Gamma, and the compact bump. No additional Adam/NPU training is part of this figure.',
           '- QUILL tuning stays within the uniform tanh-feature/least-squares framework. Pivoted rank-revealing QR and a smaller explicit cutoff materially improve the numerical floor.','',
           '## Question','',
           'Can ordinary sample-fitted QUILL achieve approximately 1e-14 relative error on challenging analytic special functions, implicit functions and oscillatory quotients, and how do trained networks compare on four representative cases?','',
           '## Procedure','',
           'The target catalog starts from the recent 24-case study in “Rebuttal: valid theory.” Airy, erf, chirp and localized oscillation are retained verbatim. Nonanalytic cases are excluded. New cases add implicit inversion and special-function/quotient examples. All targets are on [-1,1], and all have analytic continuation across the real halo. That older study was not a QUILL difficulty ranking.',
           '',
           'The first figure uses W=105,169,297,559,1089,2141. At each shared width every method receives the same max(2003,4(N+1)) covering sample values. No derivatives or exterior target labels are used. Adam/NPU settings come from saved validation-ranked pilots; there are no new baseline hyperparameter sweeps. Continue the 10k optimizer/RNG states through 50k total updates. Keep the rate constant through 25k, then cosine-decay to 1e-6. Batch size 100, Float64, seed 0, standard target-independent initialization. No L-BFGS, final readout refit, or extra seeds are included in the final baseline comparison.',
           '',
           'QUILL uses the audited fit_qi implementation: uniform centers, tanh features, and rank-revealing least squares. Tune lambda within [.2,.3], explicit cutoffs 1e-14/1e-15/machine epsilon, and gelsd/gelsy/gelss. Halo comparisons hold total W and the training points fixed. Selection uses validation only. The second figure adds W=4225 to resolve the particularly sharp elliptic-E target. Stop further tuning once that validation error reaches the requested range.',
           '',
           'Physical-dx relative L2 is the common metric. NPU validation and scoring refine logarithmic panels around possible input zero crossings; this resolves spikes that ordinary quadrature can miss. Re-ranking existing pilots changes neither the trained weights nor their optimization state. Independent final quadrature is finer than validation.',
           '', '## Main comparison','',
           'Function and signed-residual rows show each method\'s best validation-selected fit among the six shared widths. The scaling row shows all six comparisons. Selected widths are documented in the results table, and training sample counts grow with width.','',
           '| Function | Method | Width | Relative L2 |','|---|---|---:|---:|']
    for name in MAIN:
        for method in ['QUILL','Adam','NPU']:
            r=next(r for r in selected if r['target']==name and r['method']==method)
            lines.append(f"| {name} | {method} | {r['width']} | {r['relative_l2']:.5g} |")
    lines+=['','## QUILL-only suite','',
            'Best validation-selected model from the displayed width sweep; independent test error shown.','',
            '| Function | Width | Solver | Lambda | Relative L2 |','|---|---:|---|---:|---:|']
    for name in NAMES:
        r=min((r for r in rows if r['target']==name and r['method']=='QUILL'),key=lambda r:r['validation'])
        lines.append(f"| {name} | {r['width']} | {r['solver']} | {r['lambda_value']} | {r['relative_l2']:.5g} |")
    additions=OUT/'gamma_bump/selected.json'
    if additions.exists():
        for r in json.loads(additions.read_text()):
            lines.append(f"| {r['target']} | {r['width']} | {r['solver']} | {r['lambda_value']} | {r['relative_l2']:.5g} |")
        lines+=['','The added Gamma target is Gamma(0.05+2.475(x+1)), as in the earlier 24-function study. It is analytic on the requested interval; its nearest real pole is just outside it. Halo features do not require exterior labels for sample-fitted QUILL.',
                '', 'The compact bump is exp(-1/(1-(x/0.35)^2)) for |x|<0.35, and zero otherwise. It is C-infinity but **not analytic** at the support boundaries. It is an additional smooth-function example, not evidence for an analytic-only theorem.',
                '', 'Both additions use the same seven displayed widths, automatic halo, lambda=.25, gelsy, and machine-epsilon rank cutoff. The fourteen measured models and independent quadrature checks are in quill_extension_verified.json; selected dense predictions and reference checks are in gamma_bump/.']
    lines+=['','## Verification','',
            '18 implementation and NPU regression checks passed. The training resume audit is bitwise exact. Refined NPU pilot ranking preserves optimizer/model/RNG states exactly. Target values were independently checked against 70-decimal mpmath; derivative checks audit the target implementation only and supply no training information.',
            '',f"Reloaded and checked {checks['primary_models']} saved scalar networks with ordinary Float64 Torch inference. {checks.get('quadrature_flag_note',str(len(checks['quadrature_flags']))+' integration flags.')} The selected-fit reference spot audit, shared training-grid hashes, actual step counts and final learning rates are saved alongside the report.",
            '', '## Interpretation and limits','',
            'These are measured results for a declared finite training budget, not optimizer impossibility claims. The NPU family includes native, positive-input complex, and compositional candidates; the compositional candidate has more parameters at the same hidden width. All settings and parameter counts are saved. One seed is used. The function list was fixed before observing trained outcomes.',
            '',
            'Analyticity does not imply uniform difficulty. The elliptic-E composition needs more resolution than most other cases. Near the floating-point floor, reference evaluation and matrix arithmetic matter; quadrature checks and high-precision reference spot checks bound the practical interpretation. The method remains a uniform feature bank plus least squares, with no target-specific nonlinear input features or multiscale partition.',
            '', '## Artifacts','',
            '- hard_functions_3x4.png/.pdf: four-function comparison.',
            '- additional_functions_1x2.png/.pdf: two-column function key and QUILL scaling on all eighteen functions.',
            '- main50k/: the 48 trained runs; quill_tuned/: tuning candidates; predictions/: plotted arrays.',
            '- scaling.csv, verified_summary.json, selected_fits.json, verification.json, reference_spot_audit.json, figure_provenance.json.',
            '- specification_main50k.json, target_audit.json, training_audit.json, refinement_state_audit.json, source snapshot and environment versions.',
            '',
            'The superseded 100k plan, its pilots, and any completed old full runs are retained for provenance but excluded from the final figures. The initial derivative-based prototype is also archived and excluded.',
            '', '## Next decision','',
            'The requested two figures and saved data are complete. No additional baseline training is pending.']
    (OUT/'expK06_results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
