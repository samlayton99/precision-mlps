"""Export case identities and common-horizon measurements, without report prose."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import design, history, core


def training_window(folder,horizon):
    """All update-start losses in the final fifth; disclose missing trace coverage."""
    lower=int(.8*horizon);blocks=[]
    for _,data in history.arrays(folder,'trace_*.npz'):
        start=int(data['start']);end=int(data['end'])
        if end<=lower or start>=horizon:continue
        block=data['trace'][max(0,lower-start):min(end,horizon)-start]
        blocks.append(block[:,:18])  # Stable prefix shared by every campaign trace.
    result=dict(id=folder.name,horizon=horizon,expected_samples=horizon-lower)
    if not blocks:return dict(result,samples=0)
    a=np.concatenate(blocks);finite=np.isfinite(a[:,0]);values=a[finite,0]
    result.update(samples=len(a),finite_samples=len(values),coverage=len(a)/(horizon-lower))
    if not len(values):return result
    result.update(mean_train_mse=float(np.mean(values)),median_train_mse=float(np.median(values)),
                  q90_train_mse=float(np.quantile(values,.9)),max_train_mse=float(np.max(values)))
    for name in ('eta','physical_readout_gradient','physical_gamma_gradient','delta_readout_rms',
                 'delta_gamma_rms','coarse_residual_norm','adam_guard_fraction','zero_physical_step'):
        result['mean_'+name]=float(np.mean(a[finite,core.TRACE.index(name)]))
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(root, out):
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in sorted(root.glob('*/case.json')):
        folder = path.parent
        if not (folder / 'latest.json').exists():
            continue
        config = json.loads(path.read_text())
        latest = json.loads((folder / 'latest.json').read_text())
        row = dict(id=folder.name, **{k: v for k, v in config.items()
                                      if not isinstance(v, (dict, list))})
        origin = config.get('origin', {})
        row.update(parent_checkpoint=origin.get('checkpoint', ''),
                   parent_sha256=origin.get('sha256', ''),
                   carry_optimizer=origin.get('carry_optimizer', ''),
                   reset_filter=origin.get('reset_filter', True) if origin else '',
                   saved_step=latest['step'], failed_update=latest.get('failed_update', 0),
                   status=latest.get('status', ''),
                   final_validation_mse=latest.get('validation_mse'),
                   final_relative_mse=latest.get('validation_relative_mse'))
        failed = latest.get('failed_update', 0)
        row['finite_updates'] = min(latest['step'], failed - 1) if failed else latest['step']
        for field in ('lambda_q10', 'lambda_median', 'lambda_q90', 'readout_l2', 'max_gamma'):
            row[field] = latest.get(field)
        cases.append(row)
    write_csv(out / 'case_ledger.csv', cases)
    scores = []
    for horizon in (20000, 100000, 200000, 300000, 500000):
        for row in design.rank(root, horizon=horizon):
            mse = [e['validation_mse'] for e in design.selection_window(
                history.evaluations(root / row['id']), horizon)]
            scores.append(dict(id=row['id'], horizon=horizon, mean_relative_mse=row['score'],
                               mean_mse=float(np.mean(mse)), median_mse=float(np.median(mse)),
                               q90_mse=float(np.quantile(mse, .9)),
                               median_relative_mse=row['median'], q90_relative_mse=row['upper'],
                               best_relative_mse=row['best']))
    write_csv(out / 'horizon_scores.csv', scores)
    return dict(cases=len(cases), failures=sum(bool(r['failed_update']) for r in cases),
                score_rows=len(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.root, args.out)))


if __name__ == '__main__':
    main()
