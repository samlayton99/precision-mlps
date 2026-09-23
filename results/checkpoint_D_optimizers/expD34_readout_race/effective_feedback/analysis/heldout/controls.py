"""Direct matched-time numerical sensitivity audit, not an error certificate."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from experiments.expD34_readout_race.effective_feedback_analysis import write_csv, _ratio

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARMS = ('joint', 'freeze_map', 'clamp_residual')


def key(case):
    return case['target'], int(case['seed']), int(case['start'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--miss', action='store_true', help='Posthoc gauss_right sign check only.')
    args = parser.parse_args()
    arms = ('joint', 'clamp_residual') if args.miss else ARMS
    control_root = 'heldout_miss_controls' if args.miss else 'heldout_controls'
    prefix = 'miss_control' if args.miss else 'control'
    records, hashes = {}, {}
    for control in ('primary', 'degree129', 'halfstep'):
        for arm in arms:
            folder = (ROOT/'raw/heldout'/arm if control == 'primary' else
                      ROOT/'raw'/control_root/f'{control}_{arm}')
            path = folder/'manifest.json'
            manifest = json.loads(path.read_text())
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(folder/'snapshots/000000000.npz') as data:
                initial = data['p'].copy()
            states = {}
            for offset in ((50000,) if args.miss else (10000, 50000, 200000)):
                path = folder/'snapshots'/f'{offset:09d}.npz'
                if path.exists():
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(path) as data:
                        states[offset] = {k: data[k].copy() for k in ('p', 'count', 'failed', 'metric_eH')}
            records[control, arm] = manifest, initial, states, {key(c): i for i, c in enumerate(manifest['cases'])}
    horizons = sorted(set.intersection(*(set(r[2]) for r in records.values())))
    state_rows, contrasts = [], []
    cases = records['degree129', 'joint'][0]['cases']
    for control in ('degree129', 'halfstep'):
        for offset in horizons:
            for case in cases:
                k = key(case)
                for arm in arms:
                    cm, ci, cs, cl = records[control, arm]
                    bm, bi, bs, bl = records['primary', arm]
                    i, j = cl[k], bl[k]
                    np.testing.assert_array_equal(ci[i], bi[j])
                    assert not cs[offset]['failed'][i] and not bs[offset]['failed'][j]
                    assert cs[offset]['count'][i]*cm['eta'] == bs[offset]['count'][j]*bm['eta'] == .002*offset
                    pc, pb = cs[offset]['p'][i], bs[offset]['p'][j]
                    width = (len(pc)-1)//3
                    change = pb[:width]-bi[j, :width]
                    difference = pc[:width]-pb[:width]
                    base = dict(control=control, target=k[0], seed=k[1], start=k[2], arm=arm,
                                offset=offset, total_reference_updates=k[2]+offset)
                    state_rows.append(dict(base, slope_difference_norm=float(np.linalg.norm(difference)),
                        slope_motion_relative_difference=_ratio(np.linalg.norm(difference), np.linalg.norm(change)),
                        mean_gamma_difference=float(np.mean(abs(pc[:width])-abs(pb[:width]))),
                        modal_degree2_to65_difference_norm=float(np.linalg.norm(
                            cs[offset]['metric_eH'][i, :64]-bs[offset]['metric_eH'][j, :64]))))
                    if arm == 'joint':
                        continue
                    _, _, cjstates, cjlookup = records[control, 'joint']
                    _, _, bjstates, bjlookup = records['primary', 'joint']
                    cj = cjstates[offset]['p'][cjlookup[k], :width]
                    bj = bjstates[offset]['p'][bjlookup[k], :width]
                    actual = pb[:width]-bj
                    control_contrast = pc[:width]-cj
                    actual_mean = float(np.mean(abs(pb[:width])-abs(bj)))
                    control_mean = float(np.mean(abs(pc[:width])-abs(cj)))
                    spread = abs(control_mean-actual_mean)
                    contrasts.append(dict(base, baseline_mean_gamma_contrast=actual_mean,
                        control_mean_gamma_contrast=control_mean,
                        mean_gamma_contrast_absolute_difference=spread,
                        raw_sign_agrees=bool(np.sign(actual_mean) == np.sign(control_mean)),
                        baseline_exact_zero=actual_mean == 0, control_exact_zero=control_mean == 0,
                        sign_margin_over_observed_difference=_ratio(abs(actual_mean), spread),
                        slope_contrast_relative_difference=_ratio(np.linalg.norm(control_contrast-actual), np.linalg.norm(actual))))
    write_csv(OUT/f'{prefix}_states.csv', state_rows)
    write_csv(OUT/f'{prefix}_contrasts.csv', contrasts)
    (OUT/f'{prefix}_provenance.json').write_text(json.dumps(dict(horizons=horizons,
        case_count=len(cases), all_initial_states_bitwise_equal=True, all_physical_times_equal=True,
        caveat='Two refinement levels measure numerical sensitivity; they do not certify an error bound.',
        artifact_sha256=hashes), indent=2)+'\n')
    print('Common control horizons:', horizons, 'state comparisons:', len(state_rows))


if __name__ == '__main__':
    main()
