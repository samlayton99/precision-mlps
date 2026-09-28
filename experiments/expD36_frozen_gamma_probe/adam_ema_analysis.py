"""Fixed-rule EMA of every-update archived Adam relative squared errors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

HALF_LIVES = [10, 30, 100, 300, 1000, 3000]
DEFAULT_OUT = Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access')


def ema_squared(errors, half_life):
    """M[0]=E[0]^2, M[n]=beta*M[n-1]+(1-beta)*E[n]^2."""
    errors = np.asarray(errors, dtype=float)
    if half_life <= 0 or not len(errors) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError('Finite nonnegative errors and positive half-life required')
    beta = 2.**(-1./half_life)
    squared = errors**2
    result = np.empty_like(squared)
    result[0] = squared[0]
    if len(errors) > 1:
        result[1:], _ = lfilter([1-beta], [1, -beta], squared[1:], axis=0,
                               zi=beta*squared[:1])
    return result


def crossing_summary(values, threshold=1e-4):
    """First acquisition and number of later below-to-above recrossings."""
    below = np.asarray(values) <= threshold
    any_hit = below.any(axis=0)
    first = np.where(any_hit, np.argmax(below, axis=0), -1)
    upcrossings = np.sum(below[:-1] & ~below[1:], axis=0)
    return first, upcrossings


def analyze(archive):
    archive = Path(archive)
    with np.load(archive) as z:
        errors = z['errors']; metadata = json.loads(str(z['metadata']))
        raw_first=z['first'][:,:,0]; raw_sustained=z['sustained'][:,:,0]
        raw_last=z['last_above'][:,:,0]
        assert not np.any(z['failed'])
    assert errors.shape == (200001,4,10)
    first,_=crossing_summary(errors,.01)
    np.testing.assert_array_equal(first,raw_first)
    last=np.max(np.where(errors>.01,np.arange(len(errors))[:,None,None],-1),axis=0)
    np.testing.assert_array_equal(last,raw_last)
    np.testing.assert_array_equal(np.where(last<len(errors)-1,last+1,-1),raw_sustained)
    summaries=[]; crossing_steps=[]
    for h in HALF_LIVES:
        ema=ema_squared(errors,h)
        first, recross=crossing_summary(ema)
        summaries.append((first,recross,ema[-1].copy()))
        # Retain each exact first crossing and its preceding state. Recrossings
        # are counted from the complete trace, not inferred from plotted samples.
        transitions=first[first>=0]
        crossing_steps.extend(transitions.tolist()); crossing_steps.extend(np.maximum(0,transitions-1).tolist())
    steps=np.unique(np.r_[0,np.geomspace(1,len(errors)-1,1000).astype(int),crossing_steps]).astype(int)
    curves=np.stack([ema_squared(errors,h)[steps] for h in HALF_LIVES])
    cases=[]
    for gi,gamma in enumerate(metadata['gammas']):
        for ci,column in enumerate(metadata['columns']):
            cases.append(dict(gamma=gamma,view=column['view'],target=column['target'],column=ci,
                raw_first=int(raw_first[gi,ci]),raw_last_above=int(raw_last[gi,ci]),raw_sustained=int(raw_sustained[gi,ci]),
                ema=[dict(half_life=h,first_hit=int(first[gi,ci]),later_upcrossings=int(recross[gi,ci]),
                          final_ema_squared=float(final[gi,ci]))
                     for h,(first,recross,final) in zip(HALF_LIVES,summaries)]))
    result=dict(main_half_life=100,half_lives=HALF_LIVES,threshold_squared=1e-4,horizon=200000,
        gammas=metadata['gammas'],columns=metadata['columns'],cases=cases,
        recurrence='M0=E0^2; Mn=beta*M(n-1)+(1-beta)*En^2; beta=2^(-1/half_life). No bias correction.',
        interpretation='First EMA crossing is acquisition of smoothed squared error; not sustained raw-error convergence.',
        sensitivity='Shared post-hoc window choice: main half-life100 for all40 cases; the same six half-lives report window sensitivity. Windows were explored before this analysis.',
        curve_layout='ema_squared[half_life,step,gamma,column]; raw_errors[step,gamma,column]',
        sampling='1000 geometric sample requests, plus all first threshold crossings and preceding states, all cases and half-lives. Curves omit some subsequent oscillations; recrossing counts use every update.',
        source_archive=str(archive),source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_files=metadata['source_files'],source_metadata=metadata,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=dict(raw_first_last_sustained_match=True,trajectories=40,half_life_cases=240),
        limitations=['EMA is a descriptive statistic, not a GD theorem applied to Adam.',
                     'Confirmation horizon remains200000; recrossings are reported rather than suppressed.',
                     'No hyperparameter selection or new optimizer training.'])
    arrays=dict(steps=steps,half_lives=np.array(HALF_LIVES),gammas=np.array(metadata['gammas']),
                raw_errors=errors[steps],ema_squared=curves)
    return result,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('/private/tmp/adam_raw_scalar_archive.npz'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT);args=p.parse_args()
    result,arrays=analyze(args.archive);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'ema_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'ema_curves.npz',**arrays)
    print(json.dumps(dict(sampled_steps=len(arrays['steps']),primary=[dict(gamma=c['gamma'],ema=c['ema']) for c in result['cases'] if c['view']=='common' and c['target']=='sine_mix_2_6_10'])))


if __name__ == '__main__':
    main()
