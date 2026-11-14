"""Compare completed Adam runs to the paired saved GD runs; validate their shared setup."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD64_adam_joint_training_diagnostics.run import OUT
from experiments.expD56_scale_acquisition.audit_slope_growth import constants,force_terms
GD=ROOT/'results/checkpoint_D_optimizers/expD62_joint_training_diagnostics'

def main():
    rows=[];hashes={};checked=0;violations=0
    for path in sorted(OUT.glob('width*/runs/*/checkpoint.npz')):
        status=json.loads((path.parent/'status.json').read_text());assert status['state']=='complete' and status['step']==50000
        gd=GD/path.relative_to(OUT)
        with np.load(path.parent/'initial.npz') as a,np.load(gd.parent/'initial.npz') as b:
            for key in ['theta','train_x','train_y','eval_x','eval_y']:
                np.testing.assert_array_equal(a[key],b[key])
            k=constants(a['train_x'])
        with np.load(path) as z:
            h={key:z['history'][:,i] for i,key in enumerate(z['keys'].tolist())}
            assert np.isfinite(z['history']).all()
        with np.load(gd) as z:g={key:z['history'][:,i] for i,key in enumerate(z['keys'].tolist())}
        width=status['width']
        for i in range(len(h['step'])):
            q=np.sqrt(h['parameter_energy'][i]/width)
            forces=np.asarray(force_terms(q,width,[h[f'chi_{p}'][i] for p in [4,6,10,14]],
                              h['nonaffine_residual_rms'][i],h['tau_3'][i],h['tau_5'][i],k))*np.sqrt(width)
            # These pointwise gradient inequalities are optimizer-independent, unlike the time recurrence.
            assert np.all(forces+1e-10>=h['fine_norm'][i])
            checked+=1
        assert np.min(h['coarse_kernel_min_eigenvalue'])>0
        row={'width':width,'target':status['target'],'lr':status['learning_rate'],
             'gamma_initial':h['gamma_rms'][0],'gamma_adam':h['gamma_rms'][-1],
             'gamma_gd':g['gamma_rms'][-1],'gamma_ratio_adam_gd':h['gamma_rms'][-1]/g['gamma_rms'][-1],
             'error_adam':h['eval_rel_l2'][-1],'error_gd':g['eval_rel_l2'][-1],
             'norm_adam':np.sqrt(h['parameter_energy'][-1]),'norm_gd':np.sqrt(g['parameter_energy'][-1]),
             'max_chi6_adam':h['chi_6'].max()}
        rows.append(row);hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(rows)==24
    with (OUT/'adam_vs_gd.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'validation.json').write_text(json.dumps({'matched_initialization_and_grids':24,
        'pointwise_F4_force_checks':checked,'all_four_alternatives_pass':True,'all_coarse_kernels_positive':True,
        'source_sha256':hashes,'note':'F.4 bounds the raw gradient for either optimizer. F.6 is a GD recurrence and is not certified for Adam.'},indent=2)+'\n')
    lines=['# Adam versus GD: matched 50k-step tracking runs','',
           'All 24 Adam runs completed. Initial parameters, target normalization, and both sampling grids match the saved GD runs exactly. Full-batch float64 tanh networks; half-MSE loss; Adam betas 0.9/0.999, epsilon 1e-8, no weight decay.','',
           'The original GD recurrence is plotted only as a comparator. It is not a bound on Adam. The raw-gradient F.4 inequalities pass at all '+str(checked)+' saved states; this does not transfer the GD time-recursion proof to Adam.','',
           '| Width | Function | LR | Final RMS slope: GD | Final RMS slope: Adam | Relative L2: GD | Relative L2: Adam |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['width']} | {r['target']} | {r['lr']:g} | {r['gamma_gd']:.4g} | {r['gamma_adam']:.4g} | {r['error_gd']:.3g} | {r['error_adam']:.3g} |")
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'results.md')
if __name__=='__main__':main()
