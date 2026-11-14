"""Evaluate the proposed signed-energy bound on saved GD data; no training."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.audit_slope_growth import constants
from experiments.expD62_joint_training_diagnostics import plot
OUT=plot.OUT/'signed_energy_audit';START=30000

def state(theta,x,y,lr):
    a,b,w=np.split(theta[:-1],3);W=len(a);u=x[:,None]*a+b;features=np.tanh(u)
    q=features@w+theta[-1];e=q-y
    coarse=np.column_stack([np.ones(len(x)),x/np.sqrt(np.mean(x*x))])
    def pc(v):return coarse@(coarse.T@v/len(x))
    def ph(v):return v-pc(v)
    sensitivity=(1-features**2)*w
    g=np.r_[np.mean(sensitivity*(e*x)[:,None],axis=0),np.mean(sensitivity*e[:,None],axis=0),features.T@e/len(x)]
    z=(features+u*(1-features**2))@w
    d5=ph((u*(1-features**2)-3*features+2*u)@w)
    qh=ph(q);fh=ph(y);eh=ph(e);ec=pc(e);zc=pc(z)
    p=theta[:-1];M=float(p@p);r2=a*a+b*b+w*w
    actual_change=float((p-lr*g)@(p-lr*g)-M)
    radial=-2*lr*float(p@g);step2=lr**2*float(g@g)
    expanded=lr*(-8*np.mean(qh*qh)-2*np.mean(qh*d5)+2*np.mean(fh*ph(z))-2*np.mean(ec*zc))+step2
    np.testing.assert_allclose(actual_change,radial+step2,atol=2e-11,rtol=1e-8)
    np.testing.assert_allclose(expanded,radial+step2,atol=2e-11,rtol=1e-8)
    np.testing.assert_allclose(ph(z),4*qh+d5,atol=2e-13,rtol=1e-11)
    return dict(M=M,EC=float(np.sqrt(np.mean(ec*ec))),EH=float(np.sqrt(np.mean(eh*eh))),
                actual_energy_increment=actual_change,exact_radial_increment=radial,exact_squared_step=step2,
                exact_self_output_increment=float(lr*(-8*np.mean(qh*qh)-2*np.mean(qh*d5))),
                squared_defect_allowance=float(lr*np.mean(d5*d5)/8),
                moment_defect_allowance=float(lr*.5*constants(x)['A5']**2*np.sum(r2**3)**2),
                w1=float(np.sum(np.abs(w))),current_S4=float(np.sum(r2**2)),current_S6=float(np.sum(r2**3)),current_S8=float(np.sum(r2**4)))

def main():
    OUT.mkdir(exist_ok=True);rows=[]
    sources=[(p,False) for p in sorted(plot.OUT.glob('width*/runs/*/checkpoint.npz'))]
    extra=ROOT/'results/checkpoint_D_optimizers/expD60_feature_bound_reconstruction'
    sources += [(extra/n/'checkpoint.npz',True) for n in ['glorot__legendre_5','pytorch_default__legendre_5']]
    for path,reconstruction in sources:
        status=json.loads((path.parent/'status.json').read_text());W=status['width'];lr=status['learning_rate']
        with np.load(path) as d:
            h={key:d['history'][:,i] for i,key in enumerate(d['keys'].tolist())};snaps=d['snapshots'].copy()
        with np.load(path.parent/'initial.npz') as d:
            x=d['x' if reconstruction else 'train_x'];y=d['y' if reconstruction else 'train_y']
        indices=np.flatnonzero((h['step']>=START)&(h['step']<=50000));i=indices[0]
        states=[state(snaps[j],x,y,lr) for j in indices]
        H=max(s['EH'] for s in states);EC=max(s['EC'] for s in states)
        K6=max(h['chi_6'][indices]);K10=max(h['chi_10'][indices]);M0=states[0]['M'];k=constants(x)
        tau3=h['tau_3'][i];tau5=h['tau_5'][i];fH=h['target_nonaffine_rms'][i]
        def terms(M):
            S4=math.sqrt(K6)*M*M/W;S6=K6*M**3/W**2;S8=math.sqrt(K6*K10)*M**4/W**3
            U={'self_remainder':.5*k['A5']**2*S6*S6,'target_cubic':8*k['A3']*tau3*S4,
               'target_quintic':12*k['A5']*tau5*S6,'target_remainder':16*k['A7']*fH*S8,
               'affine_residual':3*EC*math.sqrt(W*M)}
            G=k['C3']*H*math.sqrt(S6)+EC*math.sqrt(1+2*M)
            return {**{name:lr*v for name,v in U.items()},'discrete_correction':lr*lr*G*G}
        for st in states:
            point=lr*(.5*k['A5']**2*st['current_S6']**2+8*k['A3']*tau3*st['current_S4']+12*k['A5']*tau5*st['current_S6']+16*k['A7']*fH*st['current_S8']+3*st['EC']*st['w1'])+st['exact_squared_step']
            assert st['actual_energy_increment']<=point+1e-10
        # An additional old-bound comparison uses exactly the same constant suffix maxima.
        maxchi=[max(h[f'chi_{p}'][indices]) for p in [4,6,10,14]]
        maxR=max(h['tracking_norm'][indices]);B=math.sqrt(M0);old_initial=None
        for oldn in range(1,20001):
            force=min(plot.force_terms(B/math.sqrt(W),W,maxchi,H,tau3,tau5,k))*math.sqrt(W)
            Bnext=B+lr*(force+maxR)
            if old_initial is None:old_initial=Bnext*Bnext-M0
            B=Bnext
            if B*B>=2*M0:break
        matched_old_step=oldn if B*B>=2*M0 else None
        M=M0;trace=[M]
        for n in range(1,20001):
            M+=sum(terms(M).values());trace.append(M)
            if M>=2*M0:break
        double_step=n if M>=2*M0 else None
        plot.RESTART=START
        # Compare equal barrier: doubled parameter energy, not an arbitrary display cap.
        old=plot.ceiling(h,W,lr,k,cap=math.sqrt(2*M0))
        row={'case':str(path.parent.relative_to(ROOT)),'width':W,'target':status['target'],'lr':lr,
             'start':START,'end_allowance_window':50000,'initial_B':math.sqrt(M0),'initial_energy':M0,
             'H':H,'EC':EC,'K6':K6,'K10':K10,'tau3':tau3,'tau5':tau5,
             'at_restart':states[0],'increment_terms_at_start':terms(M0),
             'new_initial_energy_increment_bound':sum(terms(M0).values()),
             'new_initial_norm_increment_bound':math.sqrt(M0+sum(terms(M0).values()))-math.sqrt(M0),
             'new_steps_to_double_energy_ceiling':double_step,
             'old_steps_to_double_energy_constant_maxima':matched_old_step,
             'old_initial_energy_increment_constant_maxima':old_initial,
             'old_steps_to_double_energy_ceiling':old['cap_crossing_step']-START if old['cap_crossing_step'] else None,
             'first_exit_lower_bound_using_constant_barrier_allowance':M0/sum(terms(2*M0).values()),
             'actual_energy_final':states[-1]['M'],
             'checkpoints_checked':len(states)}
        rows.append(row)
        print(W,status['target'],lr,'B0',round(math.sqrt(M0),4),'new/old ceiling doubling',double_step,row['old_steps_to_double_energy_ceiling'],flush=True)
    report={'scope':'Read-only evaluation of the proposed signed-energy comparison. All allowances are suffix maxima of saved 30k–50k checkpoints, not certified between checkpoints or beyond the interval. Original comparator uses its existing interpolated allowances.',
            'rows':rows}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
