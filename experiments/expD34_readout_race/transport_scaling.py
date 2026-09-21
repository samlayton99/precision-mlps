"""Check formal width powers at the affine-law reference, not at trained GD states.

This isolates the power-counting calculation from finite-width nonlinear drift
and random seeds. It is not a nonlinear trajectory forecast or a barrier proof.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from . import targets, transport as tr
from .recovery import write_table


def affine_law(width, x, y, horizon=40., nodes=12):
    z,d,w=tr.law_initial(width,nodes)
    G0=(z*w) @ z.T
    mu2=np.mean(x*x); y0=np.mean(y); y1=np.mean(x*y)
    def rhs(t,state):
        U=state[:9].reshape(3,3); G=U @ G0 @ U.T
        e0=state[9]+width*G[1,2]-y0; e1=width*mu2*G[0,2]-y1
        M=np.array([[0.,0.,e1],[0.,0.,e0],[e1,e0,0.]])
        return np.r_[(-M @ U).ravel(),-e0]
    solution=solve_ivp(rhs,(0.,horizon),np.r_[np.eye(3).ravel(),0.],method='DOP853',rtol=2e-12,atol=2e-13,max_step=.5)
    if not solution.success: raise RuntimeError(solution.message)
    U=solution.y[:9,-1].reshape(3,3); d=solution.y[9,-1]; z=U @ z
    affine=x*(width*np.sum(w*z[0]*z[2]))+width*np.sum(w*z[1]*z[2])+d-y
    defect=max(abs(np.mean(affine)),abs(np.mean(x*affine)))
    if defect>1e-10: raise ValueError(f'Affine coarse relaxation not resolved: {defect}')
    # Affine flow exactly preserves each particle's a^2+b^2-c^2.
    old,_,_=tr.law_initial(width,nodes)
    balance=np.max(abs(z[0]**2+z[1]**2-z[2]**2-(old[0]**2+old[1]**2-old[2]**2)))
    return z,d,w,defect,balance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    x=targets.grid(2048); rows=[]
    for target in ('sine','moment3','moment9'):
        y=targets.data(len(x),target)['y']
        for width in (89,177,353,705,1409,2817,5633):
            z,d,w,defect,balance=affine_law(width,x,y)
            scalar,arrays=tr.modal_diagnostics(z,d,x,y,w,width,17)
            K=sum(arrays['K_'+k] for k in 'abcd'); B=np.linalg.solve(K[:2,:2],K[:2,2:])
            ja=arrays['J_a']; T=ja[2:].T-ja[:2].T @ B; e=arrays['residual_modes']
            direct9=np.linalg.norm(T[:,7]*e[9])
            effective=T @ e[2:]
            alignment=-(np.sqrt(w)*np.sign(z[0])) @ effective/np.linalg.norm(effective)
            rows.append(dict(target=target,width=width,affine_coarse_defect=defect,affine_balance_error=balance,
                effective_slope_norm=scalar['effective_slope_norm'],full_slope_norm=scalar['full_slope_norm'],
                effective_outward_alignment=float(alignment),
                direct_mode9_effective_norm=direct9,omitted_slope_norm=scalar['omitted_slope_norm']))
    write_table(args.output/'width_tangent_scaling.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,target,power in zip(axes,('sine','moment3','moment9'),(1,1,2)):
        rr=[r for r in rows if r['target']==target]; width=np.array([r['width'] for r in rr])
        norm=np.array([r['effective_slope_norm'] for r in rr])
        ax.loglog(width,norm,'o-',label='Effective force at affine-law state')
        ax.loglog(width,norm[-1]*(width[-1]/width)**power,'--',label=f'W^(-{power}), anchored at largest W')
        if target=='moment9':
            ax.loglog(width,[r['direct_mode9_effective_norm'] for r in rr],':',label='Direct mode-9 effective force')
        ax.set(title=target,xlabel='Physical width',ylabel='Slope-force norm');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(args.output/'width_tangent_scaling.png',dpi=160);plt.close(fig)
    print(f'Wrote {len(rows)} affine-law operator checks')


if __name__=='__main__':main()
