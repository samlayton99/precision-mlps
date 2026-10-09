"""Audit saved solved weights independently; preserve original solve timings."""
import json
import numpy as np
from run import geometry,disk,boundary,exact,source,OUT


def main():
    for path in sorted(OUT.glob('free_*.json')):
        row=json.loads(path.read_text())
        arrays=np.load(OUT/row['arrays'])
        weights=arrays['weights'];bias=float(arrays['bias'])
        bank=geometry(*weights.shape)
        train=bank.bind(disk(row['interior_points'],row['seed']))
        b=bank.bind(boundary(row['boundary_points'],.173))
        residual=-train.forward(weights,2)-source(train.points)
        if row.get('nonlinear',False):residual+=(train.forward(weights)+bias)**3-exact(train.points)**3
        row['final']['train_pde_rms']=float(np.sqrt(np.mean(residual**2)))
        row['final']['train_boundary_rms']=float(np.sqrt(np.mean((b.forward(weights)+bias-exact(b.points))**2)))
        # Full ordinary tanh bank, all centers, 16 independent validation points.
        plan=bank.bind(arrays['points'][:16]);audit={}
        for order in (0,2):
            dense=np.zeros(plan.k)
            for m in range(bank.m):
                z=bank.gamma*(plan.projection[m,:,None]-bank.centers)
                phi=np.tanh(z)
                if order==2:
                    e=np.exp(-2*abs(z));phi=-2*bank.gamma**2*phi*(4*e/(1+e)**2)
                dense+=phi@weights[m]
            audit[str(order)]=float(np.max(abs(plan.forward(weights,order)-dense)))
        row['dense_network_audit']=audit
        row['post_solve_audit_note']='Independent dense solved-weight and training-residual audits from saved arrays; original solve timings preserved.'
        row.setdefault('status','iteration_budget_exhausted' if row['final']['lsmr_stop']==7 else 'inner_solver_stopped')
        path.write_text(json.dumps(row,indent=2))
        assert audit['0']<1e-10 and audit['2']<1e-8,(path,audit)
    print('All saved free-readout solutions pass independent dense value/Laplacian audits.')


if __name__=='__main__':main()
