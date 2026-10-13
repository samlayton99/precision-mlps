"""One-step counterfactuals at saved intervention states; never used for fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD35_optimization_exploration import core,run as old,ssb
from experiments.expD06_fixed_center_scales import higher_order as higher
from . import accessibility as access,diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
    old.verify_gpu(args.root)
    cases=[]
    for path in sorted(args.root.glob('*/case.json')):
        c=json.loads(path.read_text())
        if c.get('implementation')=='primed_guard_v2' and c['policy']=='adaptive' and c['n']==128 and c['seed']<2:cases.append((path.parent,c))
    for index,(folder,c) in enumerate(cases):
        if index%args.workers!=args.worker:continue
        g,loss,_=ssb.problem(c);solver=higher.ssb_solver(args.source,1e-30,1e-15,'accepted_step')
        template=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
        _,_,residual,_,_=access.problem(c['n'],c['coordinates'],c['target'])
        svd=diagnostics.kernels(c['n'],c['coordinates'],c['target'])[-1]
        hvp=diagnostics.kernels(c['n'],c['coordinates'],c['target'])[-2]
        tangent_fn=jax.jit(lambda zz,dd:jax.jvp(residual,(zz,),(dd,))[1])
        taus=jnp.asarray(json.loads((folder/'controller.json').read_text())['taus'])
        one=ssb.kernel(c['n'],c['coordinates'],c['target'],args.source,1e-30,1e-15,'non_descent',1000,1)
        for path in sorted(folder.glob('diagnostic_*.json')):
            row=json.loads(path.read_text())
            if row.get('event')!='metric_mix':continue
            at=row['step'];parent=folder/f'solver_{at:09d}.npz';state=ssb.load_solver(parent,template)
            z=state['z'];r=residual(z);grad=jax.grad(loss)(z);h=state['solver'].f_info.hessian_inv.pytree
            mixed,_=access.mix_metric(h,grad,row['beta'])
            before=np.asarray(svd(z[g.width+1:],r,taus)[0]);records=[]
            options=[('history_only',h),('metric_mix',mixed)]
            options.extend((f'metric_beta_times_{factor}',access.mix_metric(h,grad,row['beta']*factor)[0]) for factor in (.01,.1))
            for name,matrix in options:
                start=ssb.restart(state,solver,loss,matrix);end,trace=one(start)
                delta=end['z']-z
                direction=-matrix@grad;norm=jnp.linalg.norm(direction);unit=direction/norm
                action=hvp(z,unit)
                tangent=tangent_fn(z,unit)
                slope=grad@unit;curvature=unit@action;gn_curvature=tangent@tangent
                linear_length=-slope/curvature
                exposure=[]
                for tau in taus:
                    _,derivative=access.problem(c['n'],c['coordinates'],c['target'])[-1](z[g.width+1:],r,tau)
                    exposure.append(float(derivative@unit[g.width+1:]))
                after=np.asarray(svd(end['z'][g.width+1:],r,taus)[0])
                records.append(dict(arm=name,status=int(end['status']),accepted=int(end['count'])-at,
                    before_mse=float(2*loss(z)),after_mse=float(2*loss(end['z'])),
                    actual_frozen_residual_gain=(after-before).tolist(),before_gain=before.tolist(),
                    native_step_norm=float(jnp.linalg.norm(delta)),geometry_step_norm=float(jnp.linalg.norm(delta[g.width+1:])),
                    proposed_norm=float(norm),unit_slope=float(slope),unit_true_curvature=float(curvature),
                    unit_gn_curvature=float(gn_curvature),quadratic_optimal_length=float(linear_length),
                    unit_exposure=exposure,quadratic_exposure=(np.asarray(exposure)*float(linear_length)).tolist(),
                    emergency_reset=int(trace[0,14]),first_step_direction_cosine=float(core.cosine(delta,-matrix@grad))))
            old.write_json(folder/f'probe_curvature_{at:09d}.json',dict(step=at,beta=row['beta'],arms=records,
                checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                source_commit=os.environ.get('EXPLORATION_SOURCE_COMMIT'),
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            print(folder.name,at,records,flush=True)


if __name__=='__main__':main()
