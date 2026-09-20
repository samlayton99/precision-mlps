"""Audit saved failed inverse metrics, including an 80-digit quadratic form."""
import argparse
import json
from pathlib import Path
import jax
import mpmath as mp
import numpy as np
from . import core,run,ssb
from experiments.expD06_fixed_center_scales import higher_order as higher


def audit(folder,source):
    c=json.loads((folder/'case.json').read_text())
    initial,_=run.restore(folder/'initial.npz');g,loss,physical=ssb.problem(c)
    solver=higher.ssb_solver(source,c.get('curvature_epsilon',1e-30),c.get('search_threshold',1e-15),'accepted_step')
    state=ssb.load_solver(folder/'solver.npz',higher.ssb_initial(solver,loss,initial['z']))
    matrix=np.asarray(state['solver'].f_info.hessian_inv.pytree)
    stored=np.asarray(state['solver'].f_info.grad);recomputed=np.asarray(jax.grad(loss)(state['z']))
    eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2)
    with mp.workdps(80):
        mm=mp.matrix(matrix.tolist());gg=mp.matrix(stored.tolist())
        exact_slope=-mp.fdot(gg,mm*gg)
    search=state['solver'].search_state
    return dict(id=folder.name,stored_directional_derivative_fp64=float(-stored@matrix@stored),
        stored_directional_derivative_mp80=mp.nstr(exact_slope,80),search_slope=float(search.slope_init),
        actual_stored_direction_slope=float(stored@(-np.asarray(state['solver'].descent_state.newton))),
        recomputed_directional_derivative_fp64=float(-recomputed@matrix@recomputed),
        stored_gradient_norm=float(np.linalg.norm(stored)),
        cpu_gpu_gradient_difference_relative=float(np.linalg.norm(stored-recomputed)/np.linalg.norm(stored)),
        smallest_eigenvalue=float(eigenvalues[0]),largest_eigenvalue=float(eigenvalues[-1]),
        search_interval=float(abs(search.stepsize_hi-search.stepsize_lo)),
        search_stepsize=float(search.stepsize),search_iterations=int(search.ls_iter_num))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=[]
    for folder in sorted(args.root.glob('ssbroyden*')):
        if not (folder/'solver.npz').exists():continue
        c=json.loads((folder/'case.json').read_text());latest=json.loads((folder/'latest.json').read_text())
        if latest.get('status')!='search_failed' or c['curvature_epsilon']!=1e-30:continue
        result=audit(folder,args.source);rows.append(result);print(json.dumps(result),flush=True)
    run.write_json(args.out,rows)


if __name__=='__main__':main()
