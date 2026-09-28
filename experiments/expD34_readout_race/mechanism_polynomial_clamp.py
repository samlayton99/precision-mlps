"""Secondary causal diagnostic: evolving geometry with the fine residual clamped."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from . import effective_feedback as ef, transport, mechanism_polynomial as poly
from .mechanism_splitting_baselines import matrices
from .mechanism_splitting_diagnostics import write_csv


def clamped_field(p, transform, e0):
    _, jacobian = poly.coefficients_jacobian(p, 5)
    modal = transform@jacobian; JC, JH = modal[:2], modal[2:]
    raw = JH.T@e0
    return raw-JC.T@jnp.linalg.solve(JC@JC.T, JC@raw)


def predictor(steps=20000):
    def one(p, transform, e0, correction):
        return jax.lax.fori_loop(0, steps, lambda _, state: state-.002*(clamped_field(state, transform, e0)+correction), p)
    return jax.jit(jax.vmap(one, in_axes=(0, None, 0, 0)))


def residual(p, x, y):
    a,b,c = p[:-1].reshape(3,-1)
    return np.tanh(x[:,None]*a+b)@c+p[-1]-y


def run(root, confirmation):
    out = root/'clamped_quintic'; out.mkdir(exist_ok=False)
    rows = []; sources = {}
    for n in (128,512,1024):
        source = root/('fork_inputs' if confirmation else 'width_inputs')/(f'N{n}.npz' if confirmation else f'N{n}_fork20000.npz')
        pp,x,yy,cases = ef.load_inputs(source)
        transform,_ = poly.modal_setup(x,yy[0],5)
        q = transport.basis(x,65)
        errors=[]; corrections=[]
        for p,y in zip(pp,yy):
            _,target=poly.modal_setup(x,y,5)
            coefficients,_=poly.coefficients_jacobian(jnp.asarray(p),5)
            e0=(transform@np.asarray(coefficients)-target)[2:]
            _,T,_,exacte=matrices(p,x,y,np.ones(len(p)),q)
            force=np.asarray(clamped_field(jnp.asarray(p),jnp.asarray(transform),jnp.asarray(e0)))
            errors.append(e0); corrections.append(T@exacte-force)
        states=np.asarray(predictor()(jnp.asarray(pp),jnp.asarray(transform),jnp.asarray(errors),jnp.asarray(corrections)))
        np.savez_compressed(out/f'N{n}.npz', p0=pp, clamped5=states)
        future=root/('continuation' if confirmation else 'widths_post20k')/f'N{n}'/'snapshots/000020000.npz'
        with np.load(future) as data: actual=data['p'].copy()
        ownpath=root/('forecasts' if confirmation else 'polynomial_anchor')/f'N{n}.npz'
        with np.load(ownpath) as data: own=data['anchored5'].copy()
        w=(len(pp[0])-1)//3
        for i,case in enumerate(cases):
            r0=residual(pp[i],x,yy[i]); r1=residual(actual[i],x,yy[i])
            e0=q.T@r0/len(x); e1=q.T@r1/len(x)
            full0=r0-q[:,:2]@e0[:2]; full1=r1-q[:,:2]@e1[:2]
            motion=np.linalg.norm(actual[i,:w]-pp[i,:w])
            row=dict(case, actual_motion=float(motion),
                clamped_relative_error=float(np.linalg.norm(states[i,:w]-actual[i,:w])/motion),
                own_error_relative_error=float(np.linalg.norm(own[i,:w]-actual[i,:w])/motion),
                actual_retained65_residual_fraction_change=float(np.linalg.norm(e1[2:]-e0[2:])/np.linalg.norm(e0[2:])),
                actual_full_complement_residual_fraction_change=float(np.linalg.norm(full1-full0)/np.linalg.norm(full0)))
            for k in range(2,6):
                row[f'e{k}_initial']=float(e0[k]); row[f'e{k}_final']=float(e1[k]); row[f'e{k}_change']=float(e1[k]-e0[k])
            rows.append(row)
        sources[str(source)]=ef.digest(source); sources[str(future)]=ef.digest(future)
    write_csv(out/'scores.csv',rows)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=ef.digest(__file__),kernel_sha256=ef.digest(poly.__file__),
        sources=sources, issued_utc=datetime.now(timezone.utc).isoformat(), role='retrospective secondary causal diagnostic',
        statement='Quintic fine residual fixed at fork; own-state polynomial Jacobian and coarse projection recomputed; exact initial full-parameter force correction fixed'),indent=2))
    for n in (128,512,1024):
        part=[r for r in rows if r['nref']==n]
        print(n, 'median clamped/own',np.median([r['clamped_relative_error'] for r in part]),np.median([r['own_error_relative_error'] for r in part]),'max actual residual change',max(r['actual_full_complement_residual_fraction_change'] for r in part))
        for r in part:
            if r['target']=='moment5': print(r)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path); parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args(); run(args.root,args.confirmation)
