"""80-digit fixed-input Woodbury arithmetic audit; not a tanh certificate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2]
from experiments.expD36_frozen_gamma_probe import uniform_grid_spectrum as c
from experiments.expD36_frozen_gamma_probe.structured_resolvent import signed_lowrank_solver
DEFAULT=ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/structured_gamma'
def cv(x):
    return mp.mpc(float(np.real(x)),float(np.imag(x)))
def inner(a,b): return mp.fsum(mp.conj(x)*y for x,y in zip(a,b))
def norm(a): return mp.sqrt(mp.re(inner(a,a)))
def pair(z): return [float(mp.re(z)),float(mp.im(z))]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    mp.mp.dps=80
    source_paths=[DEFAULT/'summary.json',Path(c.__file__),Path(__file__),
        Path(__file__).with_name('structured_resolvent.py'),
        DEFAULT.parent.parent/'common/N512/arrays.npz']
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    results=[]
    run_cases(args.output/'precision_audit.json',sources,results)

def run_cases(output,sources,results):
    for gamma,eta,cutoff in [(8,.002068412912130894,1.1029830912616472e-7),(64,.0020173468150207307,.0001480046700289772)]:
        start=time.time()
        a=c.construct(512,16,23,gamma)
        d,u,s=c.low_rank_gram(a)
        keep=np.abs(s)>1e-12
        d=eta*d; u=u[:,keep]; s=eta*s[keep]
        y=np.load(ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/common/N512/arrays.npz')['y_train'][:,0]
        rawg=np.sqrt(eta)*a['approximate'].T@y
        g=c.to_bulk_basis(rawg[:,None],a)[:,0]
        z=cutoff/1.25*np.exp(1j*np.pi/64)
        # Freeze the actual FP64 weighted factors used internally by the solver.
        w=u*np.sqrt(np.abs(s)); signs=np.sign(s)
        inv=1/(d-z)
        small=np.diag(signs)+w.conj().T@(inv[:,None]*w)
        solve=signed_lowrank_solver(d,u,s)
        vz=solve(g,z); vb=solve(g,np.conj(z))
        dpaction=lambda v:d*v+w@(signs*(w.conj().T@v))
        rz=g-(dpaction(vz)-z*vz)
        rb=g-(dpaction(vb)-np.conj(z)*vb)
        qdp=np.vdot(g,vz); corrected_dp=qdp+np.vdot(vb,rz)
        print('MP start',gamma,len(d),len(s),flush=True)
        dm=[mp.mpf(float(x)) for x in d]; gm=[cv(x) for x in g]
        wm=[[cv(x) for x in row] for row in w]
        zm=cv(z); im=[1/(x-zm) for x in dm]; rank=len(s)
        sm=mp.matrix(rank); rhs=mp.matrix(rank,1)
        for r in range(rank):
            rhs[r]=mp.fsum(mp.conj(wm[i][r])*im[i]*gm[i] for i in range(len(d)))
            for t in range(rank):
                sm[r,t]=(float(signs[r]) if r==t else 0)+mp.fsum(mp.conj(wm[i][r])*im[i]*wm[i][t] for i in range(len(d)))
        coeff=mp.lu_solve(sm,rhs)
        vm=[im[i]*(gm[i]-mp.fsum(wm[i][r]*coeff[r] for r in range(rank))) for i in range(len(d))]
        qmp=inner(gm,vm)
        def residual(v,shift):
            v=[cv(x) for x in v]
            wt=[mp.fsum(mp.conj(wm[i][r])*v[i] for i in range(len(d)))*float(signs[r]) for r in range(rank)]
            return [gm[i]-(dm[i]-shift)*v[i]-mp.fsum(wm[i][r]*wt[r] for r in range(rank)) for i in range(len(d))]
        rzm=residual(vz,zm); rbm=residual(vb,mp.conj(zm))
        qc=inner(gm,[cv(x) for x in vz])+inner([cv(x) for x in vb],rzm)
        radius=norm(rzm)*norm(rbm)/abs(mp.im(zm))
        # Difference between computed FP64 residual and its fixed-input MP value.
        residual_error=norm([cv(x)-t for x,t in zip(rz,rzm)])
        row=dict(gamma=gamma,n=512,dimension=len(d),signed_rank=rank,eta=eta,cutoff=cutoff,pole=pair(z),
            input_definition='Fixed FP64 normalized diagonal, weighted signed factors, complex forcing, and pole; all small-matrix sums and solve recomputed at 80 decimal digits. Weighted factors include the original FP64 square root.',
            input_sha256=hashlib.sha256(b''.join(x.tobytes() for x in [d,w,signs,g,np.asarray(z)])).hexdigest(),
            small_condition_number_fp64=float(np.linalg.cond(small)),
            small_formation_max_absolute_error=float(max(abs(cv(small[r,t])-sm[r,t]) for r in range(rank) for t in range(rank))),
            q_mp=pair(qmp),q_dp=pair(qdp),q_dp_corrected=pair(corrected_dp),
            uncorrected_quadratic_error=float(abs(cv(qdp)-qmp)),
            corrected_using_mp_residual_error=float(abs(qc-qmp)),
            corrected_using_dp_residual_error=float(abs(cv(corrected_dp)-qmp)),
            exact_input_residual_product_bound=float(radius),
            fp64_residual_product_bound=float(np.linalg.norm(rz)*np.linalg.norm(rb)/abs(z.imag)),
            mp_residual_norm=float(norm(rzm)),dp_residual_norm=float(np.linalg.norm(rz)),
            residual_formation_error=float(residual_error),
            upper_half_filter_pole_weight=2/64/float(y@y),
            corrected_filter_pole_absolute_error=float(abs(mp.re(cv(corrected_dp)-qmp)))*2/64/float(y@y),
            elapsed_seconds=time.time()-start,
            status='Arithmetic diagnostic for the fixed compressed model only; excludes construction, dropped terms, transforms, and input-rounding certification.')
        assert abs(qc-qmp)<=radius
        results.append(row)
        output.write_text(json.dumps(dict(decimal_digits=80,source_sha256=sources,cases=results),indent=2)+'\n')
        print(json.dumps(row,indent=2),flush=True)


if __name__ == '__main__':
    main()
