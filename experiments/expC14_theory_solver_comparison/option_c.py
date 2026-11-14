"""Option C: original theorem, unchanged A.51 bound and A.65 cutoff.

Only admissible geometry and validated nearby-problem bounds are improved.
No practical-cutoff replacement, norm-guard algorithm or posterior shortcut.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import gmpy2 as gp
import numpy as np
from scipy.linalg import lstsq,svd
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.theory_parameters import prepare,PREC,fhi
from experiments.expC14_theory_solver_comparison import tight_products as tp

OUT=r.OUT/'option_c_jacobi'


def fnorm(a):
    return tp.frobenius_upper(np.asarray(a).reshape(-1,1))


def upper(fn):
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):return fhi(fn())


def mp_upper(fn):
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):return +fn()


def bound_value(value,upward=True):
    """Finite JSON scalar; huge analytic bounds retain MPFR exponent range."""
    if abs(value)<=np.finfo(float).max:
        return float(np.nextafter(float(value),np.inf if upward else -np.inf))
    with gp.context(gp.get_context(),precision=PREC):
        v=gp.next_above(value) if upward else gp.next_below(value)
        digits,exponent,_=gp.digits(v,10,90)
        sign='-' if digits.startswith('-') else ''
        digits=digits.lstrip('-')
        return f'{sign}{digits[0]}.{digits[1:]}e{exponent-1:+d}'


def weighted_gram_bound(H,error,singular,left=False):
    weighted=singular[:,None]*H if left else H*singular
    measured=tp.spectral_upper(weighted)
    # Elementwise multiplication rounding and enclosure perturbation.
    rounding=r.gamma(2)*fnorm(weighted)
    return upper(lambda:gp.mpfr(measured)+gp.mpfr(rounding)+gp.mpfr(error)*gp.mpfr(singular[0]))


def certify_factorization(F,U,s,Vt,params):
    start=time.perf_counter();n=F.shape[1]
    hu=tp.enclosed_product(U.T,U,target=np.eye(n))
    hv=tp.enclosed_product(Vt,Vt.T,target=np.eye(n))
    oU=hu.spectral_upper();oV=hv.spectral_upper()
    assert 0<=oU<1 and 0<=oV<1
    Vplus=upper(lambda:gp.sqrt(1+gp.mpfr(oV)))
    Uplus=upper(lambda:gp.sqrt(1+gp.mpfr(oU)))
    residual=tp.enclosed_product(U*s,Vt,target=F)
    # U*s is rounded before the enclosed GEMM. Account for its multiplication
    # error using ||U diag(s)||_F <= Uplus*||s||2 and ||Vt||2 <= Vplus.
    multiply_allowance=upper(lambda:gp.mpfr(r.gamma(1))*gp.mpfr(Uplus)*gp.mpfr(fnorm(s))*gp.mpfr(Vplus))
    rF=upper(lambda:gp.mpfr(residual.spectral_upper())+gp.mpfr(multiply_allowance))
    hSigma=weighted_gram_bound(hu.value,hu.error_frobenius,s)
    SigmaG=weighted_gram_bound(hv.value,hv.error_frobenius,s,left=True)
    def sqrt_tail(o):return gp.mpfr(o)**2/(8*(1-gp.mpfr(o))**gp.mpfr('1.5'))
    aW=upper(lambda:gp.mpfr(params['eF'])+gp.mpfr(rF)+
             (gp.mpfr(hSigma)/2+sqrt_tail(oU)*gp.mpfr(s[0]))*gp.mpfr(Vplus)+
             gp.mpfr(SigmaG)/2+sqrt_tail(oV)*gp.mpfr(s[0]))
    info=dict(aW=aW,eF=params['eF'],rF=rF,multiplication_allowance=multiply_allowance,
              factor_residual_midpoint_frobenius=fnorm(residual.value),
              factor_residual_enclosure_error=residual.error_frobenius,
              oU=oU,oV=oV,Uplus=Uplus,Vplus=Vplus,H_Sigma_bound=hSigma,Sigma_G_bound=SigmaG,
              U_gram_enclosure_error=hu.error_frobenius,V_gram_enclosure_error=hv.error_frobenius,
              certificate_seconds=time.perf_counter()-start)
    return hu,info


def projection_bounds(U,yn,p,hu,bounds,s):
    # p is the actual FP64 pairwise projection; residual enclosed before
    # final rounding, so cancellation cannot erase the diagnostic accuracy.
    diff=tp.enclosed_product(U.T,yn,target=p)
    qhat=p+diff.value
    bs=[];details=[]
    for j in range(yn.shape[1]):
        qnorm=fnorm(qhat[:,j])
        qerror=upper(lambda:gp.mpfr(diff.error_frobenius)+gp.mpfr(r.gamma(1))*gp.mpfr(qnorm))
        dp0=upper(lambda:gp.mpfr(fnorm(diff.value[:,j]))+gp.mpfr(diff.error_frobenius))
        hq=hu.value@qhat[:,j]
        hqerr=upper(lambda:gp.mpfr(r.gamma(len(s)+2))*gp.mpfr(fnorm(hu.value))*gp.mpfr(qnorm)+
                    gp.mpfr(hu.error_frobenius)*(gp.mpfr(qnorm)+gp.mpfr(qerror))+
                    gp.mpfr(bounds['oU'])*gp.mpfr(qerror))
        hqnorm=upper(lambda:gp.mpfr(fnorm(hq))+gp.mpfr(hqerr))
        Y=fnorm(yn[:,j]);ey=upper(lambda:gp.mpfr(r.gamma(4))*gp.mpfr(Y)+gp.mpfr('1e-40'))
        def tail():
            e=gp.mpfr(bounds['oU'])
            return 3*e*e/(8*(1-e)**gp.mpfr('2.5'))
        dp=upper(lambda:gp.mpfr(dp0)+gp.mpfr(hqnorm)/2+tail()*(gp.mpfr(qnorm)+gp.mpfr(qerror)))
        b=upper(lambda:gp.mpfr(ey)+gp.mpfr(dp)+gp.mpfr(r.UNIT)*gp.mpfr(fnorm(p[:,j]))+
                gp.mpfr(r.SUBNORMAL)*gp.mpfr(fnorm(s)))
        bs.append(b);details.append(dict(ey=ey,projection_error=dp,working_projection_error=dp0,
            weighted_polar_projection=hqnorm/2,projection_product_enclosure=diff.error_frobenius,bW=b))
    return bs,details


def solve_original_cutoff(U,s,Vt,p,params,bounds,bW,j):
    D=params['D']
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
        B=gp.mpfr(params['B'][j])
        S_hi=B*gp.mpfr(params['L_upper'])
        Ebar=B*gp.mpfr(params['E_per_B_upper'])
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
        S_lo=gp.mpfr(B)*gp.mpfr(params['L_lower'])
    tau=upper(lambda:(Ebar+gp.mpfr(bW))/S_lo+gp.mpfr(bounds['aW']))
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
        slack=(gp.mpfr(tau)-gp.mpfr(bounds['aW']))*S_lo-Ebar-gp.mpfr(bW)
    assert slack>=0,'Original cutoff lower inequality failed'
    rho=mp_upper(lambda:(gp.mpfr(tau)-gp.mpfr(bounds['aW']))*S_hi-Ebar-gp.mpfr(bW))
    keep=s>tau
    z=np.zeros_like(p[:,j]);z[keep]=p[keep,j]/s[keep]
    c=r.export(Vt,z,D)
    # Enclose the prescribed native export, including stored D error.
    mass=upper(lambda:gp.mpfr(fnorm(D))*gp.mpfr(fnorm(Vt))*gp.mpfr(fnorm(z)))
    kexport=2+math.ceil(math.log2(len(z)))
    eexport=upper(lambda:gp.mpfr(r.gamma(kexport))*gp.mpfr(mass)+
        gp.mpfr(fnorm(params['Derr']))*gp.mpfr(bounds['Vplus'])*gp.mpfr(fnorm(z))+gp.mpfr('1e-40'))
    deltaV=upper(lambda:gp.mpfr(bounds['oV'])/(1+gp.sqrt(1-gp.mpfr(bounds['oV']))))
    ec=mp_upper(lambda:gp.mpfr(eexport)+2*gp.mpfr(params['L_upper'])*S_hi*gp.mpfr(deltaV))
    T=mp_upper(lambda:2*gp.mpfr(params['L_upper'])*S_hi+ec)
    RX=mp_upper(lambda:2*Ebar+4*gp.mpfr(bounds['aW'])*S_hi+3*gp.mpfr(bW)+rho+ec)
    with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
        normpass=gp.mpfr(fnorm(z))<=2*S_lo
    assert normpass
    return c,dict(tau=tau,rank=int(keep.sum()),cutoff_pass=True,cutoff_slack_lower=bound_value(slack,False),
                  cutoff_analytic=upper(lambda:Ebar/S_lo),cutoff_matrix=bounds['aW'],
                  cutoff_labels=upper(lambda:gp.mpfr(bW)/S_lo),
                  Ebar=bound_value(Ebar),S_upper=bound_value(S_hi),S_lower=bound_value(S_lo,False),
                  rhoW=bound_value(rho),coefficient_bound=bound_value(T),sample_bound=bound_value(RX),export_error=bound_value(ec),
                  transformed_norm=fnorm(z),transformed_norm_pass=bool(normpass),
                  coefficient_l1=upper(lambda:sum(gp.mpfr(abs(float(x))) for x in c)))


def run_one(N,cfg,xe,ye,xv,yv):
    start=time.perf_counter();params=prepare(N,cfg)
    approximation_info=dict(recipe='original_A51',E_per_B_upper=params['E_per_B_upper'])
    if cfg.get('approximation_bound')=='sharp_existing_lemmas':
        from experiments.expC14_theory_solver_comparison.sharp_bound_audit import sharp_per_B
        sharp=sharp_per_B(N,cfg)
        approximation_info.update(recipe='minimum_original_A51_and_unsimplified_bounds',
                                  original_E_per_B_upper=params['E_per_B_upper'],**sharp)
        params['E_per_B_upper']=min(params['E_per_B_upper'],sharp['Esharp_per_B_upper'])
        approximation_info['E_per_B_upper']=params['E_per_B_upper']
    x=np.linspace(-1,1,N*cfg['sampling_density']+1)
    affine=cfg.get('feature_parameterization')=='exact_affine'
    if affine:
        from experiments.expC14_theory_solver_comparison import affine_grid
        A,y=affine_grid.training(N,params['R'],cfg['sampling_density'])
    else:
        y=r.target_values(x)
        A=r.feature_matrix(x,params['centers'],params['slope'])
    actual=[]
    for j in range(6):
        c,_,rank,_=lstsq(A,y[:,j],cond=cfg['actual_relative_cutoff'],lapack_driver='gelsd',check_finite=False)
        actual.append(c)
    actual=np.column_stack(actual)
    F=A*params['D']/math.sqrt(len(x));yn=y/math.sqrt(len(x))
    print(f'N={N}: computing FP64 {cfg["theory_driver"]} SVD, shape {F.shape}',flush=True)
    if cfg['theory_driver']=='gejsv':
        from experiments.expC14_theory_solver_comparison.jacobi_probe import jacobi_svd
        U,s,Vt,driver_info=jacobi_svd(F,joba=0,jobv=1)
    else:
        U,s,Vt=svd(F,full_matrices=False,lapack_driver=cfg['theory_driver'],check_finite=False)
        driver_info=dict(driver=cfg['theory_driver'])
    print(f'N={N}: enclosing SVD errors',flush=True)
    hu,bounds=certify_factorization(F,U,s,Vt,params)
    p=np.column_stack([r.projection(U,yn[:,j]) for j in range(6)])
    bs,projection_info=projection_bounds(U,yn,p,hu,bounds,s)
    theory=[];solver_info=[]
    for j in range(6):
        c,info=solve_original_cutoff(U,s,Vt,p,params,bounds,bs[j],j)
        sample_error=fnorm(F@(c/params['D'])-yn[:,j])
        info.update(projection_info[j]);info['sample_error_observed']=sample_error
        with gp.context(gp.get_context(),precision=PREC):
            info['sample_bound_pass']=bool(gp.mpfr(sample_error)<=gp.mpfr(info['sample_bound']))
            info['coefficient_bound_pass']=bool(gp.mpfr(info['coefficient_l1'])<=gp.mpfr(info['coefficient_bound']))
        if cfg.get('input_error_route')=='uniform_A81':
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
                # A.68 implies max sampled residual <=sqrt(M)*R_X.
                # Use this valid (usually loose) bound in the ORIGINAL A.81.
                T=gp.mpfr(info['coefficient_bound']);B=gp.mpfr(params['B'][j])
                max_res=gp.sqrt(len(x))*gp.mpfr(info['sample_bound'])
                rho=gp.mpfr(cfg['ellipse_rho']);degree=cfg['sampling_degree']
                ean=2*(B+T)/(rho-1)*rho**(-degree)
                # Affine rounding of gamma*t+beta: <=2u*max|gamma*t+beta|
                # before cancellation, then correctly-rounded tanh adds u.
                u=gp.mpfr(2)**-53
                maxbias=max(abs(params['slope']-cfg['lambda']*k) for k in range(-params['R'],N+params['R']+1))
                epsphi=4*u*(gp.mpfr(abs(params['slope']))+gp.mpfr(maxbias)+1)+gp.mpfr('1e-40')
                ksum=1+math.ceil(math.log2(len(c)))
                gam=ksum*u/(1-ksum*u)
                Eev=(epsphi+gam*(1+epsphi))*T+gp.mpfr('1e-40')
                uniform=2*max_res+3*ean+B*u/gp.mpfr(cfg['delta'])+Eev
                info['original_A81_uniform_bound']=bound_value(uniform)
                info['original_A81_sample_max_bound']=bound_value(max_res)
                info['uniform_activation_allowance']=fhi(epsphi)
        assert info['sample_bound_pass'] and info['coefficient_bound_pass']
        theory.append(c);solver_info.append(info)
    theory=np.column_stack(theory)
    del A,F,U,Vt,hu
    coefficients=np.column_stack((actual,theory))
    if affine:
        errors,maxima=affine_grid.evaluate(N,params['R'],coefficients,np.tile(ye,(1,2)))
        check,_=affine_grid.evaluate(N,params['R'],coefficients,np.tile(yv,(1,2)))
    else:
        errors,maxima=r.evaluate(xe,params['centers'],params['slope'],coefficients,np.tile(ye,(1,2)))
        check,_=r.evaluate(xv,params['centers'],params['slope'],coefficients,np.tile(yv,(1,2)))
    biases=params['slope']-cfg['lambda']*np.arange(-params['R'],N+params['R']+1)
    np.savez_compressed(OUT/f'coefficients_N{N}.npz',centers=params['centers'],gamma=params['slope'],hidden_biases=biases,actual=actual,theory=theory,D=params['D'],singular=s)
    rows=[]
    for j,name in enumerate(r.NAMES):
        rows.append(dict(N=N,W=len(params['centers']),R=params['R'],target=name,
            actual_error=float(errors[j]),theory_error=float(errors[j+6]),
            actual_validation_error=float(check[j]),theory_validation_error=float(check[j+6]),
            actual_linf=float(maxima[j]),theory_linf=float(maxima[j+6]),
            actual_l1=float(np.sum(abs(actual[:,j]))),
            geometry_checks=params['checks'],matrix_bounds=bounds,theory=solver_info[j],driver_info=driver_info,
            approximation_bound=approximation_info,
            input_error_route=cfg.get('input_error_route','RMS_A80'),feature_parameterization=cfg.get('feature_parameterization','centered'),
            design=dict(a_H=cfg['a_H'],lambda_value=cfg['lambda'],delta=cfg['delta'],Delta=cfg['Delta']),
            total_seconds=time.perf_counter()-start))
    (OUT/f'N{N}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(f'N={N} complete ({time.perf_counter()-start:.1f}s), aW={bounds["aW"]:.3e}: '+
          ', '.join(f"{v['target']} {v['actual_error']:.2e}/{v['theory_error']:.2e}" for v in rows),flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--N',nargs='+',type=int);parser.add_argument('--driver',choices=['gesvd','gesdd','gejsv']);parser.add_argument('--output',type=Path);parser.add_argument('--config',type=Path,default=r.HERE/'option_c_config.yaml');args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if args.N:cfg['N']=args.N
    if args.driver:cfg['theory_driver']=args.driver
    if args.output:OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'config.yaml').exists():
        previous=yaml.safe_load((OUT/'config.yaml').read_text())
        keys=['lambda','a_H','delta','Delta','sampling_density','sampling_degree','ellipse_rho',
              'eval_points','validation_points','actual_relative_cutoff','theory_driver',
              'input_error_route','feature_parameterization','extended_scalar_bounds','approximation_bound']
        if any(previous.get(k)!=cfg.get(k) for k in keys):
            raise ValueError('Output directory belongs to a different configuration; use --output')
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    sources=['option_c.py','theory_parameters.py','tight_products.py']
    if cfg['theory_driver']=='gejsv':sources.append('jacobi_probe.py')
    if cfg.get('feature_parameterization')=='exact_affine':sources.append('affine_grid.py')
    if cfg.get('approximation_bound')=='sharp_existing_lemmas':sources.append('sharp_bound_audit.py')
    manifest=dict(config=cfg,source_hashes={f:hashlib.sha256((r.HERE/f).read_bytes()).hexdigest() for f in sources},
        configuration_source=dict(path=str(args.config.resolve()),sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()),
        claim='Original theorem conditions and cutoff retained; measured accuracy compared on matched geometry',
        no_substitutions=['no norm guard','no practical cutoff in theoretical arm','no posterior-only proof','no high-precision SVD or coefficient refinement'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    xe=-1+2*(np.arange(cfg['eval_points'])+.5)/cfg['eval_points'];xv=-1+2*(np.arange(cfg['validation_points'])+.5)/cfg['validation_points']
    ye,yv=r.target_values(xe),r.target_values(xv)
    with threadpool_limits(limits=cfg['threads']):
        for N in cfg['N']:
            if not (OUT/f'N{N}.json').exists():run_one(N,cfg,xe,ye,xv,yv)


if __name__=='__main__':main()
