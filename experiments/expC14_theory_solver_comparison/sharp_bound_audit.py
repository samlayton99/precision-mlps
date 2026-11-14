"""Outward evaluation of existing, unsimplified analytic-witness bounds.

Same witness, envelopes, D and S. This audits allowed cutoffs; it does not
rerun fits or claim that the practical 2**-52 cutoff meets A.65.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import gmpy2 as gp
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import theory_parameters as tp
from experiments.expC14_theory_solver_comparison.posterior import (
    point,add,sub,mul,div,neg,directed,PREC)
from experiments.expC14_theory_solver_comparison.option_c import upper


def sharp_per_B(N,cfg):
    with gp.context(gp.get_context(),precision=PREC):
        pi=(directed(gp.const_pi,gp.RoundDown),directed(gp.const_pi,gp.RoundUp))
        log2=(directed(lambda:gp.log(2),gp.RoundDown),directed(lambda:gp.log(2),gp.RoundUp))
        h=div(point(2),point(N));lam=point(cfg['lambda']);delta=point(cfg['delta'])
        gamma=div(lam,h);d=div(pi,mul(point(2),gamma))
        krange=div(delta,mul(point(8),d))
        k=int(gp.ceil(krange[1]));assert int(gp.ceil(krange[0]))==k
        sigma=mul(point(2*k),d)
        assert sigma[0]>=delta[1]/4 and sigma[1]<=delta[0]/2
        R=math.ceil(cfg['a_H']*math.sqrt(N));m=(R+1)//2
        theta=tp.exp(neg(mul(lam,point(R+.5))))
        Dlambda=add(div(pi,mul(point(2),lam)),div(mul(point(4),log2),pi))
        # Existing analytic-density bound, before relaxing it to 4 B / delta.
        M_per_B=div(point(1),sub(sub(delta,sigma),d))
        horizontal=div(mul(point(4),M_per_B),sub(tp.exp(div(mul(mul(point(2),pi),sigma),h)),point(1)))
        pole_sum=tp.total([div(point(1),sub(tp.exp(div(mul(point(2*j+1),mul(pi,pi)),lam)),point(1))) for j in range(k)])
        pole=mul(div(mul(mul(point(4),pi),M_per_B),gamma),pole_sum)
        boundary=mul(mul(div(mul(mul(point(4),h),Dlambda),delta),
                         div(theta,tp.power(sub(point(1),theta),2))),
                     tp.power(mul(point(2),theta),m))
        total=tp.total([horizontal,pole,boundary])
        return dict(Esharp_per_B_upper=tp.fhi(total[1]),
                    horizontal_per_B_upper=tp.fhi(horizontal[1]),
                    pole_per_B_upper=tp.fhi(pole[1]),boundary_per_B_upper=tp.fhi(boundary[1]),
                    R=R,m=m,k=k)


def main():
    source=ROOT/'results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_review_64_768'
    cfg=yaml.safe_load((source/'config.yaml').read_text())
    output=source.parent/'sharp_bound_audit_local_density';output.mkdir(exist_ok=False)
    rows=[]
    for N in cfg['N']:
        sharp=sharp_per_B(N,cfg);params=tp.prepare(N,cfg)
        prior=json.loads((source/f'N{N}.json').read_text())
        smax=float(np.load(source/f'coefficients_N{N}.npz')['singular'][0])
        Eb=min(params['E_per_B_upper'],sharp['Esharp_per_B_upper'])
        for j,row in enumerate(prior):
            old=row['theory']
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
                Slo=gp.mpfr(params['B'][j])*gp.mpfr(params['L_lower'])
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundUp):
                Ebar=gp.mpfr(params['B'][j])*gp.mpfr(Eb)
            tau=upper(lambda:gp.mpfr(row['matrix_bounds']['aW'])+(Ebar+gp.mpfr(old['bW']))/Slo)
            with gp.context(gp.get_context(),precision=PREC,round=gp.RoundDown):
                slack=(gp.mpfr(tau)-gp.mpfr(row['matrix_bounds']['aW']))*Slo-Ebar-gp.mpfr(old['bW'])
            assert slack>=0
            minimum_floor=upper(lambda:(gp.mpfr(row['matrix_bounds']['aW'])+gp.mpfr(old['bW'])/Slo)/gp.mpfr(smax))
            rows.append(dict(N=N,W=row['W'],target=row['target'],**sharp,
                old_E_per_B_upper=params['E_per_B_upper'],
                old_absolute_cutoff=old['tau'],new_absolute_cutoff=tau,
                old_relative_cutoff=old['tau']/smax,new_relative_cutoff=tau/smax,
                numerical_relative_floor=minimum_floor,cutoff_inequality_pass=True))
        e=rows[-6]
        print(f"N={N}: old rcond {e['old_relative_cutoff']:.6e}, new {e['new_relative_cutoff']:.6e}; Esharp/B {Eb:.6e}")
    (output/'thresholds.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(
        claim='Same analytic witness; minimum of old bound and sum of existing Hbound/Pbound/bd-general.',
        computation='256-bit outward interval arithmetic; existing SVD numerical bounds held fixed; no solver rerun.',
        source=str(source),config=cfg,source_hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest()
        for f in ['sharp_bound_audit.py','theory_parameters.py','posterior.py','option_c.py']}),indent=2)+'\n')


if __name__=='__main__':main()
