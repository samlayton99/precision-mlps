"""Initial-state aggregate curvature certificates for exact effective flow.

The FP64 calculation proposes travel radii; Arb verifies every reported
certificate from the archived binary64 data. No future trajectory is used.
All numerical execution is delegated to Modal by population_initial_modal.py.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from flint import arb, ctx


def initial_float(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    h = np.tanh(x[:, None]*a+b)
    s = 1-h*h
    basis = np.column_stack((np.ones_like(x), (x-x.mean())/x.std()))
    J = np.column_stack((s*c*x[:, None], s*c, h, np.ones_like(x)))
    residual = h@c+p[-1]-y
    ec = basis.T@residual/len(x)
    eh = residual-basis@ec
    jc = basis.T@J/len(x)
    gram = jc@jc.T
    raw = J.T@eh/len(x)
    ell = np.linalg.solve(gram, jc@raw)
    F = raw-jc.T@ell
    psi = eh-basis@ell
    loaded = blocks_float(p, x, psi)
    r2 = (p[:-1].reshape(3, -1)**2).sum(axis=0)
    return dict(f0=np.linalg.norm(F), Y0=np.sqrt(np.mean(eh*eh)),
                target_norm=np.sqrt(np.mean(y*y)), sigma0=np.sqrt(np.linalg.eigvalsh(gram)[0]),
                P0=np.sqrt(r2.sum()), B0=(r2**3).sum()**(1/6),
                H0=np.linalg.norm(loaded), M4=len(r2)*np.sum(r2*r2),
                coarse_norm=np.linalg.norm(ec), ell_norm=np.linalg.norm(ell))


def blocks_float(p, x, psi):
    a, b, c = p[:-1].reshape(3, -1)
    h = np.tanh(x[:, None]*a+b); s = 1-h*h
    aa = -2*psi[:, None]*c*h*s
    ac = psi[:, None]*s
    blocks = np.zeros((len(a), 3, 3))
    blocks[:, 0, 0] = (aa*x[:, None]**2).mean(axis=0)
    blocks[:, 0, 1] = blocks[:, 1, 0] = (aa*x[:, None]).mean(axis=0)
    blocks[:, 1, 1] = aa.mean(axis=0)
    blocks[:, 0, 2] = blocks[:, 2, 0] = (ac*x[:, None]).mean(axis=0)
    blocks[:, 1, 2] = blocks[:, 2, 1] = ac.mean(axis=0)
    return blocks


def constants(state, radius, sqrt):
    """Same algebra for floating estimates and Arb outward enclosures."""
    P, b = state['P0']+radius, state['B0']+radius
    h2 = sqrt(2)+4*P
    h3 = 4*sqrt(2)*P+8/sqrt(3)
    sigma = state['sigma0']-h2*radius
    if not sigma > 0:
        return None
    j = 3*b**3
    ell = j*state['Y0']/sigma
    dell = 6*h2*state['Y0']*b**3/sigma**2+(9*b**6+6*sqrt(2)*state['Y0']*b**2)/sigma
    psi = sqrt(state['Y0']**2+ell**2)
    L = h2*(j+dell)+h3*psi
    return dict(sigma=sigma, L=L, h2=h2, h3=h3, dell=dell)


def initial_arb(p, xx, yy):
    """Two streaming passes; no sample-by-parameter Arb array is stored."""
    started = time.monotonic()
    w, m = (len(p)-1)//3, len(xx)
    pp = [arb(float(v)) for v in p]
    a, b, c = pp[:w], pp[w:2*w], pp[2*w:3*w]
    xs, ys = [arb(float(v)) for v in xx], [arb(float(v)) for v in yy]
    mean = sum(xs, arb(0))/m
    std = (sum(((x-mean)**2 for x in xs), arb(0))/m).sqrt()
    qs = [(x-mean)/std for x in xs]
    jc0, jc1, g = ([arb(0) for _ in pp] for _ in range(3))
    residuals = []; ec0 = arb(0); ec1 = arb(0)
    for x, y, q in zip(xs, ys, qs):
        hs = [(aj*x+bj).tanh() for aj,bj in zip(a,b)]
        ss = [1-h*h for h in hs]
        r = pp[-1]+sum((cj*h for cj,h in zip(c,hs)),arb(0))-y
        residuals.append(r); ec0 += r; ec1 += q*r
        js = [cj*s*x for cj,s in zip(c,ss)]+[cj*s for cj,s in zip(c,ss)]+hs+[arb(1)]
        for k,j in enumerate(js):
            jc0[k] += j; jc1[k] += q*j; g[k] += r*j
    jc0, jc1, g = ([v/m for v in row] for row in (jc0,jc1,g))
    ec0 /= m; ec1 /= m
    dot = lambda u,v: sum((a*b for a,b in zip(u,v)),arb(0))
    k00,k01,k11 = dot(jc0,jc0),dot(jc0,jc1),dot(jc1,jc1)
    det = k00*k11-k01*k01
    if not det > 0:
        raise ValueError('Coarse rank not certified')
    raw = [v-u*ec0-wv*ec1 for v,u,wv in zip(g,jc0,jc1)]
    v0,v1 = dot(jc0,raw),dot(jc1,raw)
    ell0,ell1 = (k11*v0-k01*v1)/det,(k00*v1-k01*v0)/det
    F = [v-u*ell0-wv*ell1 for v,u,wv in zip(raw,jc0,jc1)]
    eh = [r-ec0-q*ec1 for r,q in zip(residuals,qs)]
    blocks = [[arb(0) for _ in range(5)] for _ in a]
    for x,q,r in zip(xs,qs,eh):
        psi = r-ell0-q*ell1
        for j,(aj,bj,cj) in enumerate(zip(a,b,c)):
            h=(aj*x+bj).tanh(); s=1-h*h
            aa=-2*psi*cj*h*s; ac=psi*s
            for k,value in enumerate((aa*x*x,aa*x,aa,ac*x,ac)):
                blocks[j][k] += value
    loaded2 = arb(0)
    for row in blocks:
        aa,ab,bb,ac,bc = [v/m for v in row]
        loaded2 += aa*aa+bb*bb+2*(ab*ab+ac*ac+bc*bc)
    r2 = [aj*aj+bj*bj+cj*cj for aj,bj,cj in zip(a,b,c)]
    Y = (dot(eh,eh)/m).sqrt()
    sigma = ((k00+k11-((k00-k11)**2+4*k01*k01).sqrt())/2).sqrt()
    result = dict(f0=dot(F,F).sqrt().upper(), Y0=Y.upper(), Y0_lower=Y.lower(),
                  target_norm=(dot(ys,ys)/m).sqrt().upper(), sigma0=sigma.lower(),
                  P0=sum(r2,arb(0)).sqrt().upper(), B0=sum((r**3 for r in r2),arb(0)).root(6).upper(),
                  H0=loaded2.sqrt().upper(), M4=(w*dot(r2,r2)).upper(),
                  coarse_norm=(ec0*ec0+ec1*ec1).sqrt().upper(),
                  ell_norm=(ell0*ell0+ell1*ell1).sqrt().upper())
    return result,time.monotonic()-started


def certify(state, radius, intervals=2048):
    A = arb(float(radius))
    const = constants(state,A,lambda x:arb(x).sqrt())
    if const is None:
        return dict(radius=radius,valid=False,reason='rank margin')
    L,H,f = const['L'].upper(),state['H0'],state['f0']
    # U is increasing. The right-endpoint Riemann sum bounds the exact
    # scalar comparison time from below, without a quadrature assumption.
    T = arb(0)
    for k in range(1,intervals+1):
        right = A*k/intervals
        T += (A/intervals)/(f+H*right+L*right*right/2)
    energy = state['Y0_lower']**2-2*(f*A+H*A*A/2+L*A**3/6)
    floor = energy.sqrt()/state['target_norm'] if energy > 0 else arb(0)
    # Round returned binary64 summaries down using nextafter; Arb strings
    # retain the actual proof enclosures.
    down = lambda value:float(np.nextafter(float(value.lower()),-np.inf))
    return dict(radius=radius,valid=True,time_lower=max(0.,down(T)),
                energy_relative_floor_lower=max(0.,down(floor)),
                enclosure={k:str(v) for k,v in dict(time=T,floor=floor,L=L,sigma=const['sigma']).items()})


def run(args):
    from scipy.integrate import quad
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.inputs,allow_pickle=False) as data:
        ps,x,ys = data['p'],data['x'],data['y']
        cases=json.loads(str(data['cases']))
    if np.max(np.abs(x)) > 1:
        raise ValueError('The derivative bounds require |x| <= 1')
    if ps.nbytes+ys.nbytes+x.nbytes > 4*1024**2:
        raise ValueError('Only the small explicit input capsule is allowed')
    radii=np.geomspace(1e-6,.1,41)
    results=[]
    for i,(p,y,case) in enumerate(zip(ps,ys,cases)):
        state=initial_float(p,x,y)
        proposals=[]
        for radius in radii:
            const=constants(state,radius,math.sqrt)
            if const is None: continue
            U=lambda u:state['f0']+state['H0']*u+const['L']*u*u/2
            T=quad(lambda u:1/U(u),0,radius,epsabs=1e-9)[0]
            energy=state['Y0']**2-2*(state['f0']*radius+state['H0']*radius**2/2+const['L']*radius**3/6)
            proposals.append(dict(radius=float(radius),time=float(T),floor=math.sqrt(max(0.,energy))/state['target_norm']))
        useful=[v for v in proposals if v['floor'] > .01]
        proposal=max(useful,key=lambda v:v['time']) if useful else None
        record=dict(index=i,case=case,initial=state,proposals=proposals,best_float=proposal)
        print(json.dumps(record),flush=True)
        if args.certify and proposal:
            ctx.prec=128
            exact,elapsed=initial_arb(p,x,y)
            record['arb_seconds']=elapsed
            record['arb_initial']={k:str(v) for k,v in exact.items()}
            record['certificate']=certify(exact,proposal['radius'])
            print(json.dumps(dict(index=i,certificate=record['certificate'],seconds=elapsed)),flush=True)
        results.append(record)
        (args.output/'initial.json').write_text(json.dumps(dict(scope='Exact effective flow; empirical binary64 samples; no GD certificate',
            precision=128,results=results),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--certify',action='store_true')
    run(parser.parse_args())
