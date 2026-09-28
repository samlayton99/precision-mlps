"""One-instance Arb checkpoint and chord-reference certificate feasibility probe."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from flint import arb, ctx

ctx.prec = 100

def upper(x):
    return x.upper()

def norm(v):
    return sum((z*z for z in v), arb(0)).sqrt()

def exact(v):
    return [arb(float(z)) for z in v]

def determinant(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))

def positive(a):
    return (a[0][0] > 0 and a[0][0]*a[1][1]-a[0][1]*a[1][0] > 0
            and determinant(a) > 0)

def spectral_bounds(a):
    # Floating eigenvalues propose endpoints only; Sylvester verifies each.
    lam = np.linalg.eigvalsh([[float(z.mid()) for z in row] for row in a])
    pad = arb(2)**-40
    for _ in range(60):
        lo = arb(float(lam[0]))-pad
        hi = arb(float(lam[-1]))+pad
        left = [[a[i][j]-(lo if i == j else 0) for j in range(3)] for i in range(3)]
        right = [[(hi if i == j else 0)-a[i][j] for j in range(3)] for i in range(3)]
        if positive(left) and positive(right):
            return lo.lower(), hi.upper()
        pad *= 2
    raise RuntimeError('Cannot verify 3x3 spectral endpoints')

def point(p, xx, yy):
    start = time.perf_counter()
    w = (len(p)-1)//3
    a,b,c = [exact(p[k*w:(k+1)*w]) for k in range(3)]
    d = arb(float(p[-1]))
    g = [arb(0) for _ in p]
    blocks = [[[arb(0) for _ in range(3)] for _ in range(3)] for _ in range(w)]
    j2 = arb(0); r2 = arb(0); m = len(xx)
    for xf,yf in zip(xx, yy):
        x,y = arb(float(xf)),arb(float(yf))
        hs = [(aj*x+bj).tanh() for aj,bj in zip(a,b)]
        ss = [1-z*z for z in hs]
        residual = d+sum((cj*hj for cj,hj in zip(c,hs)),arb(0))-y
        r2 += residual*residual
        j2 += 1
        g[-1] += residual
        for j,(cj,h,s) in enumerate(zip(c,hs,ss)):
            t = -2*h*s
            g[j] += residual*cj*s*x
            g[w+j] += residual*cj*s
            g[2*w+j] += residual*h
            j2 += h*h+cj*cj*s*s*(1+x*x)
            aa = residual*cj*t; ac=residual*s
            blocks[j][0][0] += aa*x*x
            blocks[j][0][1] += aa*x
            blocks[j][1][1] += aa
            blocks[j][0][2] += ac*x
            blocks[j][1][2] += ac
    negative = arb(0); curvature = arb(0)
    for block in blocks:
        for i in range(3):
            for j in range(i,3):
                block[i][j] /= m
                block[j][i] = block[i][j]
        lo,hi=spectral_bounds(block)
        negative=max(negative, -lo)
        curvature=max(curvature, -lo, hi)
    return dict(g=[z/m for z in g], jacobian=upper((j2/m).sqrt()),
                residual=upper((r2/m).sqrt()), negative=negative,
                curvature=curvature, cmax=max(abs(z).upper() for z in c),
                seconds=time.perf_counter()-start)

def constants(s, radius, eta):
    cmax=s['cmax']+radius
    m2=8*cmax/(3*arb(3).sqrt())+arb(2).sqrt()
    m3=4*arb(2).sqrt()*cmax+8/arb(3).sqrt()
    drift=s['jacobian']*radius+m2*radius*radius/2
    cd=m2*drift+m3*s['residual']*radius
    negative=upper(s['negative']+cd)
    ceiling=upper((s['jacobian']+m2*radius)**2+s['curvature']+cd)
    # Spectrum is [-negative, ceiling]; 1+eta*ceiling is always safe,
    # but use both signed endpoint magnitudes to retain GD damping.
    beta=max(upper(1+eta*negative), upper(abs(1-eta*ceiling)))
    return beta,ceiling

def load(inputs, enclosure):
    raw=np.load(inputs)
    print('input_keys',raw.files,flush=True)
    evidence=np.load(enclosure)
    p0=evidence['p0']
    pp=raw['p']; yy=raw['y']; xx=raw['x']
    candidates=np.flatnonzero(np.all(pp == p0, axis=1))
    if len(candidates) != 1: raise ValueError('Need unique exact p0 match')
    return p0,xx,yy[candidates[0]],evidence

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--enclosure',required=True)
    parser.add_argument('--chunk',type=int,default=1000)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p0,x,y,archive=load(args.inputs,args.enclosure)
    if max(abs(x)) > 1: raise ValueError('|x| must be <=1')
    centers=archive['p']
    eta=arb(float(.002)); h=arb(1)/64; threshold=arb(1)/4
    radius=arb(0); w=(len(p0)-1)//3
    prefix=max(abs(arb(float(v))) for v in p0[:w])*h
    rows=[]; p=p0
    for start in range(0,args.steps,args.chunk):
        end=min(start+args.chunk,args.steps); length=end-start
        pn=centers[end-1]
        state=point(p,x,y)
        delta=[v-u for u,v in zip(exact(p),exact(pn))]
        distance=upper(norm(delta))
        velocity=[v/length for v in delta]
        defect0=upper(norm([eta*g+v for g,v in zip(state['g'],velocity)]))
        _,u_ref=constants(state,distance,eta)
        defect=upper(defect0+eta*u_ref*distance)
        for k in range(length):
            beta,_=constants(state,upper(radius+distance),eta)
            radius=upper(beta*radius+defect)
            if not radius.is_finite() or radius > 100:
                raise RuntimeError('Vacuous growing radius')
        # Chord coordinate absolute values are maximized at endpoints;
        # radius is nondecreasing since beta>=1 and defect>=0.
        chordmax=max(abs(arb(float(v))) for v in np.r_[p[:w],pn[:w]])
        prefix=max(prefix,upper(h*(chordmax+radius)))
        row=dict(start=start,end=end,radius=str(radius),prefix=str(prefix),
                 point_seconds=state['seconds'],distance=str(distance),
                 defect0=str(defect0),defect_uniform=str(defect),
                 all_excluded=bool(prefix < threshold))
        rows.append(row)
        print(json.dumps(row),flush=True)
        p=pn
    Path(args.output).write_text(json.dumps(dict(precision=ctx.prec,
        exact_data='archived binary64 x,y,p0; exact GD with binary64 eta',
        reference='exact chord interpolation of archived binary64 centers',
        rows=rows),indent=2))

if __name__ == '__main__': main()
