"""Fixed exploratory target suite: analytic, nonanalytic, and discontinuous."""
from __future__ import annotations
import numpy as np
from scipy import special

TARGETS={
    "erf_transition": {"label":r"$\mathrm{erf}(12(x-0.15))$", "formula":"erf(12*(x-0.15))",
                       "regularity":"entire", "feature":"steep special-function transition"},
    "gamma_shifted": {"label":r"$\Gamma(0.05+2.475(x+1))$", "formula":"Gamma(0.05+2.475*(x+1))",
                      "regularity":"analytic on a neighborhood of [-1,1]", "feature":"near-pole boundary variation; argument stays in [0.05,5]"},
    "airy_transition": {"label":r"$\mathrm{Ai}(-30x)$", "formula":"Ai(-30*x)",
                        "regularity":"entire", "feature":"transition from rapid decay to oscillation"},
    "signed_sqrt_cusp": {"label":r"$\mathrm{sgn}(x-0.137)\sqrt{|x-0.137|}$",
                         "formula":"sign(x-0.137)*sqrt(abs(x-0.137))", "regularity":"continuous, unbounded derivative at 0.137",
                         "feature":"off-grid cusp"},
    "smooth_bump": {"label":"Smooth compact bump", "formula":"exp(-1/(1-(x/0.35)**2)) for abs(x)<0.35; 0 otherwise",
                    "regularity":"C-infinity, nonanalytic at support boundaries", "feature":"localized signal with exactly flat tails"},
    "oscillatory_cusp": {"label":r"$x\sin(1/x)$", "formula":"x*sin(1/x) for x!=0; 0 at x=0",
                         "regularity":"continuous, nondifferentiable at zero", "feature":"infinitely increasing local frequency with vanishing amplitude"},
    "shifted_step": {"label":r"$\mathbf{1}_{x\geq 0.137}$", "formula":"1 if x>=0.137 else 0",
                     "regularity":"discontinuous", "feature":"one off-grid jump"},
    "square_wave": {"label":r"$\mathrm{sgn}(\sin(7\pi x+0.37))$", "formula":"sign(sin(7*pi*x+0.37))",
                    "regularity":"discontinuous", "feature":"multiple jumps and alternating plateaus"},
}


def target(name,x):
    x=np.asarray(x,dtype=np.float64)
    if name=="erf_transition": return special.erf(12*(x-.15))
    if name=="gamma_shifted": return special.gamma(.05+2.475*(x+1))
    if name=="airy_transition": return special.airy(-30*x)[0]
    if name=="signed_sqrt_cusp":
        z=x-.137
        return np.sign(z)*np.sqrt(np.abs(z))
    if name=="smooth_bump":
        z=x/.35
        out=np.zeros_like(x)
        inside=np.abs(z)<1
        out[inside]=np.exp(-1/(1-z[inside]**2))
        return out
    if name=="oscillatory_cusp":
        out=np.zeros_like(x)
        nonzero=x!=0
        out[nonzero]=x[nonzero]*np.sin(1/x[nonzero])
        return out
    if name=="shifted_step": return (x>=.137).astype(np.float64)
    if name=="square_wave": return np.sign(np.sin(7*np.pi*x+.37))
    raise KeyError(name)


def special_probe_points():
    """Fixed near-cusp, support-boundary, and one-sided jump probes."""
    centers=[0.,.137,-.35,.35,-1.,1.]
    centers += [(k*np.pi-.37)/(7*np.pi) for k in range(-7,8)]
    offsets=[0.,-1e-8,1e-8,-1e-6,1e-6,-1e-4,1e-4]
    return np.unique([c+d for c in centers for d in offsets if -1<=c+d<=1])
