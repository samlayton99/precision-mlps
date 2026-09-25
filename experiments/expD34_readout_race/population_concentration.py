"""Population-structure bounds for exact tanh; numerical execution on Modal.

No force or reinforcement measurements enter the bound coefficients. Recorded
forces are used only to audit slack. The geometry-shape comparison is a labeled
reference to the earlier, more informative structural theorem.
"""
import math

import numpy as np

from .population_balance_dynamics import A3, A5, A7, C3, C5, C7


def population_shape(p):
    hidden = np.asarray(p[:-1]).reshape(3, -1)
    energy = np.sum(hidden*hidden, axis=0)
    mass = float(np.sum(energy))
    width = len(energy)
    return dict(M=mass, **{f'C{k}':float(width**(k/2-1)*np.sum(energy**(k/2))/mass**(k/2))
                         for k in (4, 6, 8, 10, 14)})


def target_structure(x, y):
    """Fixed target projections and analytic constants for a symmetric measure."""
    x, y = np.asarray(x), np.asarray(y)
    if not np.allclose(x, -x[::-1], rtol=0, atol=16*np.finfo(float).eps):
        raise ValueError('The sharper constants require a symmetric grid')
    mu2, mu4, mu6 = (float(np.mean(x**k)) for k in (2, 4, 6))
    s2, s3 = mu4-mu2*mu2, mu6-mu4*mu4/mu2
    q, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, 5))
    coefficients = q.T@y/math.sqrt(len(x))
    return dict(tau3=float(np.linalg.norm(coefficients[2:4])),
                tau5=float(np.linalg.norm(coefficients[2:6])),
                fine_norm=float(np.linalg.norm(y-q[:, :2]@(q[:, :2].T@y))/math.sqrt(len(x))),
                target_norm=float(np.linalg.norm(y)/math.sqrt(len(x))),
                s2=s2, s3=s3, D3=math.sqrt(8*s2/27+3*s3/16),
                B3=math.sqrt(s2/64+3*s3/256))


def bounds(radius, row, target, residual_bound):
    """Four nested levels of information, with explicit population scalings."""
    w = row['width']
    m = radius*radius
    s4 = row['C4']*m*m/w
    s6 = row['C6']*m**3/w**2
    root6 = math.sqrt(row['C6'])*radius**3/w
    root10 = math.sqrt(row['C10'])*radius**5/w**2
    root14 = math.sqrt(row['C14'])*radius**7/w**3
    cap_generic = A3*s4
    cap = min(cap_generic, target['B3']*s4+A5*s6)
    jac_generic = C3*root6
    jac_projected = target['D3']*root6+C5*root10
    generic = residual_bound*jac_generic
    projected = min(generic, residual_bound*jac_projected)
    cubic = target['D3']*root6*(target['tau3']+cap)+C5*residual_bound*root10
    quintic = (target['D3']*root6*target['tau3']+C5*root10*target['tau5']
               +(target['D3']*root6+C5*root10)*cap+C7*residual_bound*root14)
    target_bound = min(projected, cubic, quintic)
    # Existing Section 8 comparison, retained as a reference. These polynomial
    # coherences are population observables; they are not fitted force values.
    if 'q3' in row:
        Q = row['q3']*m*m/w+row['q5']*m**3/w**2+A7*row['C8']*m**4/w**3
        J = row['j3']*radius**3/w+row['j5']*radius**5/w**2+C7*root14
        G = row['g3']*radius**3/w+row['g5']*radius**5/w**2+C7*target['fine_norm']*root14
        reference = J*Q+G
    else:
        reference = math.nan
    return dict(generic=generic, projected=projected, target=target_bound,
                reference=reference, capacity=cap, capacity_generic=cap_generic,
                jacobian_generic=jac_generic, jacobian_projected=jac_projected,
                target_cubic=target['D3']*root6*target['tau3'],
                generated_cubic=target['D3']*root6*cap,
                cubic_remainder=C5*residual_bound*root10)


def concentration_envelope(initial_radius, clock, residual_bound, width, tracking=0.):
    """Bihari bound; tracking is accumulated hidden-path upper allowance.

    Also bounds native GD with left-point sums, provided the residual bound and
    structural quantities apply at every iterate. Nonpositive denominators end
    this sufficient comparison, not the network trajectory.
    """
    start = initial_radius+np.asarray(tracking)
    denominator = 1-2*C3*residual_bound*start*start*np.asarray(clock)/width
    return np.divide(start, np.sqrt(np.maximum(denominator, 0.)),
                     out=np.full(np.shape(denominator), np.nan), where=denominator > 0)


def gd_radius_envelope(initial_radius, speeds, tracking, eta):
    """Exact scalar Euler recurrence for supplied nondecreasing speed maps."""
    radius = [float(initial_radius)]
    for speed, drift in zip(speeds, tracking, strict=True):
        radius.append(radius[-1]+eta*(speed(radius[-1])+drift))
    return np.array(radius)
