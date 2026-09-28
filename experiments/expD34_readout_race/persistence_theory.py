"""Finite-horizon bounds using a starting state and an autonomous linear forecast.

Inequalities are real-arithmetic analytical bounds. Their FP64 evaluations are
not directed-rounding interval certificates; no future true states enter them.
"""
from __future__ import annotations

import numpy as np


def tensors(p, x, y):
    w = (len(p)-1)//3; a, b, c = p[:-1].reshape(3, w)
    h = np.tanh(x[:, None]*a+b)
    ex = np.exp(-2*np.abs(x[:, None]*a+b)); s = 4*ex/(1+ex)**2
    r = h @ c+p[-1]-y
    J = np.column_stack((x[:, None]*s*c, s*c, h, np.ones(len(x))))
    gram = J.T @ J/len(x); g = J.T @ r/len(x)
    second = -2*h*s
    rr = np.zeros((w, 3, 3))
    rr[:, 0, 0] = c*((r*x*x) @ second)/len(x)
    rr[:, 0, 1] = rr[:, 1, 0] = c*((r*x) @ second)/len(x)
    rr[:, 1, 1] = c*(r @ second)/len(x)
    rr[:, 0, 2] = rr[:, 2, 0] = (r*x) @ s/len(x)
    rr[:, 1, 2] = rr[:, 2, 1] = r @ s/len(x)
    H = gram.copy()
    indices = np.arange(w)[:, None]+w*np.arange(3)[None, :]
    H[indices[:, :, None], indices[:, None, :]] += rr
    return dict(r=r, J=J, gram=gram, g=g, curvature=rr, hessian=H,
                jacobian_bound=np.linalg.norm(J)/np.sqrt(len(x)))


def ball_constants(p, state, radius, eta=.002):
    """Uniform Hessian bounds on the Euclidean ball of the given radius.

The model has |x|<=1, |tanh'|<=1, |tanh''|<=4/(3 sqrt(3)),
and |tanh'''|<=2. Output Hessians are block diagonal by neuron.
"""
    w = (len(p)-1)//3
    cmax = np.max(abs(p[2*w:3*w]))+radius
    m2 = 8*cmax/(3*np.sqrt(3))+np.sqrt(2)
    m3 = 4*np.sqrt(2)*cmax+8/np.sqrt(3)
    j0 = state['jacobian_bound']; r0 = np.linalg.norm(state['r'])/np.sqrt(len(state['r']))
    residual_drift = j0*radius+.5*m2*radius**2
    curvature_drift = m2*residual_drift+m3*r0*radius
    eig = np.linalg.eigvalsh(state['curvature'])
    negative = max(0., -eig.min())+curvature_drift
    upper = (j0+m2*radius)**2+np.max(abs(eig))+curvature_drift
    third = 3*(j0+m2*radius)*m2+(r0+residual_drift)*m3
    beta = max(1+eta*negative, abs(1-eta*upper))
    return dict(negative=negative, upper=upper, third=third, beta=beta,
                curvature_drift=curvature_drift, m2=m2, m3=m3)


def geometric(beta, n):
    """Sum of beta**j for 0 <= j < n, including beta=1."""
    if beta == 1: return float(n)
    exponent = n*np.log(beta)
    if exponent > 700: return np.inf
    return np.expm1(exponent)/(beta-1)


def acquisition_distance(a, fraction, threshold):
    k = int(np.ceil(fraction*len(a)))
    gaps = np.sort(np.maximum(threshold-abs(a), 0))
    return np.linalg.norm(gaps[:k])


def local_horizons(p, x, y, eta=.002):
    state = tensors(p, x, y); g0 = np.linalg.norm(state['g'])
    rows = []
    for radius in np.geomspace(1e-6, 1., 121):
        constants = ball_constants(p, state, radius, eta)
        beta = constants['beta']
        n = int(np.floor(np.log1p(radius*(beta-1)/(eta*g0))/np.log(beta))) if beta > 1 else int(radius/(eta*g0))
        while n > 0 and eta*g0*geometric(beta, n) >= radius: n -= 1
        rows.append(dict(radius=radius, updates=n, time=eta*n,
                         parameter_path=eta*g0*geometric(beta, n), initial_gradient=g0, **constants))
    return rows


def frozen_spectrum(p, x, y):
    state = tensors(p, x, y)
    # SVD avoids inventing positive or negative tiny eigenvalues from J.T @ J.
    _, singular, vt = np.linalg.svd(state['J']/np.sqrt(len(x)), full_matrices=False)
    values = singular**2; vectors = vt.T; loading = vt @ state['g']
    qc = np.stack((np.ones_like(x), x/np.sqrt(np.mean(x*x))))
    jc = qc @ state['J']/len(x)
    projected = vectors-jc.T @ np.linalg.solve(jc @ jc.T, jc @ vectors)
    return dict(p0=p.copy(), values=values, vectors=vectors, loading=loading,
                projected=projected, gram=state['gram'], initial=state)


def frozen_at(spectrum, n, eta=.002):
    values = spectrum['values']; loading = spectrum['loading']
    if eta*values.max() >= 1: raise ValueError('Requires nonoscillating linear GD')
    exponent = n*np.log1p(-eta*values)
    integral = np.divide(-np.expm1(exponent), values, out=np.full_like(values, eta*n), where=values > 0)
    p = spectrum['p0']-spectrum['vectors'] @ (integral*loading)
    g = spectrum['vectors'] @ (np.exp(exponent)*loading)
    w = (len(p)-1)//3
    effective = spectrum['projected'][:w] @ (np.exp(exponent)*loading)
    # Triangle inequality on each spectral component's full discrete path.
    slope_path = np.linalg.norm(spectrum['vectors'][:w], axis=0) @ (integral*abs(loading))
    return p, g, effective, float(slope_path)


def frozen_enclosure(p0, x, y, horizon=500000, block=1000, eta=.002):
    """A predictive error-tube attempt; stop at the first unclosed block.

Every candidate radius bounds the nonlinear loss on a whole ball. The defect
bound covers every predicted update in the block, not just sampled endpoints.
"""
    spectrum = frozen_spectrum(p0, x, y); radius_error = 0.; rows = []
    for offset in range(0, horizon, block):
        count = min(block, horizon-offset)
        p, ghat, _, _ = frozen_at(spectrum, offset, eta)
        state = tensors(p, x, y)
        defect = np.linalg.norm(state['g']-ghat)
        hessian_defect = np.linalg.norm(state['hessian']-spectrum['gram'])
        predicted_path = eta*count*np.linalg.norm(ghat)
        base = radius_error+predicted_path+eta*count*defect
        closed = False; next_error = np.inf; chosen = None
        for radius in np.maximum(base, 1e-14)*np.geomspace(1.000001, 128., 121):
            if radius > 1.: break
            constants = ball_constants(p, state, radius, eta)
            max_defect = defect+hessian_defect*predicted_path+.5*constants['third']*predicted_path**2
            amplification = 1+(constants['beta']-1)*geometric(constants['beta'], count)
            next_error = amplification*radius_error+eta*max_defect*geometric(constants['beta'], count)
            chosen = constants | dict(ball_radius=radius, maximum_defect=max_defect)
            if next_error+predicted_path <= radius:
                closed = True; break
        if chosen is None: chosen = dict(ball_radius=base, maximum_defect=np.nan,
            negative=np.nan, upper=np.nan, third=np.nan, beta=np.nan, curvature_drift=np.nan, m2=np.nan, m3=np.nan)
        row = dict(offset=offset, end=offset+count, closed=closed, initial_error=radius_error,
            end_error=next_error, predicted_path=predicted_path, point_defect=defect,
            hessian_defect=hessian_defect, **chosen)
        for threshold in (1., 3.2, 16.):
            # Distance to an acquisition event is 1-Lipschitz. This covers
            # all intermediate predicted states and all actual GD updates.
            gaps = np.sort(np.maximum(threshold-abs(p[:(len(p)-1)//3]), 0))
            distances = np.sqrt(np.cumsum(gaps*gaps))
            row[f'maximum_acquired_count_{threshold:g}'] = int(np.count_nonzero(distances <= next_error+predicted_path)) if closed else -1
        rows.append(row)
        if not closed: break
        radius_error = next_error
    return rows
