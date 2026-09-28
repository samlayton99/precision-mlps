"""Gamma-explicit antiperiodic bulk and finite-interval tanh corrections.

Raw readout coordinates are preserved. Analytic bounds are exact-arithmetic
statements; the floating-point diagnostics are not interval certificates.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, svd

from . import core, gamma_filter as gf


def bulk(n, q, gamma, length=2., tolerance=1e-20):
    """Half-odd Fourier blocks for q*n samples and n centers on [0,length).

    The half-period is length, so there is no square-wave parity nullspace
    within this half-period core. The full-period model is a different object.
    """
    harmonics = n
    while True:
        tail = gf.feature_remainder(gamma, harmonics, length, 0.)['tail']
        if tail <= tolerance:
            break
        harmonics += n
    modes = np.arange(-harmonics, harmonics)
    odd = 2*modes+1
    omega = np.pi*odd/length
    coefficient = 2/(1j*np.pi*odd)*gf.multiplier(gamma, omega)
    phase = np.arange(q)*length/(q*n)
    eigen = np.zeros((q, n), dtype=complex)
    for s in range(q):
        np.add.at(eigen[s], modes % n,
                  n*coefficient*np.exp(1j*omega*phase[s]))
    column = np.fft.ifft(eigen, axis=1)*np.exp(1j*np.pi*np.arange(n)/n)
    assert np.max(np.abs(column.imag)) < 1e-12
    rows = np.arange(q*n)
    displacement = rows[:, None]//q-np.arange(n)[None, :]
    values = column.real[rows[:, None] % q, displacement % n]
    values *= np.where(displacement < 0, -1., 1.)
    return values, eigen, float(tail), harmonics


def image_tail(gamma, length, distance, terms):
    """Alternating image-series remainder, summed over image locations."""
    return float(4*np.exp(-2*(terms+1)*gamma*distance)
                 /(-np.expm1(-2*(terms+1)*gamma*length)))


def choose_boundary(n, gamma, width, length=2., kernel_tolerance=1e-15):
    """Minimize the explicit core correction rank on a disclosed finite scan."""
    candidates = []
    h = length/n
    for boundary in range(1, n//2):
        for terms in range(1, 257):
            eps = np.sqrt(n-2*boundary)*image_tail(gamma, length, boundary*h, terms)
            delta = (2*np.sqrt(width+1)+eps)*eps
            if delta <= kernel_tolerance:
                candidates.append((2*boundary+2*terms, boundary, terms, eps))
                break
    if not candidates:
        raise ValueError('No nontrivial boundary split satisfies this scan')
    return min(candidates)[1:]


def construct(n, q, halo, gamma, length=2., boundary=None, terms=None):
    """Return J0+U V^T on endpoint grids, including bias and halo columns.

    Coordinates are shifted to [0,length] only for this construction. Column
    ordering is bias, then centers -halo,...,n+halo, matching the archive.
    """
    h = length/n
    x = np.arange(q*n+1)*h/q
    centers = np.arange(-halo, n+halo+1)*h
    m, width = len(x), len(centers)
    if boundary is None:
        boundary, terms, _ = choose_boundary(n, gamma, width, length)
    assert terms is not None and 0 < boundary < n/2
    values, phase_eigen, fourier_tail, harmonics = bulk(n, q, gamma, length)
    core_columns = 1+halo+np.arange(n)
    j0 = np.zeros((m, width+1))
    j0[:-1, core_columns] = values/np.sqrt(m)
    left, right = [], []

    def append(u, v):
        left.append(u)
        right.append(v)

    boundary_centers = np.r_[np.arange(boundary), np.arange(n-boundary, n)]
    for j in boundary_centers:
        u = np.zeros(m)
        u[:-1] = (np.tanh(gamma*(x[:-1]-j*h))-values[:, j])/np.sqrt(m)
        v = np.zeros(width+1)
        v[core_columns[j]] = 1.
        append(u, v)
    interior = np.arange(boundary, n-boundary)
    for ell in range(1, terms+1):
        rate = 2*gamma*ell
        factor = 2*(-1.)**(ell-1)/(1+np.exp(-rate*length))
        for positive in [True, False]:
            u = np.zeros(m)
            u[:-1] = np.exp(-rate*((length-x[:-1]) if positive else x[:-1]))/np.sqrt(m)
            v = np.zeros(width+1)
            v[core_columns[interior]] = factor*(1 if positive else -1)*np.exp(
                -rate*(interior*h if positive else length-interior*h))
            append(u, v)
    extra_columns = np.setdiff1d(np.arange(width+1), core_columns)
    for col in extra_columns:
        u = np.ones(m) if col == 0 else np.tanh(gamma*(x-centers[col-1]))
        v = np.zeros(width+1)
        v[col] = 1.
        append(u/np.sqrt(m), v)
    u = np.zeros(m)
    u[-1] = 1/np.sqrt(m)
    v = np.zeros(width+1)
    v[core_columns] = np.tanh(gamma*(length-np.arange(n)*h))
    append(u, v)
    u, v = np.column_stack(left), np.column_stack(right)
    approximate = j0+u@v.T
    epsilon = np.sqrt(n-2*boundary)*(fourier_tail+image_tail(gamma, length, boundary*h, terms))
    delta = (2*np.sqrt(width+1)+epsilon)*epsilon
    diagonal = np.zeros(width+1)
    diagonal[core_columns] = np.sum(np.abs(phase_eigen)**2, axis=0)/m
    return dict(j0=j0, u=u, v=v, approximate=approximate,
        core_columns=core_columns, diagonal=diagonal, phase_eigen=phase_eigen,
        x=x-length/2, centers=centers-length/2,
        n=n, q=q, halo=halo, gamma=gamma, length=length,
        boundary=boundary, terms=terms, harmonics=harmonics,
        fourier_tail=fourier_tail, feature_error=float(epsilon),
        analytic_kernel_error=float(delta), feature_correction_rank=u.shape[1])


def to_bulk_basis(values, construction):
    """Apply embedded half-odd DFT adjoint in coefficient coordinates."""
    result = np.asarray(values, dtype=complex).copy()
    indices = construction['core_columns']
    n = len(indices)
    phase = np.exp(-1j*np.pi*np.arange(n)/n)
    result[indices] = np.fft.fft(phase[:, None]*result[indices], axis=0)/np.sqrt(n)
    return result


def low_rank_gram(construction):
    """Return D+Q diag(s) Q*, retaining signed boundary cross terms.

    QR only changes coordinates; no small singular values are dropped.
    """
    j0, u, v = (construction[k] for k in ['j0', 'u', 'v'])
    r = u.shape[1]
    z = to_bulk_basis(np.column_stack((j0.T@u, v)), construction)
    b = np.block([[np.zeros((r, r)), np.eye(r)], [np.eye(r), u.T@u]])
    q, rmat = np.linalg.qr(z, mode='reduced')
    small = rmat@b@rmat.conj().T
    signed, rotation = eigh((small+small.conj().T)/2)
    return construction['diagonal'], q@rotation, signed


def secular_count(diagonal, vectors, signed, cutoff, drop=0.):
    """Count corrected eigenvalues below cutoff by a small inertia problem.

    Optional dropping has explicit operator error <=max(abs(dropped signed)).
    Cutoff must avoid bulk poles; reported zero pivots are not a certificate.
    """
    keep = np.abs(signed) > drop
    values, q = signed[keep], vectors[:, keep]
    d = diagonal-cutoff
    if np.any(d == 0):
        raise ValueError('Cutoff equals a bulk pole')
    small = -np.diag(1/values)-q.conj().T@(q/d[:, None])
    pivots = eigh((small+small.conj().T)/2, eigvals_only=True)
    count = np.sum(d < 0)+np.sum(pivots < 0)-np.sum(values > 0)
    return int(count), float(np.min(np.abs(pivots))), int(np.sum(keep))


def slow_mass(model, cutoff):
    return model['floor']+np.sum(model['weights'][model['rates'] <= cutoff], axis=0)


def buffered_mass(model, cutoff, buffer, normalized_error):
    """Two-sided target-mass transfer; no individual eigengap assumption."""
    assert 0 < buffer < cutoff
    lower = np.maximum(0., np.sqrt(slow_mass(model, cutoff-buffer))-normalized_error/buffer)**2
    upper = np.minimum(1., (np.sqrt(slow_mass(model, cutoff+buffer))+normalized_error/buffer)**2)
    return lower, upper
