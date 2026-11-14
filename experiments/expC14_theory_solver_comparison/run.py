"""Matched FP64 experiment: published gelsd readout vs Appendix A.3 recipe.

The attachment QI_MLPs___ICLR_2027_Submission (18).pdf is authoritative.
Geometry, envelope, scaling and cutoff equations: A.1,A.3,A.55--A.65.
This is NOT a claim of interval-certified near-machine continuous error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precisionmlps-c14-mpl")
import gmpy2 as gp
import numpy as np
from scipy.linalg import lstsq, svd
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/checkpoint_C_geometry/expC14_theory_solver_comparison"
UNIT = 2.0**-53
SUBNORMAL = float(np.nextafter(0., 1.))
NAMES = ["exp", "sine4", "sine24", "chirp", "weak_high_frequency", "packet"]


def up(value):
    return float(np.nextafter(float(value), np.inf))


def gamma(k):
    assert k * UNIT < 1
    return up(k * UNIT / (1 - k * UNIT))


def norm(a):
    return float(np.linalg.norm(a))


def pairwise_sum(a, axis=0):
    """Explicit balanced addition tree, including the unpaired last entry."""
    v = np.moveaxis(np.asarray(a, dtype=np.float64), axis, 0).copy()
    while len(v) > 1:
        half = len(v) // 2
        out = v[:2*half:2] + v[1:2*half:2]
        v = np.concatenate((out, v[-1:]), axis=0) if len(v) % 2 else out
    return v[0]


def projection(U, y):
    return np.concatenate([
        pairwise_sum(U[:, k:k+64] * y[:, None])
        for k in range(0, U.shape[1], 64)
    ])


def export(Vt, z, D):
    return D * np.array([pairwise_sum(Vt[:, k] * z) for k in range(len(z))])


def target_values(x, precision=160):
    """High-precision oracle, rounded once to FP64; no solver high precision."""
    values = np.empty((len(x), len(NAMES)))
    with gp.context(gp.get_context(), precision=precision):
        pi = gp.const_pi()
        def sinpi(t):
            # Preserve exact zeros instead of evaluating sin(k * rounded pi).
            return gp.mpfr(0) if t == gp.floor(t) else gp.sin(pi*t)
        for i, xx in enumerate(x):
            z = gp.mpfr(float(xx))
            values[i] = [
                float(gp.exp(z)), float(sinpi(4*z)),
                float(sinpi(24*z)), float(sinpi(8*(z+1)**2)),
                float(sinpi(2*z) + gp.mpfr(1)/1000*sinpi(40*z)),
                float(gp.exp(-8*(z-gp.mpfr(1)/5)**2)*sinpi(8*z)),
            ]
    return values


def analytic_B(cfg):
    """Valid rectangle bounds on U_delta([-1-Delta,1+Delta])."""
    with gp.context(gp.get_context(), precision=160):
        delta, Delta, pi = map(gp.mpfr, [cfg['delta'], cfg['Delta'], gp.const_pi()])
        return np.array([up(v) for v in [
            gp.exp(1+Delta+delta), gp.cosh(4*pi*delta),
            gp.cosh(24*pi*delta), gp.cosh(16*pi*(2+Delta+delta)*delta),
            gp.cosh(2*pi*delta)+gp.cosh(40*pi*delta)/1000,
            gp.exp(8*delta**2)*gp.cosh(8*pi*delta),
        ]])


def geometry(N, cfg):
    h = 2.0/N
    R = math.ceil(cfg['a_H']*math.sqrt(N))
    centers = -1 + np.arange(-R, N+R+1)*h
    slope = cfg['lambda']*N/2
    checks = dict(halo=(R+.5)*h <= cfg['Delta'],
                  strip=slope*cfg['delta'] >= 4*np.pi,
                  halo_order=cfg['lambda']*R >= 2*np.log(2),
                  minimum_N=N >= 16*max(1, cfg['a_H']**2),
                  exact_endpoints=(N & (N-1)) == 0)
    rho, p, s = cfg['ellipse_rho'], cfg['sampling_degree'], cfg['sampling_density']
    checks.update(ellipse_activation=cfg['lambda']*(rho-1/rho) <= np.pi,
                  ellipse_target=h/4*(rho+1/rho) < cfg['delta'],
                  sampling=s >= 2*(p+1)**2,
                  overdetermined=N*s+1 >= len(centers)+1)
    if cfg.get('input_error_route')=='uniform_A81':
        # A.81 requires Q([-1,1]) subset [-1,1], not exact cell endpoints.
        checks.pop('exact_endpoints')
        checks['uniform_input_rounding_condition']=True
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f'Nonadmissible N={N}: {checks}')
    return h, R, slope, centers, checks


def envelopes(N, cfg, precision=160):
    """A.56: outward halo slots; bias is LAST, as in the main-figure code."""
    h, R, slope, centers, _ = geometry(N, cfg)
    with gp.context(gp.get_context(), precision=precision):
        lam, delta = gp.mpfr(cfg['lambda']), gp.mpfr(cfg['delta'])
        hh, pi = gp.mpfr(h), gp.const_pi()
        d = pi/(2*gp.mpfr(slope))
        m = (R+1)//2
        zeta = gp.exp(-2*lam)
        products = [gp.mpfr(1)]
        for k in range(1, m+1):
            products.append(products[-1]*(1-zeta**k))
        alpha = [hh/(2*(delta-d)) for _ in centers]
        Dlam = pi/(2*lam)+4*gp.log(2)/pi
        for i in range(1, m+1):
            Li = zeta**(gp.mpfr(i*(i+1)-1)/2)/(products[i-1]*products[m-i])
            for j in range(1, m+1):
                if i != j:
                    Li *= 1+zeta**(gp.mpfr(j)-gp.mpfr('.5'))
            corr = hh*Dlam*Li/(2*delta)
            alpha[i-1] += corr
            alpha[-i] += corr
        alpha.append(1+sum(alpha))
        D = np.array([float(gp.sqrt(a)) for a in alpha])
        L = up(gp.sqrt(sum(alpha)))
        capp = min(pi*delta, lam*gp.mpfr(cfg['a_H'])**2)/8
        # Outward double enclosure of the analytic E_rep/B used by the cutoff.
        E_per_B = up(128*pi/(lam*delta)*(
            gp.exp(-capp*len(centers))+gp.exp(-pi*pi/lam)/len(centers)))
    return np.array([float(a) for a in alpha]), D, L, E_per_B


def feature_matrix(x, centers, slope):
    """Correctly rounded tanh of EXACT dyadic arguments, using a lookup table.

    All experimental grids are dyadic, so the arguments lie on one small
    uniform lattice. MPFR tanh at 53 bits rounds directly to FP64 (no FP64
    arithmetic is replaced by a high-precision readout solve).
    """
    args = slope*(x[:, None]-centers[None, :])
    step = float(slope*(x[1]-x[0]))
    base = float(args.min())
    ix = np.rint((args-base)/step).astype(np.int64)
    count = int(ix.max())+1
    lattice = base+step*np.arange(count)
    if not np.array_equal(lattice[ix], args):
        raise ValueError('Preactivation lattice is not exact')
    with gp.context(gp.get_context(), precision=53):
        table = np.array([float(gp.tanh(gp.mpfr(float(t)))) for t in lattice])
    return np.column_stack((table[ix], np.ones(len(x))))


def factorization_bounds(F, U, singular, Vt, L):
    """A.63 with Frobenius upper bounds and standard gamma_k allowances.

    gamma_k accounts for sequential dot products (also safely covers trees).
    These deliberately conservative allowances are not fitted to test error.
    Stored-array diagnostics are FP64; certification remains explicitly false.
    """
    M, n = F.shape
    reconstructed = (U*singular) @ Vt
    residual = norm(F-reconstructed)
    # || |U Sigma| |Vt| ||_F <= ||U Sigma||_F ||Vt||_F.
    reconstruction_allowance = gamma(n+2)*norm(U*singular)*norm(Vt)
    rF = up((residual+reconstruction_allowance)*(1+gamma(M*n+2)))
    del reconstructed
    u_observed = norm(U.T@U-np.eye(n))
    v_observed = norm(Vt@Vt.T-np.eye(n))
    oU = up((u_observed+gamma(M+2)*norm(U)**2)*(1+gamma(n*n+2)))
    oV = up((v_observed+gamma(n+2)*norm(Vt)**2)*(1+gamma(n*n+2)))
    if max(oU, oV) >= 1:
        raise ValueError('Orthogonality condition failed')
    deltaU = up(oU/(1+math.sqrt(1-oU)))
    deltaV = up(oV/(1+math.sqrt(1-oV)))
    Uplus, Vplus = up(math.sqrt(1+oU)), up(math.sqrt(1+oV))
    # CR tanh, D, sqrt(M), multiply/divide: eight rounding factors suffice;
    # constants are prepared at 160 bits and independently checked at 256.
    eF = up(gamma(8)*L)
    aW = up(eF+rF+singular[0]*(deltaU*Vplus+deltaV))
    return dict(eF=eF, residual_observed=residual, reconstruction_allowance=reconstruction_allowance,
                rF=rF, oU=oU, oV=oV, orthogonality_observed_U=u_observed,
                orthogonality_observed_V=v_observed, deltaU=deltaU, deltaV=deltaV,
                Uplus=Uplus, Vplus=Vplus, aW=aW)


def theorem_solve(F, y, U, singular, Vt, D, L, E_per_B, B, bounds):
    M, n = F.shape
    projection_y = projection(U, y)
    Y = up(norm(y)*(1+gamma(M+1)))
    gpw = gamma(1+math.ceil(math.log2(M)))
    beta = up((2*M-1)*SUBNORMAL/(1-(1+math.ceil(math.log2(M)))*UNIT))
    ey = up(gamma(4)*Y)
    ep = up(math.sqrt(n)*(gpw*bounds['Uplus']*Y+beta))
    bW = up(ey+bounds['deltaU']*Y+ep+UNIT*norm(projection_y)+SUBNORMAL*norm(singular))
    with gp.context(gp.get_context(), precision=160):
        Smp = gp.mpfr(B)*gp.mpfr(L)
        Emp = gp.mpfr(B)*gp.mpfr(E_per_B)
        eta = Emp+gp.mpfr(bounds['aW'])*Smp+gp.mpfr(bW)
        tau = up(eta/Smp)
        cutoff_pass = gp.mpfr(tau)*Smp >= eta
        rhoW = up(gp.mpfr(tau)*Smp-eta)
        S, Ebar = float(Smp), float(Emp)
    keep = singular > tau
    z = np.zeros_like(projection_y)
    z[keep] = projection_y[keep]/singular[keep]
    c = export(Vt, z, D)
    # A.64: |D| times the product/sum export error, plus polar-factor allowance.
    kexport = 2+math.ceil(math.log2(n))
    # Absolute dot-product mass: orthogonality does not control |V| |z|.
    export_mass = float(D @ (np.abs(Vt).T @ np.abs(z)))
    export_mass = up(export_mass*(1+gamma(2*n+3)))
    eexport = up(gamma(kexport)*export_mass)
    ec = up(eexport+2*L*S*bounds['deltaV'])
    T = up(2*L*S+ec)
    RX = up(2*Ebar+4*bounds['aW']*S+3*bW+rhoW+ec)
    return c, dict(tau=tau, rank=int(keep.sum()), bW=bW, ep=ep, ey=ey,
                   S=S, Ebar=Ebar, rhoW=rhoW, ec=ec, export_mass=export_mass, coefficient_bound=T,
                   sample_absolute_bound=RX, cutoff_pass=bool(cutoff_pass),
                   transformed_norm=norm(z), transformed_norm_pass=bool(norm(z)<=2*S),
                   cutoff_analytic=E_per_B/L, cutoff_matrix=bounds['aW'],
                   cutoff_labels=bW/S, coefficient_l1=float(np.sum(abs(c))))


def evaluate(x, centers, slope, coefficients, truth):
    residual = np.empty_like(truth)
    for start in range(0, len(x), 512):
        stop = min(start+512, len(x))
        features = feature_matrix(x[start:stop], centers, slope)
        for j in range(truth.shape[1]):
            pred = pairwise_sum(features*coefficients[:, j], axis=1)
            residual[start:stop, j] = pred-truth[start:stop, j]
    return np.linalg.norm(residual, axis=0)/np.linalg.norm(truth, axis=0), np.max(abs(residual), axis=0)


def run_one(N, cfg, out, xe, ye, xv, yv):
    started = time.perf_counter()
    h, R, slope, centers, checks = geometry(N, cfg)
    x = np.linspace(-1, 1, N*cfg['sampling_density']+1)
    y = target_values(x)
    A = feature_matrix(x, centers, slope)
    alpha, D, L, E_per_B = envelopes(N, cfg)
    B = analytic_B(cfg)
    actual = []
    actual_rank = []
    for j in range(len(NAMES)):
        c, _, rank, _ = lstsq(A, y[:, j], cond=cfg['actual_relative_cutoff'],
                              lapack_driver='gelsd', check_finite=False)
        actual.append(c)
        actual_rank.append(int(rank))
    actual = np.column_stack(actual)
    print(f'N={N}: actual solves finished ({time.perf_counter()-started:.1f}s)', flush=True)
    F = A*D/math.sqrt(len(x))
    del A
    yn = y/math.sqrt(len(x))
    U, singular, Vt = svd(F, full_matrices=False, lapack_driver='gesdd', check_finite=False)
    bounds = factorization_bounds(F, U, singular, Vt, L)
    theory, diagnostics = [], []
    for j in range(len(NAMES)):
        c, info = theorem_solve(F, yn[:, j], U, singular, Vt, D, L, E_per_B, B[j], bounds)
        # Predictions in scaled coordinates equal native ones up to rounding;
        # direct A c is reconstructed by dividing each column's scale back.
        sample_residual = norm(F @ (c/D)-yn[:, j])
        info['sample_absolute_error'] = sample_residual
        info['sample_bound_pass'] = sample_residual <= info['sample_absolute_bound']
        info['coefficient_bound_pass'] = info['coefficient_l1'] <= info['coefficient_bound']
        theory.append(c)
        diagnostics.append(info)
    theory = np.column_stack(theory)
    del F, U, Vt
    coefficients = np.column_stack((actual, theory))
    errors, maxima = evaluate(xe, centers, slope, coefficients, np.tile(ye, (1,2)))
    validation, _ = evaluate(xv, centers, slope, coefficients, np.tile(yv, (1,2)))
    np.savez_compressed(out/f'coefficients_N{N}.npz', centers=centers, gamma=slope,
                        actual=actual, theory=theory, alpha=alpha, D=D, singular=singular)
    rows = []
    for j, name in enumerate(NAMES):
        rows.append(dict(N=N, W=len(centers), R=R, samples=len(x), target=name,
                         actual_error=float(errors[j]), theory_error=float(errors[j+len(NAMES)]),
                         actual_linf=float(maxima[j]), theory_linf=float(maxima[j+len(NAMES)]),
                         actual_validation_error=float(validation[j]),
                         theory_validation_error=float(validation[j+len(NAMES)]),
                         actual_rank=actual_rank[j], actual_l1=float(np.sum(abs(actual[:,j]))),
                         B=float(B[j]), L=L, E_per_B=E_per_B,
                         geometry_checks=checks, numerical_bounds=bounds,
                         theory=diagnostics[j], certified=False))
    (out/f'N{N}.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    print(f'N={N} complete in {time.perf_counter()-started:.1f}s: '+
          ', '.join(f"{r['target']} {r['actual_error']:.2e}/{r['theory_error']:.2e}" for r in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', nargs='+', type=int)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE/'config.yaml').read_text())
    if args.N:
        cfg['N'] = args.N
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest = dict(config=cfg, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        numpy=np.__version__, baseline='expC09/convergence.py gelsd single-target solve',
                        evaluation='correctly rounded tanh, explicit pairwise output sum for both arms',
                        caveat='No interval-certified oracle/constants/norms; s=8,p=1 continuous bound is loose. Numerical rounding allowances are conservative and declared, not optimized.')
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        xe = -1+2*(np.arange(cfg['eval_points'])+.5)/cfg['eval_points']
        xv = -1+2*(np.arange(cfg['validation_points'])+.5)/cfg['validation_points']
        ye, yv = target_values(xe), target_values(xv)
        with threadpool_limits(limits=cfg['threads']):
            for N in cfg['N']:
                if not (out/f'N{N}.json').exists():
                    run_one(N, cfg, out, xe, ye, xv, yv)
    from experiments.expC14_theory_solver_comparison.plot import draw
    draw(out)


if __name__ == '__main__':
    main()
