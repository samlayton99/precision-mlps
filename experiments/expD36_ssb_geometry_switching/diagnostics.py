"""Same-state direction comparisons, numerical audits, and spectral evidence."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from . import accessibility as access
from experiments.expD35_optimization_exploration import core, ssb


@lru_cache(maxsize=16)
def kernels(n, coordinates, target):
    g, matrix, residual, _, value_grad = access.problem(n, coordinates, target)
    _, loss, _ = ssb.problem(dict(n=n, coordinates=coordinates, target=target))
    grad = jax.jit(jax.grad(loss))
    hvp = jax.jit(lambda z,v:jax.jvp(jax.grad(loss), (z,), (v,))[1])
    @jax.jit
    def svd_values(lam, r, taus):
        u, s, _ = jnp.linalg.svd(matrix(lam), full_matrices=False)
        energy = (u.T@r)**2/(r@r)
        gains = jax.vmap(lambda tau:jnp.sum(-jnp.expm1(-2*tau*s*s)*energy))(taus)
        return gains, s, energy
    return g, matrix, residual, value_grad, grad, hvp, svd_values


def diagnostic(z, metric, config, taus, audit=False):
    g, matrix, residual, value_grad, grad_fn, hvp, svd_values = kernels(config['n'],config['coordinates'],config['target'])
    split = g.width+1
    gradient = grad_fn(z)
    r = residual(z)
    lam = z[split:]
    hg = metric@gradient
    norm_hg, norm_g = float(jnp.linalg.norm(hg)), float(jnp.linalg.norm(gradient))
    if min(norm_hg, norm_g, float(r@r)) <= 0:
        return dict(resolved=False, reason='zero_gradient_or_residual'), dict(residual=np.asarray(r))
    directions = jnp.stack((-hg/norm_hg, -gradient/norm_g))
    gained=[]; derivatives=[]
    for tau in taus:
        value, derivative=value_grad(lam,r,tau)
        gained.append(float(value));derivatives.append(np.asarray(derivative))
    derivatives=np.stack(derivatives)
    exposure=derivatives@np.asarray(directions[:,split:]).T
    reference,spectrum,energy=svd_values(lam,r,jnp.asarray(taus))
    reference=np.asarray(reference)
    value_error=np.abs(np.asarray(gained)-reference)
    uncertainty=np.maximum(128*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(exposure),axis=1)),value_error)
    finite=np.full((2,2,2),np.nan)
    finite_gain=np.full((2,2),np.nan)
    # Full SVD differences audit only when scheduled or an intervention is possible.
    candidate=exposure[1,0]<=3*uncertainty[1] and exposure[1,1]>3*uncertainty[1]
    epsilons=[]
    if audit or candidate:
        for k,direction in enumerate(directions[:,split:]):
            rms=float(jnp.sqrt(jnp.mean(direction**2)))
            epsilon=float(np.clip(1e-4*max(float(jnp.sqrt(jnp.mean(lam**2))),1e-3)/max(rms,1e-30),1e-7,1e-2))
            epsilons.append(epsilon)
            for j,factor in enumerate((1.,.5)):
                e=epsilon*factor
                plus=np.asarray(svd_values(lam+e*direction,r,jnp.asarray(taus))[0])
                minus=np.asarray(svd_values(lam-e*direction,r,jnp.asarray(taus))[0])
                finite[:,k,j]=(plus-minus)/(2*e)
                if j==1:finite_gain[:,k]=plus-reference
            discrepancy=np.maximum(np.abs(finite[:,k,1]-exposure[:,k]),np.abs(finite[:,k,1]-finite[:,k,0]))
            rounding=128*np.finfo(float).eps*np.maximum(reference,1e-15)/(.5*epsilon)
            uncertainty=np.maximum(uncertainty,np.maximum(discrepancy,rounding))
    reliability=[]
    for direction in directions:
        action=hvp(z,direction)
        # H C p - p compares the inverse metric with current *true* curvature.
        error=float(jnp.linalg.norm(metric@action-direction))
        curvature=float(direction@action)
        epsilon=1e-5/max(1.,float(jnp.linalg.norm(direction[split:]))/max(float(jnp.linalg.norm(lam)),1e-3))
        difference=(grad_fn(z+epsilon*direction)-grad_fn(z-epsilon*direction))/(2*epsilon)
        numerical=float(jnp.linalg.norm(difference-action)/jnp.maximum(jnp.linalg.norm(action),1e-300))
        reliability.append(dict(inverse_action_error=error,curvature=curvature,hvp_fd_relative=numerical,
                                reliable=bool(error<=.25 and curvature>0 and numerical<=.1)))
    c,gamma=core.physical(z,g,config['coordinates'])
    r_np=np.asarray(r)
    # Endpoint-grid DFT is a discrete diagnostic; Parseval normalization is exact.
    fourier=np.abs(np.fft.rfft(r_np))**2/len(r_np)
    fourier[1:]*=2
    if len(r_np)%2==0:fourier[-1]/=2
    beta=access.exposure_beta(exposure[1,0],exposure[1,1],uncertainty[1],
                              exposure[0,0],exposure[0,1],uncertainty[0])
    descending=bool(np.all(np.asarray(directions@gradient)<0))
    resolved=bool(np.all(np.isfinite(exposure)) and np.all(value_error<1e-7) and descending)
    row=dict(resolved=resolved, audited=bool(audit or candidate), gain=gained, gain_svd=reference.tolist(),
             gain_value_error=value_error.tolist(),exposure=exposure.tolist(),uncertainty=uncertainty.tolist(),
             beta_candidate=beta if resolved else None, direction_cosine=float(directions[0]@directions[1]),
             scale=norm_hg/norm_g,gradient_norm=norm_g,readout_gradient=float(jnp.linalg.norm(gradient[:split])),
             geometry_gradient=float(jnp.linalg.norm(gradient[split:])),curvature=reliability,
             mse=float(r@r),lambda_quantiles=np.quantile(np.abs(lam),[0,.1,.5,.9,1]).tolist(),
             readout_norm=float(jnp.linalg.norm(c[1:])),finite_difference_steps=epsilons)
    arrays=dict(gradient=np.asarray(gradient),directions=np.asarray(directions),gain_gradients=derivatives,
                exposure=exposure,finite_difference=finite,finite_gain=finite_gain,
                residual=r_np,singular_values=np.asarray(spectrum),residual_singular_energy=np.asarray(energy),
                fourier_mse=fourier,physical_readout=np.asarray(c),physical_gamma=np.asarray(gamma))
    return row, arrays
