"""Targets, exact full-batch GD, and population diagnostics for D56."""
from __future__ import annotations

import numpy as np
import torch


TARGETS = {
    "square": ("easy", r"$x^2$", "x**2"),
    "cubic": ("easy", r"$x^3$", "x**3"),
    "exponential": ("easy", r"$e^x$", "exp(x)"),
    "sine_pi": ("easy", r"$\sin(\pi x)$", "sin(pi*x)"),
    "cosine_pi": ("easy", r"$\cos(\pi x)$", "cos(pi*x)"),
    "sine_4pi": ("medium", r"$\sin(4\pi x)$", "sin(4*pi*x)"),
    "legendre_5": ("medium", r"$P_5(x)$", "Legendre polynomial degree 5"),
    "runge_25": ("medium", r"$(1+25x^2)^{-1}$", "1/(1+25*x**2)"),
    "gaussian_25": ("medium", r"$e^{-25(x-0.2)^2}$", "exp(-25*(x-0.2)**2)"),
    "softplus_5": ("medium", r"$\log(1+e^{5x})/5$", "logaddexp(0,5*x)/5"),
    "mixed_sine": ("hard", "Mixed sine", "sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)"),
    "chirp": ("hard", r"$\sin(8\pi(x+1)^2)$", "sin(8*pi*(x+1)**2)"),
    "legendre_12": ("hard", r"$P_{12}(x)$", "Legendre polynomial degree 12"),
    "runge_100": ("hard", r"$(1+100x^2)^{-1}$", "1/(1+100*x**2)"),
    "tanh_20": ("hard", r"$\tanh(20(x-0.15))$", "tanh(20*(x-0.15))"),
    "localized_oscillation": ("hard", r"$e^{-40x^2}\sin(16\pi x)$", "exp(-40*x**2)*sin(16*pi*x)"),
}


def target(name: str, x: np.ndarray) -> np.ndarray:
    if name == "square": return x**2
    if name == "cubic": return x**3
    if name == "exponential": return np.exp(x)
    if name == "sine_pi": return np.sin(np.pi*x)
    if name == "cosine_pi": return np.cos(np.pi*x)
    if name == "sine_4pi": return np.sin(4*np.pi*x)
    if name.startswith("legendre_"):
        degree = int(name.split("_")[-1])
        return np.polynomial.legendre.legval(x, [0]*degree+[1])
    if name == "runge_25": return 1/(1+25*x*x)
    if name == "gaussian_25": return np.exp(-25*(x-.2)**2)
    if name == "softplus_5": return np.logaddexp(0, 5*x)/5
    if name == "mixed_sine": return np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(14*np.pi*x)
    if name == "chirp": return np.sin(8*np.pi*(x+1)**2)
    if name == "runge_100": return 1/(1+100*x*x)
    if name == "tanh_20": return np.tanh(20*(x-.15))
    if name == "localized_oscillation": return np.exp(-40*x*x)*np.sin(16*np.pi*x)
    raise KeyError(name)


def midpoint_grid(n: int) -> np.ndarray:
    return -1+(np.arange(n, dtype=np.float64)+.5)*(2/n)


def initial_parameters(width: int, seed: int, multiplier: float) -> torch.Tensor:
    """Literal torch Linear defaults, created in float64; only first layer scaled."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        hidden = torch.nn.Linear(1, width, dtype=torch.float64)
        readout = torch.nn.Linear(width, 1, dtype=torch.float64)
    return torch.cat((hidden.weight.detach().flatten()*multiplier,
                      hidden.bias.detach()*multiplier,
                      readout.weight.detach().flatten(), readout.bias.detach()))


@torch.jit.script
def gd_chunk(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
             lr: float, count: int) -> tuple[torch.Tensor, int, bool]:
    """Simultaneous analytical gradients, equivalent to plain SGD on half MSE.

    No clipping, schedules, regularization, readout refits, or frozen parameters.
    Last finite state is returned if an update becomes nonfinite.
    """
    width = (theta.numel()-1)//3
    a, b, c, d = theta[:width], theta[width:2*width], theta[2*width:3*width], theta[-1]
    for k in range(count):
        h = torch.tanh(a[:, None]*x[None, :] + b[:, None])
        e = torch.matmul(c, h)+d-y
        scaled = e / x.numel()
        dh = (1-h*h)*c[:, None]*scaled[None, :]
        da = torch.sum(dh*x[None, :], dim=1)
        db = torch.sum(dh, dim=1)
        dc = torch.matmul(h, scaled)
        dd = torch.sum(scaled)
        anew, bnew, cnew, dnew = a-lr*da, b-lr*db, c-lr*dc, d-lr*dd
        next_theta = torch.cat((anew, bnew, cnew, dnew.reshape(1)))
        if not bool(torch.all(torch.isfinite(next_theta))):
            return torch.cat((a, b, c, d.reshape(1))), k, False
        a, b, c, d = anew, bnew, cnew, dnew
    return torch.cat((a, b, c, d.reshape(1))), count, True


def prediction(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c = np.split(theta[:-1], 3)
    return np.tanh(x[:, None]*a+b)@c+theta[-1]


def diagnostics(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                xe: np.ndarray, ye: np.ndarray, h_ref: float) -> dict[str, float]:
    a, b, c = np.split(theta[:-1], 3)
    width, n = len(a), len(x)
    features = np.tanh(x[:, None]*a+b)
    pred = features@c+theta[-1]
    residual = pred-y
    # Full native parameter Jacobian, including output bias.
    sensitivity = (1-features*features)*c
    jac = np.column_stack((sensitivity*x[:, None], sensitivity, features, np.ones(n)))
    gradient = jac.T@residual/n
    # Columns of coarse_basis are orthonormal under the empirical RMS inner product.
    xc = x-x.mean()
    coarse_basis = np.column_stack((np.ones(n), xc/np.sqrt(np.mean(xc*xc))))
    coarse_jac = coarse_basis.T@jac/n
    kc = coarse_jac@coarse_jac.T
    eigenvalues = np.linalg.eigvalsh(kc)
    if eigenvalues[0] > 0:
        tracking = coarse_jac.T@np.linalg.solve(kc, coarse_jac@gradient)
    else:
        tracking = np.full_like(gradient, np.nan)
    fine = gradient-tracking
    residual_h = residual-coarse_basis@(coarse_basis.T@residual/n)
    target_h = y-coarse_basis@(coarse_basis.T@y/n)
    r2 = a*a+b*b+c*c
    energy = r2.sum()
    slope_rms = np.sqrt(np.mean(a*a))
    out = {
        "gamma_rms": float(slope_rms), "gamma_mean_abs": float(np.mean(np.abs(a))),
        "gamma_p95_neurons": float(np.quantile(np.abs(a), .95)),
        "gamma_max": float(np.max(np.abs(a))), "lambda_rms_reference": float(h_ref*slope_rms),
        "train_rel_l2": float(np.linalg.norm(residual)/np.linalg.norm(y)),
        "eval_rel_l2": float(np.linalg.norm(prediction(theta, xe)-ye)/np.linalg.norm(ye)),
        "parameter_energy": float(energy), "gradient_norm": float(np.linalg.norm(gradient)),
        "tracking_norm": float(np.linalg.norm(tracking)), "fine_norm": float(np.linalg.norm(fine)),
        "nonaffine_residual_rms": float(np.sqrt(np.mean(residual_h**2))),
        "target_nonaffine_rms": float(np.sqrt(np.mean(target_h**2))),
        "coarse_kernel_min_eigenvalue": float(eigenvalues[0]),
        "coarse_kernel_condition": float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0]>0 else float("inf"),
        "q_generic_current": float(np.sqrt(6)/8*np.sum(r2**2)),
    }
    for k in (4, 6, 10, 14):
        out[f"chi_{k}"] = float(width**(k/2-1)*np.sum((r2/energy)**(k/2))) if energy>0 else float("nan")
    poly = np.polynomial.legendre.legvander(x, 5)
    basis, _ = np.linalg.qr(poly)
    for degree in (3, 5):
        out[f"tau_{degree}"] = float(np.linalg.norm(basis[:, :degree+1].T@target_h)/np.sqrt(n))
    return out
