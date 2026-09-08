"""expD23 -- every zero-order solver in Fchaubard/zero_order_rnn, ported to
this repo's flat-vector fp64 full-batch setting, plus the two *oracle*
zero-order arms that bound the whole class (coordinate finite-difference
gradient, coordinate finite-difference Hessian).

Upstream (rge_series_experiments.py, commit 9201389, 2026-01-08) exposes five
solvers through train_step():

  1SPSA       cdrge_optimize   CD-RGE: theta += (lr/eps) * sum_j c_j z_j,
                               c_j = -(f_j^+ - f_j^-)/(2 n).  Optional beta1
                               momentum, beta2 RMSProp with v initialised at
                               ONES (no bias correction), probe preconditioning.
  1.5-SPSA    SPSA1_5          same, but c_j /= max(curv_j^alpha, lambda_reg),
                               curv_j = |f_j^+ - 2 f_0 + f_j^-| / eps^2.
                               NOTE: in the cached path (the only path
                               train_step and the LR search call) the division
                               is computed and then NOT applied -- the update
                               uses the raw c_j and ignores lr.  Both readings
                               are ported: curv_alpha=None reproduces the code
                               as run; curv_alpha=a reproduces the intent.
  2SPSA       SPSA2            6 evals per probe (theta +- eps v, theta +- eps v
                               +- eps u); c_j = -(f^{+v} - f^{-v}) /
                               (2 n kappa_j), kappa_j = mixed second difference
                               / (4 eps^2), replaced by +1 when |kappa_j| < 1,
                               otherwise used WITH its sign.  No lr.
  BanditSPSA  BanditSPSA       CD-RGE whose probe seeds are re-drawn from a
                               reservoir of past seeds ranked by |EMA of the
                               finite difference| (softmax, temperature 1e-4);
                               c_j = -(f^+ - f^-)/2 (no 1/n), step (lr/eps).
  Sanger-SPSA SangerSPSA       g = sum_j c_j z_j with c_j = -(f^+-f^-)/(2 n eps);
                               theta += lr_t (W W^T g + alpha g), W (m x r)
                               updated by Sanger's rule on g/|g|.
  binary LR search             probe 3 log-spaced lr, narrow `depth` times;
                               lr = eps tied; re-search on EMA plateau.

Everything here is a pure function of a loss oracle that returns fp64 loss
values; nothing touches gradients.  The oracle is batched (FlatLoss) so that
one step's 2n probes are one chunked tensor evaluation.
"""

from __future__ import annotations

import math

import torch

EPS_MACH = 2.0 ** -52


# ----------------------------------------------------------------------------- oracles

class FlatLoss:
    """Full-batch MSE of the standard-parameterization QIMlp as a function of
    the flat parameter vector in parameters_to_vector order
    [w (W), b (W), v (W), c (1)]:  f(x) = c + sum_k v_k tanh(w_k x + b_k).
    Batched over B parameter vectors.  Values agree with QIMlp(x) to summation
    order (~1e-16 relative)."""

    def __init__(self, X, y, W, chunk=16):
        self.x = X.reshape(-1).to(torch.float64)
        self.y = y.reshape(-1).to(torch.float64)
        self.W = int(W)
        self.n = self.x.numel()
        self.m = 3 * self.W + 1
        self.chunk = int(chunk)
        self.n_evals = 0

    def split(self, theta):
        W = self.W
        return theta[..., :W], theta[..., W:2 * W], theta[..., 2 * W:3 * W], theta[..., 3 * W]

    def predict(self, thetas, x=None):
        """thetas (B, m) -> outputs (B, n) on grid x (default: train grid)."""
        x = self.x if x is None else x.reshape(-1)
        w, b, v, c = self.split(thetas)
        pre = x[None, :, None] * w[:, None, :] + b[:, None, :]
        return torch.einsum("bnw,bw->bn", torch.tanh(pre), v) + c[:, None]

    def losses(self, thetas):
        thetas = thetas.reshape(-1, self.m)
        out = torch.empty(thetas.shape[0], dtype=torch.float64)
        for s in range(0, thetas.shape[0], self.chunk):
            pred = self.predict(thetas[s:s + self.chunk])
            out[s:s + self.chunk] = ((pred - self.y) ** 2).mean(dim=1)
        self.n_evals += int(thetas.shape[0])
        return out

    def loss(self, theta):
        return float(self.losses(theta.reshape(1, -1))[0])

    def features(self, theta, x=None):
        """Phi (n, W) = tanh(w x + b) at a single theta."""
        x = self.x if x is None else x.reshape(-1)
        w, b, _, _ = self.split(theta)
        return torch.tanh(x[:, None] * w[None, :] + b[None, :])

    def rel_l2(self, theta, X_ev, y_ev, y_norm):
        pred = self.predict(theta.reshape(1, -1), x=X_ev)[0]
        return float(torch.linalg.norm(pred - y_ev.reshape(-1)) / y_norm)


class ReadoutLoss:
    """The same loss oracle restricted to the readout block [v (W), c] with the
    inner geometry frozen, so Phi is fixed: L(v, c) = mean((Phi v + c - y)^2).
    Exactly what FlatLoss computes for those coordinates (Phi is bitwise the
    same tanh), cached once."""

    def __init__(self, Phi, y, chunk=64):
        self.Phi = Phi.to(torch.float64)
        self.y = y.reshape(-1).to(torch.float64)
        self.n, self.W = self.Phi.shape
        self.m = self.W + 1
        self.chunk = int(chunk)
        self.n_evals = 0

    def predict(self, thetas, Phi=None):
        Phi = self.Phi if Phi is None else Phi
        return thetas[:, :self.W] @ Phi.T + thetas[:, self.W:self.W + 1]

    def losses(self, thetas):
        thetas = thetas.reshape(-1, self.m)
        out = torch.empty(thetas.shape[0], dtype=torch.float64)
        for s in range(0, thetas.shape[0], self.chunk):
            pred = self.predict(thetas[s:s + self.chunk])
            out[s:s + self.chunk] = ((pred - self.y) ** 2).mean(dim=1)
        self.n_evals += int(thetas.shape[0])
        return out

    def loss(self, theta):
        return float(self.losses(theta.reshape(1, -1))[0])


# ----------------------------------------------------------------------------- probes

def rademacher(m, gen, n_rows=1):
    """n_rows x m Rademacher(+-1) rows drawn one row at a time -- the same
    stream as expD22's cdrge.py `_probe` so trajectories can be cross-checked."""
    rows = [(torch.randint(0, 2, (m,), generator=gen, dtype=torch.int8)
             .to(torch.float64) * 2.0 - 1.0) for _ in range(n_rows)]
    return torch.stack(rows, 0)


def rademacher_from_seed(m, seed):
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    return rademacher(m, g, 1)[0]


def _lr_at(step, peak, warmup, total, cosine, end_frac=1e-3):
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    if not cosine:
        return peak
    end = end_frac * peak
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (peak - end) * (1.0 + math.cos(math.pi * prog))


class EMAPlateauDetector:
    """Upstream verbatim (rge_series_experiments.py::EMAPlateauDetector)."""

    def __init__(self, alpha=0.1, patience=10, threshold=0.01):
        self.alpha, self.patience, self.threshold = alpha, patience, threshold
        self.ema = self.best_ema = None
        self.iters_without_improvement = 0

    def update(self, loss):
        if self.ema is None:
            self.ema = self.best_ema = loss
            return False
        self.ema = self.alpha * loss + (1 - self.alpha) * self.ema
        rel = (self.best_ema - self.ema) / (self.best_ema + 1e-8)
        if rel > self.threshold:
            self.best_ema = self.ema
            self.iters_without_improvement = 0
        else:
            self.iters_without_improvement += 1
        return self.iters_without_improvement >= self.patience

    def reset(self):
        self.iters_without_improvement = 0


# ----------------------------------------------------------------------------- CD-RGE family

class CDRGEState:
    def __init__(self, m, beta1, beta2, upstream_beta2):
        self.mom = torch.zeros(m, dtype=torch.float64) if (beta1 > 0 or beta2 > 0) else None
        if beta2 > 0:
            self.var = (torch.ones if upstream_beta2 else torch.zeros)(m, dtype=torch.float64)
        else:
            self.var = None
        self.t = 0
        self.reservoir = {}          # BanditSPSA
        self.bandit_step = 0


def cdrge_step(theta, oracle, st, *, n_perturb, eps, lr, gen, beta1=0.0, beta2=0.0,
               upstream_beta2=False, adam_lr=None, curv_alpha=None, lam_reg=1.0,
               probe_precond=False, bandit=None, adam_delta=1e-16):
    """One step of the CD-RGE family on a flat theta. Returns
    (theta_new, mean_probe_loss). `st` carries momentum/variance/reservoir.

    lr enters as theta += (lr/eps) * sum_j c_j z_j with c_j = -(f_j^+ - f_j^-)/(2n)
    [upstream factoring; c_j carries a factor eps].  With curv_alpha set the
    1.5-SPSA intent divides c_j by max(curv_j^alpha, lam_reg).  With beta2 > 0
    and adam_lr None the upstream RMSProp path (v init ones, no bias correction,
    stabiliser 1e-8) is used; with adam_lr set, expD22's Adam-style path
    (bias-corrected moments on ghat = c/eps, step adam_lr, stabiliser 1e-16).
    bandit=dict(temperature, min_fd, ema) turns on the seed reservoir."""
    m = theta.numel()
    st.t += 1
    if bandit is not None:
        Z, seeds = _bandit_probes(m, n_perturb, st, gen, bandit)
    else:
        Z = rademacher(m, gen, n_perturb)
        seeds = None
    if probe_precond and st.var is not None:
        Z = Z / (st.var.sqrt() + 1e-8)
    thetas = torch.cat([theta[None, :] + eps * Z, theta[None, :] - eps * Z], 0)
    f = oracle.losses(thetas)
    fp, fm = f[:n_perturb], f[n_perturb:]
    if curv_alpha is not None:
        f0 = oracle.loss(theta)
        curv = (fp - 2.0 * f0 + fm).abs() / (eps ** 2)
        denom = torch.clamp(curv ** curv_alpha, min=lam_reg)
    else:
        denom = torch.ones(n_perturb, dtype=torch.float64)
    if bandit is not None:
        coeff = -(fp - fm) / 2.0 / denom                       # upstream: no 1/n
        _bandit_update(st, seeds, -(fp - fm) / 2.0, bandit)
    else:
        coeff = -(fp - fm) / (2.0 * n_perturb) / denom
    buf = coeff @ Z                                             # sum_j c_j z_j  (carries eps)
    mean_loss = float(0.5 * (fp + fm).mean())

    if beta2 > 0 and adam_lr is not None:                      # expD22 Adam-style
        ghat = buf / (-eps)
        b1 = beta1 if beta1 > 0 else 0.9
        st.mom.mul_(b1).add_(ghat, alpha=1.0 - b1)
        st.var.mul_(beta2).addcmul_(ghat, ghat, value=1.0 - beta2)
        m_hat = st.mom / (1.0 - b1 ** st.t)
        v_hat = st.var / (1.0 - beta2 ** st.t)
        return theta - lr * m_hat / (v_hat.sqrt() + adam_delta), mean_loss
    g = buf
    if beta1 > 0:
        st.mom.mul_(beta1).add_(g, alpha=1.0 - beta1)
        m_hat = st.mom
    else:
        m_hat = g
    if beta2 > 0:                                               # upstream RMSProp path
        st.var.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        v_hat = st.var.sqrt() + 1e-8
        if probe_precond:
            return theta + (lr / eps) * m_hat, mean_loss        # "v_hat already in the probe"
        return theta + (lr / eps) * m_hat / v_hat, mean_loss
    return theta + (lr / eps) * m_hat, mean_loss


def _bandit_probes(m, n_perturb, st, gen, bandit):
    """Upstream BanditSPSA seed selection: exploit share pt(t) (logistic 0.2 ->
    0.9 over the first ~500 steps), softmax over |ema| at `temperature`,
    sampled without replacement; the rest are fresh seeds."""
    step = st.bandit_step
    st.bandit_step += 1
    if step and step % 1000 == 0:
        st.reservoir.clear()
    pt = 0.20 + 0.70 / (1.0 + math.exp(-(step - 250) / 75.0))
    n_exploit = min(int(round(n_perturb * pt)), len(st.reservoir))
    seeds = []
    if n_exploit:
        keys = list(st.reservoir.keys())
        scores = [abs(st.reservoir[k]["ema"]) for k in keys]
        mx = max(scores)
        w = torch.tensor([math.exp((s - mx) / bandit["temperature"]) for s in scores],
                         dtype=torch.float64)
        if not torch.isfinite(w).all() or float(w.sum()) <= 0:
            w = torch.ones(len(keys), dtype=torch.float64)       # diverged run: uniform
        while len(seeds) < n_exploit and keys:
            tot = float(w.sum())
            probs = w / tot if (math.isfinite(tot) and tot > 0) else torch.full_like(w, 1.0 / w.numel())
            i = int(torch.multinomial(probs, 1, generator=gen))   # temperature 1e-4 underflows all but the top
            seeds.append(keys.pop(i))
            w = torch.cat([w[:i], w[i + 1:]])
    while len(seeds) < n_perturb:
        s = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))
        if s not in st.reservoir and s not in seeds:
            seeds.append(s)
    Z = torch.stack([rademacher_from_seed(m, s) for s in seeds], 0)
    return Z, seeds


def _bandit_update(st, seeds, raw_fd, bandit):
    for s, fd in zip(seeds, raw_fd.tolist()):
        if math.isfinite(fd) and abs(fd) > bandit["min_fd"]:
            rec = st.reservoir.get(s)
            if rec is None:
                st.reservoir[s] = {"ema": fd, "pulls": 1}
            else:
                rec["pulls"] += 1
                rec["ema"] = bandit["ema"] * rec["ema"] + (1 - bandit["ema"]) * fd
    if len(st.reservoir) > 10_000:
        for s, _ in sorted(st.reservoir.items(), key=lambda kv: abs(kv[1]["ema"]))[:len(st.reservoir) - 10_000]:
            st.reservoir.pop(s, None)


def cdrge_minimize(theta0, oracle, *, steps, n_perturb, eps, lr=None, beta1=0.0, beta2=0.0,
                   upstream_beta2=False, adam_lr=None, cosine=False, warmup=0,
                   curv_alpha=None, lam_reg=1.0, probe_precond=False, bandit=None,
                   lr_search=None, eps_halve_every=None, eps_floor=1e-15,
                   seed=0, callback=None, max_evals=None):
    """Run the CD-RGE family for `steps` steps (or until max_evals oracle
    calls).  lr defaults to eps (the author's tie).  lr_search=dict(lr_min,
    lr_max, depth, probe_steps, patience, threshold, ema_alpha) runs the
    upstream binary LR search at the start and on every plateau, with lr = eps
    retied to the found value.  eps_halve_every halves eps (and, when lr is
    tied, lr) on that cadence.  callback(step, theta, mean_loss, eps, evals)."""
    theta = theta0.detach().clone()
    m = theta.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    st = CDRGEState(m, beta1, beta2, upstream_beta2)
    tied = lr is None
    lr_now = eps if tied else lr
    eps_now = eps
    detector = None
    evals0 = oracle.n_evals
    n_searches = 0
    kw = dict(n_perturb=n_perturb, beta1=beta1, beta2=beta2, upstream_beta2=upstream_beta2,
              adam_lr=adam_lr, curv_alpha=curv_alpha, lam_reg=lam_reg,
              probe_precond=probe_precond, bandit=bandit)

    def search(theta):
        nonlocal n_searches
        n_searches += 1
        return _binary_search_lr(theta, oracle, st, gen, lr_search, kw)

    if lr_search is not None:
        detector = EMAPlateauDetector(lr_search.get("ema_alpha", 0.1),
                                      lr_search.get("patience", 10),
                                      lr_search.get("threshold", 0.01))
        lr_now = eps_now = search(theta)
    trace = []
    diverged = False
    for step in range(int(steps)):
        if eps_halve_every and step and step % int(eps_halve_every) == 0:
            eps_now = max(eps_now * 0.5, eps_floor)
            if tied:
                lr_now = eps_now
        lr_t = _lr_at(step, adam_lr if adam_lr is not None else lr_now, warmup, steps, cosine)
        theta, mean_loss = cdrge_step(theta, oracle, st, eps=eps_now, lr=lr_t, gen=gen, **kw)
        trace.append(mean_loss)
        if not (math.isfinite(mean_loss) and mean_loss < 1e30):
            diverged = True
            if callback is not None:
                callback(step + 1, theta, mean_loss, eps_now, oracle.n_evals - evals0)
            break
        if detector is not None and detector.update(mean_loss):
            lr_now = eps_now = search(theta)
            detector.reset()
        if callback is not None:
            callback(step + 1, theta, mean_loss, eps_now, oracle.n_evals - evals0)
        if max_evals is not None and oracle.n_evals - evals0 >= max_evals:
            break
    return theta, {"evals": oracle.n_evals - evals0, "steps_run": len(trace),
                   "final_eps": eps_now, "final_lr": lr_now, "n_lr_searches": n_searches,
                   "loss_trace": trace, "reservoir_size": len(st.reservoir), "diverged": diverged}


def _probe_loss_at_lr(theta, oracle, st, gen, lr, n_steps, kw):
    """Upstream probe_loss_at_lr: take n_steps at lr = eps = candidate from a
    saved theta (moments frozen: upstream mutates them; we snapshot), evaluate,
    restore."""
    mom = None if st.mom is None else st.mom.clone()
    var = None if st.var is None else st.var.clone()
    t = st.t
    th = theta.clone()
    for _ in range(n_steps):
        th, _ = cdrge_step(th, oracle, st, eps=lr, lr=lr, gen=gen, **kw)
    loss = oracle.loss(th)
    if mom is not None:
        st.mom.copy_(mom)
    if var is not None:
        st.var.copy_(var)
    st.t = t
    return loss


def _binary_search_lr(theta, oracle, st, gen, cfg, kw):
    """Upstream binary_search_lr, log-space; returns best lr."""
    lo, hi = math.log10(cfg["lr_min"]), math.log10(cfg["lr_max"])
    n_steps = cfg.get("probe_steps", 1)
    pts = [lo, 0.5 * (lo + hi), hi]
    res = [(p, _probe_loss_at_lr(theta, oracle, st, gen, 10 ** p, n_steps, kw)) for p in pts]
    best = min(range(3), key=lambda i: res[i][1])
    for _ in range(2, cfg.get("depth", 3) + 1):
        if best == 0:
            hi = res[1][0]
        elif best == len(res) - 1:
            lo = res[-2][0]
        else:
            lo, hi = res[best - 1][0], res[best + 1][0]
        mid = 0.5 * (lo + hi)
        res.append((mid, _probe_loss_at_lr(theta, oracle, st, gen, 10 ** mid, n_steps, kw)))
        res.sort(key=lambda r: r[0])
        best = min(range(len(res)), key=lambda i: res[i][1])
    return 10 ** min(res, key=lambda r: r[1])[0]


# ----------------------------------------------------------------------------- 2SPSA (upstream)

def spsa2_minimize(theta0, oracle, *, steps, n_perturb, eps, seed=0, callback=None,
                   max_evals=None, curv_floor=1.0):
    """Upstream SPSA2 verbatim: per probe (v, u) six losses; kappa = mixed second
    difference/(4 eps^2), set to +1 when |kappa| < curv_floor and otherwise used
    with its sign; theta += sum_j -(f^{+v}-f^{-v}) v_j / (2 n kappa_j)."""
    theta = theta0.detach().clone()
    m = theta.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    evals0 = oracle.n_evals
    trace = []
    diverged = False
    for step in range(int(steps)):
        V = rademacher(m, gen, n_perturb)
        U = rademacher(m, gen, n_perturb)
        T = theta[None, :]
        pts = torch.cat([T + eps * V, T - eps * V,
                         T + eps * V + eps * U, T + eps * V - eps * U,
                         T - eps * V + eps * U, T - eps * V - eps * U], 0)
        f = oracle.losses(pts).reshape(6, n_perturb)
        raw = (f[2] - f[3] - f[4] + f[5]) / (4.0 * eps ** 2)
        denom = torch.where(raw.abs() < curv_floor, torch.ones_like(raw), raw)
        coeff = -(f[0] - f[1]) / (2.0 * n_perturb * denom)
        theta = theta + coeff @ V
        mean_loss = float(0.5 * (f[0] + f[1]).mean())
        trace.append(mean_loss)
        if callback is not None:
            callback(step + 1, theta, mean_loss, eps, oracle.n_evals - evals0)
        if max_evals is not None and oracle.n_evals - evals0 >= max_evals:
            break
        if not (math.isfinite(mean_loss) and mean_loss < 1e30):
            diverged = True
            break
    return theta, {"evals": oracle.n_evals - evals0, "steps_run": len(trace), "diverged": diverged,
                   "final_eps": eps, "loss_trace": trace}


# ----------------------------------------------------------------------------- Sanger-SPSA (upstream)

def sanger_minimize(theta0, oracle, *, steps, n_perturb, eps, lr, rank=1, alpha_eye=1.0,
                    beta_eig=0.1, warmup=100, base_lr=1e-4, qr_every=10_000_000,
                    probe_precond=False, seed=0, callback=None, max_evals=None):
    """Upstream SangerSPSA (the live, third version): g = sum_j c_j z_j with
    c_j = -(f^+ - f^-)/(2 n eps) [so g ~ -grad]; theta += lr_t (W W^T g +
    alpha_eye g), lr_t = base_lr + lr * min(t/warmup, 1); W (m x rank) updated
    column-wise by Sanger's rule on g/|g| with rate beta_eig, columns renormalised."""
    theta = theta0.detach().clone()
    m = theta.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(m, rank, generator=gen, dtype=torch.float64)
    W, _ = torch.linalg.qr(W, mode="reduced")
    evals0 = oracle.n_evals
    trace, var_ratios = [], []
    diverged = False
    for step in range(int(steps)):
        warm = min(step / max(1, warmup), 1.0)
        lr_t = base_lr + lr * warm
        Z = rademacher(m, gen, n_perturb)
        if probe_precond:
            Z = Z @ W @ W.T
            Z = Z * (math.sqrt(m) / (Z.norm(dim=1, keepdim=True) + 1e-12))
        pts = torch.cat([theta[None, :] + eps * Z, theta[None, :] - eps * Z], 0)
        f = oracle.losses(pts)
        fp, fm = f[:n_perturb], f[n_perturb:]
        coeff = -(fp - fm) / (2.0 * n_perturb * eps)
        g = coeff @ Z
        g_norm = float(g.norm())
        pre_g = W @ (W.T @ g) + alpha_eye * g
        theta = theta + lr_t * pre_g
        g_unit = g / max(g_norm, 1e-12)
        proj = W.T @ g_unit
        acc = torch.zeros(m, dtype=torch.float64)
        for i in range(rank):
            acc.add_(W[:, i], alpha=float(proj[i]))
            W[:, i].add_((g_unit - acc) * float(proj[i]), alpha=beta_eig)
            cn = float(W[:, i].norm())
            if cn > 0:
                W[:, i].div_(max(cn, 1e-6))
        if (step + 1) % qr_every == 0:
            W, _ = torch.linalg.qr(W, mode="reduced")
        var_ratios.append(float((W.T @ g).pow(2).sum() / max(g.pow(2).sum(), 1e-12)))
        mean_loss = float(0.5 * (fp + fm).mean())
        trace.append(mean_loss)
        if callback is not None:
            callback(step + 1, theta, mean_loss, eps, oracle.n_evals - evals0)
        if max_evals is not None and oracle.n_evals - evals0 >= max_evals:
            break
        if not (math.isfinite(mean_loss) and mean_loss < 1e30):
            diverged = True
            break
    return theta, {"evals": oracle.n_evals - evals0, "steps_run": len(trace), "diverged": diverged,
                   "final_eps": eps, "loss_trace": trace, "var_ratio_trace": var_ratios}


# ----------------------------------------------------------------------------- oracle ZO arms

def fd_gradient(theta, oracle, eps):
    """Coordinate central differences: g_i = (f(theta+eps e_i) - f(theta-eps e_i))/(2 eps).
    2m oracle calls.  eps may be a scalar or an (m,) tensor."""
    m = theta.numel()
    E = torch.eye(m, dtype=torch.float64) * eps
    pts = torch.cat([theta[None, :] + E, theta[None, :] - E], 0)
    f = oracle.losses(pts)
    return (f[:m] - f[m:]) / (2.0 * (E.diagonal()))


def fd_hessian(theta, oracle, eps):
    """Coordinate second differences on the full vector:
    H_ii = (f(+e_i) - 2 f0 + f(-e_i))/eps^2,
    H_ij = (f(+e_i+e_j) - f(+e_i-e_j) - f(-e_i+e_j) + f(-e_i-e_j))/(4 eps^2), i<j.
    2m + 1 + 2 m(m-1) oracle calls.  Returns the symmetric (m, m) matrix."""
    m = theta.numel()
    f0 = oracle.loss(theta)
    E = torch.eye(m, dtype=torch.float64) * eps
    f1 = oracle.losses(torch.cat([theta[None, :] + E, theta[None, :] - E], 0))
    H = torch.zeros(m, m, dtype=torch.float64)
    H.diagonal().copy_((f1[:m] - 2.0 * f0 + f1[m:]) / eps ** 2)
    iu, ju = torch.triu_indices(m, m, offset=1)
    npair = iu.numel()
    Ei, Ej = E[iu], E[ju]
    T = theta[None, :]
    vals = torch.empty(npair, dtype=torch.float64)
    B = 4096
    for s in range(0, npair, B):
        a, b = Ei[s:s + B], Ej[s:s + B]
        pts = torch.cat([T + a + b, T + a - b, T - a + b, T - a - b], 0)
        f = oracle.losses(pts).reshape(4, -1)
        vals[s:s + B] = (f[0] - f[1] - f[2] + f[3]) / (4.0 * eps ** 2)
    H[iu, ju] = vals
    H[ju, iu] = vals
    return H


def truncated_newton_step(theta, g, H, tau):
    """theta - H_tau^+ g where H_tau keeps eigenvalues > tau * lambda_max."""
    lam, Q = torch.linalg.eigh(H)
    keep = lam > tau * lam.max()
    inv = torch.where(keep, 1.0 / torch.where(keep, lam, torch.ones_like(lam)), torch.zeros_like(lam))
    return theta - Q @ (inv * (Q.T @ g)), int(keep.sum())


def zo_estimator_covariance(g):
    """Exact covariance of the Rademacher CD-RGE estimate ghat = (1/n) sum_j (z_j^T g) z_j
    on a quadratic (no truncation term):  Cov = (|g|^2 I + g g^T - 2 diag(g^2)) / n."""
    m = g.numel()
    return (g.dot(g) * torch.eye(m, dtype=torch.float64) + torch.outer(g, g)
            - 2.0 * torch.diag(g * g))
