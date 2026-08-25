"""CD-RGE (central-difference random gradient estimation), Chaubard & Kochenderfer 2025.

Faithful port of `cdrge_optimize` (solver "1SPSA") from
https://github.com/Fchaubard/zero_order_rnn to this repo's flat-vector fp64
full-batch setting. Per step, with n probes z_j ~ Rademacher(+-1)^m:

    f_j^+ = L(theta + eps * z_j),   f_j^- = L(theta - eps * z_j)
    buf   = -(1/n) sum_j [(f_j^+ - f_j^-)/2] z_j          # = -eps * (zz^T-avg) grad + O(eps^2)
    m_t   = beta1 * m + (1-beta1) * buf                    # optional momentum
    theta <- theta + (lr/eps) * m_t                        # lr = eps => scale-free step

The upstream code never divides by eps ("no epsilon bc we mult later so it
cancels out"): the buffer carries a factor eps, and the update multiplies by
lr/eps, so the realized step is theta - lr * ghat with ghat the standard
two-sided RGE gradient estimate. With the author-recommended lr = eps the step
magnitude is set entirely by the measured loss differences.

Differences from upstream, all mechanical:
  - perturbed points are formed as theta + eps*z from a stored copy of theta
    (upstream restores by re-adding seeded probes; at machine-precision targets
    the fp64 restore drift of that scheme is a confound we cannot afford);
  - no chunking (largest model here is ~10^3 parameters);
  - probes come from one torch.Generator over the flat vector rather than
    per-(param, chunk) manual_seed reseeding (upstream's seed arithmetic
    collides across params; the distribution is identical).

Schedules (the author's advice: "lr = eps, start eps high, cut both by half
with every step, n_perturb high (1000)"):
  - "constant":  eps_t = eps0
  - "halve_k":   eps_t = eps0 * 0.5^(t // k)   (k = 1 is the author's recipe)
  - "plateau":   halve eps when the mean probe loss stops improving
                 (rel_improve < threshold for `patience` consecutive steps).
                 NOTE: this is a control decision on loss values, which the
                 repo kill list caps near 1e-10; it is included as the
                 measured-driver contrast to the pure schedules.
Every schedule floors at eps_floor.
"""

from __future__ import annotations

import math

import torch


def _probe(m, gen, distribution, device):
    if distribution == "rad":
        return (torch.randint(0, 2, (m,), generator=gen, dtype=torch.int8,
                              device=device).to(torch.float64) * 2.0 - 1.0)
    if distribution == "normal":
        return torch.randn(m, generator=gen, dtype=torch.float64, device=device)
    if distribution == "uniform":
        return torch.rand(m, generator=gen, dtype=torch.float64, device=device) * 2 - 1
    raise ValueError(f"unknown distribution {distribution!r}")


def cdrge_minimize(x0, loss_fn, *, max_steps, n_perturb, eps0, lr_over_eps=1.0,
                   schedule="constant", halve_every=1, eps_floor=1e-15,
                   plateau_patience=5, plateau_threshold=1e-3,
                   distribution="rad", beta1=0.0, beta2=0.0, adam_lr=None,
                   adam_delta=1e-16, adam_cosine=False, adam_warmup=0, seed=0,
                   step_callback=None, log_every=1, max_evals=None):
    """Minimise loss_fn (x -> float) from x0. Returns (x_final, info).

    step_callback(step, x, mean_probe_loss, eps) is invoked every `log_every`
    steps and on the last step. `max_evals` optionally caps total loss
    evaluations (2 * n_perturb per step) so variants can be cost-matched.

    beta2 > 0 switches to an Adam-style update on the gradient estimate
    ghat = -buf/eps (bias-corrected first/second moments, step size adam_lr,
    stabilizer adam_delta). This deviates from upstream's literal beta2 code,
    whose v-initialized-at-ones accumulator never adapts at fp64 regression
    loss scales; Adam's bias-corrected form is the battle-tested equivalent
    (REQUIREMENTS section 2: battle-tested wins the toss-up).
    """
    x = x0.detach().clone()
    m = x.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mom = torch.zeros_like(x) if (beta1 > 0 or beta2 > 0) else None
    var = torch.zeros_like(x) if beta2 > 0 else None
    if beta2 > 0 and adam_lr is None:
        raise ValueError("beta2 > 0 requires adam_lr")

    eps = float(eps0)
    n_halvings = 0
    best_loss, since_improve = float("inf"), 0
    evals = 0
    loss_trace = []

    for step in range(int(max_steps)):
        if schedule == "halve_k":
            eps = max(eps0 * 0.5 ** (step // int(halve_every)), eps_floor)

        buf = torch.zeros_like(x)
        sum_losses = 0.0
        for _ in range(n_perturb):
            z = _probe(m, gen, distribution, x.device)
            f_plus = float(loss_fn(x + eps * z))
            f_minus = float(loss_fn(x - eps * z))
            evals += 2
            coeff = -(f_plus - f_minus) / (2.0 * n_perturb)
            buf.add_(z, alpha=coeff)
            sum_losses += 0.5 * (f_plus + f_minus)
        mean_loss = sum_losses / n_perturb
        loss_trace.append(mean_loss)

        if beta2 > 0:
            ghat = buf.div(-eps)                # unbiased-scale gradient estimate
            b1 = beta1 if beta1 > 0 else 0.9
            mom.mul_(b1).add_(ghat, alpha=1.0 - b1)
            var.mul_(beta2).addcmul_(ghat, ghat, value=1.0 - beta2)
            t = step + 1
            m_hat = mom / (1.0 - b1 ** t)
            v_hat = var / (1.0 - beta2 ** t)
            lr_t = adam_lr
            if adam_warmup and step < adam_warmup:
                lr_t = adam_lr * (step + 1) / adam_warmup
            elif adam_cosine:
                prog = min(1.0, (step - adam_warmup)
                           / max(1, max_steps - adam_warmup))
                end = 1e-3 * adam_lr
                lr_t = end + 0.5 * (adam_lr - end) * (1.0 + math.cos(math.pi * prog))
            x.add_(m_hat / (v_hat.sqrt() + adam_delta), alpha=-lr_t)
        elif mom is not None:
            mom.mul_(beta1).add_(buf, alpha=1.0 - beta1)
            x.add_(mom, alpha=lr_over_eps)      # buf already carries -eps*ghat
        else:
            x.add_(buf, alpha=lr_over_eps)

        if schedule == "plateau":
            rel = (best_loss - mean_loss) / best_loss if best_loss > 0 else 0.0
            if mean_loss < best_loss:
                best_loss = mean_loss
            if rel < plateau_threshold:
                since_improve += 1
                if since_improve >= plateau_patience:
                    eps = max(eps * 0.5, eps_floor)
                    n_halvings += 1
                    since_improve = 0
            else:
                since_improve = 0

        if step_callback is not None and ((step + 1) % log_every == 0
                                          or step + 1 == max_steps):
            step_callback(step + 1, x, mean_loss, eps)

        if max_evals is not None and evals >= max_evals:
            break
        if schedule == "halve_k" and eps <= eps_floor and (
                step // int(halve_every)) * math.log(2) > math.log(eps0 / eps_floor) + 3 * math.log(2):
            break                               # eps floored for >=3 windows: nothing left to do

    info = {"evals": evals, "final_eps": eps, "steps_run": len(loss_trace),
            "n_halvings": n_halvings, "loss_trace": loss_trace}
    return x, info
