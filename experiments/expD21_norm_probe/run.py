"""expD21 -- which feature normalization should the big factorial carry?

A SELECTION probe, not the main study. expD19 found that repairing the GELU
QI-init's feature-column scale fixes the inverse-PINN divergence, but it could
not say which normalization to adopt: BatchNorm won the PINN row while being
worst on 2-D interpolation, LayerNorm was most consistent, and the static
reparameterization achieved the exact spread fix at no parameter cost. Two
confounds blocked the decision:

  (1) expD19's BN/LN arms carried 2W extra learnable parameters, so they were
      not compared at equal capacity;
  (2) BN won the PINN row while barely reducing the column-norm spread, which
      the pure-scaling story cannot explain. The obvious missing ingredient is
      CENTERING: BN subtracts the column mean, the static reparameterization
      does not.

Five variants, identical parameter counts, halo HELD FIXED at the standard rule
in every one (the halo question is deliberately not asked here):

  baseline             no normalization
  rms_nocenter         h -> h / rms(h_init)              (expD19's static_colnorm)
  rms_center           h -> (h - mean(h_init)) / std(h_init)
  batchnorm_noaffine   nn.BatchNorm1d(W, affine=False)
  layernorm_noaffine   nn.LayerNorm(W, elementwise_affine=False)

`rms_center` is exactly a FROZEN BatchNorm: same affine map, statistics taken
once from the init feature matrix instead of tracked. So the pair
(rms_nocenter, rms_center) isolates centering from scaling, and the pair
(rms_center, batchnorm_noaffine) isolates frozen-at-init statistics from
running statistics. That is the mechanistic question expD19 left open.

With the readout zeroed, variants 2, 3, 5 and primed-4 are all pure
REPARAMETERIZATIONS at init: the represented function is identical (zero), only
the gradient geometry differs. Nothing here changes what the network can express.

SEEDS. expD19's runs are fully deterministic -- the QI init is a formula, the
readout starts at zero, the loss is full-batch, and the sample sets were fixed
-- so repeating a cell would give bitwise-identical numbers and a "3 seeds"
robustness claim would be vacuous. The seed axis here is therefore the DATA
REALIZATION: the 1-D training grid is jittered within its own spacing (expB01
established x-jitter is harmless to the floor), the 2-D disk sample is redrawn,
and the PINN interior data points are redrawn. Everything else -- geometry,
init, schedule -- is identical across seeds by construction.

Usage:
    uv run --extra dev python experiments/expD21_norm_probe/run.py --driver
    uv run --extra dev python experiments/expD21_norm_probe/run.py --job interp1d:128:gelu:rms_center:0
    uv run --extra dev python experiments/expD21_norm_probe/run.py --plot
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# expD17's machinery: targets, 2-D geometry, PDE residuals, lr schedule.
P = _load_by_path("expD21_expD17_parent",
                  REPO_ROOT / "experiments" / "expD17_geometry_motion" / "run.py")
_geo2d = P._geo2d

from src.construction.qi_mpmath import default_halo          # noqa: E402
from src.data.targets import get_target                      # noqa: E402
from src.data.targets2d import get_target_2d                 # noqa: E402
from src.data.sampling2d import disk_uniform, disk_grid      # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD21_norm_probe"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

LAMBDA = {"tanh": 0.25, "gelu": 0.707}
LAM_RATIO = LAMBDA["gelu"] / LAMBDA["tanh"]
RCOND = 1e-13

STEPS = 2000                        # short probe (expD17/expD19 used 5000)
EVAL_EVERY = 20
ADAM_LR = P.ADAM_LR
ADAM_WARMUP = P.ADAM_WARMUP

DEAD_TOL = 1e-10
DEAD_SNAPS = 30
SNAP_KEEP = 35

VARIANTS = ["baseline", "rms_nocenter", "rms_center",
            "batchnorm_noaffine", "layernorm_noaffine"]
ACTS = ["tanh", "gelu"]
SEEDS = [0, 1, 2]

PROBLEMS = {
    "interp1d": ["sine", "runge", "sine_8pi"],
    "interp2d": ["gauss_bump", "sine2d", "mixed2d"],
    "pinn_inverse": ["burgers_nu", "bratu_lam", "allencahn_k"],
}
WIDTHS = {"interp1d": 128, "interp2d": 576, "pinn_inverse": 512}

COLLAR = 2.5                        # expD17's standard radon collar, fixed


# ----------------------------- model -----------------------------

class ProbeModel(nn.Module):
    """d_in -> W -> act -> [(h - shift) * scale] -> [norm] -> readout.

    `col_shift` / `col_scale` are fixed buffers set once from the init feature
    matrix; `norm` is parameter-free. All five variants therefore have exactly
    the same trainable parameters, which is what expD19 lacked.
    """

    def __init__(self, d_in, W, act_name, variant):
        super().__init__()
        self.act_name = act_name
        self.variant = variant
        self.inner = nn.Linear(d_in, W, dtype=torch.float64)
        self.readout = nn.Linear(W, 1, dtype=torch.float64)
        if variant == "batchnorm_noaffine":
            self.norm = nn.BatchNorm1d(W, momentum=None, affine=False,
                                       dtype=torch.float64)
        elif variant == "layernorm_noaffine":
            self.norm = nn.LayerNorm(W, elementwise_affine=False,
                                     dtype=torch.float64)
        else:
            self.norm = None
        self.register_buffer("col_shift", torch.zeros(W, dtype=torch.float64))
        self.register_buffer("col_scale", torch.ones(W, dtype=torch.float64))

    def raw_features(self, x):
        z = self.inner(x)
        return torch.tanh(z) if self.act_name == "tanh" else F.gelu(z)

    def features(self, x):
        h = (self.raw_features(x) - self.col_shift) * self.col_scale
        return self.norm(h) if self.norm is not None else h

    def forward(self, x):
        return self.readout(self.features(x))


def set_static_(model, x_ref, center: bool):
    """Freeze the init column statistics into the buffers.

    center=False -> divide by the raw column RMS (scaling only).
    center=True  -> subtract the column mean, divide by the column std,
                    i.e. exactly a BatchNorm frozen at its init statistics.
    """
    with torch.no_grad():
        A = model.raw_features(x_ref)
        mu = A.mean(dim=0) if center else torch.zeros(A.shape[1], dtype=A.dtype)
        rms = (A - mu).pow(2).mean(dim=0).sqrt()
        rms = torch.where(rms < 1e-300, torch.ones_like(rms), rms)
        model.col_shift.copy_(mu)
        model.col_scale.copy_(1.0 / rms)


def prime_bn_(model, x_ref, freeze=False):
    """One full-batch forward in train mode primes the running statistics.

    `freeze` (PINN only) pins BN to eval mode: the PINN loss makes three
    separate forward passes (collocation, boundary, data), so train-mode batch
    statistics would normalize each block differently and the PDE residual
    would be inconsistent with the boundary fit. Frozen, BN is a fixed
    data-derived affine map -- the fair test of the normalization, not of a
    known BN/PINN incompatibility. Kept identical to expD19.
    """
    if isinstance(model.norm, nn.BatchNorm1d):
        model.train()
        with torch.no_grad():
            model.features(x_ref)
        if freeze:
            model.norm.eval()
            model.norm.train = lambda mode=True: model.norm


def col_norm_stats(model, x_ref, sign_class):
    with torch.no_grad():
        was = model.training
        model.eval()
        A = model.features(x_ref)
        model.train(was)
        cn = A.pow(2).mean(dim=0).sqrt().cpu().numpy()
    sc = np.asarray(sign_class)
    live = cn > 1e-300
    out = {"max_over_min": float(cn.max() / max(cn.min(), 1e-300)),
           "max_over_min_live": float(cn.max() / max(cn[live].min(), 1e-300))
           if live.any() else float("inf"),
           "frac_zero": float((~live).mean())}
    for tag, msk in (("span", sc == "span"), ("pos", sc == "pos"),
                     ("neg", sc == "neg")):
        out[f"mean_{tag}"] = float(cn[msk].mean()) if msk.any() else float("nan")
    return out


# ----------------------------- probes -----------------------------

def _tsvd(M, y):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > RCOND * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])


def probe(model, X_fit, y_fit, X_ev, y_ev):
    """Truncated-SVD readout solve on the CURRENT geometry, eval rel L2.
    Model params untouched; BN taken in eval mode so one deterministic map."""
    was = model.training
    model.eval()
    with torch.no_grad():
        Ff = model.features(X_fit).cpu().numpy()
        Fe = model.features(X_ev).cpu().numpy()
    model.train(was)
    beta = _tsvd(np.hstack([Ff, np.ones((len(Ff), 1))]),
                 y_fit.cpu().numpy().ravel())
    pred = np.hstack([Fe, np.ones((len(Fe), 1))]) @ beta
    y = y_ev.cpu().numpy().ravel()
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def _eval(model, X, y, y_norm):
    was = model.training
    model.eval()
    with torch.no_grad():
        out = float(torch.linalg.norm(model(X) - y) / y_norm)
    model.train(was)
    return out


# ----------------------------- builders -----------------------------

def build_interp1d(problem, act_name, variant, N, seed):
    lam = LAMBDA[act_name]
    h = 2.0 / N
    halo = default_halo(N, lambda_star=lam)          # STANDARD rule, all variants
    idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + idx.astype(np.float64) * h
    W = centers.size
    gamma = lam / h

    model = ProbeModel(1, W, act_name, variant)
    with torch.no_grad():
        model.inner.weight.copy_(torch.full((W, 1), gamma))
        model.inner.bias.copy_(torch.tensor(-gamma * centers))
        model.readout.weight.zero_()
        model.readout.bias.zero_()

    t = get_target(problem)
    # Seed axis = data realization: jitter the training grid within its spacing.
    n_tr = 2003
    base = torch.linspace(-1, 1, n_tr)
    if seed > 0:
        g = torch.Generator().manual_seed(1000 + seed)
        jit = (torch.rand(n_tr, generator=g) - 0.5) * (2.0 / (n_tr - 1))
        base = torch.clamp(base + jit, -1.0, 1.0)
    X_tr = base.reshape(-1, 1)
    X_ev = torch.linspace(-1, 1, 4001).reshape(-1, 1)
    y_tr, y_ev = t.fn(X_tr), t.fn(X_ev)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(model=model, net=model, inner=model.inner, x_ref=X_tr,
                geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
                eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
                probe_fn=lambda: probe(model, X_tr, y_tr, X_ev, y_ev),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "halo": int(halo), "lambda": lam})


def build_interp2d(problem, act_name, variant, N_req, seed):
    dirs, offs = _geo2d.build_radon_tensor(N_req, radius=COLLAR)
    W = dirs.shape[0]
    lam = round(P.LAMBDA_2D[problem] * (LAM_RATIO if act_name == "gelu" else 1.0), 4)
    h_ref = 2.8 / math.sqrt(N_req)
    gamma = lam / h_ref

    model = ProbeModel(2, W, act_name, variant)
    with torch.no_grad():
        model.inner.weight.copy_(gamma * torch.tensor(dirs))
        model.inner.bias.copy_(-gamma * torch.tensor(offs))
        model.readout.weight.zero_()
        model.readout.bias.zero_()

    t = get_target_2d(problem)
    X_tr_np = disk_uniform(8000, radius=1.0, seed=seed)   # seed axis
    X_ev_np = disk_grid(120, radius=1.0)
    X_tr, X_ev = torch.tensor(X_tr_np), torch.tensor(X_ev_np)
    y_tr = torch.tensor(t.fn_numpy(X_tr_np)).reshape(-1, 1)
    y_ev = torch.tensor(t.fn_numpy(X_ev_np)).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(model=model, net=model, inner=model.inner, x_ref=X_tr,
                geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
                eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
                probe_fn=lambda: probe(model, X_tr, y_tr, X_ev, y_ev),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "lambda": lam, "collar": COLLAR})


class ProbePinn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.log_p = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, x):
        return self.net(x)


def build_pinn(problem, act_name, variant, N_req, seed):
    dirs, offs = _geo2d.build_radon_tensor(N_req, radius=COLLAR)
    W = dirs.shape[0]
    lam = round(P.LAMBDA_PINN * (LAM_RATIO if act_name == "gelu" else 1.0), 4)
    h_ref = 2.8 / math.sqrt(N_req)
    gamma = lam / h_ref

    net = ProbeModel(2, W, act_name, variant)
    model = ProbePinn(net)
    u_fn, f_fn, p_true = P._ustar_and_f(problem)
    with torch.no_grad():
        model.log_p.fill_(math.log(p_true / 10.0))          # start 10x low
        net.inner.weight.copy_(gamma * torch.tensor(dirs))
        net.inner.bias.copy_(-gamma * torch.tensor(offs))
        net.readout.weight.zero_()
        net.readout.bias.zero_()

    s = torch.linspace(-0.95, 0.95, 28)
    Xc = torch.cartesian_prod(s, s)
    tb = torch.linspace(-1, 1, 64)
    ones = torch.ones(64)
    Xb = torch.cat([torch.stack([tb, ones], 1), torch.stack([tb, -ones], 1),
                    torch.stack([ones, tb], 1), torch.stack([-ones, tb], 1)])
    g = torch.Generator().manual_seed(1 + seed)             # seed axis
    Xd = (torch.rand(100, 2, generator=g) * 2 - 1) * 0.9
    f_c, u_b, u_d = f_fn(Xc), u_fn(Xb), u_fn(Xd)

    se = torch.linspace(-1, 1, 61)
    X_ev = torch.cartesian_prod(se, se)
    u_ev = u_fn(X_ev)
    u_norm = float(torch.linalg.norm(u_ev))

    def loss_fn():
        r = P._pde_residual(problem, model, Xc, f_c)
        bc = net(Xb).squeeze(-1) - u_b
        dd = net(Xd).squeeze(-1) - u_d
        return (r ** 2).mean() + (bc ** 2).mean() + (dd ** 2).mean()

    def eval_fn():
        was = net.training
        net.eval()
        with torch.no_grad():
            out = float(torch.linalg.norm(net(X_ev).squeeze(-1) - u_ev) / u_norm)
        net.train(was)
        return out

    X_fit = torch.cat([Xb, Xd])
    y_fit = torch.cat([u_b, u_d]).reshape(-1, 1)

    return dict(model=model, net=net, inner=net.inner,
                x_ref=torch.cat([Xc, X_fit]),
                geom_params=[net.inner.weight, net.inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn,
                probe_fn=lambda: probe(net, X_fit, y_fit, X_ev,
                                       u_ev.reshape(-1, 1)),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "lambda": lam, "p_true": p_true,
                       "collar": COLLAR},
                param_cb=lambda: float(torch.exp(model.log_p).detach()))


BUILDERS = {"interp1d": build_interp1d, "interp2d": build_interp2d,
            "pinn_inverse": build_pinn}


# ----------------------------- one run -----------------------------

def preact_sign(inner, x_ref):
    with torch.no_grad():
        Z = inner(x_ref)
        zmin, zmax = Z.min(dim=0).values, Z.max(dim=0).values
    return ["neg" if hi < 0 else ("pos" if lo > 0 else "span")
            for lo, hi in zip(zmin.tolist(), zmax.tolist())]


def snap_schedule(total=STEPS, dense_until=25, factor=1.2, cap=50):
    out, s, iv = [], 1, 1.0
    while s <= total:
        out.append(s)
        if s >= dense_until:
            iv = min(iv * factor, cap)
        s += max(1, int(round(iv)))
    return out


def run_one(cls, problem, act_name, variant, width, seed):
    b = BUILDERS[cls](problem, act_name, variant, width, seed)
    model, geom, inner, net = b["model"], b["geom_params"], b["inner"], b["net"]

    sign_class = preact_sign(inner, b["x_ref"])
    if variant == "rms_nocenter":
        set_static_(net, b["x_ref"], center=False)
    elif variant == "rms_center":
        set_static_(net, b["x_ref"], center=True)
    prime_bn_(net, b["x_ref"], freeze=(cls == "pinn_inverse"))
    model.train()

    cn_init = col_norm_stats(net, b["x_ref"], sign_class)
    g0 = P.geom_flat(geom)
    n0 = float(torch.linalg.norm(g0))
    pre_probe = b["probe_fn"]()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    sched = set(snap_schedule())

    drift, drift_abs, evals, param_traj = [], [], [], []
    snap_steps, snap_upd = [], []
    t0 = time.time()
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = P.lr_at(step, b["lr"], ADAM_WARMUP, STEPS)
        snap = (step + 1) in sched and len(snap_steps) < SNAP_KEEP
        if snap:
            w_before = inner.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        b["loss_fn"]().backward()
        opt.step()

        g = P.geom_flat(geom)
        da = float(torch.linalg.norm(g - g0))
        drift_abs.append(da)
        drift.append(da / n0)
        if snap:
            with torch.no_grad():
                upd = torch.linalg.norm(inner.weight.detach() - w_before, dim=1)
            snap_steps.append(step + 1)
            snap_upd.append([float(f"{v:.5g}") for v in upd.tolist()])
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            e = b["eval_fn"]()
            evals.append([step + 1, e if math.isfinite(e) else None])
            if "param_cb" in b:
                param_traj.append([step + 1, b["param_cb"]()])

    cn_final = col_norm_stats(net, b["x_ref"], sign_class)
    row = {"class": cls, "problem": problem, "activation": act_name,
           "variant": variant, "width": width, "seed": seed, "steps": STEPS,
           "n_params": n_params, "g0_norm": n0, "n_geom": int(g0.numel()),
           "abs_drift_end": drift_abs[-1], "rel_drift_end": drift[-1],
           "pre_probe": pre_probe, "post_probe": b["probe_fn"](),
           "final_err": b["eval_fn"](),
           "colnorm_init": cn_init, "colnorm_final": cn_final,
           "sign_class": sign_class, "snap_steps": snap_steps,
           "snap_upd": snap_upd, "evals": evals,
           "wall_s": round(time.time() - t0, 1), "extra": b["extra"]}
    if param_traj:
        row["param_traj"] = param_traj
        row["param_true"] = b["extra"]["p_true"]
        row["param_final"] = param_traj[-1][1]
    print(f"  {cls}/{problem}/{act_name}/{variant}/s{seed}: "
          f"run={row['final_err']:.3e} pre={pre_probe:.2e} "
          f"post={row['post_probe']:.2e} ({row['wall_s']}s)", flush=True)
    return row


def dead_stats(row):
    upd = np.asarray(row["snap_upd"][:DEAD_SNAPS], dtype=float)
    if upd.size == 0:
        return {}
    dead = (upd < DEAD_TOL).all(axis=0)
    zero = (upd == 0.0).all(axis=0)
    sc = np.asarray(row["sign_class"])
    out = {"dead_frac": float(dead.mean()), "zero_frac": float(zero.mean())}
    for tag in ("span", "pos", "neg"):
        m = sc == tag
        out[f"frac_{tag}"] = float(m.mean())
        out[f"dead_of_{tag}"] = float(dead[m].mean()) if m.any() else float("nan")
    return out


# ----------------------------- jobs -----------------------------

def job_list():
    return [(cls, act, var, seed)
            for cls in PROBLEMS for act in ACTS
            for var in VARIANTS for seed in SEEDS]


def run_job(cls, act, variant, seed):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{cls}_{act}_{variant}_s{seed}.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for problem in PROBLEMS[cls]:
            f.write(json.dumps(run_one(cls, problem, act, variant,
                                       WIDTHS[cls], seed)) + "\n")
            f.flush()
    tmp.replace(out)                     # atomic: partial files never land
    print(f"Saved {out}", flush=True)


def job_done(cls, act, variant, seed):
    p = DATA_DIR / f"{cls}_{act}_{variant}_s{seed}.jsonl"
    return p.exists() and sum(1 for _ in open(p)) >= len(PROBLEMS[cls])


def driver(max_procs=9):
    jobs = [j for j in job_list() if not job_done(*j)]
    print(f"expD21: {len(jobs)} jobs x 3 problems = {3*len(jobs)} runs, "
          f"{max_procs}-way parallel", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    running, queue = [], list(jobs)
    while queue or running:
        while queue and len(running) < max_procs:
            cls, act, var, seed = queue.pop(0)
            cmd = [sys.executable, __file__, "--job",
                   f"{cls}:{act}:{var}:{seed}"]
            running.append((subprocess.Popen(cmd, env=env),
                            (cls, act, var, seed)))
        time.sleep(3)
        still = []
        for pr, tag in running:
            if pr.poll() is None:
                still.append((pr, tag))
            elif pr.returncode != 0:
                print(f"  FAILED {tag} rc={pr.returncode}", flush=True)
        running = still
    print("all jobs done", flush=True)


def param_gate():
    """All five variants must have IDENTICAL trainable parameter counts."""
    counts = {}
    for v in VARIANTS:
        b = build_interp1d("sine", "gelu", v, 64, 0)
        counts[v] = sum(p.numel() for p in b["model"].parameters()
                        if p.requires_grad)
    print("trainable parameter counts:", counts)
    assert len(set(counts.values())) == 1, f"parameter mismatch: {counts}"
    print("GATE PASS: identical parameter counts across all variants")
    return counts


if __name__ == "__main__":
    if "--job" in sys.argv:
        cls, act, var, seed = sys.argv[sys.argv.index("--job") + 1].split(":")
        run_job(cls, act, var, int(seed))
    elif "--gate" in sys.argv:
        param_gate()
    elif "--plot" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analysis
        analysis.main()
    else:
        param_gate()
        driver()
