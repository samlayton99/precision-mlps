"""expD19 -- does a scale-aware GELU init fix the pathologies expD17 found?

expD17 measured three GELU/QI-init pathologies, all absent under tanh:
  (a) 68-92% of right-halo neurons receive bitwise-zero Adam updates,
  (b) floor-quality geometries are damaged 2-3 orders more than under tanh,
  (c) the inverse-PINN runs diverge at width (bratu field error 87 at W=1024).

The diagnosis: at lambda* = 0.707 with gamma = O(N), GELU's unbounded right
limit (gelu(z) -> z) makes left-halo feature columns grow linearly in width
(RMS norm 21 -> 72 -> 138 as N goes 64 -> 128 -> 256) while the right-halo
columns underflow to zero. Phi^T Phi's diagonal therefore spans ~10 orders,
self-inflicted by the init. lstsq is scale-robust and still reaches 2.3e-13;
gradient training is not, and that is where GELU fails.

Six arms test the fix, all with the QI init, plain Adam, zero-init readout:

  baseline        expD17's GELU QI init as-is (halo ~0.4N, bias only)
  halo8           same, halo 8 per side (2-D/PINN analog: radon collar 2.5 -> 1.3)
  static_colnorm  full halo, features divided by their init column norms.
                  A pure REPARAMETERIZATION: a fixed buffer, no trainable
                  parameters, no running statistics, no batch dependence, so
                  it is the variant compatible with the docs/REQUIREMENTS.md
                  practicality gate (optimizer state stays O(m), Adam's class).
                  The function at init is unchanged; only the parameterization,
                  and therefore the optimization trajectory, differs.
  recommended     halo8 + static_colnorm + an explicit linear term. GELU is
                  kernel order r=2 (K = gelu''), so the paper's integration
                  polynomial p_{r-1} is AFFINE, not the single bias tanh (r=1)
                  needs; our init has only ever supplied the bias.
  batchnorm       BatchNorm1d on the hidden features (affine=True), full halo
  layernorm       LayerNorm on the hidden features (affine=True), full halo

Prediction under test, reported honestly either way: BN normalizes per neuron
across samples, which is the pathology's axis, so it should help; LN normalizes
per sample across neurons, so the norm-138 halo entries dominate each sample's
statistics and should compress the informative interior features.

tanh cross-check: `baseline` and `recommended` on the 1-D and 2-D classes, to
confirm the fixes do not harm the activation that already worked and to test
whether the R ~ 0.4N halo rule is oversized for tanh too.

Usage:
    uv run --extra dev python experiments/expD19_gelu_init/run.py --driver
    uv run --extra dev python experiments/expD19_gelu_init/run.py --job interp1d:128:gelu:halo8
    uv run --extra dev python experiments/expD19_gelu_init/run.py --plot
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# expD17's machinery: builders' data/targets, glorot, lr schedule, sign classes.
P = _load_by_path("expD19_expD17_parent",
                  REPO_ROOT / "experiments" / "expD17_geometry_motion" / "run.py")
_geo2d = P._geo2d

from src.construction.qi_mpmath import default_halo          # noqa: E402
from src.data.targets import get_target                      # noqa: E402
from src.data.targets2d import get_target_2d                 # noqa: E402
from src.data.sampling2d import disk_uniform, disk_grid      # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD19_gelu_init"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

LAMBDA = {"tanh": 0.25, "gelu": 0.707}
LAM_RATIO = LAMBDA["gelu"] / LAMBDA["tanh"]
RCOND = 1e-13

STEPS = P.STEPS                     # 5000, expD17's budget
EVAL_EVERY = 25
ADAM_LR = P.ADAM_LR
ADAM_WARMUP = P.ADAM_WARMUP

# Dead-neuron criterion (expD17's): applied update < 1e-10 across the first
# DEAD_SNAPS snapshots. Snapshots are dense early, so this is early training.
DEAD_TOL = 1e-10
DEAD_SNAPS = 30
SNAP_KEEP = 35                      # store per-row update norms this many times

PROBLEMS = {
    "interp1d": ["sine", "runge", "sine_8pi"],
    "interp2d": ["gauss_bump", "sine2d", "mixed2d"],
    "pinn_inverse": ["burgers_nu", "bratu_lam", "allencahn_k"],
}
# PINN runs at W=512 only: W=1024 was cut for runtime. The GELU+QI divergence
# already appears at 512 (expD17: bratu 67% parameter error, field error 0.366),
# so the divergence test is intact at the smaller width. expD17's W=1024 numbers
# (bratu 90.3%, field error 87) remain the record of the pathology there; what
# is untested is only whether these fixes still hold at W=1024.
WIDTHS = {"interp1d": [128, 256], "interp2d": [576], "pinn_inverse": [512]}

GELU_ARMS = ["baseline", "halo8", "static_colnorm", "recommended",
             "batchnorm", "layernorm"]
TANH_ARMS = ["baseline", "recommended"]
# tanh cross-check skips the expensive PINN class (see writeup).
TANH_CLASSES = ["interp1d", "interp2d"]

ARM_SPEC = {
    #                 small_halo  colnorm  linear  norm
    "baseline":       (False,     False,   False,  None),
    "halo8":          (True,      False,   False,  None),
    "static_colnorm": (False,     True,    False,  None),
    "recommended":    (True,      True,    True,   None),
    "batchnorm":      (False,     False,   False,  "bn"),
    "layernorm":      (False,     False,   False,  "ln"),
}

SMALL_HALO_1D = 8
COLLAR_FULL = 2.5                   # expD17's radon collar
COLLAR_SMALL = 1.3                  # the 2-D/PINN "small halo" analog


# ----------------------------- model -----------------------------

class ArmModel(nn.Module):
    """d_in -> W -> act -> [scale] -> [norm] -> readout, plus optional linear skip.

    `col_scale` is a fixed buffer, not a parameter: features are divided by
    their init column norms. Setting the readout to zero (as every arm does)
    means the represented function is unchanged at init; what changes is the
    parameterization, hence the gradient geometry.
    """

    def __init__(self, d_in, W, act_name, norm=None, linear=False):
        super().__init__()
        self.act_name = act_name
        self.inner = nn.Linear(d_in, W, dtype=torch.float64)
        self.readout = nn.Linear(W, 1, dtype=torch.float64)
        self.lin = nn.Linear(d_in, 1, bias=False, dtype=torch.float64) if linear else None
        if norm == "bn":
            # momentum=None -> cumulative average, so one full-batch forward
            # pass sets running stats to the exact training statistics.
            self.norm = nn.BatchNorm1d(W, momentum=None, dtype=torch.float64)
        elif norm == "ln":
            self.norm = nn.LayerNorm(W, dtype=torch.float64)
        else:
            self.norm = None
        self.register_buffer("col_scale", torch.ones(W, dtype=torch.float64))
        if self.lin is not None:
            with torch.no_grad():
                self.lin.weight.zero_()

    def raw_features(self, x):
        z = self.inner(x)
        return torch.tanh(z) if self.act_name == "tanh" else torch.nn.functional.gelu(z)

    def features(self, x):
        h = self.raw_features(x) * self.col_scale
        return self.norm(h) if self.norm is not None else h

    def forward(self, x):
        y = self.readout(self.features(x))
        return y + self.lin(x) if self.lin is not None else y


def set_col_scale_(model, x_ref):
    """col_scale_k = 1/RMS_k, from the init feature matrix on x_ref."""
    with torch.no_grad():
        A = model.raw_features(x_ref)
        rms = A.pow(2).mean(dim=0).sqrt()
        rms = torch.where(rms < 1e-300, torch.ones_like(rms), rms)
        model.col_scale.copy_(1.0 / rms)


def prime_bn_(model, x_ref, freeze=False):
    """One full-batch forward in train mode: running stats become the exact
    training statistics, so eval-mode probes are meaningful from step 0.

    `freeze` puts BN permanently in eval mode. Required for the PINN class,
    whose loss makes THREE separate forward passes (collocation, boundary,
    data): in train mode each block would be normalized by its own batch
    statistics, so the three terms would see three different networks and the
    PDE residual would be inconsistent with the boundary fit. Freezing after
    priming makes BN a fixed, data-derived affine map -- the fair test of
    "does normalizing the feature columns fix the pathology", rather than a
    test of a known BN/PINN incompatibility. Full-batch single-block losses
    (interp1d, interp2d) need no such handling: with momentum=None and one
    repeated batch, train-mode BN and the primed running stats coincide.
    """
    if isinstance(model.norm, nn.BatchNorm1d):
        model.train()
        with torch.no_grad():
            model.features(x_ref)
        if freeze:
            model.norm.eval()
            model.norm.train = lambda mode=True: model.norm    # stays frozen


def col_norm_stats(model, x_ref, sign_class):
    """RMS feature-column norms (what training's curvature sees), by region."""
    with torch.no_grad():
        A = model.features(x_ref)
        cn = A.pow(2).mean(dim=0).sqrt().cpu().numpy()
    sc = np.asarray(sign_class)
    live = cn > 1e-300
    out = {"max_over_min": float(cn.max() / max(cn.min(), 1e-300)),
           "max_over_min_live": float(cn.max() / max(cn[live].min(), 1e-300))
           if live.any() else float("inf"),
           "frac_zero": float((~live).mean())}
    for tag, msk in (("span", sc == "span"), ("pos", sc == "pos"), ("neg", sc == "neg")):
        out[f"mean_{tag}"] = float(cn[msk].mean()) if msk.any() else float("nan")
    return out


# ----------------------------- probes -----------------------------

def _tsvd(M, y):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > RCOND * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])


def probe(model, X_fit, y_fit, X_ev, y_ev, linear):
    """Truncated-SVD readout solve on the CURRENT geometry, eval rel L2.

    Arms with a linear term get the x columns in the solve, so every arm is
    scored against the best readout ITS parameterization can express.
    BatchNorm features are taken in eval mode (running stats) so the fit and
    eval halves use one deterministic feature map.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        Ff = model.features(X_fit).cpu().numpy()
        Fe = model.features(X_ev).cpu().numpy()
    model.train(was_training)
    xf, xe = X_fit.cpu().numpy(), X_ev.cpu().numpy()
    cols = [Ff, np.ones((len(xf), 1))]
    colse = [Fe, np.ones((len(xe), 1))]
    if linear:
        cols.append(xf)
        colse.append(xe)
    beta = _tsvd(np.hstack(cols), y_fit.cpu().numpy().ravel())
    pred = np.hstack(colse) @ beta
    y = y_ev.cpu().numpy().ravel()
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


# ----------------------------- builders -----------------------------

def build_interp1d(problem, act_name, arm, N):
    small, colnorm, linear, norm = ARM_SPEC[arm]
    lam = LAMBDA[act_name]
    h = 2.0 / N
    halo = SMALL_HALO_1D if small else default_halo(N, lambda_star=lam)
    idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + idx.astype(np.float64) * h
    W = centers.size
    gamma = lam / h

    model = ArmModel(1, W, act_name, norm=norm, linear=linear)
    with torch.no_grad():
        model.inner.weight.copy_(torch.full((W, 1), gamma))
        model.inner.bias.copy_(torch.tensor(-gamma * centers))
        model.readout.weight.zero_()
        model.readout.bias.zero_()

    t = get_target(problem)
    X_tr = torch.linspace(-1, 1, 2003).reshape(-1, 1)
    X_ev = torch.linspace(-1, 1, 4001).reshape(-1, 1)
    y_tr, y_ev = t.fn(X_tr), t.fn(X_ev)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(model=model, inner=model.inner, x_ref=X_tr, linear=linear,
                colnorm=colnorm, geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
                eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
                probe_fn=lambda: probe(model, X_tr, y_tr, X_ev, y_ev, linear),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "halo": int(halo), "lambda": lam})


def build_interp2d(problem, act_name, arm, N_req):
    small, colnorm, linear, norm = ARM_SPEC[arm]
    collar = COLLAR_SMALL if small else COLLAR_FULL
    dirs, offs = _geo2d.build_radon_tensor(N_req, radius=collar)
    W = dirs.shape[0]
    lam = round(P.LAMBDA_2D[problem] * (LAM_RATIO if act_name == "gelu" else 1.0), 4)
    h_ref = 2.8 / math.sqrt(N_req)
    gamma = lam / h_ref

    model = ArmModel(2, W, act_name, norm=norm, linear=linear)
    with torch.no_grad():
        model.inner.weight.copy_(gamma * torch.tensor(dirs))
        model.inner.bias.copy_(-gamma * torch.tensor(offs))
        model.readout.weight.zero_()
        model.readout.bias.zero_()

    t = get_target_2d(problem)
    X_tr_np = disk_uniform(8000, radius=1.0, seed=0)
    X_ev_np = disk_grid(120, radius=1.0)
    X_tr, X_ev = torch.tensor(X_tr_np), torch.tensor(X_ev_np)
    y_tr = torch.tensor(t.fn_numpy(X_tr_np)).reshape(-1, 1)
    y_ev = torch.tensor(t.fn_numpy(X_ev_np)).reshape(-1, 1)
    y_norm = float(torch.linalg.norm(y_ev))

    return dict(model=model, inner=model.inner, x_ref=X_tr, linear=linear,
                colnorm=colnorm, geom_params=[model.inner.weight, model.inner.bias],
                loss_fn=lambda: ((model(X_tr) - y_tr) ** 2).mean(),
                eval_fn=lambda: _eval(model, X_ev, y_ev, y_norm),
                probe_fn=lambda: probe(model, X_tr, y_tr, X_ev, y_ev, linear),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "lambda": lam, "collar": collar})


class ArmPinn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.log_p = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, x):
        return self.net(x)


def build_pinn(problem, act_name, arm, N_req):
    small, colnorm, linear, norm = ARM_SPEC[arm]
    collar = COLLAR_SMALL if small else COLLAR_FULL
    dirs, offs = _geo2d.build_radon_tensor(N_req, radius=collar)
    W = dirs.shape[0]
    lam = round(P.LAMBDA_PINN * (LAM_RATIO if act_name == "gelu" else 1.0), 4)
    h_ref = 2.8 / math.sqrt(N_req)
    gamma = lam / h_ref

    net = ArmModel(2, W, act_name, norm=norm, linear=linear)
    model = ArmPinn(net)
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
    g = torch.Generator().manual_seed(1)
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
            pred = net(X_ev).squeeze(-1)
            out = float(torch.linalg.norm(pred - u_ev) / u_norm)
        net.train(was)
        return out

    X_fit = torch.cat([Xb, Xd])
    y_fit = torch.cat([u_b, u_d]).reshape(-1, 1)

    return dict(model=model, inner=net.inner, x_ref=torch.cat([Xc, X_fit]),
                linear=linear, colnorm=colnorm, net=net,
                geom_params=[net.inner.weight, net.inner.bias],
                loss_fn=loss_fn, eval_fn=eval_fn,
                probe_fn=lambda: probe(net, X_fit, y_fit, X_ev,
                                       u_ev.reshape(-1, 1), linear),
                lr=ADAM_LR,
                extra={"W": W, "gamma": gamma, "lambda": lam, "p_true": p_true,
                       "collar": collar},
                param_cb=lambda: float(torch.exp(model.log_p).detach()))


def _eval(model, X, y, y_norm):
    was = model.training
    model.eval()
    with torch.no_grad():
        out = float(torch.linalg.norm(model(X) - y) / y_norm)
    model.train(was)
    return out


BUILDERS = {"interp1d": build_interp1d, "interp2d": build_interp2d,
            "pinn_inverse": build_pinn}


# ----------------------------- one run -----------------------------

def run_one(cls, problem, act_name, arm, width):
    b = BUILDERS[cls](problem, act_name, arm, width)
    model, geom, inner = b["model"], b["geom_params"], b["inner"]
    net = b.get("net", model)

    sign_class = P_preact_sign(inner, b["x_ref"])
    if b["colnorm"]:
        set_col_scale_(net, b["x_ref"])
    prime_bn_(net, b["x_ref"], freeze=(cls == "pinn_inverse"))
    model.train()

    cn_init = col_norm_stats(net, b["x_ref"], sign_class)
    g0 = P.geom_flat(geom)
    n0 = float(torch.linalg.norm(g0))
    prev = g0.clone()
    pre_probe = b["probe_fn"]()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    sched = set(P_snap_schedule())

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
        prev = g
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
    row = {"class": cls, "problem": problem, "activation": act_name, "arm": arm,
           "width": width, "steps": STEPS, "n_params": n_params,
           "g0_norm": n0, "n_geom": int(g0.numel()),
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
    print(f"  {cls}/{problem}/{act_name}/{arm}/w{width}: "
          f"run={row['final_err']:.3e} pre={pre_probe:.2e} "
          f"post={row['post_probe']:.2e} cn={cn_init['max_over_min_live']:.1e} "
          f"({row['wall_s']}s)", flush=True)
    return row


def P_preact_sign(inner, x_ref):
    with torch.no_grad():
        Z = inner(x_ref)
        zmin, zmax = Z.min(dim=0).values, Z.max(dim=0).values
    return ["neg" if hi < 0 else ("pos" if lo > 0 else "span")
            for lo, hi in zip(zmin.tolist(), zmax.tolist())]


def P_snap_schedule(total=STEPS, dense_until=25, factor=1.2, cap=50):
    out, s, iv = [], 1, 1.0
    while s <= total:
        out.append(s)
        if s >= dense_until:
            iv = min(iv * factor, cap)
        s += max(1, int(round(iv)))
    return out


def dead_stats(row):
    """Fraction of neurons with |update| < DEAD_TOL across the first
    DEAD_SNAPS snapshots, split by preactivation sign class."""
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
    jobs = []
    for cls in WIDTHS:
        for w in WIDTHS[cls]:
            for arm in GELU_ARMS:
                jobs.append((cls, w, "gelu", arm))
            if cls in TANH_CLASSES:
                for arm in TANH_ARMS:
                    jobs.append((cls, w, "tanh", arm))
    return jobs


def run_job(cls, width, act_name, arm):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{cls}_w{width}_{act_name}_{arm}.jsonl"
    with open(out, "w") as f:
        for problem in PROBLEMS[cls]:
            f.write(json.dumps(run_one(cls, problem, act_name, arm, width)) + "\n")
            f.flush()
    print(f"Saved {out}", flush=True)


def job_done(cls, width, act_name, arm):
    """A job is complete when its file holds one row per problem."""
    p = DATA_DIR / f"{cls}_w{width}_{act_name}_{arm}.jsonl"
    if not p.exists():
        return False
    return sum(1 for _ in open(p)) >= len(PROBLEMS[cls])


def driver(max_procs=9):
    jobs = [j for j in job_list() if not job_done(*j)]
    print(f"expD19: {len(jobs)} jobs x 3 problems = {3*len(jobs)} runs, "
          f"{max_procs}-way parallel", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    running, queue = [], list(jobs)
    while queue or running:
        while queue and len(running) < max_procs:
            cls, w, a, arm = queue.pop(0)
            cmd = [sys.executable, __file__, "--job", f"{cls}:{w}:{a}:{arm}"]
            running.append((subprocess.Popen(cmd, env=env), (cls, w, a, arm)))
        time.sleep(3)
        still = []
        for pr, tag in running:
            if pr.poll() is None:
                still.append((pr, tag))
            elif pr.returncode != 0:
                print(f"  FAILED {tag} rc={pr.returncode}", flush=True)
        running = still
    print("all jobs done", flush=True)


if __name__ == "__main__":
    if "--job" in sys.argv:
        cls, w, a, arm = sys.argv[sys.argv.index("--job") + 1].split(":")
        run_job(cls, int(w), a, arm)
    elif "--plot" in sys.argv:
        import analysis
        analysis.main()
    else:
        driver()
