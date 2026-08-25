"""expD20 -- the shared evaluation harness: how much nonlinear headroom does a task have?

One question per task: how far below the best LINEAR model can a strong NONLINEAR
model get? The ratio

    headroom = (best linear or poly-2 ridge) / (best nonlinear)

is what decides whether a task can discriminate optimizers at all. The current
expF04 suite sits at 1.2-1.8x, which is inside the noise of any optimizer
comparison; we want >= 3x.

Every task is scored identically:
  * fixed split (the task's own if it has one, else a seeded 80/20),
  * inputs standardized then squashed to [-1,1] (this repo's domain convention),
  * targets standardized on the TRAIN split only,
  * metric = test relative L2, ||yhat - y|| / ||y||, the repo-wide metric.

Models: OLS, ridge (swept), poly-2 ridge (with interactions when d is small
enough), MLPs (1/2/3 hidden layers, several widths, early-stopped on a
validation split), HistGradientBoostingRegressor, random forest, kNN.

Run with sklearn injected:
    uv run --with scikit-learn --extra dev python experiments/expD20_tabular_suite/evaluate.py
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np

warnings.filterwarnings("ignore")

SEED = 0


# --------------------------------------------------------------------------
# metric + preprocessing
# --------------------------------------------------------------------------

def rel_l2(pred, y):
    """The repo-wide metric. Guard against a degenerate all-zero target."""
    den = np.linalg.norm(y)
    return float(np.linalg.norm(pred - y) / den) if den > 0 else float("nan")


def prep(Xtr, ytr, Xte, yte):
    """Standardize on train, squash x to [-1,1] via a robust scale.

    Targets are standardized (not min-maxed) so rel L2 is comparable across
    tasks: a model predicting the train mean scores ~1.0 on every task.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # squash the tails so the domain matches the repo's [-1,1] convention
    q = np.quantile(np.abs(Xtr), 0.999, axis=0)
    q = np.where(q < 1e-12, 1.0, q)
    Xtr, Xte = np.clip(Xtr / q, -1, 1), np.clip(Xte / q, -1, 1)
    ym, ys = ytr.mean(), ytr.std()
    ys = ys if ys > 1e-12 else 1.0
    return Xtr, (ytr - ym) / ys, Xte, (yte - ym) / ys


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def fit_linear(Xtr, ytr, Xte, yte):
    """OLS and ridge over a wide lambda sweep. Returns (ols, best_ridge)."""
    A = np.c_[Xtr, np.ones(len(Xtr))]
    Ae = np.c_[Xte, np.ones(len(Xte))]
    b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    ols = rel_l2(Ae @ b, yte)
    G, rhs, I = A.T @ A, A.T @ ytr, np.eye(A.shape[1])
    best = min(rel_l2(Ae @ np.linalg.solve(G + lam * I, rhs), yte)
               for lam in 10.0 ** np.arange(-8.0, 6.0))
    return ols, best


def poly2(X, interactions):
    cols = [X, X ** 2]
    if interactions:
        i, j = np.triu_indices(X.shape[1], 1)
        cols.append(X[:, i] * X[:, j])
    cols.append(np.ones((len(X), 1)))
    return np.hstack(cols)


def fit_poly2(Xtr, ytr, Xte, yte, max_feats=4000):
    """Degree-2 ridge. Interactions included when the expansion stays sane."""
    d = Xtr.shape[1]
    inter = (d * (d + 1) // 2 + 2 * d + 1) <= max_feats
    P, Pe = poly2(Xtr, inter), poly2(Xte, inter)
    G, rhs, I = P.T @ P, P.T @ ytr, np.eye(P.shape[1])
    best = np.inf
    for lam in 10.0 ** np.arange(-6.0, 6.0):
        try:
            best = min(best, rel_l2(Pe @ np.linalg.solve(G + lam * I, rhs), yte))
        except np.linalg.LinAlgError:
            continue
    return float(best), inter


def fit_trees(Xtr, ytr, Xte, yte, quick=False):
    """GBDT is the true strong baseline on tabular data; RF/kNN are references."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    out = {}
    grid = [dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31)] if quick else [
        dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31),
        dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63),
        dict(max_iter=1500, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=5),
    ]
    best = np.inf
    for kw in grid:
        m = HistGradientBoostingRegressor(random_state=SEED, early_stopping=True,
                                          validation_fraction=0.15, **kw).fit(Xtr, ytr)
        best = min(best, rel_l2(m.predict(Xte), yte))
    out["gbdt"] = float(best)
    n_est = 100 if not quick else 40
    m = RandomForestRegressor(n_estimators=n_est, random_state=SEED, n_jobs=-1,
                              min_samples_leaf=2).fit(Xtr, ytr)
    out["rf"] = rel_l2(m.predict(Xte), yte)
    kbest = np.inf
    for k in (5, 10, 25):
        m = KNeighborsRegressor(n_neighbors=k, n_jobs=-1).fit(Xtr, ytr)
        kbest = min(kbest, rel_l2(m.predict(Xte), yte))
    out["knn"] = float(kbest)
    return out


def fit_mlps(Xtr, ytr, Xte, yte, archs=None, seeds=(0,), max_epochs=400, quick=False):
    """Torch MLPs, early-stopped on a validation slice carved from train.

    This is the "did WE underfit?" arm, so it must be a serious attempt:
    several depths and widths, Adam + cosine, patience-based stopping.
    """
    import torch
    import torch.nn as nn

    dev = torch.device("cpu")
    torch.manual_seed(SEED)
    if archs is None:
        archs = [(256,), (1024,), (4096,), (256, 256), (512, 512), (512, 512, 512)]
        if quick:
            archs = [(256,), (1024,), (256, 256), (512, 512)]

    n = len(Xtr)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    nval = max(256, int(0.15 * n))
    vidx, tidx = perm[:nval], perm[nval:]
    Xt = torch.tensor(Xtr[tidx], dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr[tidx], dtype=torch.float32, device=dev).view(-1, 1)
    Xv = torch.tensor(Xtr[vidx], dtype=torch.float32, device=dev)
    yv = torch.tensor(ytr[vidx], dtype=torch.float32, device=dev).view(-1, 1)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)

    best_overall, best_arch = np.inf, None
    bs = 512
    for arch in archs:
        for seed in seeds:
            torch.manual_seed(seed)
            layers, prev = [], Xtr.shape[1]
            for h in arch:
                layers += [nn.Linear(prev, h), nn.GELU()]
                prev = h
            layers += [nn.Linear(prev, 1)]
            net = nn.Sequential(*layers).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_val, best_state, bad = np.inf, None, 0
            nb = max(1, len(Xt) // bs)
            for ep in range(max_epochs):
                net.train()
                idx = torch.randperm(len(Xt), device=dev)
                for b in range(nb):
                    sl = idx[b * bs:(b + 1) * bs]
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(Xt[sl]), yt[sl])
                    loss.backward()
                    opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    v = nn.functional.mse_loss(net(Xv), yv).item()
                if v < best_val * 0.9995:
                    best_val, bad = v, 0
                    best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 30:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                e = rel_l2(net(Xe).cpu().numpy().ravel(), yte)
            if e < best_overall:
                best_overall, best_arch = e, arch
    return float(best_overall), best_arch


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

@dataclass
class Result:
    task: str
    n_train: int
    n_test: int
    d: int
    ols: float = np.nan
    ridge: float = np.nan
    poly2: float = np.nan
    poly2_inter: bool = False
    mlp: float = np.nan
    mlp_arch: str = ""
    gbdt: float = np.nan
    rf: float = np.nan
    knn: float = np.nan
    wall_s: float = 0.0
    note: str = ""

    @property
    def best_linear(self):
        return float(np.nanmin([self.ols, self.ridge]))

    @property
    def best_linear_or_poly(self):
        return float(np.nanmin([self.ols, self.ridge, self.poly2]))

    @property
    def best_nonlinear(self):
        return float(np.nanmin([self.mlp, self.gbdt, self.rf, self.knn]))

    @property
    def headroom_vs_linear(self):
        return self.best_linear / self.best_nonlinear

    @property
    def headroom_vs_poly(self):
        return self.best_linear_or_poly / self.best_nonlinear


def evaluate_task(name, Xtr, ytr, Xte, yte, quick=False, note=""):
    t0 = time.time()
    Xtr, ytr, Xte, yte = prep(Xtr.astype(np.float64), ytr.astype(np.float64).ravel(),
                              Xte.astype(np.float64), yte.astype(np.float64).ravel())
    r = Result(task=name, n_train=len(Xtr), n_test=len(Xte), d=Xtr.shape[1], note=note)
    r.ols, r.ridge = fit_linear(Xtr, ytr, Xte, yte)
    r.poly2, r.poly2_inter = fit_poly2(Xtr, ytr, Xte, yte)
    tr = fit_trees(Xtr, ytr, Xte, yte, quick=quick)
    r.gbdt, r.rf, r.knn = tr["gbdt"], tr["rf"], tr["knn"]
    r.mlp, arch = fit_mlps(Xtr, ytr, Xte, yte, quick=quick)
    r.mlp_arch = "x".join(map(str, arch)) if arch else ""
    r.wall_s = time.time() - t0
    return r


def print_row(r: Result):
    print(f"{r.task:26s} n={r.n_train:7d} d={r.d:4d} | "
          f"ols {r.ols:7.4f} ridge {r.ridge:7.4f} poly2 {r.poly2:7.4f} | "
          f"mlp {r.mlp:7.4f} ({r.mlp_arch:11s}) gbdt {r.gbdt:7.4f} rf {r.rf:7.4f} knn {r.knn:7.4f} | "
          f"HR_lin {r.headroom_vs_linear:6.2f}x HR_poly {r.headroom_vs_poly:6.2f}x  [{r.wall_s:.0f}s]")


def save(results, path):
    with open(path, "w") as f:
        for r in results:
            d = asdict(r)
            d.update(best_linear=r.best_linear, best_linear_or_poly=r.best_linear_or_poly,
                     best_nonlinear=r.best_nonlinear,
                     headroom_vs_linear=r.headroom_vs_linear,
                     headroom_vs_poly=r.headroom_vs_poly)
            f.write(json.dumps(d) + "\n")
