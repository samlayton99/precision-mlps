"""expF11 driver: D0 (random init) + three QI inits, low-data Darcy + convergence.

  D0: random-init FNO, train on labels
  1 : pretrain on (a, u_QI), fine-tune on labels
  2 : train on (u_ref - u_QI); infer u_QI + FNO
  3 : QI spectral init, train on labels

Usage: run.py --method all [--smoke] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for d in ("expF10_qi_operator", "expF11_qi_fno_init"):
    sys.path.append(str(REPO_ROOT / "experiments" / d))

import fno2d
import data as dd
import qi_codec as qc
import qi_solve as qs
import init_methods as im

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF11_qi_fno_init"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    res: int = 64
    n_qi: int = 500
    n_lab: int = 100
    n_test: int = 200
    epochs: int = 100
    pre_epochs: int = 100
    batch: int = 32
    lr: float = 1e-3
    W: int = 576
    fno_kw: dict = field(default_factory=lambda: dict(width=32, modes=12, n_layers=4))


SMOKE_CFG = Cfg(res=32, n_qi=8, n_lab=8, n_test=8, epochs=3, pre_epochs=3, batch=4, W=128)


def _t(x):
    return torch.tensor(np.asarray(x, np.float32))


def _rel(pred, tgt):
    return (torch.linalg.vector_norm(pred - tgt, dim=1)
            / torch.linalg.vector_norm(tgt, dim=1)).mean()


def _field(a):
    return _t(a)[:, None]                       # [n,1,res,res]


def _flat(u):
    return _t(u).reshape(len(u), -1)


def _fit(net, xin, ytgt, cfg, epochs, log=None, logx=None, logy=None):
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xin, ytgt = xin.to(DEV), ytgt.to(DEV)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xin), device=DEV)
        for i in range(0, len(xin), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = _rel(net(xin[idx]).reshape(len(idx), -1), ytgt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if log is not None:
            log.append(dict(epoch=ep, rel=_test(net, logx, logy)))


@torch.no_grad()
def _test(net, xte, yte):
    net.eval()
    return _rel(net(xte.to(DEV)).reshape(len(xte), -1), yte.to(DEV)).item()


def _load(cfg):
    a_tr, u_tr = dd.load_darcy("train", cfg.n_qi, cfg.res)      # n_qi >= n_lab
    a_te, u_te = dd.load_darcy("test", cfg.n_test, cfg.res)
    uq_tr = qs.batch_u_qi(a_tr, cfg.res, "train")
    uq_te = qs.batch_u_qi(a_te, cfg.res, "test")
    return a_tr, u_tr, uq_tr, a_te, u_te, uq_te


def train_eval(method, cfg):
    torch.manual_seed(0)
    a_tr, u_tr, uq_tr, a_te, u_te, uq_te = _load(cfg)
    L = cfg.n_lab
    xte, yte = _field(a_te), _flat(u_te)
    net = fno2d.FNO2d(**cfg.fno_kw).to(DEV)
    conv = []
    t0 = time.time()
    if method == "3":
        im.qi_spectral_init(net, qc.QICodec(cfg.W, 0.25), res=cfg.res)
    if method == "1":
        _fit(net, _field(a_tr), _flat(uq_tr), cfg, cfg.pre_epochs)   # pretrain
    if method == "2":
        _fit(net, _field(a_tr[:L]), _flat(u_tr[:L] - uq_tr[:L]), cfg, cfg.epochs,
             conv, _field(a_te), _flat(u_te - uq_te))
        pred = (_t(uq_te).reshape(len(uq_te), -1)
                + net(xte.to(DEV)).reshape(len(xte), -1).cpu())
        rel = _rel(pred, yte).item()
    else:
        _fit(net, _field(a_tr[:L]), _flat(u_tr[:L]), cfg, cfg.epochs,
             conv, xte, yte)
        rel = _test(net, xte, yte)
    return dict(method=method, n_lab=L, res=cfg.res, test_rel_l2=rel,
                conv=conv, t_train=time.time() - t0)


def _start_loss(net, cfg):
    a_tr, u_tr, *_ = _load(cfg)
    return _test(net, _field(a_tr[:cfg.n_lab]), _flat(u_tr[:cfg.n_lab]))


def random_start_loss(cfg):
    torch.manual_seed(0)
    return _start_loss(fno2d.FNO2d(**cfg.fno_kw).to(DEV), cfg)


def pretrain_start_loss(cfg):
    torch.manual_seed(0)
    a_tr, u_tr, uq_tr, *_ = _load(cfg)
    net = fno2d.FNO2d(**cfg.fno_kw).to(DEV)
    _fit(net, _field(a_tr), _flat(uq_tr), cfg, cfg.pre_epochs)
    return _start_loss(net, cfg)


def save(recs, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(recs, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = json.loads((RESULTS_DIR / "data.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = sorted({r["n_lab"] for r in recs})
    methods = ["D0", "1", "2", "3"]
    x = np.arange(len(labs))
    for k, m in enumerate(methods):
        ys = [next((r["test_rel_l2"] for r in recs
                    if r["method"] == m and r["n_lab"] == L), np.nan) for L in labs]
        ax.bar(x + k * 0.2, ys, width=0.2, label=f"method {m}")
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([f"N={L}" for L in labs])
    ax.set_ylabel("test rel L2")
    ax.legend(fontsize=8)
    ax.set_title("expF11: QI-init methods vs random (D0)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "init_accuracy.png", dpi=150)
    Lmin = labs[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        r = next((r for r in recs if r["method"] == m and r["n_lab"] == Lmin), None)
        if r and r["conv"]:
            ax.plot([c["epoch"] for c in r["conv"]],
                    [c["rel"] for c in r["conv"]], label=f"method {m}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test rel L2")
    ax.set_title(f"expF11: convergence (N={Lmin})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "init_convergence.png", dpi=150)
    print("figures written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
        return
    cfg = SMOKE_CFG if args.smoke else Cfg()
    methods = ["D0", "1", "2", "3"] if args.method == "all" else [args.method]
    recs = []
    for L in ([cfg.n_lab] if args.smoke else [100, 300]):
        c = cfg if args.smoke else Cfg(**{**cfg.__dict__, "n_lab": L})
        for m in methods:
            rec = train_eval(m, c)
            recs.append(rec)
            print({k: rec[k] for k in ("method", "n_lab", "test_rel_l2", "t_train")},
                  flush=True)
            save(recs, "data.json")


if __name__ == "__main__":
    main()
