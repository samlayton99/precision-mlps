"""expF10 driver: train A/B/C on Darcy, eval accuracy + discretization
invariance + data efficiency.

Usage:
  uv run --extra dev python experiments/expF10_qi_operator/run.py --config all
  uv run --extra dev python experiments/expF10_qi_operator/run.py --eval-invariance
  uv run --extra dev python experiments/expF10_qi_operator/run.py --eval-data-eff
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
sys.path.append(str(HERE))

import qi_codec as qc
import data as dd
import models as mo

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF10_qi_operator"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    W: int = 576
    lam: float = 0.25
    res: int = 64
    n_train: int = 1000
    n_test: int = 200
    epochs: int = 100
    batch: int = 32
    lr: float = 1e-3
    fno_kw: dict = field(default_factory=lambda: dict(width=32, modes=12, n_layers=4))


SMOKE_CFG = Cfg(W=128, res=16, n_train=16, n_test=16, epochs=2, batch=8,
                fno_kw=dict(width=8, modes=6, n_layers=2))


def _prep(config, cfg, codec, res=None, n_train=None):
    """Return (train_in, train_tgt, test_in, test_tgt, kind) tensors at res.
    Targets are the flattened reference solution fields."""
    res = res or cfg.res
    n_train = n_train if n_train is not None else cfg.n_train
    a_tr, u_tr = dd.load_darcy("train", n_train, res)
    a_te, u_te = dd.load_darcy("test", cfg.n_test, res)
    u_tr_f = torch.tensor(u_tr.reshape(len(u_tr), -1), dtype=torch.float32)
    u_te_f = torch.tensor(u_te.reshape(len(u_te), -1), dtype=torch.float32)
    if config == "A":
        Pinv = codec.pinv(res)
        enc = lambda A: torch.tensor(
            (Pinv @ A.reshape(len(A), -1).T).T, dtype=torch.float32)
        return enc(a_tr), u_tr_f, enc(a_te), u_te_f, "coeff"

    def to_field(A):
        if config == "B":
            Pinv = codec.pinv(res)
            Phi = codec.basis(codec.grid(res))
            A = (Phi @ (Pinv @ A.reshape(len(A), -1).T)).T.reshape(A.shape)
        return torch.tensor(A[:, None], dtype=torch.float32)   # [n,1,res,res]
    return to_field(a_tr), u_tr_f, to_field(a_te), u_te_f, "field"


def _rel_l2(pred, tgt):
    """Mean per-sample relative L2 as a tensor (keeps grad for the loss)."""
    return (torch.linalg.vector_norm(pred - tgt, dim=1)
            / torch.linalg.vector_norm(tgt, dim=1)).mean()


def _build(config, cfg, codec, res):
    if config == "A":
        Phi_out = codec.basis(codec.grid(res))
        net, _ = mo.build_model("A", D=codec.D, Phi_out=Phi_out)
    else:
        net, _ = mo.build_model(config, fno_kw=cfg.fno_kw)
    return net.to(DEV)


def _train(net, xin, ytr, cfg):
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    xin, ytr = xin.to(DEV), ytr.to(DEV)
    for ep in range(cfg.epochs):
        net.train()
        perm = torch.randperm(len(xin), device=DEV)
        for i in range(0, len(xin), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            pred = net(xin[idx]).reshape(len(idx), -1)
            loss = _rel_l2(pred, ytr[idx])
            loss.backward()
            opt.step()
        sched.step()


@torch.no_grad()
def _test(net, xte, yte):
    net.eval()
    pred = net(xte.to(DEV)).reshape(len(xte), -1).cpu()
    return _rel_l2(pred, yte).item()


def train_eval(config, cfg, res=None, n_train=None):
    torch.manual_seed(0)
    res = res or cfg.res
    codec = qc.QICodec(cfg.W, cfg.lam)
    xin, ytr, xte, yte, _ = _prep(config, cfg, codec, res=res, n_train=n_train)
    net = _build(config, cfg, codec, res)
    t0 = time.time()
    _train(net, xin, ytr, cfg)
    return dict(config=config, res=res, n_train=n_train or cfg.n_train,
                test_rel_l2=_test(net, xte, yte),
                n_params=sum(p.numel() for p in net.parameters()),
                t_train=time.time() - t0)


def eval_invariance(config, cfg, test_res=(16, 32, 64, 128, 256)):
    """Train once at cfg.res, evaluate zero-shot at each test_res."""
    torch.manual_seed(0)
    codec = qc.QICodec(cfg.W, cfg.lam)
    xin, ytr, _, _, _ = _prep(config, cfg, codec)
    net = _build(config, cfg, codec, cfg.res)
    _train(net, xin, ytr, cfg)
    out = []
    for r in test_res:
        _, _, xte, yte, _ = _prep(config, cfg, codec, res=r)
        if config == "A":                 # rebuild the decode basis at res r
            net.Phi_out = torch.tensor(codec.basis(codec.grid(r)),
                                       dtype=torch.float32).to(DEV)
        out.append(dict(config=config, train_res=cfg.res, test_res=r,
                        test_rel_l2=_test(net, xte, yte)))
        print(out[-1], flush=True)
    return out


def save(recs, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(recs, indent=1))


def _load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else []


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"A": "tab:red", "B": "tab:blue", "C": "0.4"}
    LAB = {"A": "A: QI-coeff MLP", "B": "B: QI->FNO", "C": "C: plain FNO"}

    acc = _load("data.json")
    if acc:
        fig, ax = plt.subplots(figsize=(5, 4))
        cs = [r["config"] for r in acc]
        ax.bar(cs, [r["test_rel_l2"] for r in acc],
               color=[COL[c] for c in cs])
        for r in acc:
            ax.text(r["config"], r["test_rel_l2"], f"{r['test_rel_l2']:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("test rel L2 @ 64^2")
        ax.set_title("expF10: Darcy accuracy (1000 train)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "accuracy_bar.png", dpi=150)
        plt.close(fig)

    inv = _load("invariance.json")
    if inv:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in ("A", "B", "C"):
            pts = sorted((r for r in inv if r["config"] == c),
                         key=lambda r: r["test_res"])
            if pts:
                ax.plot([r["test_res"] for r in pts],
                        [r["test_rel_l2"] for r in pts], "o-",
                        color=COL[c], label=LAB[c])
        ax.axvline(64, color="k", ls=":", alpha=0.5, label="train res")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("test resolution")
        ax.set_ylabel("test rel L2")
        ax.set_title("expF10: discretization invariance (train @ 64^2)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "invariance_vs_res.png", dpi=150)
        plt.close(fig)

    de = _load("data_eff.json")
    if de:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in ("A", "B", "C"):
            pts = sorted((r for r in de if r["config"] == c),
                         key=lambda r: r["n_train"])
            if pts:
                ax.plot([r["n_train"] for r in pts],
                        [r["test_rel_l2"] for r in pts], "o-",
                        color=COL[c], label=LAB[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_train")
        ax.set_ylabel("test rel L2 @ 64^2")
        ax.set_title("expF10: data efficiency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "data_eff.png", dpi=150)
        plt.close(fig)
    print("figures written to", RESULTS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-invariance", action="store_true")
    ap.add_argument("--eval-data-eff", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
        return
    cfg = SMOKE_CFG if args.smoke else Cfg()
    configs = ["A", "B", "C"] if args.config == "all" else [args.config]

    if args.eval_invariance:
        recs = [r for c in configs for r in eval_invariance(c, cfg)]
        save(recs, "invariance.json")
        return
    if args.eval_data_eff:
        recs = [train_eval(c, cfg, n_train=n)
                for c in configs for n in (100, 300, 1000)]
        for r in recs:
            print(r, flush=True)
        save(recs, "data_eff.json")
        return

    recs = [train_eval(c, cfg) for c in configs]
    for r in recs:
        print(r, flush=True)
    save(recs, "data.json")


if __name__ == "__main__":
    main()
