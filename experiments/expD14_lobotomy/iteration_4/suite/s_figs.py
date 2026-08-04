"""iteration_4 suite figures.

  S1_scaling.png       2x2 (the four scenarios): error vs width N, fgen vs
                       adam, reached (open) and after-finisher floor (filled)
  S2_losscurves.png    2x3 (six targets): eval error vs iteration from
                       stock-random init, fgen vs adam, floors dashed
  S3_signals_1d.png    2x2 (four scenarios, sine_8pi): alpha, f_gen, shat,
                       cooling factor over the run (from t8)
  S4_signals_2d.png    same, the four 2-D ridge cases (from the suite), with
                       mu on a twin axis
  S5_dl.png            the 2-hidden-layer task and the three real-data tasks:
                       train/test curves, fgen vs adam
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("suite_common", HERE / "common.py")
sc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sc)

FIGS = sc.FIGS
FIGS.mkdir(parents=True, exist_ok=True)
suite = sc.load(sc.RESULTS / "suite.jsonl")
CF = {"fgen": "#2ca02c", "adam": "0.45"}


def cells(kind, **kw):
    out = [r for r in suite if r["kind"] == kind
           and all(r.get(k) == v for k, v in kw.items())]
    return out


# ---- S1: scaling ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
for ax, case in zip(axes.flat, ("qi", "clustered", "datagap", "rand")):
    for arm in ("adam", "fgen"):
        Ns, reach, floor = [], [], []
        for N in (32, 64, 128, 256):
            cc = cells("scale", case=case, N=N, arm=arm)
            if cc:
                Ns.append(N)
                reach.append(np.median([c["final_rel"] for c in cc]))
                floor.append(np.median([c["floor_final"] for c in cc]))
        ax.plot(Ns, reach, "o--", color=CF[arm], mfc="none", lw=1.0)
        ax.plot(Ns, floor, "s-", color=CF[arm], lw=1.6)
    cc0 = cells("scale", case=case, arm="adam")
    f0 = {N: np.median([c["floor0"] for c in cells("scale", case=case, N=N,
                                                   arm="adam")])
          for N in (32, 64, 128, 256)}
    ax.plot(list(f0), list(f0.values()), "k:", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3e0)
    ax.set_title(case)
    ax.grid(True, alpha=0.3)
axes[1, 0].set_xlabel("width $N$")
axes[1, 1].set_xlabel("width $N$")
axes[0, 0].set_ylabel("eval rel $L_2$")
axes[1, 0].set_ylabel("eval rel $L_2$")
fig.legend(handles=[
    plt.Line2D([], [], color=CF["fgen"], marker="s", lw=1.6,
               label="optimizer + finisher (floor)"),
    plt.Line2D([], [], color=CF["fgen"], marker="o", ls="--", mfc="none",
               lw=1.0, label="optimizer, reached in-run"),
    plt.Line2D([], [], color=CF["adam"], marker="s", lw=1.6,
               label="Adam + finisher"),
    plt.Line2D([], [], color=CF["adam"], marker="o", ls="--", mfc="none",
               lw=1.0, label="Adam, reached in-run"),
    plt.Line2D([], [], color="k", ls=":", lw=1.0,
               label="no training + finisher")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.suptitle("width scaling, sine_8pi", y=0.90, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIGS / "S1_scaling.png", dpi=150)
plt.close(fig)

# ---- S2: loss curves --------------------------------------------------------
TARGETS6 = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
for ax, fn in zip(axes.flat, TARGETS6):
    for arm in ("adam", "fgen"):
        cc = cells("loss", fn=fn, arm=arm)
        if not cc:
            continue
        h, fh = cc[0]["hist"], cc[0]["fhist"]
        ax.plot(h["it"], h["rel"], color=CF[arm], lw=1.2)
        ax.plot(fh["it"], fh["floor"], color=CF[arm], lw=1.0, ls="--",
                alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3e0)
    ax.set_title(fn)
    ax.grid(True, alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("iteration")
for ax in axes[:, 0]:
    ax.set_ylabel("eval rel $L_2$")
fig.legend(handles=[
    plt.Line2D([], [], color=CF["fgen"], lw=1.2, label="optimizer, reached"),
    plt.Line2D([], [], color=CF["fgen"], lw=1.0, ls="--",
               label="optimizer, floor (after finisher)"),
    plt.Line2D([], [], color=CF["adam"], lw=1.2, label="Adam, reached"),
    plt.Line2D([], [], color=CF["adam"], lw=1.0, ls="--",
               label="Adam, floor")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
fig.suptitle("six target families, stock-random init, N = 128", y=0.90,
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIGS / "S2_losscurves.png", dpi=150)
plt.close(fig)


# ---- S3 / S4: the signal plots ---------------------------------------------
def signal_grid(recs_by_case, title, fname, with_mu=False):
    cases = list(recs_by_case)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, case in zip(axes.flat, cases):
        rec = recs_by_case[case]
        h = rec["hist"]
        ax.plot(h["it"], h["alpha"], color="#1f77b4", lw=1.3)
        ax.plot(h["it"], h["fgen"], color="#2ca02c", lw=1.3)
        ax.plot(h["it"], h["shat"], color="#9467bd", lw=1.0, alpha=0.8)
        ax.plot(h["it"], h["rel"], color="k", lw=0.8, alpha=0.5)
        if with_mu and "mu" in h:
            ax.plot(h["it"], np.maximum(h["mu"], 1e-32), color="#d62728",
                    lw=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 3e1)
        ax.set_title(case)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    handles = [
        plt.Line2D([], [], color="#1f77b4", lw=1.3,
                   label=r"$\alpha$ (damping = $r_{entry}$)"),
        plt.Line2D([], [], color="#2ca02c", lw=1.3,
                   label=r"$f_{gen}$ (held-out inexpressible fraction)"),
        plt.Line2D([], [], color="#9467bd", lw=1.0,
                   label=r"$\hat s$ (geometry SNR)"),
        plt.Line2D([], [], color="k", lw=0.8, alpha=0.5,
                   label="eval rel $L_2$")]
    if with_mu:
        handles.append(plt.Line2D([], [], color="#d62728", lw=1.0,
                                  label=r"$\mu = (\alpha\sigma_1)^2$"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle(title, y=0.88, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)


t8 = sc.load(sc.core4.RESULTS / "t8_fgen.jsonl")
one_d = {}
for case in ("qi", "clustered", "datagap", "rand"):
    cc = [r for r in t8 if r["arm"] == "fgen" and r["fn"] == "sine_8pi"
          and r["case"] == case and r["seed"] == 0]
    if cc:
        one_d[case] = cc[0]
if len(one_d) == 4:
    signal_grid(one_d, "the four scenarios, 1-D (sine_8pi, N = 128)",
                "S3_signals_1d.png", with_mu=False)

two_d = {}
for case in ("qi", "clustered", "datagap", "random"):
    cc = cells("twod", case=case, arm="fgen")
    if cc:
        two_d[case] = cc[0]
if len(two_d) == 4:
    signal_grid(two_d, "the four scenarios, 2-D ridge geometry",
                "S4_signals_2d.png", with_mu=True)

# ---- S5: the DL tasks -------------------------------------------------------
DL = [("dl2", "sine_mixture", "2-layer MLP, sine_mixture"),
      ("dltest", "airfoil", "airfoil"),
      ("dltest", "parkinsons", "parkinsons"),
      ("dltest", "bike_sharing", "bike_sharing")]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (kind, fn, title) in zip(axes, DL):
    for arm in ("adam", "fgen"):
        cc = cells(kind, fn=fn, arm=arm)
        if not cc:
            continue
        h = cc[0]["hist"]
        ax.plot(h["it"], h["train_mse"], color=CF[arm], lw=1.2)
        ax.plot(h["it"], h["test_mse"], color=CF[arm], lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("MSE")
fig.legend(handles=[
    plt.Line2D([], [], color=CF["fgen"], lw=1.2, label="optimizer, train"),
    plt.Line2D([], [], color=CF["fgen"], lw=1.0, ls="--",
               label="optimizer, test"),
    plt.Line2D([], [], color=CF["adam"], lw=1.2, label="Adam, train"),
    plt.Line2D([], [], color=CF["adam"], lw=1.0, ls="--",
               label="Adam, test")],
    loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIGS / "S5_dl.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("suite figures written to", FIGS)
