"""Render the parameter-movement gifs from the suite's snapshot runs.

One gif per (case, arm): left panel the model against the target (the fit
forming), right panel the neuron geometry -- center position c_k = -b_k/w_k
against bandwidth w_k, every neuron colored red-to-blue by its INITIAL
center position, so migration is visible as color separation.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("suite_common", HERE / "common.py")
sc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sc)

FIGS = sc.FIGS
FIGS.mkdir(parents=True, exist_ok=True)
recs = [r for r in sc.load(sc.RESULTS / "suite.jsonl") if r["kind"] == "gif"]

d08run = sc.core4.core1.d08run
x_plot = np.linspace(-1, 1, 800)


def render(rec):
    W = rec["W"]
    fn = d08run.make_fn(d08run.TARGETS[rec["fn"]])
    y_true = fn(x_plot)
    snaps = sorted(((int(k), np.asarray(v)) for k, v in rec["snaps"].items()),
                   key=lambda kv: kv[0])
    th0 = snaps[0][1]
    c0 = -th0[W:2 * W] / np.where(np.abs(th0[:W]) > 1e-12, th0[:W], 1e-12)
    colors = plt.cm.coolwarm((np.clip(c0, -1.3, 1.3) + 1.3) / 2.6)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(top=0.82)
    axL.plot(x_plot, y_true, "k-", lw=1.0)
    fit_line, = axL.plot([], [], color="tab:green", lw=1.2)
    axL.set_xlim(-1, 1)
    pad = 0.4 * (y_true.max() - y_true.min() + 1e-9)
    axL.set_ylim(y_true.min() - pad, y_true.max() + pad)
    axL.set_title("model (green) vs target (black)", fontsize=9)
    scat = axR.scatter(c0, np.abs(th0[:W]), c=colors, s=14)
    axR.set_xlim(-2.2, 2.2)
    axR.set_yscale("log")
    axR.set_ylim(1e-2, 1e4)
    axR.set_xlabel("center $c_k = -b_k/w_k$")
    axR.set_ylabel("bandwidth $|w_k|$")
    axR.set_title("neurons, colored by initial center", fontsize=9)
    ttl = fig.suptitle("")

    def frame(i):
        it, th = snaps[i]
        z = np.tanh(x_plot[:, None] * th[:W][None, :] + th[W:2 * W][None, :])
        pred = z @ th[2 * W:3 * W] + th[-1]
        fit_line.set_data(x_plot, pred)
        c = -th[W:2 * W] / np.where(np.abs(th[:W]) > 1e-12, th[:W], 1e-12)
        scat.set_offsets(np.c_[np.clip(c, -2.2, 2.2),
                               np.clip(np.abs(th[:W]), 1e-2, 1e4)])
        rel = np.linalg.norm(pred - fn(x_plot)) / np.linalg.norm(fn(x_plot))
        ttl.set_text(f"{rec['case']} / {rec['fn']} / {rec['arm']}   "
                     f"it {it}   eval rel $L_2$ = {rel:.1e}")
        return fit_line, scat, ttl

    anim = FuncAnimation(fig, frame, frames=len(snaps), blit=False)
    out = FIGS / f"gif_{rec['case']}_{rec['arm']}.gif"
    anim.save(out, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)
    print("wrote", out)


for rec in recs:
    render(rec)
print(f"{len(recs)} gifs rendered")
