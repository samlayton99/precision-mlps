"""Exact signed population-balance identities on the small saved capsule.

This is checkpoint post-processing, not a trajectory or a persistence certificate.
Run through population_target_gap_modal.py --study balance.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import zipfile

import numpy as np


def balance(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    m = len(x)
    u = x[:, None]*a+b
    phi = np.tanh(u)
    derivative = 1-phi*phi
    output = phi@c+p[-1]
    q, _ = np.linalg.qr(np.column_stack((np.ones_like(x), x)))
    q *= np.sqrt(m)
    coarse = lambda v: q.T@v/m
    fine = lambda v: v-q@coarse(v)
    fh, g = fine(output), fine(y)
    residual = output-y
    ec, eh = coarse(residual), fine(residual)
    J = np.column_stack((derivative*c*x[:, None], derivative*c,
                         phi, np.ones_like(x)))
    jc = coarse(J)
    gram = jc@jc.T
    raw = J.T@eh/m
    ell = np.linalg.solve(gram, jc@raw)
    F = raw-jc.T@ell
    R = jc.T@(ec+ell)
    full = J.T@residual/m
    v = np.r_[a, b, -c, 0.]
    K = (phi-u*derivative)@c
    remainder = (3*phi-u*derivative-2*u)@c
    generated = -2*np.mean(fh*fh)
    target = 2*np.mean(g*fh)
    higher = np.mean(eh*fine(remainder))
    compensation = -ell@coarse(K)
    tracking = (ec+ell)@coarse(K)
    gen_raw = J.T@fh/m
    gen_F = gen_raw-jc.T@np.linalg.solve(gram, jc@gen_raw)
    geometry2 = float(a@a+b@b)
    readout2 = float(c@c)
    slope2 = float(a@a)
    ac = float(a@c)
    return dict(width=len(a), imbalance=(geometry2-readout2)/2,
                geometry2=geometry2, slope2=slope2, readout2=readout2,
                ac=ac, alignment=ac/np.sqrt(slope2*readout2),
                output_fine_norm=float(np.sqrt(np.mean(fh*fh))),
                generated=float(generated), target_loading=float(target), higher=float(higher),
                compensation=float(compensation), tracking=float(tracking),
                effective_drift=float(-v@F), full_drift=float(-v@full),
                gd_quadratic_coefficient=float((np.sum(full[:2*len(a)]**2)
                                               -np.sum(full[2*len(a):-1]**2))/2),
                generated_effective_drift=float(-v@gen_F),
                target_effective_drift=float(-v@(F-gen_F)),
                F_norm=float(np.linalg.norm(F)), R_norm=float(np.linalg.norm(R)),
                direct_identity_error=float(abs(-v@full-np.mean(residual*K))),
                effective_identity_error=float(abs(-v@F-generated-target-higher-compensation)),
                full_identity_error=float(abs(-v@full-generated-target-higher-compensation-tracking)),
                force_split_error=float(np.linalg.norm(full-F-R)),
                coarse_rank_min=float(np.linalg.eigvalsh(gram)[0]))


def audit(capsule, endpoints, output):
    for path in (capsule, endpoints):
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 4*1024**2:
                raise ValueError("Only small checkpoint archives are authorized for this probe")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(capsule, allow_pickle=False) as pack:
        cases = json.loads(str(pack["cases"]))
        rows = [dict(target=case["target"], seed=case.get("seed"), start=case.get("start"),
                     phase="initial", kind="gd", dt=case["eta"], time=0.,
                     **balance(p, pack["x"], y))
                for case, p, y in zip(cases, pack["p"], pack["y"], strict=True)]
        initial = list(rows)
        with np.load(endpoints, allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["p0"], pack["p"], rtol=0, atol=0)
            for label, p in zip(saved["labels"], saved["endpoints"], strict=True):
                target, kind, dt = ast.literal_eval(str(label))
                index = next(i for i, case in enumerate(cases) if case["target"] == target)
                case = cases[index]
                rows.append(dict(target=target, seed=case["seed"], start=case["start"],
                                 phase="final", kind=kind, dt=dt, time=200.,
                                 **balance(p, pack["x"], pack["y"][index])))
    with (output/"balance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    positions = np.arange(len(initial))
    for label, marker, color, title in (
            ("generated_effective_drift", "o", "#1764ab", "Generated error"),
            ("target_effective_drift", "^", "#b45d00", "Target coupling"),
            ("effective_drift", "x", "black", "Net effective fine")):
        axes[0].scatter([r[label] for r in initial], positions, marker=marker, color=color,
                        label=title)
    axes[0].axvline(0, color=".7", lw=1)
    axes[0].set(xscale="symlog", yticks=positions, yticklabels=[r["target"] for r in initial],
                xlabel="Derivative of population imbalance (flow time)",
                title="Signed contributions at the 20k checkpoint")
    axes[0].set_xscale("symlog", linthresh=1e-8)
    axes[0].set_xlim(-3e-3, 3e-3)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)
    final = [next(r for r in rows if r["target"] == row["target"]
                  and r["phase"] == "final" and r["kind"] == "gd" and r["dt"] == .002)
             for row in initial]
    for position, first, last in zip(positions, initial, final, strict=True):
        axes[1].plot([first["alignment"], last["alignment"]], [position]*2, color=".6")
    axes[1].scatter([r["alignment"] for r in initial], positions, color="#1764ab", label="20k updates")
    axes[1].scatter([r["alignment"] for r in final], positions, facecolors="none",
                    edgecolors="#b45d00", label="120k updates")
    axes[1].set(xlim=(-1, 1), yticks=positions, yticklabels=[r["target"] for r in initial],
                xlabel="Slope–readout cosine alignment", title="Alignment at the two saved endpoints")
    axes[1].axvline(0, color=".7", lw=1)
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2)
    fig.savefig(output/"population_balance.png", dpi=180)
    plt.close(fig)
    pairs = [dict(target=first["target"],
                  initial_imbalance=first["imbalance"], final_imbalance=last["imbalance"],
                  initial_alignment=first["alignment"], final_alignment=last["alignment"],
                  initial_ac=first["ac"], final_ac=last["ac"],
                  initial_slope_rms=np.sqrt(first["slope2"]/first["width"]),
                  final_slope_rms=np.sqrt(last["slope2"]/last["width"]))
             for first, last in zip(initial, final, strict=True)]
    summary = dict(scope="Six targets, two endpoints; no interval persistence claim", pairs=pairs,
                   maximum_identity_error=max(r[k] for r in rows for k in
                       ("direct_identity_error", "effective_identity_error", "full_identity_error")),
                   rows=rows)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.capsule, args.endpoints, args.output)
