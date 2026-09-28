"""Audit the target-gap theorem from cached scalars and one small capsule.

Execute numerical work on Modal using population_target_gap_modal.py. No new
training, large checkpoint loading, interval certification, or report writing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

import numpy as np

from . import targets, effective_feedback_holdout as holdout

A3 = math.sqrt(6) / 8
A5 = (2 / 15) * 2**2.5 * 5**2.5 / 6**3
A7 = (17 / 315) * 2**3.5 * 7**3.5 / 8**4
SCOPE = "FP64 evaluation of proved formulas; not an outward-rounded certificate"


def target_data(name):
    """Original 2048-point training measure and original target normalization."""
    if name in holdout.TARGETS:
        return holdout.data(name)[:2]
    x = targets.grid(2048)
    mapping = targets.polynomial_map(x)
    if name in targets.TARGETS:
        return x, targets.values(name, x, mapping)
    q = np.polynomial.legendre.legvander(x, 9) @ mapping
    blends = dict(blend_m010=-.1, blend_p001=.01, blend_p010=.1, blend_p030=.3)
    if name == "moment4":
        return x, .3*q[:, 0] + .4*q[:, 1] + math.sqrt(.75)*q[:, 4]
    if name in blends:
        s = blends[name]
        return x, .3*q[:, 0]+.4*q[:, 1]+math.sqrt(.75)*(s*q[:, 3]+math.sqrt(1-s*s)*q[:, 9])
    if name in ("mixed_sine", "localized_sine"):
        y = np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(14*np.pi*x)
        if name == "localized_sine":
            y *= np.exp(-.5*(x/.4)**2)
    elif name == "chirp":
        u = (x+1)/2
        y = np.sin(2*np.pi*(u+4*u*u))
    else:
        raise ValueError("Unknown archived target: " + name)
    return x, y/np.sqrt(np.mean(y*y))


def target_split(x, y, degree):
    basis, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, degree))
    coarse = basis[:, :2] @ (basis[:, :2].T @ y)
    fine = y-coarse
    low = basis[:, 2:] @ (basis[:, 2:].T @ y)
    norm = lambda v: float(np.linalg.norm(v)/math.sqrt(len(x)))
    return dict(target_norm=norm(y), target_fine_norm=norm(fine),
                leakage=norm(low), high_norm=norm(fine-low))


def coupling(q, width, degree, leakage, high_norm):
    if degree not in (3, 5):
        raise ValueError("Only the proved degree-three and degree-five gaps")
    constant = A5 if degree == 3 else A7
    power = degree+3
    low = leakage*A3*q**4/width
    high = constant*high_norm*q**power/width**(power/4)
    return low+high, low, high


def candidate(state, split, degree, factor, mode, method="fourth_only"):
    """One chosen radius; all inputs are from the restart checkpoint only."""
    width, eta = state["width"], state["eta"]
    q0 = state["M4"]**.25
    if method == "fourth_only":
        qstar = q0*factor
        radius = (qstar-q0)/width**.25
        qbound = 2*qstar-q0 if mode == "gd" else qstar
        B, low, high = coupling(qbound, width, degree, split["leakage"], split["high_norm"])
        capacity = A3*qstar**4/width
        parameter_bound = qbound
        inner_parameter_bound = qstar
    elif method == "separate_moments":
        power = degree+3
        s4 = (state["M4"]/width)**.25
        sk = (state[f"M{power}"]/width**(power/2-1))**(1/power)
        radius = factor*sk
        outer = (2 if mode == "gd" else 1)*radius
        low = split["leakage"]*A3*(s4+outer)**4
        high = (A5 if degree == 3 else A7)*split["high_norm"]*(sk+outer)**power
        B = low+high
        capacity = A3*(s4+radius)**4
        qstar = width**.25*(s4+radius)
        parameter_bound = math.sqrt(state["M"])+outer
        inner_parameter_bound = math.sqrt(state["M"])+radius
    else:
        raise ValueError("Unknown population bound")
    L0 = state["loss"] if mode != "effective_flow" else .5*state["fine_norm"]**2
    excess = L0-.5*split["target_fine_norm"]**2
    available = excess+B
    floor = max(0., split["target_fine_norm"]-capacity)/split["target_norm"]
    jo = math.sqrt(1+2*parameter_bound**2)
    H = jo**2+(math.sqrt(2*L0)+2*radius*jo)*(math.sqrt(2)+4*parameter_bound)
    G = math.sqrt(1+2*inner_parameter_bound**2)*math.sqrt(2*L0)
    guarded = mode != "gd" or (eta*H <= 1 and eta*G <= radius)
    if available < -1e-12*max(1., L0):
        raise ValueError("Energy lower bound inconsistent with checkpoint")
    valid = available > 0 and guarded and floor > .01
    time = radius**2 / ((2 if mode == "gd" else 1)*available) if valid else 0.
    return dict(degree=degree, mode=mode, method=method, radius_parameter=factor,
                factor=qstar/q0, qstar=qstar, radius=radius,
                flow_time=time, eta_time_units=time/eta,
                # Strict inequality n*eta<T; these are actual updates only for GD.
                guaranteed_updates=max(0, math.ceil(time/eta)-1) if mode == "gd" else None,
                relative_floor=floor, available_loss=available, initial_excess=excess,
                leakage_budget=low, high_budget=high,
                hessian_step=eta*H, guard_step_ratio=eta*G/radius,
                guard_pass=guarded, valid=valid,
                rms_gamma_bound=inner_parameter_bound/math.sqrt(width))


def select(state, split, degree, mode, method="fourth_only"):
    # Finite search over theorem radii, never over trajectories or checkpoint outcomes.
    lower = 1.002 if method == "fourth_only" else .002
    options = [candidate(state, split, degree, float(f), mode, method)
               for f in np.geomspace(lower, 4., 801)]
    return max(options, key=lambda r: r["flow_time"])


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def capsule_audit(path):
    # Bound decompressed size before numpy sees the archive.
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 4*1024**2:
            raise ValueError("This audit accepts only the named small capsule")
    rows = []
    with np.load(path, allow_pickle=False) as pack:
        x, yy, pp = pack["x"], pack["y"], pack["p"]
        cases = json.loads(str(pack["cases"]))
        for case, p, y in zip(cases, pp, yy, strict=True):
            xr, yr = target_data(case["target"])
            np.testing.assert_allclose(x, xr, atol=0, rtol=0)
            np.testing.assert_allclose(y, yr, atol=3e-14, rtol=3e-14)
            a, b, c = p[:-1].reshape(3, -1)
            output = np.tanh(x[:, None]*a+b)@c+p[-1]
            basis, _ = np.linalg.qr(np.column_stack((np.ones_like(x), x)))
            fine = lambda v: v-basis@(basis.T@v)
            z, g, residual = fine(output), fine(y), output-y
            ec = basis.T@residual/math.sqrt(len(x))
            energy = .5*float(ec@ec)
            generated = .5*float(np.mean(z*z))
            correlation = float(np.mean(z*g))
            squares = a*a+b*b+c*c
            state = dict(width=len(a), eta=.002, M4=len(a)*np.sum(squares**2),
                         M=float(np.sum(squares)), M6=float(len(a)**2*np.sum(squares**3)),
                         M8=float(len(a)**3*np.sum(squares**4)),
                         loss=float(np.mean(residual*residual)/2),
                         fine_norm=float(np.sqrt(np.mean(fine(residual)**2))))
            identity_error = abs(state["loss"]-np.mean(g*g)/2-energy-generated+correlation)
            for degree in (3, 5):
                split = target_split(x, y, degree)
                for mode in ("full_flow", "effective_flow", "gd"):
                    for method in ("fourth_only", "separate_moments"):
                        result = select(state, split, degree, mode, method)
                        rows.append(dict(target=case["target"], seed=case.get("seed"),
                                         **state, **split, **result, coarse_energy=energy,
                                         generated_energy=generated, target_correlation=correlation,
                                         energy_identity_error=identity_error))
    return rows


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with args.source.open() as stream:
        archive = list(csv.DictReader(stream))
    targets_seen = sorted({r["target"] for r in archive if r.get("status") == "finite"})
    splits = {(name, degree): target_split(*target_data(name), degree)
              for name in targets_seen for degree in (3, 5)}
    results, unique, duplicate = [], set(), 0
    for row in archive:
        if row.get("status") != "finite":
            continue
        key = (row["state_sha256"], row["target"])
        if key in unique:
            duplicate += 1
            continue
        unique.add(key)
        state = {k: float(row[k]) for k in ("width", "eta", "M", "M4", "M6", "M8", "loss", "fine_norm")}
        for degree in (3, 5):
            split = splits[row["target"], degree]
            for field in ("target_norm", "target_fine_norm"):
                np.testing.assert_allclose(float(row[field]), split[field], atol=2e-12, rtol=2e-12)
            for mode in ("full_flow", "effective_flow", "gd"):
                for method in ("fourth_only", "separate_moments"):
                    result = select(state, split, degree, mode, method)
                    meta = {k: row.get(k, "") for k in
                            ("target", "seed", "start", "horizon", "role", "panel", "arm", "state_sha256")}
                    results.append(dict(**meta, **state, **split, **result,
                                        relative_l2=float(row["relative_l2"]),
                                        R_over_F=float(row["R_norm"])/max(float(row["F_norm"]), 1e-300),
                                        coarse_energy=max(0., state["loss"]-.5*state["fine_norm"]**2)))
    write_csv(args.output/"checkpoint_bounds.csv", results)
    capsules = capsule_audit(args.capsule)
    write_csv(args.output/"capsule_bounds.csv", capsules)
    dense = list(csv.DictReader(args.dense.open()))
    comparisons = []
    for target in sorted({r["target"] for r in dense}):
        available = [r for r in capsules if r["target"] == target and r["mode"] == "gd"]
        selected = max(available, key=lambda r: r["flow_time"])
        path = [r for r in dense if r["target"] == target and r["kind"] == "gd" and float(r["dt"]) == .002]
        path.sort(key=lambda r: float(r["time"]))
        if not path:
            raise ValueError("No matching dense GD path")
        np.testing.assert_allclose(float(path[0]["M4"]), selected["M4"], rtol=1e-12)
        first = max((r for r in available if r["method"] == "fourth_only"), key=lambda r: r["flow_time"])
        violations = [float(r["time"])/.002 for r in path if float(r["M4"]) >= selected["qstar"]**4]
        comparisons.append(dict(target=target, degree=selected["degree"], method=selected["method"],
            guaranteed_updates=selected["guaranteed_updates"],
            fourth_only_updates=first["guaranteed_updates"],
            relative_floor=selected["relative_floor"],
            saved_updates=float(path[-1]["time"])/.002,
            saved_peak_M4_ratio=max(float(r["M4"]) for r in path)/selected["M4"],
            allowed_M4_ratio=selected["factor"]**4,
            saved_endpoint_relative_error=float(path[-1]["relative_error"]),
            first_saved_moment_violation_updates=min(violations) if violations else None,
            sampled_region_holds=all(float(r["M4"]) < selected["qstar"]**4 for r in path)))
    write_csv(args.output/"dense_comparison.csv", comparisons)
    plot(results, capsules, comparisons, args.output)
    grouped = defaultdict(list)
    for r in results:
        if r["mode"] == "gd":
            grouped[r["target"], r["degree"], r["method"], r["width"]].append(r)
    groups = []
    for (name, degree, method, width), items in sorted(grouped.items()):
        valid = [r for r in items if r["valid"]]
        groups.append(dict(target=name, degree=degree, method=method, width=width, count=len(items),
            valid=len(valid), min_updates=min((r["guaranteed_updates"] for r in valid), default=0),
            median_updates=float(np.median([r["guaranteed_updates"] for r in valid])) if valid else 0,
            max_updates=max((r["guaranteed_updates"] for r in valid), default=0),
            leakage=items[0]["leakage"],
            min_R_over_F=min(r["R_over_F"] for r in items), max_R_over_F=max(r["R_over_F"] for r in items)))
    write_csv(args.output/"target_summary.csv", groups)
    summary = dict(scope=SCOPE, rows_read=len(archive), distinct_states=len(unique),
                   duplicate_states=duplicate, targets=targets_seen,
                   widths=sorted({r["width"] for r in results}),
                   counts=dict(Counter(r["target"] for r in results if r["mode"] == "gd" and r["degree"] == 5 and r["method"] == "fourth_only")),
                   invalid_gd=sum(not r["valid"] for r in results if r["mode"] == "gd"),
                   capsule_identity_error=max(r["energy_identity_error"] for r in capsules),
                   dense=comparisons,
                   gap_target_summaries=[r for r in groups if (r["target"], r["degree"]) in (("moment5", 3), ("moment9", 5))],
                   constants=dict(A3=A3, A5=A5, A7=A7))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def plot(rows, capsules, comparisons, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    labels = [r["target"] for r in comparisons]
    for i, row in enumerate(comparisons):
        axes[0].plot([max(.1, row["guaranteed_updates"]), row["saved_updates"]], [i, i], color=".8")
    axes[0].scatter([max(.1, r["fourth_only_updates"]) for r in comparisons], range(len(labels)), label="Fourth moment only", marker="x", color="#888888")
    axes[0].scatter([max(.1, r["guaranteed_updates"]) for r in comparisons], range(len(labels)), label="Best GD bound", color="#1764ab")
    axes[0].scatter([r["saved_updates"] for r in comparisons], range(len(labels)), label="Saved continuation", marker="|", s=100, color="#b45d00")
    axes[0].set(xscale="log", yticks=range(len(labels)), yticklabels=labels,
                xlabel="Additional updates (η = 0.002)", title="A proved duration can still be too short")
    axes[0].legend(fontsize=8, loc="upper center")
    for i, row in enumerate(comparisons):
        r = next(r for r in capsules if r["target"] == row["target"] and r["mode"] == "gd" and r["degree"] == row["degree"] and r["method"] == row["method"])
        # Positive reservoirs and allowances are shown separately; signed correlation is in CSV.
        axes[1].scatter(max(r["coarse_energy"], 1e-20), i, marker="s", color="#888888", label="Initial coarse error" if i == 0 else None)
        axes[1].scatter(max(r["generated_energy"], 1e-20), i, marker="o", color="#1764ab", label="Initial generated error" if i == 0 else None)
        axes[1].scatter(r["leakage_budget"]+r["high_budget"], i, marker="^", color="#b45d00", label="Allowed target coupling" if i == 0 else None)
    axes[1].set(xscale="log", yticks=range(len(labels)), yticklabels=labels,
                xlabel="Squared-error loss units", title="Which term makes the bound short?")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(output/"target_gap_audit.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, color in (("moment5", "#b45d00"), ("moment9", "#1764ab")):
        degree = 3 if name == "moment5" else 5
        for method, marker in (("fourth_only", "x"), ("separate_moments", "o")):
            group = [r for r in rows if r["target"] == name and r["degree"] == degree and r["mode"] == "gd" and r["valid"] and r["method"] == method]
            label = "fourth only" if method == "fourth_only" else "separate moments"
            ax.scatter([r["width"] for r in group], [r["eta_time_units"] for r in group],
                       color=color, marker=marker, alpha=.55, label=f"{name}: {label}")
    ax.set(yscale="log", xscale="log", xlabel="Actual neuron count W",
           ylabel="Bound in update units at each archived η", title="The finite-width constants matter")
    ax.legend(fontsize=8)
    fig.savefig(output/"target_gap_widths.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "capsule", "dense", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    audit(parser.parse_args())
