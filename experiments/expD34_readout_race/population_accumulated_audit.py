"""Retrospective accumulated-concentration envelopes from archived GD scalars.

Integrate sqrt(I_F) in physical time, piecewise linearly between saved states.
No interpolation envelope, effective-flow trajectory, or GD certificate is
inferred. This helper writes numerical evidence and figures, never report prose.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCOPE = "sampled-GD retrospective substitution into effective-flow Theorem 12; not a certificate"
CAPACITY_CONSTANT = 2 * math.sqrt(2) / 3
IDENTITY = ("dataset", "panel", "target", "seed", "index", "arm")


def number(row, key):
    return float(row[key])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cumulative(t, values):
    """Trapezoidal integral with the correct unequal physical-time intervals."""
    t, values = np.asarray(t, float), np.asarray(values, float)
    if len(t) != len(values) or not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("Require matching arrays with strictly increasing times.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite quadrature input.")
    return np.r_[0., np.cumsum(np.diff(t) * (values[:-1] + values[1:]) / 2)]


def crossing_time(t, speed, budget):
    """First budget crossing in the piecewise-linear speed interpolant.

    None means right-censored at the final saved time; never extrapolate.
    """
    if budget <= 0:
        return 0.
    t, speed = np.asarray(t, float), np.asarray(speed, float)
    if np.any(speed < 0):
        raise ValueError("Budget speed must be nonnegative.")
    accumulated = cumulative(t, speed)
    if budget > accumulated[-1]:
        return None
    index = int(np.searchsorted(accumulated, budget, side="left"))
    if index == 0:
        return float(t[0])
    dt = t[index] - t[index - 1]
    remaining = budget - accumulated[index - 1]
    left = speed[index - 1]
    slope = (speed[index] - left) / dt
    # Rationalized positive root of left*u+slope*u^2/2=remaining.
    denominator = left + math.sqrt(max(0., left * left + 2 * slope * remaining))
    offset = 2 * remaining / denominator if denominator > 0 else 0.
    return float(t[index - 1] + min(dt, max(0., offset)))


def envelope(width, m40, y0, target_norm, target_fine, clock, time, h):
    """Theorem 12 with supplied concentration integrals, without a rank guard."""
    clock, time = np.asarray(clock, float), np.asarray(time, float)
    if min(width, m40, y0, target_norm, h) <= 0 or target_fine < 0:
        raise ValueError("Invalid positive initial data.")
    if clock.shape != time.shape or np.any(clock < 0) or np.any(time < 0):
        raise ValueError("Invalid clock/time.")
    consumption = 6 * y0 * math.sqrt(m40) * clock / width
    valid = consumption < 1
    denominator = np.where(valid, 1 - consumption, np.nan)
    m4 = m40 / denominator**2
    increment = m40 * np.expm1(-2 * np.log(denominator))
    q_increment = m40**.25 * np.expm1(-.5 * np.log(denominator))
    dissipation = 3 * y0 * increment / (4 * width)
    capacity_floor = np.maximum(0., target_fine - CAPACITY_CONSTANT * m4 / width) / target_norm
    retention_floor = y0 * np.exp(-3 * increment / (4 * y0 * width)) / target_norm
    # P4b with lambda0=.125, lambda*=.25. Bound initial p0 by the
    # population fourth moment, rather than an individual-neuron assumption.
    p0_bound = min(1., h**4 * m40 / (width**2 * .125**4))
    fraction = np.minimum(1., p0_bound + h**4 * q_increment**4 / (width**2 * .125**4))
    return dict(clock=clock, clock_consumption=consumption, envelope_defined=valid,
                M4_envelope=m4, weighted_travel_envelope=q_increment,
                dissipation_envelope=dissipation,
                total_travel_envelope=np.sqrt(time * dissipation),
                capacity_relative_floor=capacity_floor,
                retention_relative_floor=retention_floor,
                combined_relative_floor=np.maximum(capacity_floor, retention_floor),
                ever_lambda025_fraction_envelope=fraction)


def budgets(width, m40, y0, target_norm, target_fine, tolerance=.01):
    """Clock budgets for moment doubling, loss floors, and denominator expiry."""
    critical = width / (6 * y0 * math.sqrt(m40))
    margin = target_fine - tolerance * target_norm
    capacity0 = CAPACITY_CONSTANT * m40 / width
    capacity = critical * (1 - math.sqrt(capacity0 / margin)) if margin > capacity0 else 0.
    if y0 > tolerance * target_norm:
        limit = m40 + 4 * y0 * width / 3 * math.log(y0 / (tolerance * target_norm))
        retention = critical * (1 - math.sqrt(m40 / limit))
    else:
        retention = 0.
    return dict(denominator=critical, moment_double=critical * (1 - 1 / math.sqrt(2)),
                capacity=capacity, retention=retention, combined=max(capacity, retention))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean(rows))


def stats(values):
    values = np.asarray([v for v in values if v is not None], float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), **dict(zip(
        ("min", "p10", "median", "p90", "max"), np.quantile(values, [0, .1, .5, .9, 1]))))


def normalize(row, dataset):
    result = {key: row.get(key, "") for key in IDENTITY}
    result["dataset"] = dataset
    for key in ("width", "start", "horizon", "eta", "h", "M4", "fine_norm",
                "target_norm", "target_fine_norm", "relative_l2", "relative_eval_l2",
                "M", "mean_gamma", "rms_gamma", "fine_loss_dot_tracking"):
        result[key] = number(row, key)
    result.update(scale=float(row.get("scale") or 1.), reference=row.get("reference", ""),
                  state_sha256=row["state_sha256"], source=row["source"])
    for key, source in (
        ("I_F", "hidden_force_energy_concentration"), ("F_norm", "F_norm"),
        ("sqrt_M4_dot_effective", "sqrt_M4_dot_effective"), ("sigma", "coarse_sigma")):
        result[key] = number(row, "reinforcement_" + source)
    result["flow_time"] = result["eta"] * result["horizon"]
    if not all(math.isfinite(result[key]) and result[key] > 0 for key in
               ("width", "eta", "h", "M4", "fine_norm", "target_norm", "I_F", "F_norm")):
        raise ValueError("Nonpositive or nonfinite force/moment data.")
    q = result["M4"]**.25
    force_bound = 3 * result["fine_norm"] * result["I_F"]**.25 * q**3 / result["width"]
    transport_bound = result["I_F"]**.25 * result["F_norm"]
    qdot = result["sqrt_M4_dot_effective"] / (2 * q)
    result.update(force_bound=force_bound, force_bound_over_actual=force_bound / result["F_norm"],
                  q_dot_effective=qdot, transport_speed_bound=transport_bound,
                  transport_bound_over_absolute_qdot=transport_bound / abs(qdot) if qdot != 0 else None,
                  signed_moment_rate_saturation=qdot / (result["I_F"]**.25 * force_bound))
    return result


def audit_path(path):
    path = sorted(path, key=lambda r: r["flow_time"])
    first, last = path[0], path[-1]
    t = np.array([row["flow_time"] for row in path])
    if t[0] != 0:
        raise ValueError("Trajectory must include its initial state.")
    for key in ("width", "eta", "h", "target_norm", "target_fine_norm"):
        if not np.allclose([row[key] for row in path], first[key], rtol=1e-12, atol=1e-14):
            raise ValueError("Inconsistent trajectory field: " + key)
    speed = np.sqrt([row["I_F"] for row in path])
    clock = cumulative(t, speed)
    fields = envelope(first["width"], first["M4"], first["fine_norm"],
                      first["target_norm"], first["target_fine_norm"], clock, t, first["h"])
    actual_f = np.array([row["F_norm"] for row in path])
    f_integral = cumulative(t, actual_f)
    f2_integral = cumulative(t, actual_f**2)
    tracking_energy = cumulative(t, np.abs([row["fine_loss_dot_tracking"] for row in path]))
    for i, row in enumerate(path):
        row.update({key: value[i] for key, value in fields.items()})
        row.update(M4_over_initial=row["M4"] / first["M4"],
                   estimated_F_integral=f_integral[i], estimated_F2_integral=f2_integral[i],
                   estimated_absolute_tracking_energy=tracking_energy[i],
                   fine_squared_error_reduction=first["fine_norm"]**2 - row["fine_norm"]**2)
    endpoint = {key: first[key] for key in (*IDENTITY, "width", "start", "eta", "h", "scale", "reference")}
    endpoint.update(snapshots=len(path), horizon=last["horizon"], flow_time=t[-1],
                    initial_M4=first["M4"], initial_fine_norm=first["fine_norm"],
                    initial_relative_error=first["relative_l2"],
                    final_relative_error=last["relative_l2"], final_M4_ratio=last["M4_over_initial"],
                    sampled_peak_M4_ratio=max(row["M4_over_initial"] for row in path),
                    final_mean_gamma_ratio=last["mean_gamma"] / first["mean_gamma"],
                    final_clock=clock[-1], final_clock_consumption=fields["clock_consumption"][-1],
                    average_sqrt_I=clock[-1] / t[-1], sampled_peak_I=max(speed)**2,
                    sampled_peak_to_average_sqrt_I=max(speed) / (clock[-1] / t[-1]),
                    constant32_clock_over_integrated=math.sqrt(32) * t[-1] / clock[-1],
                    final_envelope_defined=bool(fields["envelope_defined"][-1]),
                    last_sampled_combined_floor=fields["combined_relative_floor"][-1])
    dt = np.diff(t)
    endpoint["endpoint_min_clock_proxy"] = float(np.sum(dt * np.minimum(speed[:-1], speed[1:])))
    endpoint["endpoint_max_clock_proxy"] = float(np.sum(dt * np.maximum(speed[:-1], speed[1:])))
    # These are sensitivity proxies, not lower/upper bounds on an unknown path.
    endpoint["endpoint_proxy_relative_spread"] = (
        endpoint["endpoint_max_clock_proxy"] - endpoint["endpoint_min_clock_proxy"]) / clock[-1]
    coarse_indices = [i for i, row in enumerate(path) if row["horizon"] in
                      (0, 1000, 20000, 40000, 100000)]
    coarse_indices = sorted(set([0, len(path) - 1, *coarse_indices]))
    coarse_clock = cumulative(t[coarse_indices], speed[coarse_indices])[-1]
    endpoint.update(coarsened_snapshots=len(coarse_indices),
                    coarsened_clock_relative_difference=abs(coarse_clock - clock[-1]) / clock[-1])
    for tol in (.01, .001, .0001):
        limits = budgets(first["width"], first["M4"], first["fine_norm"],
                         first["target_norm"], first["target_fine_norm"], tol)
        for label, limit in limits.items():
            if tol != .01 and label != "combined":
                continue
            key = label if tol == .01 else label + "_" + str(tol)
            crossed = crossing_time(t, speed, limit)
            endpoint[key + "_crossing_updates"] = crossed / first["eta"] if crossed is not None else None
            endpoint[key + "_right_censored"] = crossed is None
            endpoint[key + "_constant32_updates"] = limit / (math.sqrt(32) * first["eta"])
    endpoint["combined_duration_over_constant32"] = (
        endpoint["combined_crossing_updates"] / endpoint["combined_constant32_updates"]
        if endpoint["combined_crossing_updates"] is not None and endpoint["combined_constant32_updates"] > 0
        else None)
    c_rank = math.sqrt(2) + 4 * math.sqrt(first["M"])
    rank_travel = first["sigma"] / (math.sqrt(c_rank**2 + 4 * first["sigma"]) + c_rank)
    q0 = first["M4"]**.25
    clock_limit = budgets(first["width"], first["M4"], first["fine_norm"],
                          first["target_norm"], first["target_fine_norm"])["denominator"]
    endpoint["former_rank_guard_explicit_updates"] = (
        clock_limit * (1 - (q0 / (q0 + 32**.25 * rank_travel))**2)
        / (math.sqrt(32) * first["eta"]) if first["I_F"] <= 32 else None)
    endpoint["sampled_tracking_energy_over_F2"] = tracking_energy[-1] / f2_integral[-1]
    return path, endpoint


def summarize(rows, endpoints):
    groups = defaultdict(list)
    for row in endpoints:
        kind = ("natural" if row["arm"] == "natural" else
                "original" if row["arm"] == "original" else
                "repaired" if row["arm"] == "repaired" else "injected")
        groups[(row["width"], row["start"], row["eta"], row["horizon"],
                kind, row["scale"], row["reference"])].append(row)
    numeric = ("final_clock_consumption", "average_sqrt_I", "sampled_peak_I",
               "sampled_peak_to_average_sqrt_I", "constant32_clock_over_integrated",
               "endpoint_proxy_relative_spread", "coarsened_clock_relative_difference",
               "initial_M4", "final_M4_ratio", "sampled_peak_M4_ratio", "final_relative_error",
               "combined_crossing_updates", "capacity_crossing_updates", "retention_crossing_updates",
               "denominator_crossing_updates", "moment_double_crossing_updates",
               "combined_constant32_updates", "former_rank_guard_explicit_updates",
               "combined_duration_over_constant32", "combined_0.001_crossing_updates",
               "combined_0.0001_crossing_updates", "sampled_tracking_energy_over_F2", "final_mean_gamma_ratio")
    result = []
    for key, group in sorted(groups.items()):
        item = dict(zip(("width", "start", "eta", "horizon", "kind", "scale", "reference"), key))
        item.update(trajectories=len(group), targets=sorted({r["target"] for r in group}),
                    seeds=sorted({r["seed"] for r in group}),
                    endpoints_with_defined_envelope=sum(r["final_envelope_defined"] for r in group),
                    error_above_one_percent=sum(r["final_relative_error"] > .01 for r in group),
                    combined_crossing_right_censored=sum(r["combined_right_censored"] for r in group))
        item.update({metric: stats([r[metric] for r in group]) for metric in numeric})
        group_keys = {tuple(e[k] for k in IDENTITY) for e in group}
        selected = [r for r in rows if tuple(r[k] for k in IDENTITY) in group_keys]
        for metric in ("force_bound_over_actual", "transport_bound_over_absolute_qdot",
                       "signed_moment_rate_saturation"):
            item[metric] = stats([r[metric] for r in selected])
        item["states_with_positive_qdot"] = sum(r["q_dot_effective"] > 0 for r in selected)
        item["states"] = len(selected)
        matched = [r for r in group if r.get("first_hit_relative_l2_0.01") is not None]
        item["trajectories_with_every_update_counters"] = len(matched)
        item["trajectories_ever_reaching_one_percent"] = sum(r["first_hit_relative_l2_0.01"] >= 0 for r in matched)
        item["observed_ever_lambda025_fraction"] = stats([r["actual_ever_lambda025_fraction"] for r in matched])
        item["sampled_envelope_comparison"] = {
            "defined_states": sum(r["envelope_defined"] for r in selected),
            "moment_exceedances": sum(r["M4"] > r["M4_envelope"] * (1 + 1e-10) for r in selected),
            "output_floor_exceedances": sum(r["combined_relative_floor"] > r["relative_l2"] + 1e-10
                                            for r in selected),
            "scope": "sampled GD diagnostic, not continuous effective-flow validation",
        }
        result.append(item)
    return result


def plot_results(rows, endpoints, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {177: "#bc6c25", 705: "#007f86", 1409: "#7251a0"}
    baseline = [r for r in endpoints if r["arm"] == "original" and r["eta"] == .002
                and r["horizon"] == 20000]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for width in (177, 705, 1409):
        group = [r for r in baseline if r["width"] == width]
        positions = np.linspace(-.13, .13, len(group)) + (177, 705, 1409).index(width)
        for ax, metric in zip(axes, ("combined_crossing_updates", "final_M4_ratio", "final_relative_error")):
            ax.scatter(positions, [r[metric] for r in group], s=14, alpha=.7, color=colors[width])
            ax.set_xticks(range(3), ["177", "705", "1409"])
            ax.set_xlabel("Neuron count W")
    axes[0].axhline(20000, color="black", ls="--", lw=1, label="Observed continuation")
    axes[0].set(yscale="log", ylabel="Nominal updates: t / eta",
                title="When the estimated 1% floor ends")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color="black", ls=":", lw=1)
    axes[1].set(yscale="log", ylabel="M4(end) / M4(start)", title="Actual population-moment change")
    axes[2].axhline(.01, color="black", ls="--", lw=1)
    axes[2].set(yscale="log", ylabel="Raw relative training L2 error", title="Actual output error at 20k")
    fig.suptitle("Original GD: 23 targets at W=177,705; six at W=1409; two seeds\n"
                 "Restart age: 600k at W=177, 20k at wider sizes. Left is retrospective, not a GD certificate.", fontsize=11)
    fig.savefig(output / "bound_and_observation.png", dpi=180)
    plt.close(fig)

    wide = [r for r in baseline if r["width"] == 705]
    targets = sorted({r["target"] for r in wide})
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, constrained_layout=True)
    for i, target in enumerate(targets):
        for j, row in enumerate(sorted([r for r in wide if r["target"] == target], key=lambda r: r["seed"])):
            y = i + (j - .5) * .18
            axes[0].scatter(row["combined_crossing_updates"], y, s=24, color=colors[705])
            axes[0].scatter(row["combined_constant32_updates"], y, s=20,
                            facecolors="none", edgecolors="0.55")
            axes[1].scatter(row["final_relative_error"], y, s=24, color=colors[705])
    axes[0].axvline(20000, color="black", ls="--", lw=1)
    axes[0].set(xscale="log", xlabel="Nominal updates until 1% floor ends",
                title="Integrated estimate (filled); fixed I=32 (open)")
    axes[1].axvline(.01, color="black", ls="--", lw=1)
    axes[1].set(xscale="log", xlabel="Actual relative training error at 20k", title="Both archived seeds")
    axes[0].set_yticks(range(len(targets)), targets)
    axes[0].invert_yaxis()
    fig.suptitle("All 23 targets at width 705, restarting at 20k; retrospective effective-flow envelopes")
    fig.savefig(output / "all_targets.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for count, endpoint in enumerate(wide):
        group = sorted([r for r in rows if all(r[k] == endpoint[k] for k in IDENTITY)],
                       key=lambda r: r["flow_time"])
        t = np.array([r["flow_time"] for r in group])
        speed = np.sqrt([r["I_F"] for r in group])
        clock = cumulative(t, speed)
        grid = np.unique(np.r_[0., np.geomspace(.001, t[-1], 300), t])
        segments = np.minimum(np.searchsorted(t, grid, side="right") - 1, len(t) - 2)
        offset = grid - t[segments]
        partial = (clock[segments] + speed[segments] * offset
                   + .5 * (speed[segments + 1] - speed[segments])
                   / (t[segments + 1] - t[segments]) * offset**2)
        first = group[0]
        bound = envelope(first["width"], first["M4"], first["fine_norm"],
                         first["target_norm"], first["target_fine_norm"], partial, grid, first["h"])
        for ax, measured, predicted in zip(
            axes, ([r["M4_over_initial"] for r in group], [r["relative_l2"] for r in group]),
            (bound["M4_envelope"] / first["M4"], bound["combined_relative_floor"])):
            ax.plot(t / first["eta"], measured, ".-", color="#a35b32", alpha=.35,
                    lw=.8, markersize=2, label="Observed GD" if count == 0 else None)
            ax.plot(grid / first["eta"], predicted, color=colors[705], alpha=.25, lw=.8,
                    label="Effective-flow envelope estimate" if count == 0 else None)
            ax.set(xscale="symlog", yscale="log", xlabel="Additional GD updates")
            ax.set_xlim(0, 20000)
    axes[0].set(ylabel="Fourth moment / initial fourth moment", ylim=(.8, 100),
                title="Population growth: measured and bounded")
    axes[1].set(ylabel="Raw relative training L2 error", ylim=(.001, 1.2),
                title="Observed error and estimated lower floor")
    axes[1].axhline(.01, color="black", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Width 705: all 23 targets and both seeds; each curve is one branch\n"
                 "Smooth envelopes use interpolated concentration; no resetting or extrapolation beyond expiry", fontsize=11)
    fig.savefig(output / "wide_population_envelopes.png", dpi=180)
    plt.close(fig)

    selected = [r for r in rows if r["arm"] == "original" and r["eta"] == .002]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for width in (177, 705, 1409):
        group = [r for r in selected if r["width"] == width]
        for ax, metric in zip(axes, ("force_bound_over_actual", "transport_bound_over_absolute_qdot")):
            values = [r[metric] for r in group if r[metric] is not None]
            positions = np.linspace(-.15, .15, len(values)) + (177, 705, 1409).index(width)
            ax.scatter(positions, values, color=colors[width], s=9, alpha=.4)
            ax.set_xticks(range(3), ["177", "705", "1409"])
            ax.set(xlabel="Neuron count W", yscale="log", ylabel="Bound / measured quantity")
            ax.axhline(1, color="black", ls=":", lw=1)
    axes[0].set_title("Cubic population bound / effective force")
    axes[1].set_title("Population speed bound / absolute moment rate")
    fig.suptitle("Where the moment closure loses information: effective-flow probes at GD states\n"
                 "Restart age: 600k at W=177, 20k at wider sizes; repeated states are not independent.", fontsize=11)
    fig.savefig(output / "inequality_slack.png", dpi=180)
    plt.close(fig)

    long_rows = [r for r in rows if r["panel"] == "wide_N512_long" and r["arm"] == "repaired"]
    if long_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for target in sorted({r["target"] for r in long_rows}):
            group = sorted([r for r in long_rows if r["target"] == target], key=lambda r: r["horizon"])
            for ax, metric in zip(axes, ("I_F", "clock_consumption", "relative_l2")):
                ax.plot([r["horizon"] for r in group], [r[metric] for r in group], ".-", label=target)
                ax.set(xlabel="Additional GD updates", yscale="log")
        for ax, line, title in zip(axes, (32, 1, .01),
                                  ("Force concentration; old ceiling 32",
                                   "Consumed concentration budget; limit 1", "Actual output error; criterion 1%")):
            ax.axhline(line, color="black", ls="--", lw=1)
            ax.set_title(title, fontsize=10)
        axes[0].legend(fontsize=8)
        fig.suptitle("Six-target width-705 repaired continuations, restarting at 20k: 100k further updates\n"
                     "Budget consumption uses the original initial state throughout", fontsize=11)
        fig.savefig(output / "long_concentration_budget.png", dpi=180)
        plt.close(fig)


def run(args):
    paths, excluded, sources = defaultdict(list), Counter(), {}
    for source in args.source:
        sources[str(source)] = digest(source)
        with source.open(newline="") as stream:
            for original in csv.DictReader(stream):
                if original.get("role") != "trajectory":
                    continue
                if original.get("reinforcement_status") != "finite":
                    excluded[original.get("reinforcement_status", "missing_status")] += 1
                    continue
                row = normalize(original, source.parent.name)
                paths[tuple(row[key] for key in IDENTITY)].append(row)
    rows, endpoints = [], []
    for path in paths.values():
        states, endpoint = audit_path(path)
        rows.extend(states)
        endpoints.append(endpoint)
    if not rows:
        raise ValueError("No valid trajectories.")
    if args.motion_source:
        sources[str(args.motion_source)] = digest(args.motion_source)
        with args.motion_source.open(newline="") as stream:
            motion = list(csv.DictReader(stream))
        lookup = {(Path(r["run"]).name, r["target"], r["seed"], r["arm"],
                   float(r["start"]), float(r["eta"]), float(r["step"])): r for r in motion}
        for row in rows:
            match = lookup.get((row["panel"], row["target"], row["seed"], row["arm"],
                                row["start"], row["eta"], row["horizon"]))
            if match:
                row["actual_ever_lambda025_fraction"] = (
                    float(match["initially_above_lambda025"]) + float(match["learned_new_hits_lambda025"])) / row["width"]
                row["first_hit_relative_l2_0.01"] = int(match["first_hit_relative_l2_0.01"])
                row["motion_counter_scope"] = "every-update normalized-slope crossing and raw-training-error counters"
        endpoint_lookup = {tuple(r[k] for k in (*IDENTITY, "horizon")): r for r in rows}
        for endpoint in endpoints:
            row = endpoint_lookup[tuple(endpoint[k] for k in (*IDENTITY, "horizon"))]
            for key in ("actual_ever_lambda025_fraction", "first_hit_relative_l2_0.01"):
                endpoint[key] = row.get(key)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "states.csv", rows)
    write_csv(args.output / "trajectories.csv", endpoints)
    facts = dict(scope=SCOPE, sources=sources, helper_sha256=digest(__file__),
                 trajectory_count=len(endpoints), state_count=len(rows), excluded=dict(excluded),
                 integration="trapezoidal in t=eta*horizon; sqrt(I_F) interpolated linearly",
                 sensitivity="endpoint-range proxies and nested coarsening; neither certifies unsampled excursions",
                 initial_state="one fixed start per branch; no resetting the envelope at later checkpoints",
                 groups=summarize(rows, endpoints))
    (args.output / "facts.json").write_text(json.dumps(clean(facts), indent=2, allow_nan=False) + "\n")
    plot_results(rows, endpoints, args.output)
    print(json.dumps(dict(trajectories=len(endpoints), states=len(rows), excluded=dict(excluded))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--motion-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
