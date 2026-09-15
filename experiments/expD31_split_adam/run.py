"""Apply mu outside independent Adam streams for DF_tau and DG_tau."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD28_loss_gradient_decomposition import run as profile
from experiments.expD29_weighted_profile import run as gd
from experiments.expD29_weighted_profile import mu_sweep as gd_sweep

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD31_split_adam"


def config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def learning_rate_at(cfg, step):
    """Optional common cosine schedule; multiplier and Adam histories stay unchanged."""
    schedule = cfg.get("lr_schedule")
    if not schedule or step <= schedule["start_step"]:
        return cfg["learning_rate"]
    if schedule["kind"] != "cosine":
        raise ValueError(f"Unknown schedule: {schedule['kind']}")
    progress = np.clip((step-schedule["start_step"])/(cfg["steps"]-schedule["start_step"]), 0., 1.)
    factor = schedule["min_factor"]+(1-schedule["min_factor"])*.5*(1+np.cos(np.pi*progress))
    return cfg["learning_rate"]*factor


class AdamStream:
    """Bias-corrected, unit-learning-rate Adam; preview does not change moments."""

    def __init__(self, size, betas=(0.9, 0.999), epsilon=1e-8):
        self.m = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.float64)
        self.t = 0
        self.betas = betas
        self.epsilon = epsilon

    def direction(self, gradient, *, advance=False):
        b1, b2 = self.betas
        t = self.t + 1
        m = b1 * self.m + (1 - b1) * gradient
        s = b2 * self.s + (1 - b2) * gradient**2
        direction = (m / (1 - b1**t)) / (np.sqrt(s / (1 - b2**t)) + self.epsilon)
        if advance:
            self.m, self.s, self.t = m, s, t
        return direction


def balanced_geometry_direction(uF, uG, ratio):
    """Return the requested post-Adam norm balance, without a denominator floor."""
    def norm(vector):
        scale = float(np.max(np.abs(vector)))
        if not np.isfinite(scale) or scale == 0:
            raise FloatingPointError("post-Adam ratio undefined: zero or nonfinite direction")
        return scale * float(np.linalg.norm(vector / scale))

    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("target update ratio must be finite and positive")
    norm_F, norm_G = norm(uF), norm(uG)
    effective_mu = ratio * (norm_G / norm_F)
    if not np.isfinite(effective_mu) or effective_mu <= 0:
        raise FloatingPointError("effective multiplier is not finite and positive")
    return effective_mu * uF + uG, effective_mu


def train(target, arm, mu, cfg):
    """mu=0 denotes ordinary Adam, not the sum of two normalized streams."""
    x = profile.previous.midpoint_grid(cfg["n_train"])
    y = profile.previous.matched.target_values(target, x, cfg)
    initial = profile.initial_state(arm, cfg)
    p = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for k, v in initial.items()}
    tx, ty = torch.tensor(x), torch.tensor(y)
    eta = cfg["learning_rate"]
    optimizer = torch.optim.Adam([p["v"]] if mu else p.values(), lr=eta,
                                 betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
    width = len(initial["a"])
    streams = [AdamStream(2 * width, cfg["adam_betas"], cfg["adam_epsilon"]) for _ in range(2)]
    audit_steps = {0, 1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 7500, cfg["steps"]}
    chosen = set(profile.snapshots(cfg["steps"], cfg["diagnostic_snapshots"])) | audit_steps
    history, saved_steps, audits = [], [], []
    states = {k: [] for k in p}
    status = "complete"
    last_state = None
    best_F, best_step, best_state = np.inf, None, None
    for step in range(cfg["steps"] + 1):
        eta = learning_rate_at(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = eta
        for parameter in p.values():
            parameter.grad = None
        prediction = torch.tanh(tx[:, None] * p["a"] + p["b"]) @ p["v"][:-1] + p["v"][-1]
        loss = 0.5 * torch.mean((prediction - ty).square())
        if not torch.isfinite(loss):
            status = f"nonfinite loss at step {step}"
            break
        loss.backward()
        state = {k: value.detach().numpy() for k, value in p.items()}
        gL = np.r_[p["a"].grad.numpy(), p["b"].grad.numpy()]
        try:
            diag = profile.one_diagnostic(state, x, y, cfg["readout_rcond"])
        except np.linalg.LinAlgError:
            status = f"SVD failed at step {step}"
            break
        np.testing.assert_allclose(gL, diag["gL"], rtol=1e-8, atol=2e-13)
        gF = diag["gF"]
        gG = gL - gF
        uF, uG = [s.direction(g) for s, g in zip(streams, (gF, gG))]
        effective_mu = mu
        if mu and cfg.get("target_update_ratio") is not None:
            try:
                direction, effective_mu = balanced_geometry_direction(uF, uG, cfg["target_update_ratio"])
            except FloatingPointError as exc:
                status = f"{exc} at step {step}"
                break
            delta = eta * direction
        else:
            delta = eta * (mu * uF + uG) if mu else np.zeros_like(gL)
        if not (np.isfinite(delta).all() and np.isfinite(gF).all()
                and all(torch.isfinite(v.grad).all() for v in p.values())):
            status = f"nonfinite gradient or direction at step {step}"
            break
        row = dict(step=step, learning_rate=eta, effective_mu=float(effective_mu),
                   L=float(loss.detach()), F=float(diag["F"]), G=float(loss.detach())-diag["F"],
                   refit_train_relative_l2=diag["refit_train_relative_l2"],
                   mean_gamma=float(np.mean(abs(state["a"]))),
                   gamma_displacement=float(np.mean(abs(abs(state["a"])-abs(initial["a"])))),
                   geometry_displacement=float(np.linalg.norm(np.r_[state["a"]-initial["a"], state["b"]-initial["b"]])),
                   readout_norm=float(np.linalg.norm(state["v"])), rank=diag["rank"],
                   profile_coefficient_norm=float(diag["vstar_norm"]),
                   gL_norm=float(np.linalg.norm(gL)), gF_norm=float(np.linalg.norm(gF)), gG_norm=float(np.linalg.norm(gG)),
                   proposed_step_norm=float(np.linalg.norm(delta)) if mu else np.nan,
                   F_step_norm=float(eta*effective_mu*np.linalg.norm(uF)) if mu else np.nan,
                   G_step_norm=float(eta*np.linalg.norm(uG)) if mu else np.nan,
                   proposed_max_coordinate_step=float(np.max(abs(delta))) if mu else np.nan,
                   F_fraction_above_epsilon=float(np.mean(abs(gF)>cfg["adam_epsilon"])))
        history.append(row)
        last_state = {k: state[k].copy() for k in p}
        if row["F"] < best_F:
            best_F, best_step = row["F"], step
            best_state = {k: state[k].copy() for k in p}
        if step in chosen:
            saved_steps.append(step)
            for k in p:
                states[k].append(state[k].copy())
        if mu and step in audit_steps:
            alternatives = [profile.one_diagnostic(state, x, y, cfg["readout_rcond"], driver="gesdd"),
                            profile.one_diagnostic(state, x, y, cfg["readout_rcond"], backend="torch")]
            variation = max(np.linalg.norm(gF-d["gF"]) for d in alternatives)
            if cfg.get("target_update_ratio") is not None:
                alternative_deltas = []
                for d in alternatives:
                    try:
                        alternate, _ = balanced_geometry_direction(
                            streams[0].direction(d["gF"]), streams[1].direction(gL-d["gF"]),
                            cfg["target_update_ratio"])
                        alternative_deltas.append(eta * alternate)
                    except FloatingPointError:
                        alternative_deltas.append(np.full_like(delta, np.inf))
            else:
                alternative_deltas = [eta * (mu*streams[0].direction(d["gF"])
                                             + streams[1].direction(gL-d["gF"])) for d in alternatives]
            step_variation = max(np.linalg.norm(delta-d) for d in alternative_deltas)
            agree = all(d["rank"] == diag["rank"] for d in alternatives)
            audits.append(dict(step=step, gF_variation=float(variation), ranks_agree=bool(agree),
                               gF_resolved=bool(agree and row["gF_norm"] > cfg["resolution_factor"]*variation),
                               proposed_step_variation=float(step_variation), proposed_step_norm=row["proposed_step_norm"],
                               relative_step_variation=float(step_variation/max(row["proposed_step_norm"], 1e-300))))
        if step in (0, 100, 250, cfg["steps"]) or (step and step % 1000 == 0):
            setting = f"r={cfg['target_update_ratio']}" if cfg.get("target_update_ratio") is not None else f"mu={mu or 'ordinary Adam'}"
            print(f"{arm}/{target}/{setting}, step {step}: "
                  f"L={row['L']:.5g}, F={row['F']:.5g}, mean gamma={row['mean_gamma']:.6g}", flush=True)
        if step < cfg["steps"]:
            if mu:
                for s, g in zip(streams, (gF, gG)):
                    s.direction(g, advance=True)
                with torch.no_grad():
                    p["a"].sub_(torch.as_tensor(delta[:width]))
                    p["b"].sub_(torch.as_tensor(delta[width:]))
            # Gradients all came from the same pre-step state. This optimizer
            # touches only v in split runs, and all parameters in the control.
            optimizer.step()
    if not history:
        raise FloatingPointError(status)
    if saved_steps[-1] != history[-1]["step"]:
        saved_steps.append(history[-1]["step"])
        for k in p:
            states[k].append(last_state[k])
    if cfg.get("save_best_state") and best_step not in saved_steps:
        position = int(np.searchsorted(saved_steps, best_step))
        saved_steps.insert(position, best_step)
        for k in p:
            states[k].insert(position, best_state[k])
    case = {k: np.asarray([row[k] for row in history]) for k in history[0]}
    case.update({k: np.asarray(value) for k, value in states.items()})
    case.update(saved_steps=np.asarray(saved_steps), audits_json=np.array(json.dumps(audits)),
                target=np.array(target), arm=np.array(arm), mu=np.array(mu), status=np.array(status),
                best_step=np.array(best_step))
    return case


def audit(cases, cfg):
    x = profile.previous.midpoint_grid(cfg["n_train"])
    xe = profile.previous.midpoint_grid(cfg["n_eval"])
    summary, cutoffs = [], []
    for (arm, target, mu), c in cases.items():
        for k in ("a", "b", "v"):
            np.testing.assert_array_equal(c[k][0], cases[arm, target, 0][k][0])
        assert c["saved_steps"][-1] == c["step"][-1]
        state = {k: c[k][-1] for k in ("a", "b", "v")}
        y = profile.previous.matched.target_values(target, x, cfg)
        ye = profile.previous.matched.target_values(target, xe, cfg)
        h = profile.hidden(state["a"], state["b"], x)
        he = profile.hidden(state["a"], state["b"], xe)
        A = np.c_[h, np.ones(len(x))]/np.sqrt(len(x))
        primary = None
        for cutoff in (1e-12, 1e-13, 1e-14):
            d = profile.profiled_matrix(A, y/np.sqrt(len(x)), cutoff)
            fitted = he@d["vstar"][:-1]+d["vstar"][-1]
            rec = dict(arm=arm, target=target, mu=mu, step=int(c["step"][-1]), rcond=cutoff,
                       F=d["F"], rank=d["rank"], coefficient_norm=float(np.linalg.norm(d["vstar"])),
                       refit_eval_relative_l2=float(np.linalg.norm(fitted-ye)/np.linalg.norm(ye)))
            cutoffs.append(rec)
            if cutoff == cfg["readout_rcond"]:
                primary = rec
        actual = he@state["v"][:-1]+state["v"][-1]
        numerical = json.loads(str(c["audits_json"]))
        summary.append(dict(arm=arm, target=target, mu=mu, status=str(c["status"]), step=int(c["step"][-1]),
                            L=float(c["L"][-1]), F=float(c["F"][-1]), G=float(c["G"][-1]),
                            actual_eval_relative_l2=float(np.linalg.norm(actual-ye)/np.linalg.norm(ye)),
                            refit_eval_relative_l2=primary["refit_eval_relative_l2"],
                            coefficient_norm=primary["coefficient_norm"], rank=primary["rank"],
                            gamma_initial=float(c["mean_gamma"][0]),
                            gamma_after_one_step=float(c["mean_gamma"][1]) if len(c["step"])>1 else None,
                            gamma_final=float(c["mean_gamma"][-1]), gamma_max=float(np.max(abs(state["a"]))),
                            first_F_step_norm=float(c["F_step_norm"][0]) if mu else None,
                            first_max_coordinate_step=float(c["proposed_max_coordinate_step"][0]) if mu else None,
                            unresolved_audits=sum(not d["gF_resolved"] for d in numerical), audits=len(numerical),
                            maximum_relative_step_variation=max((d["relative_step_variation"] for d in numerical), default=0.),
                            seconds=float(c["seconds"])))
    (RESULTS/"data"/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (RESULTS/"data"/"terminal_cutoff_audit.json").write_text(json.dumps(cutoffs, indent=2)+"\n")
    # The first figure exposes transient minima; check their predictions on
    # an independent grid instead of trusting only the projected norm.
    early = []
    for target in cfg["targets"]:
        c = cases["xavier", target, 1000]
        i = int(np.argmin(c["F"][c["saved_steps"]]))
        step = int(c["saved_steps"][i])
        state = {k: c[k][i] for k in ("a", "b", "v")}
        y = profile.previous.matched.target_values(target, x, cfg)
        ye = profile.previous.matched.target_values(target, xe, cfg)
        A = np.c_[profile.hidden(state["a"], state["b"], x), np.ones(len(x))]/np.sqrt(len(x))
        d = profile.profiled_matrix(A, y/np.sqrt(len(x)), cfg["readout_rcond"])
        fit = profile.hidden(state["a"], state["b"], xe)@d["vstar"][:-1]+d["vstar"][-1]
        early.append(dict(arm="xavier", target=target, mu=1000, best_saved_step=step, F=float(c["F"][step]),
                          independent_refit_relative_l2=float(np.linalg.norm(fit-ye)/np.linalg.norm(ye)),
                          coefficient_norm=float(np.linalg.norm(d["vstar"])), gradient_F_norm=float(c["gF_norm"][step]),
                          proposed_F_step_norm=float(c["F_step_norm"][step])))
    (RESULTS/"data"/"early_refit_check.json").write_text(json.dumps(early, indent=2)+"\n")

    # Very large slopes can undersample transitions. Check terminal reported
    # errors at eight times the evaluation density, with the same fitted v.
    dense_x = profile.previous.midpoint_grid(8*cfg["n_eval"])
    grids = []
    for (arm, target, mu), c in cases.items():
        if mu != max(cfg["mu_values"]):
            continue
        state = {k: c[k][-1] for k in ("a", "b", "v")}
        y = profile.previous.matched.target_values(target, x, cfg)
        yd = profile.previous.matched.target_values(target, dense_x, cfg)
        A = np.c_[profile.hidden(state["a"], state["b"], x), np.ones(len(x))]/np.sqrt(len(x))
        d = profile.profiled_matrix(A, y/np.sqrt(len(x)), cfg["readout_rcond"])
        actual_error = refit_error = 0.
        for start in range(0, len(dense_x), 4096):
            xx = dense_x[start:start+4096]
            yy = yd[start:start+4096]
            h = profile.hidden(state["a"], state["b"], xx)
            actual_error += np.sum((h@state["v"][:-1]+state["v"][-1]-yy)**2)
            refit_error += np.sum((h@d["vstar"][:-1]+d["vstar"][-1]-yy)**2)
        grids.append(dict(arm=arm, target=target, mu=mu, n_eval=len(dense_x),
                          actual_eval_relative_l2=float(np.sqrt(actual_error)/np.linalg.norm(yd)),
                          refit_eval_relative_l2=float(np.sqrt(refit_error)/np.linalg.norm(yd))))
    (RESULTS/"data"/"dense_grid_check.json").write_text(json.dumps(grids, indent=2)+"\n")


def matched_gd(arm, target, mu, cfg):
    old_cfg = gd_sweep.config()
    for k, value in cfg.items():
        if k in old_cfg:
            assert old_cfg[k] == value, f"GD configuration mismatch: {k}"
    return gd_sweep.read_case(gd_sweep.case_path(arm, target, mu), old_cfg)


def plot(cases, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter, NullFormatter

    figdir = RESULTS/"figures"
    figdir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(cfg["mu_values"], plt.cm.viridis([.05, .48, .87])))
    for arm in cfg["arms"]:
        if not all((arm, target, mu) in cases for target in cfg["targets"] for mu in [0]+cfg["mu_values"]):
            continue
        fig, axes = plt.subplots(3, 4, figsize=(19, 11.4), dpi=160, sharex=True)
        for col, target in enumerate(cfg["targets"]):
            entries = [(cases[arm, target, 0], "#111111", ":", 2.5)]
            for mu in cfg["mu_values"]:
                entries.append((matched_gd(arm, target, mu, cfg), colors[mu], "--", 1.3))
                entries.append((cases[arm, target, mu], colors[mu], "-", 1.8))
            for c, color, style, width in entries:
                for row, key in enumerate(("L", "F", "mean_gamma")):
                    valid = np.isfinite(c[key])
                    values = np.maximum(c[key], 1e-32)
                    axes[row, col].plot(c["step"][valid], values[valid], color=color, ls=style, lw=width)
                    if str(c["status"]) != "complete":
                        last = np.flatnonzero(valid)[-1]
                        axes[row, col].plot(c["step"][last], values[last], "x", color=color, ms=7)
            axes[0, col].set_title(profile.LABELS[target], fontsize=14, pad=13)
            for row in range(3):
                ax = axes[row, col]
                ax.set_yscale("log")
                ax.set_xlim(0, cfg["steps"])
                ax.set_xticks([0, 100, 200, 300, 400, 500])
                lo, hi = ax.get_ylim()
                if hi/lo < 10:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                ax.grid(alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=9)
                ax.margins(y=.10)
            axes[2, col].set_xlabel("Training updates", fontsize=11)
        axes[0, 0].set_ylabel(r"Actual loss $L=\frac{1}{2}\,\mathrm{mean}(e^2)$"+"\n(log scale)", fontsize=12)
        axes[1, 0].set_ylabel("Refitted loss $F_\\tau$\n(log scale)", fontsize=12)
        axes[2, 0].set_ylabel(r"Mean scale $\overline{\gamma}=\mathrm{mean}|a_k|$"+"\n(log scale)", fontsize=12)
        fig.suptitle(profile.ARMS[arm]+"\nSeparate Adam streams; μ multiplies DF after normalization", fontsize=18, y=.985)
        handles = [Line2D([], [], color=colors[mu], lw=2, label=rf"$\mu=10^{{{int(np.log10(mu))}}}$") for mu in cfg["mu_values"]]
        handles += [Line2D([], [], color=".3", lw=2, label="Split Adam", ls="-"),
                    Line2D([], [], color=".3", lw=1.6, label="Previous weighted GD", ls="--"),
                    Line2D([], [], color=".1", lw=2.5, label="Ordinary Adam", ls=":")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .913), ncol=6, frameon=False, fontsize=11.5)
        fig.subplots_adjust(left=.085, right=.98, top=.845, bottom=.14, hspace=.26, wspace=.30)
        fig.text(.5, .058,
                 "Split geometry update: −η[μ uF + uG], where uF and uG have independent Adam moments. Readout uses ordinary Adam.\n"
                 "η=0.002, β=(0.9,0.999), εAdam=10⁻⁸; 500 updates; same 1,024 samples in [−1,1], N=128, 177 neurons, seed 0.\n"
                 "Each panel fits its own vertical range. Fτ uses the existing SVD cutoff 10⁻¹³σ₁; values near 10⁻³⁰ are a numerical floor.\n"
                 "No clipping, rate retuning, or coefficient replacement. Dashed curves reuse matched GD data; × would mark a stopped run.",
                 ha="center", va="center", fontsize=9.5)
        fig.savefig(figdir/f"{arm}.png")
        plt.close(fig)
    if (RESULTS/"data"/"summary.json").exists():
        plot_evaluation(cfg)


def plot_evaluation(cfg):
    """Expose generalization failures that sampled profile losses can hide."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    summary = json.loads((RESULTS/"data"/"summary.json").read_text())
    records = {(r["arm"], r["target"], r["mu"]): r for r in summary}
    fig, axes = plt.subplots(3, 4, figsize=(18, 11.2), dpi=160)
    x = profile.previous.midpoint_grid(cfg["n_train"])
    for row, arm in enumerate(cfg["arms"]):
        for col, target in enumerate(cfg["targets"]):
            ax = axes[row, col]
            y = profile.previous.matched.target_values(target, x, cfg)
            data = [records[arm, target, mu] for mu in [0]+cfg["mu_values"]]
            actual = [d["actual_eval_relative_l2"] for d in data]
            sampled = [np.sqrt(2*d["F"]/np.mean(y*y)) for d in data]
            independent = [d["refit_eval_relative_l2"] for d in data]
            ax.plot(range(4), actual, color=".15", ls=":", marker=".", ms=6, lw=1.7)
            ax.plot(range(4), sampled, color="#2878b5", ls="--", marker="o", ms=6, lw=1.7)
            ax.plot(range(4), independent, color="#d66b22", marker="s", mfc="none", ms=9, lw=1.6)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.set_xticks(range(4), ["Ordinary\nAdam", r"$10^3$", r"$10^4$", r"$10^5$"])
            ax.set_xlim(-.2, 3.2)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            ax.margins(y=.13)
            if row == 0:
                ax.set_title(profile.LABELS[target], fontsize=14, pad=13)
            if col == 0:
                arm_label = {"xavier": "Xavier", "scaled_xavier": "Scaled Xavier", "qi_zero": "QI"}[arm]
                ax.set_ylabel(arm_label+"\nRelative $L_2$ error (log)", fontsize=12)
            if row == 2:
                ax.set_xlabel("Outside multiplier μ", fontsize=11)
    fig.suptitle("Does the fitted geometry work between training samples?\nFinal state after 500 updates", fontsize=18, y=.985)
    handles = [Line2D([], [], color=".15", ls=":", marker=".", label="Actual model: independent grid"),
               Line2D([], [], color="#2878b5", ls="--", marker="o", label="Refitted: training projection reference"),
               Line2D([], [], color="#d66b22", marker="s", mfc="none", label="Refitted: independent grid")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.085, right=.98, top=.84, bottom=.17, hspace=.30, wspace=.32)
    fig.text(.5, .05,
             "Refits use 1,024 training points and the same 10⁻¹³ SVD cutoff. Both evaluated models use 8,192 separate midpoint samples.\n"
             "Blue = √[2Fτ / mean(y²)]. Orange recomputes predictions using the solved coefficients on the independent grid.\n"
             "Blue/orange overlap means agreement; separation exposes failure between samples or coefficient-reconstruction error. Each panel fits its own vertical range.\n"
             "All μ=10⁵ endpoints were also checked on 65,536 points; the large Xavier/Runge refit error persists (about 413).",
             ha="center", va="center", fontsize=9.5)
    fig.savefig(RESULTS/"figures"/"independent_evaluation.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--mu", nargs="+", type=int)
    args = parser.parse_args()
    cfg = config()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    cases = {}
    with threadpool_limits(limits=cfg["threads"]):
        for arm in args.arms or cfg["arms"]:
            for target in args.targets or cfg["targets"]:
                for mu in args.mu if args.mu is not None else [0]+cfg["mu_values"]:
                    path = RESULTS/"data"/f"{arm}__{target}__{mu}.npz"
                    if path.exists():
                        c = gd.load(path, cfg)
                    elif args.plot_only:
                        continue
                    elif args.analyze_only:
                        raise FileNotFoundError(path)
                    else:
                        start = time.perf_counter()
                        c = train(target, arm, mu, cfg)
                        c["seconds"] = np.array(time.perf_counter()-start)
                        gd.save(c, cfg, path)
                    cases[arm, target, mu] = c
        if len(cases) == len(cfg["arms"])*len(cfg["targets"])*(1+len(cfg["mu_values"])):
            if not args.plot_only:
                audit(cases, cfg)
            plot(cases, cfg)
        else:
            print(f"Subset available: {len(cases)} cases.", flush=True)


if __name__ == "__main__":
    main()
