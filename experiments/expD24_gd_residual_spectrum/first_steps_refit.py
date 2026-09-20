"""Steps 0--5 of ordinary GD, with evaluation-only readout solves.

Default: all targets use the same finite-interval samples, standard tanh model
and half-mean-square objective. --legacy-whole-line reproduces the earlier
comparison with different Gaussian objectives. Solves never enter training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad_vec
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import readout_comparison as rc
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum.signed_log import signed_log, set_signed_log_axis
from experiments.expD24_gd_residual_spectrum.spectrum import snapshot_legend
from src.data.targets import get_target

TARGETS = ["sine", "sine_mixture", "gaussian_envelope"]
STEPS = list(range(6))
OUTPUT = rc.RESULTS / "first_steps.png"
DATA = rc.RESULTS / "data/first_steps.npz"
MATCHED_OUTPUT = rc.RESULTS / "first_steps_matched.png"
MATCHED_DATA = rc.RESULTS / "data/first_steps_matched.npz"


def target_values(target, x, config):
    return whole.target(x, config) if target == "gaussian_envelope" else get_target(target).fn_numpy(x)


class MatchedProblem(rc.Problem):
    """Same design, sampling, normalization and initialization for all targets."""

    def __init__(self, target, config):
        self.target, self.config, self.whole = target, config, False
        n = config["n_train"]
        self.x = -1+(np.arange(n)+.5)*2/n
        self.weights = np.full(n, 1/n)
        self.y = target_values(target, self.x, config)
        self.tx, self.tw, self.ty = map(torch.tensor, (self.x, self.weights, self.y))
        self.target_energy = float(np.sum(self.weights*self.y**2))


def make_problem(target, arm, config, *, training=False):
    if config.get("matched_training", False):
        return MatchedProblem(target, config)
    if training and target == "gaussian_envelope" and arm == "qi_zero":
        return SpatialGaussianProblem(config)
    return rc.Problem(target, config)


class SpatialGaussianProblem(rc.Problem):
    """The original Gaussian QI GD objective, with the same raw a,b,v metric."""

    def __init__(self, config):
        self.target, self.config, self.whole = "gaussian_envelope", config, True
        spatial_config = yaml.safe_load((rc.HERE / "whole_line.yaml").read_text())
        self.x, self.weights = whole.quadrature(spatial_config)
        self.y = whole.target(self.x, config)
        self.tx, self.tw, self.ty = map(torch.tensor, (self.x, self.weights, self.y))
        self.target_energy = whole.target_energy(config)

    def features(self, a, b):
        return torch.tanh(self.tx[:, None]*a+b)-torch.tanh(self.tx)[:, None]

    def objective(self, features, v, optimal_residual=None):
        assert optimal_residual is None
        residual = features @ (v-v.mean())-self.ty
        return .5*torch.sum(self.tw*residual.square())


def evaluate(case, config):
    """Only consume saved arrays; fit every frame, including initialization."""
    problem = make_problem(case["target"], case["arm"], config)
    x = np.linspace(-2 if problem.whole else -1, 2 if problem.whole else 1, 2049)
    fine = rc.Problem(case["target"], config, refinement=2) if problem.whole else None
    if not problem.whole:
        x_eval = -1+(np.arange(config["n_eval"])+.5)*2/config["n_eval"]
        y_eval = target_values(case["target"], x_eval, config)
    fields = ("residual", "refit_v", "relative_l2", "rank", "stationarity",
              "readout_norm", "quadrature_difference", "geometry_step_max",
              "mean_abs_gamma_change", "sign_changes")
    result = {key: [] for key in fields}
    initial = {key: values[0] for key, values in case["parameters"].items()}
    for index, step in enumerate(STEPS):
        state = {key: values[index].copy() for key, values in case["parameters"].items()}
        with torch.no_grad():
            features = problem.features(torch.tensor(state["a"]), torch.tensor(state["b"])).numpy()
        refit, info = rc.solve_readout(features, problem.y, problem.weights)
        if problem.whole:
            plotted = rc.whole_spatial_features(state["a"], state["b"], x)
            residual = plotted @ refit-whole.target(x, config)
            with torch.no_grad():
                fine_features = fine.features(torch.tensor(state["a"]), torch.tensor(state["b"])).numpy()
            eval_r = fine_features @ refit-fine.y
            energy = float(np.sum(fine.weights*abs(eval_r)**2))
            target_energy = whole.target_energy(config)
            relative_l2 = np.sqrt(energy/target_energy)
            coarse = float(np.sum(problem.weights*abs(features @ refit-problem.y)**2))
            difference = abs(energy-coarse)/target_energy
            assert difference < 1e-5, difference
        else:
            def design(nodes):
                return np.column_stack((np.tanh(nodes[:, None]*state["a"]+state["b"]), np.ones(len(nodes))))
            residual = design(x) @ refit-target_values(case["target"], x, config)
            eval_r = design(x_eval) @ refit-y_eval
            relative_l2 = np.linalg.norm(eval_r)/np.linalg.norm(y_eval)
            difference = np.nan
        previous = case["parameters"]["a"][max(index-1, 0)]
        values = dict(residual=residual, refit_v=refit, relative_l2=relative_l2,
                      rank=info["rank"], stationarity=info["stationarity"],
                      readout_norm=info["readout_norm"], quadrature_difference=difference,
                      geometry_step_max=float(abs(state["a"]-previous).max()),
                      mean_abs_gamma_change=float(abs(abs(state["a"])-abs(initial["a"])).mean()),
                      sign_changes=int(np.count_nonzero(np.sign(state["a"]) != np.sign(initial["a"]))))
        for key in fields:
            result[key].append(values[key])
        for key in state:
            np.testing.assert_array_equal(state[key], case["parameters"][key][index])
    if case["arm"] == "qi_zero":
        # With zero readout, the first joint-GD update cannot move geometry.
        for key in ("a", "b"):
            np.testing.assert_array_equal(case["parameters"][key][0], case["parameters"][key][1])
        np.testing.assert_array_equal(result["refit_v"][0], result["refit_v"][1])
        np.testing.assert_array_equal(result["residual"][0], result["residual"][1])
    return {key: np.asarray(values) for key, values in result.items()}


def validate_gaussian_initial_gradient(config):
    """Check the large Xavier update using spatial integration on all of R."""
    problem = rc.Problem("gaussian_envelope", config)
    initial = problem.initial("xavier")
    params = {key: torch.tensor(value, requires_grad=True) for key, value in initial.items()}
    loss = problem.objective(problem.features(params["a"], params["b"]), params["v"])
    loss.backward()
    ga = params["a"].grad.numpy()
    j = int(np.argmax(abs(ga)))
    a, b, v = (initial[key] for key in ("a", "b", "v"))
    signs = np.sign(a)
    c = v-signs*np.mean(signs*v)

    def integrand(x):
        residual = float((rc.whole_spatial_features(a, b, np.array([x])) @ v)[0])-float(whole.target(x, config))
        t = a[j]*x+b[j]
        sech2 = 4*np.exp(-2*abs(t))/(1+np.exp(-2*abs(t)))**2
        return np.array([.5*residual**2, residual*c[j]*x*sech2])

    core, _ = quad_vec(integrand, -2, 2, epsabs=1e-10, epsrel=1e-11)
    left, _ = quad_vec(integrand, -np.inf, -2, epsabs=1e-9, epsrel=1e-11)
    right, _ = quad_vec(integrand, 2, np.inf, epsabs=1e-9, epsrel=1e-11)
    measured = core+left+right
    expected = np.array([loss.item(), ga[j]])
    np.testing.assert_allclose(measured, expected, rtol=1e-9, atol=1e-9)
    return {"neuron_zero_based": j, "initial_a": float(a[j]),
            "initial_slope_gradient": float(ga[j]), "initial_loss": loss.item(),
            "spatial_relative_errors": (abs(measured-expected)/abs(expected)).tolist(),
            "loss_and_gradient_fraction_outside_minus2_plus2": ((left+right)/measured).tolist()}


def validate_matched_training(case, problem):
    """Independently check each plain-GD step in NumPy on the actual grid."""
    x, w, y = problem.x, problem.weights, problem.y
    np.testing.assert_array_equal(x, -1+(np.arange(1024)+.5)*2/1024)
    np.testing.assert_array_equal(w, np.full(1024, 1/1024))
    assert not problem.whole
    errors = []
    for step in STEPS:
        a, b, v = (case["parameters"][key][step] for key in ("a", "b", "v"))
        assert v.size == a.size+1  # All cases have the same free output bias.
        hidden = np.tanh(x[:, None]*a+b)
        residual = hidden @ v[:-1]+v[-1]-y
        loss = .5*np.sum(w*residual**2)
        np.testing.assert_allclose(loss, case["loss"][step], rtol=2e-13, atol=2e-15)
        if step == STEPS[-1]:
            continue
        backprop = (w*residual)[:, None]*(1-hidden**2)*v[:-1]
        gradients = {"a": np.sum(x[:, None]*backprop, axis=0),
                     "b": np.sum(backprop, axis=0),
                     "v": np.r_[hidden.T @ (w*residual), np.sum(w*residual)]}
        for key, gradient in gradients.items():
            expected = case["parameters"][key][step]-problem.config["learning_rate"]*gradient
            observed = case["parameters"][key][step+1]
            np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-15)
            errors.append(float(abs(observed-expected).max()))
    return max(errors)


def run(config):
    previous, _ = rc.load_cases(rc.RESULTS / "data.npz")
    lookup = {(case["target"], case["arm"]): case for case in previous if case["method"] == "gd"}
    cases = []
    validation = {}
    for arm in config["arms"]:
        initial_reference = None
        for target in TARGETS:
            problem = make_problem(target, arm, config, training=True)
            case = rc.train(problem, arm, "gd", steps=5, displayed_steps=STEPS)
            old = lookup[target, arm]
            if not config.get("matched_training", False) or target != "gaussian_envelope":
                for step in (0, 2, 4):
                    old_index = old["snapshot_steps"].index(step)
                    for key, values in case["parameters"].items():
                        np.testing.assert_allclose(values[step], old["parameters"][key][old_index], rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(case["loss"], old["loss"][:6], rtol=2e-10, atol=2e-12)
            if config.get("matched_training", False):
                validation[f"{target}/{arm}"] = validate_matched_training(case, problem)
                if initial_reference is None:
                    initial_reference = {key: values[0] for key, values in case["parameters"].items()}
                else:
                    for key, values in case["parameters"].items():
                        np.testing.assert_array_equal(values[0], initial_reference[key])
            # Training has finished before any diagnostic refit is performed.
            case["views"] = {"gd_refit": evaluate(case, config)}
            cases.append(case)
    if config.get("matched_training", False):
        config["independent_gd_step_max_absolute_difference"] = validation
    return cases


def plot(cases, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = config.get("matched_training", False)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=180, sharex=True if matched else "col", sharey=True)
    colors = snapshot_legend(fig, STEPS, y=.92)
    lookup = {(case["target"], case["arm"]): case for case in cases}
    extent = max(float(abs(signed_log(case["views"]["gd_refit"]["residual"])).max()) for case in cases)
    for row, arm in enumerate(config["arms"]):
        for col, target in enumerate(TARGETS):
            case = lookup[target, arm]
            view = case["views"]["gd_refit"]
            ax = axes[row, col]
            half = 2 if target == "gaussian_envelope" and not matched else 1
            x = np.linspace(-half, half, view["residual"].shape[1])
            for step, color in zip(STEPS, colors):
                ax.plot(x, signed_log(view["residual"][step]), color=color, lw=1.0, alpha=.9)
            set_signed_log_axis(ax, extent)
            ax.axhline(0, color="#777777", lw=.6, zorder=0)
            ax.set_xlim(-half, half)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(("Xavier" if arm == "xavier" else "QI · initial γ = 16")+"\nSigned-log residual", fontsize=12)
            if row == 1:
                ax.set_xlabel("x · whole-line model, central view" if half == 2 else "x · finite interval", fontsize=11)
            values = view["relative_l2"]
            rank = view["rank"]
            title = rc.LABELS[target]+"\n" if row == 0 else ""
            ax.set_title(title+f"Refit rel. L₂ (steps 0 → 1 → 5): {values[0]:.3g} → {values[1]:.3g} → {values[5]:.3g}\n"
                         f"Retained rank: {rank[0]} → {rank[1]} → {rank[5]}", fontsize=10, pad=12)
    title = "Matched training samples · first five GD updates" if matched else "Earlier comparison · different training domains"
    fig.suptitle(title, fontsize=20, y=.988)
    fig.text(.5, .950, "Residual with evaluation-only readout solves · Xavier above, QI below · steps 0, 1, 2, 3, 4, 5", ha="center", fontsize=12)
    domain_note = ("All six cases: the same 1,024 midpoints in [−1, 1], half-mean-square loss, and standard tanh MLP with a free output bias. Readout SVD cutoff: 10⁻¹³."
                   if matched else "Gaussian uses the whole-line objective (only −2 ≤ x ≤ 2 is shown); sine and mixed sine use [−1, 1]. These training domains DO NOT MATCH.")
    fig.text(.5, .035,
             "Ordinary joint GD, rate 0.002, N = 128, halo 24 per side. Every plotted snapshot, INCLUDING step 0, receives a separate readout solve; training never uses it.\n"
             "Shared signed-log magnitude scale, floor 10⁻¹⁶, no residual normalization. QI steps 0 and 1 coincide exactly; later coincident curves can cover earlier ones.\n"
             + domain_note,
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.145, hspace=.43, wspace=.18)
    fig.savefig(MATCHED_OUTPUT if matched else OUTPUT)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--legacy-whole-line", action="store_true", help="Reproduce the earlier different-domain comparison")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    path = DATA if args.legacy_whole_line else MATCHED_DATA
    if args.plot_only:
        cases, config = rc.load_cases(path)
    else:
        config = yaml.safe_load((rc.HERE / "readout_comparison.yaml").read_text())
        config.update(targets=TARGETS, steps=5, displayed_steps=STEPS,
                      matched_training=not args.legacy_whole_line,
                      evaluation="Least-squares readout at EVERY step, with no feedback to GD")
        if config["matched_training"]:
            config.update(training_domain=[-1, 1], readout_solve_domain=[-1, 1],
                          model="Standard tanh MLP with unconstrained readout and output bias",
                          loss_normalization="0.5 * mean squared residual over the same 1024 midpoints",
                          evaluation_domain=[-1, 1])
        cases = run(config)
        if args.legacy_whole_line:
            config["gaussian_initial_gradient_validation"] = validate_gaussian_initial_gradient(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        rc.save_cases(cases, config, path)
    plot(cases, config)
    for case in cases:
        view = case["views"]["gd_refit"]
        print(json.dumps({"target": case["target"], "arm": case["arm"],
                          **{key: view[key].tolist() for key in ("relative_l2", "rank", "geometry_step_max", "sign_changes")}}))
    if args.legacy_whole_line:
        print(json.dumps(config["gaussian_initial_gradient_validation"]))
    else:
        print(json.dumps(config["independent_gd_step_max_absolute_difference"]))
    print(OUTPUT if args.legacy_whole_line else MATCHED_OUTPUT)


if __name__ == "__main__":
    main()
