"""Causal frozen-geometry control, matched-residual probe, and tangent spectra.

Run after gamma_comparison.py; --plot-only reuses this diagnostic's data.
These are measurements of the existing model, not a proposed optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import gamma_comparison as previous
from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum.spectrum import FiniteIntervalTransform, snapshot_legend

HERE = Path(__file__).resolve().parent
RESULTS = previous.RESULTS.parent / "direct_scale_test"


def fprime(xi):
    """Derivative of pi*xi/sinh(pi*xi/2), including its odd symmetry."""
    xi = np.asarray(xi, dtype=float)
    s = np.pi * abs(xi) / 2
    out = np.empty_like(s)
    small = s < 1e-3
    out[small] = np.pi / 2 * (-2*s[small]/3 + 7*s[small]**3/45 - 31*s[small]**5/1260)
    t = s[~small]
    inverse_sinh = 2*np.exp(-t) / (-np.expm1(-2*t))
    coth = 1 + 2*np.exp(-2*t) / (-np.expm1(-2*t))
    out[~small] = np.pi * inverse_sinh * (1 - t*coth)
    return out * np.sign(xi)


def tangent_spectrum(omega, gamma, centers):
    """Bare center-preserving scale derivative on the whole line."""
    w = np.asarray(omega)[..., None]
    return 1j*np.exp(-1j*w*centers)*fprime(w/gamma)/gamma**2


def magnitude_means(values):
    """Means of magnitudes across neurons; a true zero makes the GM zero."""
    magnitude = abs(values)
    with np.errstate(divide="ignore"):
        geometric = np.exp(np.mean(np.log(magnitude), axis=-1))
    return magnitude.mean(axis=-1), geometric


def spatial_problem(target, config):
    if target == "gaussian_envelope":
        spatial_config = yaml.safe_load((HERE / "whole_line.yaml").read_text())
        x, w = whole.quadrature(spatial_config)
        return x, w, whole.target(x, config)
    problem = comparison.Problem(target, config)
    return problem.x, problem.weights, problem.y


def replay_control(case, config):
    """Replay joint GD, measuring every step, alongside frozen-geometry GD.

    Frozen readout GD uses its fixed Gram matrix: exactly the same linear
    gradient, not a least-squares solve. Check against direct residual gradients.
    """
    x, w, y = spatial_problem(case["target"], config)
    tx, tw, ty = map(torch.tensor, (x, w, y))
    is_whole = case["target"] == "gaussian_envelope"
    params = {key: torch.nn.Parameter(torch.tensor(value[0])) for key, value in case["parameters"].items()}
    a0, b0 = (params[key].detach().clone() for key in ("a", "b"))
    reference = torch.tanh(tx)[:, None]
    initial_features = torch.tanh(tx[:, None]*a0 + b0)
    if is_whole:
        initial_features = initial_features - reference
        initial_features = initial_features - initial_features.mean(dim=1, keepdim=True)
    else:
        initial_features = torch.cat((initial_features, torch.ones((len(x), 1))), dim=1)
    G = initial_features.T @ (tw[:, None]*initial_features)
    h = initial_features.T @ (tw*ty)
    frozen_v = torch.zeros_like(params["v"])
    optimizer = torch.optim.SGD(params.values(), lr=config["learning_rate"])
    steps, chosen = config["steps"], case["snapshot_steps"]
    metrics = {name: np.zeros(steps+1) for name in
               ("joint_loss", "frozen_loss", "net_motion", "path_motion", "raw_gradient", "center_scale_gradient")}
    path = torch.zeros_like(a0)
    frozen_frames = []
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        values = torch.tanh(tx[:, None]*params["a"] + params["b"])
        if is_whole:
            c = params["v"] - params["v"].mean()
            prediction = (values - reference) @ c
        else:
            prediction = values @ params["v"][:-1] + params["v"][-1]
        residual = prediction - ty
        loss = .5*torch.sum(tw*residual.square())
        loss.backward()
        with torch.no_grad():
            frozen_r = initial_features @ frozen_v - ty
            metrics["joint_loss"][step] = loss.item()
            metrics["frozen_loss"][step] = (.5*torch.sum(tw*frozen_r.square())).item()
            metrics["net_motion"][step] = (abs(params["a"])-abs(a0)).abs().mean().item()
            metrics["path_motion"][step] = path.mean().item()
            ga, gb = params["a"].grad, params["b"].grad
            z = -params["b"] / params["a"]
            metrics["raw_gradient"][step] = ga.abs().mean().item()
            metrics["center_scale_gradient"][step] = (ga-z*gb).abs().mean().item()
            if step in chosen:
                index = chosen.index(step)
                for key in params:
                    np.testing.assert_allclose(params[key].numpy(), case["parameters"][key][index], rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose((G@frozen_v-h).numpy(),
                                           (initial_features.T@(tw*frozen_r)).numpy(), rtol=2e-9, atol=2e-13)
                frozen_frames.append(frozen_v.numpy().copy())
            if step < steps:
                old_a = params["a"].clone()
                frozen_v -= config["learning_rate"]*(G@frozen_v-h)
                optimizer.step()
                path += (abs(params["a"])-abs(old_a)).abs()
    np.testing.assert_allclose(metrics["joint_loss"], case["loss"], rtol=2e-10, atol=2e-13)
    assert params["a"].min() > 0
    return {**metrics, "frozen_v": np.asarray(frozen_frames)}


def matched_probe(config):
    """Same unit-L2 Gaussian sine residual, center=0, readout=1, varying gamma.

    This is an actual derivative of f_gamma=tanh(gamma*x)-tanh(gamma0*x)
    at gamma=gamma0, with target a normalized Gaussian sine. Every initial
    prediction is zero. The reference is fixed during differentiation.
    """
    gammas = np.unique(np.r_[np.geomspace(.5, 256, 321), config["initial_gammas"]])
    k, weights = comparison.frequency_quadrature(64)
    responses, spatial_checks = [], []
    for mode in config["target_modes"]:
        probe = {**config, "target_modes": [mode], "target_amplitudes": [1.]}
        norm = np.sqrt(whole.target_energy(probe))
        E = -whole.target_spectrum(np.pi*k, probe)/norm
        J = tangent_spectrum(np.pi*k, gammas, np.zeros_like(gammas))
        gradient = np.sum(weights[:, None]*np.real(E[:, None]*J.conj()), axis=0)
        responses.append(gradient)
        # Independent spatial integration and finite differences at resolved scales.
        roots, ws = roots_legendre(4096)
        x, w = 4*roots, 4*ws
        target = whole.target(x, probe)/norm
        for gamma in config["initial_gammas"]:
            expected = gradient[np.flatnonzero(gammas == gamma)[0]]
            tangent = x*(1-np.tanh(gamma*x)**2)
            spatial = -np.dot(w, target*tangent)
            np.testing.assert_allclose(spatial, expected, rtol=2e-7, atol=2e-13)
            eps = 1e-4*gamma
            # Subtract the two losses analytically to avoid cancelling their O(1) part.
            def difference(h):
                rp = np.tanh((gamma+h)*x)-np.tanh(gamma*x)-target
                rm = np.tanh((gamma-h)*x)-np.tanh(gamma*x)-target
                return np.dot(w, (rp-rm)*(rp+rm))/(4*h)
            finite_difference = (4*difference(eps/2)-difference(eps))/3
            np.testing.assert_allclose(finite_difference, expected, rtol=2e-5, atol=3e-11)
            spatial_checks.append({"mode": mode, "gamma": gamma, "gradient": float(expected),
                                   "spatial_difference": float(abs(spatial-expected)),
                                   "finite_difference_error": float(abs(finite_difference-expected))})
    return {"gamma": gammas, "modes": np.asarray(config["target_modes"]),
            "gradient": np.asarray(responses)}, spatial_checks


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_controls(cases, controls, config):
    plt = plotting()
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), dpi=150)
    colors = plt.cm.viridis(np.linspace(.1, .9, 4))
    for row, target in enumerate(config["targets"]):
        for color, gamma in zip(colors, config["initial_gammas"]):
            case = next(c for c in cases if c["target"] == target and previous.gamma_of(c) == gamma)
            result = controls[f"{target}_{gamma:g}"]
            energy = 2*case["loss"][0]
            t = np.arange(config["steps"]+1)
            for name, style in (("joint_loss", "-"), ("frozen_loss", "--")):
                axes[row, 0].plot(t, np.sqrt(2*result[name]/energy), style, color=color, lw=1.6)
            advantage = 100*(1-np.sqrt(result["joint_loss"]/result["frozen_loss"]))
            axes[row, 1].plot(t, advantage, color=color)
            for name, style in (("net_motion", "-"), ("path_motion", "--")):
                axes[row, 2].plot(t[1:], result[name][1:], style, color=color, lw=1.6)
        axes[row, 0].set(ylabel=f"{comparison.LABELS[target]}\nRelative L₂", yscale="log", ylim=(.009, 1.1))
        axes[row, 1].set(ylabel="Reduction in error norm at this step (%)")
        axes[row, 1].axhline(0, color="gray", lw=.6)
        axes[row, 2].set(ylabel="Mean absolute γ movement", yscale="log", ylim=(1e-12, 3e-3))
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.set_xlabel("GD step")
    for ax, title in zip(axes[0], ["Joint GD (solid) / frozen geometry (dashed)",
                                  "Advantage from allowing geometry to move", "Net displacement (solid) / total travel (dashed)"]):
        ax.set_title(title, fontsize=11, pad=14)
    extent = max(abs(ax.get_ylim()[0]) for ax in axes[:, 1])
    extent = max(extent, max(abs(ax.get_ylim()[1]) for ax in axes[:, 1]))
    for ax in axes[:, 1]:
        ax.set_ylim(-.05*extent, extent)
    fig.suptitle("Direct control: does moving the geometry improve ordinary GD?", fontsize=19, y=.985)
    fig.legend([plt.Line2D([], [], color=c, lw=2) for c in colors],
               [f"Initial γ = {g:g}" for g in config["initial_gammas"]], loc="upper center", bbox_to_anchor=(.5, .953), ncol=4, frameon=False)
    fig.subplots_adjust(top=.89, bottom=.08, hspace=.32, wspace=.27)
    fig.text(.5, .026, "Same initial centers, zero readout, width, learning rate 0.002, and 2,000 steps. Frozen control trains only the readout.\n"
             "Middle: 100 × (1 − joint L₂ / frozen L₂), using training quadrature. Right: average over all 177 neurons; travel sums absolute changes at every step.", ha="center", fontsize=10)
    fig.savefig(RESULTS / "training_control.png")
    plt.close(fig)


def plot_probe(data, config):
    plt = plotting()
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 8.7), dpi=170)
    colors = plt.cm.viridis([.15, .5, .85])
    for mode, gradient, color in zip(data["modes"], data["gradient"], colors):
        axes[0].loglog(data["gamma"], abs(gradient), color=color, lw=2.4,
                       label=rf"$k={mode:g}$: Gaussian $\times\,\sin({mode:g}\pi x)$")
        axes[1].semilogx(data["gamma"], abs(gradient)/max(abs(gradient)), color=color, lw=2.4)
    for ax in axes:
        ax.set_xlim(.5, 256)
        ax.set_xlabel(r"Neuron scale $\gamma$ in $\tanh(\gamma x)$  (log scale)", fontsize=13, labelpad=10)
        ax.xaxis.set_major_locator(FixedLocator([.5, 1, 4, 16, 64, 256]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=12)
        ax.grid(alpha=.2)
    axes[0].set_ylim(1e-32, 1)
    axes[0].set_ylabel("Gradient magnitude (log scale)\n"+r"$G_k(\gamma)$", fontsize=14, labelpad=12)
    axes[0].set_title("Actual gradient magnitude", fontsize=15, pad=18)
    axes[1].set_ylim(-.02, 1.05)
    axes[1].set_yticks([0, .25, .5, .75, 1])
    axes[1].set_ylabel("Fraction of this curve's own peak\n"+
                       r"$G_k(\gamma)\,/\,\max_{\gamma'} G_k(\gamma')$", fontsize=14, labelpad=12)
    axes[1].set_title("Same curves, each divided by its own maximum", fontsize=15, pad=18)
    fig.suptitle("Scale gradient for three fixed residual functions", fontsize=22, y=.98)
    fig.text(.5, .932, r"Within each curve, only $\gamma$ changes. Residual fixed; center $z=0$; readout $c=1$. No training.",
             ha="center", fontsize=13)
    sigma=config["envelope_sigma"]
    fig.text(.5, .882, rf"$e_k(x)=-C_k\,e^{{-x^2/(2\sigma^2)}}\sin(k\pi x),\qquad \sigma={sigma:g},\qquad \|e_k\|_{{L_2(\mathbb{{R}})}}=1$",
             ha="center", fontsize=16)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .847),
               ncol=3, frameon=False, fontsize=13, handlelength=2.8, columnspacing=2.5)
    fig.subplots_adjust(top=.725, bottom=.24, left=.085, right=.975, wspace=.32)
    fig.text(.5, .117,
             r"$G_k(\gamma)=\left|\frac{\partial L_{\mathbb{R}}}{\partial\gamma}\right|"
             r"=\left|\int_{-\infty}^{\infty}e_k(x)\,x\,\mathrm{sech}^{2}(\gamma x)\,dx\right|$",
             ha="center", fontsize=19)
    fig.text(.5, .049,
             "Cₖ normalizes each residual to unit L₂. Right: the maximum is taken over the plotted gamma sweep.\n"
             "A height of 1 on the right means that residual's strongest response; it does not mean equal absolute gradients.",
             ha="center", fontsize=12)
    fig.savefig(RESULTS / "matched_residual.png")
    plt.close(fig)


def spectrum_diagnostics(case, config):
    """Full-line analytic transforms or transforms of both windowed factors.

    Bare = d(tanh)/d gamma with center fixed; weighted = c_j times bare.
    Finite problems show the continuous windowed residual/tangent, while
    training uses midpoint quadrature. No taper or end smoothing is applied.
    """
    k = np.linspace(0, 128, 8193)
    result = {name: [] for name in ("error", "bare_am", "bare_gm", "weighted_am", "weighted_gm")}
    is_whole = case["target"] == "gaussian_envelope"
    if not is_whole:
        n = config["n_eval"]
        x = -1+(np.arange(n)+.5)*2/n
        transform = FiniteIntervalTransform(x, max_mode=128, points=len(k))
        target = comparison.get_target(case["target"]).fn_numpy(x)
    for index in range(len(case["snapshot_steps"])):
        a, b, v = (case["parameters"][key][index] for key in ("a", "b", "v"))
        assert np.all(a > 0)
        z = -b/a
        if is_whole:
            c = v-v.mean()
            E = whole.model_spectrum({"a": a, "b": b, "v": v}, np.pi*k)-whole.target_spectrum(np.pi*k, config)
            bare = tangent_spectrum(np.pi*k, a, z)
        else:
            c = v[:-1]
            residual = np.tanh(x[:, None]*a+b)@c+v[-1]-target
            E = transform(residual)
            bare = np.empty((len(k), len(a)), dtype=complex)
            for start in range(0, len(a), 16):
                t = x[None, :]-z[start:start+16, None]
                u = a[start:start+16, None]*t
                decay = np.exp(-2*abs(u))
                values = t*4*decay/(1+decay)**2
                bare[:, start:start+16] = transform(values).T
            # Check the matrix-transform orientation and phases at three frequencies.
            for pos in (0, 177, 777):
                t = x-z[0]
                direct = transform.dx*np.dot(t*(1-np.tanh(a[0]*t)**2), np.exp(-1j*np.pi*k[pos]*x))
                np.testing.assert_allclose(bare[pos, 0], direct, rtol=2e-7, atol=5e-13)
        am, gm = magnitude_means(bare)
        weighted_am, weighted_gm = magnitude_means(bare*c)
        for name, value in zip(result, (E, am, gm, weighted_am, weighted_gm)):
            result[name].append(value)
    return {"k": k, **{name: np.asarray(value) for name, value in result.items()}}


def whole_pairings(case, config):
    """Keep complex phases, integrate each neuron's gradient, then summarize."""
    k, weights = comparison.frequency_quadrature(64)
    masks = [k < 4, (k >= 4) & (k < 10), k >= 10]
    spatial_config = yaml.safe_load((HERE / "whole_line.yaml").read_text())
    x, w = whole.quadrature(spatial_config, refinement=2)
    target = whole.target(x, config)
    result = {name: [] for name in ("gradient", "band_gradient", "envelope", "band_energy", "spatial_error")}
    for index in range(len(case["snapshot_steps"])):
        params = {key: case["parameters"][key][index] for key in ("a", "b", "v")}
        a, b, v = (params[key] for key in ("a", "b", "v"))
        c, z = v-v.mean(), -b/a
        E = whole.model_spectrum(params, np.pi*k)-whole.target_spectrum(np.pi*k, config)
        J = tangent_spectrum(np.pi*k, a, z)*c
        density = weights[:, None]*np.real(E[:, None]*J.conj())
        gradient = density.sum(axis=0)
        bands = np.stack([density[mask].sum(axis=0) for mask in masks])
        e = whole.predict(params, x)-target
        t = x[:, None]-z
        decay = np.exp(-2*abs(t*a))
        direct = c*((w*e)@(t*4*decay/(1+decay)**2))
        np.testing.assert_allclose(gradient, direct, rtol=1e-7, atol=2e-12)
        np.testing.assert_allclose(bands.sum(axis=0), gradient, rtol=1e-10, atol=1e-14)
        envelope = np.sum(weights[:, None]*abs(E[:, None])*abs(J), axis=0)
        assert np.all(abs(gradient) <= envelope+1e-14)
        energy = weights*abs(E)**2
        result["gradient"].append(gradient)
        result["band_gradient"].append(bands)
        result["envelope"].append(envelope)
        result["band_energy"].append([energy[mask].sum()/energy.sum() for mask in masks])
        result["spatial_error"].append(max(abs(gradient-direct)))
    return {name: np.asarray(value) for name, value in result.items()}


def plot_spectra(target, cases, spectra, config):
    plt = plotting()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(4, 4, figsize=(20, 14), dpi=150, sharex="col")
    colors = snapshot_legend(fig, cases[0]["snapshot_steps"], y=.94)
    for row, gamma in enumerate(config["initial_gammas"]):
        result = spectra[f"{target}_{gamma:g}"]
        k = result["k"]
        for i, color in enumerate(colors):
            axes[row, 0].plot(k, np.maximum(abs(result["error"][i]), 1e-16), color=color, lw=1)
            for col, prefix in ((1, "bare"), (2, "weighted")):
                for suffix, style in (("am", "-"), ("gm", "--")):
                    axes[row, col].plot(k, np.maximum(result[f"{prefix}_{suffix}"][i], 1e-16), style, color=color, lw=1.1)
        for name, color, style in (("error", "black", "-"), ("bare_am", "#8a8a8a", ":"),
                                   ("weighted_am", "#137bb0", "-"), ("weighted_gm", "#db7420", "--")):
            values = abs(result[name][-1])
            if max(values) > 0:
                axes[row, 3].plot(k, np.maximum(values/max(values), 1e-16), style, color=color, lw=1.5)
        axes[row, 0].set_ylabel(f"Initial γ = {gamma:g}\nFourier magnitude")
        for col, ax in enumerate(axes[row]):
            ax.set(xlim=(0, 128 if col == 3 else 32), yscale="log", ylim=(1e-16, 2),
                   xticks=[0, 32, 64, 96, 128] if col == 3 else [0, 8, 16, 24, 32], yticks=10.**np.arange(-16, 1, 4))
            ax.grid(alpha=.2)
            if row == 3:
                ax.set_xlabel("Frequency ω/π")
    for ax, title in zip(axes[0], ["Residual |ê|", "Bare scale tangent: AM / GM", "Readout-weighted tangent: AM / GM", "Full-range final overlap (each peak = 1)"]):
        ax.set_title(title, fontsize=11, pad=12)
    fig.suptitle(f"{comparison.LABELS[target]}: where do residual and scale tangents overlap?", fontsize=19, y=.985)
    handles = [Line2D([], [], color="black"), Line2D([], [], color="#8a8a8a", ls=":"),
               Line2D([], [], color="#137bb0"), Line2D([], [], color="#db7420", ls="--")]
    fig.legend(handles, ["Final residual", "Final bare AM", "Final weighted AM", "Final weighted GM"],
               loc="upper center", bbox_to_anchor=(.5, .889), ncol=4, frameon=False)
    domain = "Analytic whole-line transforms; no window." if target == "gaussian_envelope" else "Both residual and tangent restricted to [−1, 1]; no taper. Window effects remain."
    fig.text(.5, .027, domain+"  Bare ψⱼ = (x − zⱼ) sech²(γⱼ(x − zⱼ)); weighted tangent = cⱼ ψⱼ.\n"
             "AM = arithmetic mean of magnitudes (solid); GM = geometric mean of magnitudes (dashed), over all 177 neurons. Exact zeros give GM = 0.\n"
             "First three columns zoom to ω/π ≤ 32 and preserve amplitudes; final column extends to 128 and compares shapes only. Phases are not shown. Floor 10⁻¹⁶.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.115, left=.065, right=.99, hspace=.20, wspace=.2)
    fig.savefig(RESULTS / "spectra" / f"{target}.png")
    plt.close(fig)


def plot_pairings(cases, pairs, config):
    plt = plotting()
    fig, axes = plt.subplots(4, 3, figsize=(16, 13), dpi=150, sharex=True)
    colors = ["#2877b4", "#db7a24", "#2a9d63"]
    for row, gamma in enumerate(config["initial_gammas"]):
        result = pairs[f"{gamma:g}"]
        case = next(c for c in cases if c["target"] == "gaussian_envelope" and previous.gamma_of(c) == gamma)
        steps = case["snapshot_steps"]
        for band, color in enumerate(colors):
            axes[row, 0].plot(steps, result["band_energy"][:, band], "o-", color=color, ms=3)
            axes[row, 1].plot(steps[1:], np.maximum(abs(result["band_gradient"][1:, band]).mean(axis=-1), 1e-16), "o-", color=color, ms=3)
        axes[row, 1].plot(steps[1:], abs(result["gradient"][1:]).mean(axis=-1), color="black", lw=1.5)
        axes[row, 2].plot(steps[1:], result["envelope"][1:].mean(axis=-1), "--", color="#9866a3")
        axes[row, 2].plot(steps[1:], abs(result["gradient"][1:]).mean(axis=-1), color="black", lw=1.5)
        axes[row, 0].set(ylabel=f"Initial γ = {gamma:g}\nFraction of residual energy", ylim=(-.03, 1.03))
        for ax in axes[row, 1:]:
            ax.set(yscale="log", ylim=(1e-10, 1e-2), ylabel="Mean absolute scale gradient")
        axes[row, 1].set_ylim(8e-17, 1e-2)
        for ax in axes[row]:
            ax.set_xscale("symlog", linthresh=2)
            ax.set(xticks=[0, 2, 10, 100, 2000], xticklabels=["0", "2", "10", "100", "2000"])
            ax.grid(alpha=.2)
            if row == 3:
                ax.set_xlabel("GD step (log spacing after step 2)")
    for ax, title in zip(axes[0], ["Where is the error?", "Which bands drive the scale gradient?", "Magnitude overlap versus actual pairing"]):
        ax.set_title(title, fontsize=11, pad=15)
    handles = [plt.Line2D([], [], color=c) for c in colors]+[plt.Line2D([], [], color="black"), plt.Line2D([], [], color="#9866a3", ls="--")]
    fig.legend(handles, ["ω/π < 4", "4 ≤ ω/π < 10", "ω/π ≥ 10", "Actual total gradient", "Magnitude-overlap upper bound"],
               loc="upper center", bbox_to_anchor=(.5, .945), ncol=5, frameon=False, fontsize=10)
    fig.suptitle("Gaussian whole-line problem: residual energy is not gradient strength", fontsize=19, y=.985)
    fig.text(.5, .026, "For each neuron, integrate Re(ê × conjugate(cⱼ ψ̂ⱼ)) in each band, then take absolute values and average.\n"
             "Band magnitudes do not add: signed contributions can cancel. Right: mean ∫ |ê| |cⱼψ̂ⱼ| versus mean |∫ Re(ê conjugate(cⱼψ̂ⱼ))|.\n"
             "All frequencies integrated to infinity; spatial-gradient checks at every state. Band display floor 10⁻¹⁶. Step 0 has zero readout and zero geometry gradient.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.875, bottom=.115, left=.07, right=.99, hspace=.25, wspace=.27)
    fig.savefig(RESULTS / "frequency_pairing.png")
    plt.close(fig)


def run_spectra(cases, config, plot_only):
    (RESULTS / "spectra").mkdir(exist_ok=True)
    path = RESULTS / "data/spectra.npz"
    spectra, pairs = {}, {}
    if path.exists():
        with np.load(path) as saved:
            for case in cases:
                key = f"{case['target']}_{previous.gamma_of(case):g}"
                spectra[key] = {name.split("__")[1]: saved[name] for name in saved.files if name.startswith(key+"__")}
    for case in cases:
        key = f"{case['target']}_{previous.gamma_of(case):g}"
        if not spectra.get(key):
            if plot_only:
                raise ValueError("Missing spectral data")
            print(f"Tangent spectra: {key}", flush=True)
            spectra[key] = spectrum_diagnostics(case, config)
            np.savez_compressed(path, **{f"{key}__{name}": value for key, result in spectra.items() for name, value in result.items()})
        if case["target"] == "gaussian_envelope":
            pairs[f"{previous.gamma_of(case):g}"] = whole_pairings(case, config)
    for target in config["targets"]:
        plot_spectra(target, cases, spectra, config)
    plot_pairings(cases, pairs, config)
    np.savez_compressed(RESULTS / "data/pairings.npz", **{f"{key}__{name}": value for key, result in pairs.items() for name, value in result.items()})
    print("All tangent spectra and phase-aware gradient checks completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data").mkdir(exist_ok=True)
    all_cases, config = comparison.load_cases(previous.RESULTS / "data/data.npz")
    torch.set_num_threads(config["threads"])
    cases = [case for case in all_cases if case["method"] == "gd" and previous.gamma_of(case) in config["initial_gammas"]]
    controls = {}
    control_path = RESULTS / "data/controls.npz"
    if control_path.exists():
        with np.load(control_path) as saved:
            for case in cases:
                key = f"{case['target']}_{previous.gamma_of(case):g}"
                controls[key] = {name.split("__")[1]: saved[name] for name in saved.files if name.startswith(key+"__")}
    for i, case in enumerate(cases):
        key = f"{case['target']}_{previous.gamma_of(case):g}"
        if controls.get(key):
            continue
        if args.plot_only:
            raise ValueError("Missing control data")
        print(f"Control {i+1}/{len(cases)}: {key}", flush=True)
        controls[key] = replay_control(case, config)
        np.savez_compressed(control_path, **{f"{key}__{name}": value for key, result in controls.items() for name, value in result.items()})
    plot_controls(cases, controls, config)
    probe, checks = matched_probe(config)
    np.savez_compressed(RESULTS / "data/matched_residual.npz", **probe)
    plot_probe(probe, config)
    summary = {key: {"joint_relative_l2": float(np.sqrt(value["joint_loss"][-1]/value["joint_loss"][0])),
                     "frozen_relative_l2": float(np.sqrt(value["frozen_loss"][-1]/value["frozen_loss"][0])),
                     "relative_l2_advantage_percent": float(100*(1-np.sqrt(value["joint_loss"][-1]/value["frozen_loss"][-1]))),
                     "mean_net_change": float(value["net_motion"][-1]), "mean_path_length": float(value["path_motion"][-1]),
                     "final_raw_gradient": float(value["raw_gradient"][-1]),
                     "final_center_scale_gradient": float(value["center_scale_gradient"][-1])}
               for key, value in controls.items()}
    (RESULTS / "data/validation.json").write_text(json.dumps({"controls": summary, "matched_probe": checks}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not args.controls_only:
        run_spectra(cases, config, args.plot_only)


if __name__ == "__main__":
    main()
