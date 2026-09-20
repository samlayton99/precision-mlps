"""Nine fixed-neuron snapshots, each with a ten-gamma tangent-spectrum sweep.

No training or readout refit: use the whole-line Gaussian GD run at initial
gamma=16. Select three neurons without replacement with a recorded random
seed, and hold each saved center and physical readout fixed during the sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad_vec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expD24_gd_residual_spectrum import direct_scale_test as direct
from experiments.expD24_gd_residual_spectrum import gamma_comparison as ladder
from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison
from experiments.expD24_gd_residual_spectrum import whole_line as whole

SEED = 20260911
STEPS = [2, 352, 2000]
GAMMAS = 2.0**np.arange(-1, 9)


def build_data():
    cases, _ = comparison.load_cases(ladder.RESULTS / "data/data.npz")
    case = next(c for c in cases if c["target"] == "gaussian_envelope"
                and c["method"] == "gd" and ladder.gamma_of(c) == 16)
    count = case["parameters"]["a"].shape[1]
    neurons = np.sort(np.random.default_rng(SEED).choice(count, 3, replace=False))
    indices = [case["snapshot_steps"].index(step) for step in STEPS]
    a, b, v = (case["parameters"][key][indices][:, neurons].T for key in ("a", "b", "v"))
    # Project over the full readout, before selecting the three neurons.
    full_v = case["parameters"]["v"][indices]
    c = (full_v-full_v.mean(axis=1, keepdims=True))[:, neurons].T
    assert np.all(a > 0) and np.all(c != 0)
    z = -b/a
    k = np.r_[0., np.geomspace(1e-3, 2048, 8193)]
    spectra = np.empty((3, 3, len(k), len(GAMMAS)), dtype=complex)
    errors = []
    for row in range(3):
        for col in range(3):
            center, readout = z[row, col], c[row, col]
            spectra[row, col] = readout*direct.tangent_spectrum(np.pi*k, GAMMAS, np.full(10, center))
            # Independently integrate the spatial tangent in transition coordinates.
            for gamma in (GAMMAS[0], 16., GAMMAS[-1]):
                omega = gamma*np.array([.3, 1., 3.])
                numerical, _ = quad_vec(
                    lambda u: readout*u/np.cosh(u)**2/gamma**2
                    *np.exp(-1j*omega*(center+u/gamma)),
                    -32, 32, epsabs=1e-14, epsrel=1e-11)
                exact = readout*direct.tangent_spectrum(omega, np.array([gamma]), np.array([center]))[:, 0]
                np.testing.assert_allclose(exact, numerical, rtol=2e-9, atol=2e-14)
                errors.append(float(max(abs(exact-numerical))))
    # All snapshot differences in magnitude must come only from the fixed readout.
    bare = abs(spectra)/abs(c[:, :, None, None])
    expected = abs(direct.tangent_spectrum(np.pi*k, GAMMAS, np.zeros(10)))
    np.testing.assert_allclose(bare, np.broadcast_to(expected, bare.shape), rtol=2e-14, atol=1e-300)
    assert np.all(spectra[:, :, 0] == 0)
    metadata = {"target": "gaussian_envelope", "method": "gd", "initial_gamma": 16,
                "selection_seed": SEED, "selection_population": "all 177 neurons, including halo",
                "neuron_indices_zero_based": neurons.tolist(), "steps": STEPS,
                "fixed_within_each_panel": ["center", "physical readout coefficient"],
                "transform": "analytic whole-line Fourier transform of c*(x-z)*sech^2(gamma*(x-z))",
                "frequency_coordinate": "k = omega/pi", "normalization": "none",
                "max_spatial_fourier_absolute_difference": max(errors)}
    return {"neuron_indices": neurons, "steps": np.asarray(STEPS), "gammas": GAMMAS,
            "k": k, "spectrum": spectra, "center": z, "readout": c, "observed_gamma": a,
            "metadata_json": json.dumps(metadata)}


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=170, sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(GAMMAS)))
    k = data["k"][1:]
    ceiling = 10.**np.ceil(np.log10(abs(data["spectrum"]).max()))
    for row, neuron in enumerate(data["neuron_indices"]):
        for col, step in enumerate(STEPS):
            ax = axes[row, col]
            for index, color in enumerate(colors):
                values = abs(data["spectrum"][row, col, 1:, index])
                # Mask the sub-floor tail instead of adding a horizontal floor line.
                ax.plot(k, np.where(values >= 1e-16, values, np.nan), color=color, lw=1.4)
                peak = np.argmax(values)
                ax.plot(k[peak], values[peak], "o", ms=3.5, color=color)
            ax.set(xscale="log", yscale="log", xlim=(1e-3, 2048), ylim=(1e-16, ceiling),
                   xticks=[1e-3, 1e-2, .1, 1, 10, 100, 1000], yticks=10.**np.arange(-16, 1, 4))
            ax.minorticks_off()
            ax.grid(alpha=.19)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"z = {data['center'][row, col]:+.4f}    c = {data['readout'][row, col]:+.3e}\n"
                         f"Stored γ = {data['observed_gamma'][row, col]:.6f}", fontsize=10, pad=12)
            if col == 0:
                ax.set_ylabel(f"Neuron {neuron+1} of 177\n"+r"$|\widehat{\partial f/\partial\gamma}(\omega)|$", fontsize=12)
            if row == 0:
                ax.text(.5, 1.23, f"GD step {step:,}", transform=ax.transAxes, ha="center", fontsize=13)
            if row == 2:
                ax.set_xlabel(r"Frequency $\omega/\pi$ (log scale)", fontsize=11)
    fig.suptitle("Scale-tangent spectrum as gamma increases", fontsize=21, y=.988)
    fig.text(.5, .952, "Same three randomly selected neurons at three saved training steps · Gaussian whole-line GD, initial γ = 16", ha="center", fontsize=12)
    fig.legend([plt.Line2D([], [], color=color, lw=2.5) for color in colors],
               [f"γ = {gamma:g}" for gamma in GAMMAS], loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=10, frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.7)
    fig.text(.5, .033,
             "Within each panel: vary only γ; hold the saved center z and readout c fixed. Curve = whole-line Fourier magnitude of c(x − z)sech²(γ(x − z)).\n"
             "All panels share absolute scales; no normalization. Dots mark spectral peaks. γ = 0.5, 1, 2, …, 256; display floor 10⁻¹⁶; DC is exactly zero and omitted on the log axis.\n"
             "Center affects phase, not magnitude. Snapshot height differences reflect the fixed readout. This is a hypothetical gamma sweep, not additional training.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.08, right=.985, bottom=.125, top=.80, hspace=.38, wspace=.17)
    fig.savefig(output)
    plt.close(fig)


def build_overlay_data(saved):
    """Use the original saved residual at each step, fixed during gamma sweeps."""
    cases, config = comparison.load_cases(ladder.RESULTS / "data/data.npz")
    case = next(c for c in cases if c["target"] == "gaussian_envelope"
                and c["method"] == "gd" and ladder.gamma_of(c) == 16)
    k = np.linspace(0, 32, 4097)
    residuals = []
    tangents = np.empty((3, 3, len(k), len(saved["gammas"])), dtype=complex)
    errors = []
    for col, step in enumerate(saved["steps"]):
        index = case["snapshot_steps"].index(int(step))
        state = {key: case["parameters"][key][index] for key in ("a", "b", "v")}
        E = whole.model_spectrum(state, np.pi*k)-whole.target_spectrum(np.pi*k, config)
        residuals.append(E)
        for row in range(3):
            tangents[row, col] = saved["readout"][row, col]*direct.tangent_spectrum(
                np.pi*k, saved["gammas"], np.full(len(saved["gammas"]), saved["center"][row, col]))
        # Independently integrate the residual at selected physical frequencies.
        indices = np.array([0, 256, 768, 1792, 3072, 4096])
        omega = np.pi*k[indices]
        numerical, _ = quad_vec(
            lambda x: (whole.predict(state, x)-whole.target(x, config))*np.exp(-1j*omega*x),
            -24, 24, points=[-2, 0, 2], epsabs=2e-12, epsrel=2e-12)
        np.testing.assert_allclose(E[indices], numerical, rtol=2e-8, atol=5e-12)
        errors.append(float(max(abs(E[indices]-numerical))))
    metadata = json.loads(str(saved["metadata_json"]))
    metadata.update({"residual_held_fixed_during_gamma_sweep": True,
                     "display_frequency_range": [0, 32], "axes": "linear",
                     "tangent_axis": "left, separate absolute limits per panel",
                     "residual_axis": "right, common absolute limits across all panels",
                     "max_residual_spatial_fourier_difference": max(errors)})
    return {**{name: saved[name] for name in ("neuron_indices", "steps", "gammas", "center", "readout", "observed_gamma")},
            "k": k, "spectrum": tangents, "residual_spectrum": np.asarray(residuals),
            "metadata_json": json.dumps(metadata)}


def plot_overlay(data, output, *, log_y=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), dpi=170, sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data["gammas"])))
    k = data["k"]
    floor = 1e-16 if log_y else 0.
    def visible(values):
        return np.where(values >= floor, values, np.nan) if log_y else values

    residual_limit = 1.08*abs(data["residual_spectrum"]).max()
    common_limit = 10.**np.ceil(np.log10(max(abs(data["residual_spectrum"]).max(),
                                            abs(data["spectrum"]).max())))
    for row, neuron in enumerate(data["neuron_indices"]):
        for col, step in enumerate(data["steps"]):
            ax = axes[row, col]
            right = ax if log_y else ax.twinx()
            E = abs(data["residual_spectrum"][col])
            right.fill_between(k, floor, visible(E), color="black", alpha=.045)
            right.plot(k, visible(E), color="black", lw=1.65, ls="--")
            if not log_y:
                right.set(ylim=(0, residual_limit), ylabel=r"Residual $|\widehat e|$ (right)")
            right.spines["top"].set_visible(False)
            right.tick_params(labelsize=9)
            for index, color in enumerate(colors):
                ax.plot(k, visible(abs(data["spectrum"][row, col, :, index])), color=color, lw=1.6)
            tangent_limit = common_limit if log_y else 1.08*abs(data["spectrum"][row, col]).max()
            ax.set(yscale="log" if log_y else "linear", xlim=(0, 32), ylim=(floor, tangent_limit),
                   xticks=[0, 4, 8, 12, 16, 20, 24, 28, 32])
            if log_y:
                ax.minorticks_off()
                right.minorticks_off()
            else:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(formatter)
            ax.grid(alpha=.15)
            ax.spines["top"].set_visible(False)
            ax.set_title(f"z = {data['center'][row, col]:+.4f}    c = {data['readout'][row, col]:+.3e}",
                         fontsize=10, pad=17)
            if col == 0:
                label = "Fourier magnitude" if log_y else "Tangent magnitude (left)"
                ax.set_ylabel(f"Neuron {neuron+1}\n{label}", fontsize=11)
            else:
                ax.set_ylabel("Fourier magnitude" if log_y else "Tangent magnitude (left)", fontsize=10)
            if row == 0:
                ax.text(.5, 1.24, f"GD step {step:,}", transform=ax.transAxes, ha="center", fontsize=13)
            if row == 2:
                ax.set_xlabel(r"Frequency $\omega/\pi$", fontsize=11)
    fig.suptitle("Scale-tangent gamma sweep overlaid with the saved residual spectrum", fontsize=19, y=.988)
    axis_description = "Linear frequency; logarithmic magnitudes" if log_y else "Linear frequency and magnitude axes"
    fig.text(.5, .952, f"{axis_description} · Gaussian whole-line run · same nine neuron snapshots", ha="center", fontsize=12)
    handles = [plt.Line2D([], [], color=color, lw=2.5) for color in colors]
    labels = [f"γ = {gamma:g}" for gamma in data["gammas"]]
    handles.append(plt.Line2D([], [], color="black", ls="--", lw=1.8))
    labels.append("Saved residual")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .921), ncol=11,
               frameon=False, fontsize=10, columnspacing=.9, handlelength=1.7)
    if log_y:
        magnitude_notes = (
            "Viridis: scale-tangent magnitude. Dashed black: saved residual magnitude. Both use the SAME logarithmic magnitude axis in every panel.\n"
            "All nine panels share identical frequency and magnitude limits. No amplitude normalization; values below 10⁻¹⁶ are hidden.\n")
    else:
        magnitude_notes = (
            "Viridis: tangent magnitude on the LEFT axis, with separate absolute limits per panel. Black: residual magnitude on the RIGHT axis, with identical limits throughout.\n"
            "No amplitude normalization. Separate vertical axes show frequency overlap; curve crossing heights do not imply equal magnitudes. All axes are linear.\n")
    fig.text(.5, .035,
             magnitude_notes + "The original residual is fixed within each column; it is not recomputed after the hypothetical gamma changes. View focuses on 0 ≤ ω/π ≤ 32; the previous plot shows wider tails.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.07, right=.95, bottom=.125, top=.80, hspace=.40, wspace=.25 if log_y else .46)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-overlay", action="store_true", help="Add a linear-axis residual overlay without replacing the original plot")
    parser.add_argument("--log-y", action="store_true", help="Render the saved residual overlay with logarithmic magnitude axes")
    args = parser.parse_args()
    output = direct.RESULTS
    if args.log_y:
        with np.load(output / "data/tangent_residual_overlay_linear.npz") as source:
            data = {key: source[key] for key in source.files}
        path = output / "tangent_residual_overlay_log_y.png"
        plot_overlay(data, path, log_y=True)
        print(path)
        return
    if args.residual_overlay:
        with np.load(output / "data/tangent_gamma_sweep.npz") as source:
            saved = {key: source[key] for key in source.files}
        data = build_overlay_data(saved)
        np.savez_compressed(output / "data/tangent_residual_overlay_linear.npz", **data)
        plot_overlay(data, output / "tangent_residual_overlay_linear.png")
        print(data["metadata_json"])
        print(output / "tangent_residual_overlay_linear.png")
        return
    data = build_data()
    np.savez_compressed(output / "data/tangent_gamma_sweep.npz", **data)
    plot(data, output / "tangent_gamma_sweep.png")
    print(data["metadata_json"])
    print(output / "tangent_gamma_sweep.png")


if __name__ == "__main__":
    main()
