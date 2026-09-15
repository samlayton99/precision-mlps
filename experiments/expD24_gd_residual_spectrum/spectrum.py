"""Dense finite-interval Fourier evaluation and the experiment's GIF renderer."""
from __future__ import annotations

import numpy as np
from PIL import GifImagePlugin, Image
from scipy.signal import ZoomFFT


class FiniteIntervalTransform:
    """Evaluate dx*sum e(x_n)*exp(-i*omega*x_n) on a dense frequency grid.

    ZoomFFT evaluates the same finite-window transform between integer DFT bins;
    it adds display detail, not new resolving power beyond the observation interval.
    """
    def __init__(self, x, max_mode=32, points=2049):
        self.x = np.asarray(x)
        self.dx = float(x[1] - x[0])
        self.length = len(x) * self.dx
        self.modes = np.linspace(0, max_mode, points)
        self.omega = 2 * np.pi * self.modes / self.length
        self.phase = self.dx * np.exp(-1j * self.omega * x[0])
        self.zoom = ZoomFFT(len(x), [0, max_mode], m=points, fs=len(x), endpoint=True)

    def __call__(self, residual):
        return self.phase * self.zoom(residual)


def frame_steps(total=1000):
    """Verbatim pacing from expD06, reused by expD08 iteration_11/gifs_qn."""
    steps, s = [0], 0
    stride, count = 2, 20
    while s < total:
        for _ in range(count):
            s += stride
            if s >= total:
                break
            steps.append(s)
        stride, count = stride * 2, 10
    steps.append(total)
    return sorted(set(steps))


def snapshot_steps(total, available=None, count=10):
    """Ten plotting states including initialization, concentrated early in GD."""
    desired = np.r_[0, np.rint(np.geomspace(min(2, total), total, count - 1))].astype(int)
    if available is not None:
        available = np.asarray(available)
        desired = np.array([available[np.argmin(abs(available - step))] for step in desired])
    return np.unique(desired).tolist()


def snapshot_legend(fig, steps, y=0.90):
    """Shared discrete viridis key; labels show the actual, nonuniform times."""
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(steps)))
    handles = [plt.Line2D([], [], color=color, lw=2.5) for color in colors]
    fig.legend(handles, [f"{step:,}" for step in steps], title="GD step",
               loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(steps),
               frameon=False, fontsize=10, title_fontsize=11,
               handlelength=2, columnspacing=1.1)
    return colors

def write_gif_frame(stream, rgb: Image.Image, palette: Image.Image | None, duration_ms: int):
    """Stream one palette frame; never hold the whole animation in memory."""
    if palette is None:
        frame = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        palette = frame.copy()
        header, _ = GifImagePlugin.getheader(frame, info={"loop": 0, "optimize": False})
        for block in header:
            stream.write(block)
    else:
        frame = rgb.quantize(palette=palette, dither=Image.Dither.NONE)
    for block in GifImagePlugin.getdata(frame, duration=duration_ms, disposal=1):
        stream.write(block)
    return palette


def frame_durations(count, seconds):
    """GIF uses 10-ms units; distribute rounding to preserve total loop length."""
    ticks = np.rint(np.linspace(0, seconds * 100, count + 1)).astype(int)
    durations = np.diff(ticks) * 10
    assert np.all(durations > 0)
    return durations


def animate(cases, config, modes, steps, output, arm_labels, target_labels, previews):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter

    color = "#1865aa"
    floor = 1e-5
    magnitude = {(case["target"], case["arm"]): np.abs(case["spectra"]) for case in cases}
    errors = {(case["target"], case["arm"]): case["relative_l2"] for case in cases}
    ceiling = 10.0 ** np.ceil(np.log10(max(values.max() for values in magnitude.values())))
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), dpi=150, sharex=True, sharey=True)
    lines, labels = {}, {}
    for i, target in enumerate(config["targets"]):
        for j, arm in enumerate(config["arms"]):
            ax = axes[i, j]
            key = (target, arm)
            lines[key], = ax.plot(modes, np.maximum(magnitude[key][0], floor), color=color, lw=1.5)
            labels[key] = ax.text(0.5, 1.035, "", transform=ax.transAxes, ha="center", va="bottom",
                                  color=color, fontsize=11.5)
            ax.set_xlim(0, config["max_mode"])
            ax.set_yscale("log")
            ax.set_ylim(floor, ceiling)
            ax.set_xticks(np.arange(0, config["max_mode"] + 1, 4))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(which="major", color="#d5dce5", linewidth=0.55)
            ax.grid(axis="x", which="minor", color="#e9edf2", linewidth=0.35)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            if i == 0:
                ax.set_title(arm_labels[arm], fontsize=12, pad=40)
            if j == 0:
                ax.set_ylabel(target_labels[target] + "\n" + r"$|\widehat e(\omega)|$", fontsize=13)
            if i == 3:
                ax.set_xlabel(r"Frequency $\omega/\pi$", fontsize=12)
    title = fig.suptitle("", fontsize=20, y=0.986)
    fig.text(0.5, 0.945, f"N = {config['resolutions'][0]} · halo {config['halo']} per side · GD learning rate {config['learning_rate']:g} · relative L₂ above each panel", ha="center", fontsize=11.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.085, top=0.83, hspace=0.35, wspace=0.12)
    fig.text(0.5, 0.026, f"{config['frequency_points']:,} frequency evaluations per curve · residual restricted to [−1, 1], no taper\n"
             "Linear frequency axis, fixed log-magnitude limits; values below 10⁻⁵ meet the display floor", ha="center", fontsize=10)
    destination = output / "spectrum.gif"
    partial = destination.with_suffix(".gif.partial")
    palette = None
    durations = frame_durations(len(steps), config["animation_seconds"])
    with partial.open("wb") as stream:
        for index, step in enumerate(steps):
            title.set_text(f"Residual Fourier spectrum · GD step {step:,} / {config['steps']:,}")
            for key, line in lines.items():
                line.set_ydata(np.maximum(magnitude[key][index], floor))
                labels[key].set_text(f"relative L₂ = {errors[key][step]:.3e}")
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if index in (0, len(steps) - 1):
                rgb.save(previews / f"step_{step:04d}.png")
            palette = write_gif_frame(stream, rgb, palette, int(durations[index]))
            if index % 10 == 0:
                print(f"Rendering {index + 1}/{len(steps)} · step {step}", flush=True)
        stream.write(b";")
    plt.close(fig)
    partial.replace(destination)
    with Image.open(destination) as gif:
        assert gif.n_frames == len(steps)
        total_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()
            assert gif.size == (2400, 1800)
            assert gif.info["duration"] == durations[index]
            total_ms += gif.info["duration"]
        assert total_ms == round(config["animation_seconds"] * 1000)
    print(f"GIF validated: {len(steps)} frames, {total_ms / 1000:g} s, {destination.stat().st_size / 1e6:.1f} MB", flush=True)
