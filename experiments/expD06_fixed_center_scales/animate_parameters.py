"""Animate saved physical parameters at their fixed centers; no training or refit."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


SOURCE_STEP = 320000
END_STEP = 1680000
DENSE_START = END_STEP - SOURCE_STEP - 2048


def load_trajectories(root):
    """Align seeds only at actual common checkpoints; retain signed parameters."""
    with np.load(root / "stall_analysis/construction.npz") as reference:
        centers = reference["centers"].copy()
    overview, late, inputs = [], [], []
    for seed in [0, 1]:
        paths = [root / "stall_analysis" / f"adam_scaled_0.001_s{seed}_history.npz",
                 root / "stall_continuation_analysis" / f"joint_decay_s{seed}_history.npz"]
        inputs.extend(paths)
        with np.load(paths[0]) as first, np.load(paths[1]) as second:
            assert second["step"][0] == SOURCE_STEP
            for field in ["c", "gamma"]:
                np.testing.assert_array_equal(first[field][-1], second[field][0])
            history = {k: np.concatenate([first[k], second[k][1:]]) for k in ["step", "c", "gamma"]}
        assert history["step"][-1] == END_STEP
        overview.append(history)
        fragments = []
        for path in sorted((root / "animations/input" / f"joint_decay_s{seed}").glob("dense_*.npz")):
            inputs.append(path)
            with np.load(path) as data:
                mask = (data["step"] >= DENSE_START) & (data["step"] < END_STEP-SOURCE_STEP)
                if mask.any():
                    fragments.append({k: data[k][mask] for k in ["step", "c", "gamma"]})
        dense = {k: np.concatenate([f[k] for f in fragments]) for k in ["step", "c", "gamma"]}
        np.testing.assert_array_equal(dense["step"], np.arange(DENSE_START, END_STEP-SOURCE_STEP))
        dense["step"] += SOURCE_STEP
        late.append(dense)
    common = np.intersect1d(overview[0]["step"], overview[1]["step"])
    overview = [{k: v[np.searchsorted(h["step"], common)] for k, v in h.items()} for h in overview]
    for history in overview + late:
        assert history["c"].shape == (len(history["step"]), len(centers)+1)
        assert history["gamma"].shape == (len(history["step"]), len(centers))
        assert np.all(np.diff(history["step"]) > 0)
        assert np.isfinite(history["c"]).all() and np.isfinite(history["gamma"]).all()
    return centers, overview, late, inputs


def player_html(animation, fps):
    """Use Matplotlib's frame slider and playback, with offline text buttons."""
    html = animation.to_jshtml(fps=fps, default_mode="once")
    html = re.sub(r'<link\b[^>]*>', '', html)
    labels = {"minus": "Slower", "fast-backward": "First", "step-backward": "Previous",
              "play fa-flip-horizontal": "Reverse", "pause": "Pause", "play": "Play",
              "step-forward": "Next", "fast-forward": "Last", "plus": "Faster"}
    for icon, label in labels.items():
        html = html.replace(f'<i class="fa fa-{icon}"></i>', label)
    return html


def render(centers, histories, seed, mode, output):
    dense = mode == "late"
    history = histories[seed]
    # The late movie shows every eighth saved state, plus the final state.
    indices = np.unique(np.r_[np.arange(0, len(history["step"]), 8), len(history["step"])-1]) if dense else np.arange(len(history["step"]))
    if not dense:
        # At 6 fps, hold each early checkpoint for one full second.
        indices = np.repeat(indices, np.where(history["step"][indices] <= SOURCE_STEP, 6, 1))
    fields = ["c", "gamma"]
    # Match each parameter's vertical scale across the two seed movies.
    paired = [[h[field][:, 1:] if field == "c" else h[field] for h in histories] for field in fields]
    if dense:
        paired = [[v-v[0] for v in pair] for pair in paired]
    values = [pair[seed] for pair in paired]
    bounds = [max(float(np.max(np.abs(v))) for v in pair)*1.12 for pair in paired]
    thresholds = [1e-7, 1e-4] if dense else [.01, .1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    core = np.abs(centers) <= 1
    points, annotations = [], []
    for row, (field, ax) in enumerate(zip(fields, axes)):
        title = "Readout weights w" if field == "c" else "Physical slopes gamma"
        ylabel = ("Change in w" if field == "c" else "Change in gamma") if dense else ("Physical w" if field == "c" else "Physical gamma")
        ax.axvspan(centers[0]-.01, -1, color=".93")
        ax.axvspan(1, centers[-1]+.01, color=".93")
        ax.axhline(0, color=".65", lw=.8)
        points.append([ax.scatter(centers[mask], values[row][0, mask], s=12 if label=="Core" else 24,
                                  marker=marker, color=color, label=label, linewidths=0)
                       for mask, label, color, marker in [(core, "Core", "#2466a6", "o"),
                                                          (~core, "Halo", "#c15a14", "^")]])
        ax.set(ylim=(-bounds[row], bounds[row]), xlim=(centers[0]-.025, centers[-1]+.025),
               ylabel=ylabel, title=title)
        ax.set_yscale("symlog", linthresh=thresholds[row])
        ax.grid(alpha=.18)
        ax.legend(loc="lower right", fontsize=9)
        annotations.append(ax.text(.015, .94, "", transform=ax.transAxes, va="top", fontsize=10))
    axes[-1].set_xlabel("Fixed physical center x (each neuron stays at the same horizontal position)")
    heading = fig.suptitle("")
    footer = f"Fixed signed-log scales; linear near zero (w: +/- {thresholds[0]:g}; gamma: +/- {thresholds[1]:g}).\n"
    footer += f"Change since update {int(history['step'][0]):,}; every 8 updates, no interpolation." if dense else "Actual checkpoints, unequal update gaps; no interpolation. First 320k: 1 checkpoint/s; later: 6/s."
    fig.supxlabel(footer, fontsize=9)

    def update(frame):
        i = indices[frame]
        step = int(history["step"][i])
        phase = "late movement magnified" if dense else "full training overview"
        eta = .001 if step <= SOURCE_STEP else 1e-6+.5*(.001-1e-6)*(1+np.cos(np.pi*min((step-SOURCE_STEP)/80000, 1)))
        heading.set_text(f"Seed {seed} — {phase}\nScaled training, Adam  |  update {step:,}  |  shared LR {eta:.3g}")
        for row, field in enumerate(fields):
            for dots, mask in zip(points[row], [core, ~core]):
                dots.set_offsets(np.column_stack([centers[mask], values[row][i, mask]]))
            if field == "c":
                bias = history["c"][i, 0]
                db = bias-history["c"][0, 0]
                annotations[row].set_text(f"Global bias b = {bias:+.6g}"+(f"; change = {db:+.3g}" if dense else ""))
            elif dense:
                annotations[row].set_text(f"Change since update {int(history['step'][0]):,}")
        return [heading, *annotations, *[p for row in points for p in row]]

    fps = 20 if dense else 6
    animation = FuncAnimation(fig, update, frames=len(indices), interval=1000/fps, blit=False, repeat=False)
    name = f"seed_{seed}_{mode}"
    animation.save(output/f"{name}.mp4", writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=1600,
                    extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
    html = player_html(animation, fps)
    for label, frame in [("first", 0), ("middle", len(indices)//2), ("last", len(indices)-1)]:
        update(frame)
        fig.savefig(output/f"{name}_{label}.png", dpi=110)
    plt.close(fig)
    return html, history["step"][indices].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local D06 result directory containing both analysis exports and downloaded dense states")
    args = parser.parse_args()
    output = args.root/"animations"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "animation.embed_limit": 100., "animation.frame_format": "png"})
    centers, overview, late, inputs = load_trajectories(args.root)
    provenance = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "inputs": {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  "centers": centers.tolist(), "signed_parameters": True, "interpolation": False,
                  "dense_saved_steps": len(late[0]["step"]),
                  "early_playback": {"through_step": SOURCE_STEP, "seconds_per_checkpoint": 1},
                  "row_order": ["physical readout w", "physical slope gamma"], "frames": {}}
    for seed in [0, 1]:
        name = f"seed_{seed}"
        players = []
        for mode, histories in [("overview", overview), ("late", late)]:
            player, steps = render(centers, histories, seed, mode, output)
            provenance["frames"][name+"_"+mode] = steps
            caption = "Full training overview" if mode == "overview" else "Late movement, magnified"
            players.append(f'<section><h2>{caption}</h2>{player}<p><a href="{name}_{mode}.mp4">Download MP4</a></p></section>')
        document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        document += f'<title>Seed {seed}: readouts and gammas</title><style>body{{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;color:#18212b;background:white}} section{{margin-bottom:48px}} .animation img{{max-width:100%;height:auto}} .anim-controls{{max-width:100%}} .anim-buttons button{{min-width:58px;font:inherit;padding:5px}} h1,h2{{font-weight:500}} .anim-slider{{width:95%}}</style>'
        document += f'<body><h1>Seed {seed}: readouts and gammas at fixed centers</h1>'+''.join(players)+'</body></html>'
        (output/f"{name}.html").write_text(document)
        print(f"Saved {output/name}.html", flush=True)
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
