"""Per-seed physical w/gamma movies from detached rate-study exports."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .animate_parameters import render
from .run import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--target", choices=["sine", "quadratic", "mixed"], default="sine")
    parser.add_argument("--labels", nargs="+", default=["high_shared", "high_both_changes"])
    args = parser.parse_args()
    records = json.loads((args.analysis / "evidence.json").read_text())
    plt.rcParams.update({"font.size": 10, "animation.embed_limit": 100., "animation.frame_format": "png"})
    for label in args.labels:
        pair = sorted([r for r in records if r["label"] == label and r["case"]["n"] == args.n
                       and r["case"]["target"] == args.target], key=lambda r: r["case"]["seed"])
        if [r["case"]["seed"] for r in pair] != [0, 1]:
            raise ValueError(f"Need both seeds for {label}, {args.n}, {args.target}")
        histories, dense, sources = [], [], []
        for record in pair:
            folder = args.analysis / label / Path(record["folder"]).name
            with np.load(folder / "history.npz") as history, np.load(folder / "dense_parameters.npz") as detail:
                histories.append({k: history[k] for k in history.files if k != "centers"})
                dense.append({k: detail[k] for k in detail.files})
                centers = history["centers"]
            sources.append(str(folder))
        common = np.intersect1d(histories[0]["step"], histories[1]["step"])
        histories = [{k: v[np.searchsorted(h["step"], common)] for k, v in h.items()} for h in histories]
        output = args.analysis / "animations" / f"N{args.n}_{args.target}_{label}"
        output.mkdir(parents=True, exist_ok=True)
        provenance = {"sources": sources, "row_order": ["physical w", "physical gamma"], "frames": {}}
        for seed in (0, 1):
            players = []
            for mode, data in [("overview", histories), ("late", dense)]:
                player, steps = render(centers, data, seed, mode, output,
                                       title=f"{args.target}, N={args.n}, {label.replace('_', ' ')}")
                players.append(player)
                provenance["frames"][f"s{seed}_{mode}"] = steps
            html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            html += f'<title>{label}: seed {seed}</title><style>body{{font:16px system-ui;max-width:1200px;margin:auto}} img{{max-width:100%}} button{{padding:5px}}</style><body>'
            html += f'<h1>{label.replace("_", " ")}, seed {seed}: physical readouts and slopes</h1>'
            html += ''.join(players) + '</body></html>'
            (output / f"seed_{seed}.html").write_text(html)
        write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
