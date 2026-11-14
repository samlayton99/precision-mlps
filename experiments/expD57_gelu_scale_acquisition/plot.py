"""Reuse the approved D56 format, with explicit GELU/optimizer identification."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style, measured_data, aggregate_figure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"results/checkpoint_D_optimizers/expD57_gelu_scale_acquisition/sweep")
    args=parser.parse_args()
    output=args.results.resolve()
    apply_style()
    for optimizer in ("gd","adam"):
        folder=output/optimizer
        config=json.loads((folder/"manifest.json").read_text())["config"]
        data,hashes=measured_data(folder)
        partial=len(data)<18 or any(row["step"][-1]<config["steps"] for row in data.values())
        if partial and data:
            config={**config,"plot_end_step":max(row["step"][-1] for row in data.values())}
        destination=folder/"figures"
        destination.mkdir(exist_ok=True)
        for field in ("gamma_distribution","eval_rel_l2"):
            aggregate_figure(config,data,destination,False,field,partial,"linear")
        if partial:
            # Stable artifact paths always show the latest explicitly labeled progress,
            # then receive the complete 200k figures at the end of the sweep.
            for stem in ("scale_acquisition","fitting_error"):
                for suffix in (".png",".pdf","_displayed.json"):
                    shutil.copy2(destination/f"{stem}_PARTIAL{suffix}",destination/f"{stem}{suffix}")
        write={"activation":"exact GELU","optimizer":optimizer,"partial":partial,"source_sha256":hashes,
               "renderers_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__),ROOT/"experiments/expD56_scale_acquisition/plot.py")},
               "band":"central 98% of absolute slopes within each function/run", "line":"median",
               "reference":"GELU lambda=0.707, h=2/147; gamma=51.9645. Construction reference, not necessary accuracy."}
        (destination/"figure_provenance.json").write_text(json.dumps(write,indent=2)+"\n")
        print(destination,flush=True)


if __name__=="__main__": main()
