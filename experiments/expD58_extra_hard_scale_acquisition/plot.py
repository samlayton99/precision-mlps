"""D58 training plots in the approved D56 format, plus the target gallery."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,measured_data,aggregate_figure
from experiments.expD56_scale_acquisition.model import midpoint_grid
from experiments.expD58_extra_hard_scale_acquisition.targets import TARGETS,target

OUT=ROOT/"results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition"


def target_gallery(destination):
    apply_style()
    x=np.linspace(-1,1,16385)
    train=midpoint_grid(1024)
    fig,axes=plt.subplots(2,4,figsize=(10.2,4.7),sharex=True)
    fig.subplots_adjust(left=.055,right=.99,bottom=.10,top=.80,wspace=.30,hspace=.57)
    for ax,(name,description) in zip(axes.flat,TARGETS.items()):
        y=target(name,x)/np.sqrt(np.mean(target(name,train)**2))
        style_axis(ax)
        ax.plot(x,y,color="#0072B2",lw=1.)
        ax.set_title(description["label"],fontsize=8.2,pad=7)
        ax.set_xlim(-1,1); ax.set_xticks([-1,0,1])
        ax.axhline(0,color="#aaaaaa",lw=.35)
    fig.suptitle("Extra-hard target functions",fontsize=13,fontweight="bold",y=.96)
    fig.text(.52,.88,"Unit training-grid RMS on [−1, 1]",ha="center",fontsize=8,color="#555555")
    destination.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination/"target_functions.png",dpi=300,bbox_inches="tight",pad_inches=.12)
    fig.savefig(destination/"target_functions.pdf",bbox_inches="tight",pad_inches=.12)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=OUT/"sweep")
    p.add_argument("--targets-only",action="store_true");a=p.parse_args()
    if a.targets_only:
        target_gallery(OUT/"targets");print(OUT/"targets");return
    output=a.results.resolve()
    config=json.loads((output/"manifest.json").read_text())["config"]
    data,hashes=measured_data(output)
    partial=len(data)<72 or any(r["step"][-1]<config["steps"] for r in data.values())
    if partial and data: config={**config,"plot_end_step":max(r["step"][-1] for r in data.values())}
    destination=output/"figures";destination.mkdir(exist_ok=True)
    apply_style()
    for field in ("gamma_distribution","eval_rel_l2"):
        aggregate_figure(config,data,destination,False,field,partial,"linear")
    if partial:
        for stem in ("scale_acquisition","fitting_error"):
            for suffix in (".png",".pdf","_displayed.json"):
                shutil.copy2(destination/f"{stem}_PARTIAL{suffix}",destination/f"{stem}{suffix}")
    (destination/"figure_provenance.json").write_text(json.dumps({"partial":partial,"source_sha256":hashes,
        "renderer_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/"experiments/expD56_scale_acquisition/plot.py")},
        "line":"median absolute slope within each run", "band":"central 98% across 196 neurons"},indent=2)+"\n")
    print(destination)


if __name__=="__main__":main()
