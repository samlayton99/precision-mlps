"""All smooth targets on one measured QUILL error-versus-width plot."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/precisionmlps-mpl")
import csv
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD59_quill_smooth_scaling.run import OUT

LABELS={"square":r"$x^2$","cubic":r"$x^3$","exponential":r"$e^x$",
        "sine_pi":r"$\sin(\pi x)$","cosine_pi":r"$\cos(\pi x)$","sine_4pi":r"$\sin(4\pi x)$",
        "legendre_5":r"$P_5(x)$","runge_25":"Runge (25)","gaussian_25":"Shifted Gaussian",
        "softplus_5":"Softplus","mixed_sine":"Mixed sine","chirp":"Chirp","legendre_12":r"$P_{12}(x)$",
        "runge_100":"Runge (100)","tanh_20":"Steep tanh","localized_oscillation":"Localized oscillation",
        "erf_transition":"Erf transition","gamma_shifted":"Gamma","airy_transition":"Airy",
        "smooth_bump":"Smooth compact bump"}


def main():
    config=json.loads((OUT/"manifest.json").read_text())["config"]
    data_path=OUT/"scaling_scalar.csv"
    with data_path.open() as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(10.3,6.0))
    plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans","pdf.fonttype":42})
    fig.subplots_adjust(left=.095,right=.985,top=.88,bottom=.30)
    palette=plt.colormaps["tab20"](np.arange(20))
    # Alternate the two halves of tab20 so adjacent legend items do not all pair.
    palette=palette[np.r_[np.arange(0,20,2),np.arange(1,20,2)]]
    displayed={}
    for name,color in zip(config["targets"],palette):
        selected=sorted([r for r in rows if r["target"]==name],key=lambda r:int(r["width"]))
        x=np.array([int(r["width"]) for r in selected]);y=np.array([float(r["relative_l2"]) for r in selected])
        ax.plot(x,y,color=color,lw=1.25,marker="o",markersize=2.5,markeredgewidth=0,label=LABELS[name])
        displayed[name]={"width":x.tolist(),"relative_l2":y.tolist(),"color":color.tolist()}
    ax.set(yscale="log",xlabel="Hidden width (including halos)",ylabel=r"Relative $L^2$ error")
    ax.set_xlim(0,max(int(r["width"]) for r in rows)*1.02)
    ax.set_xticks([0,256,512,1024,1536,2048])
    errors=[float(r["relative_l2"]) for r in rows]
    ax.set_ylim(min(3e-16,min(errors)*.5),max(2,max(errors)*1.2))
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=18))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=80))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both",labelsize=8,length=3,width=.85)
    ax.set_axisbelow(True);ax.grid(axis="y",which="major",color="#D8D8D8",lw=.45)
    for side in ("top","right"):ax.spines[side].set_visible(False)
    for side in ("left","bottom"):ax.spines[side].set_linewidth(1.)
    fig.suptitle("QUILL Accuracy Scaling: 20 smooth functions",fontsize=14,fontweight="bold",y=.96)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.54,.015),ncol=4,frameon=False,
               fontsize=8,handlelength=2.,columnspacing=1.5,labelspacing=.7)
    destination=OUT/"figures";destination.mkdir(exist_ok=True)
    fig.savefig(destination/"quill_scaling.png",dpi=300,bbox_inches="tight",pad_inches=.12)
    fig.savefig(destination/"quill_scaling.pdf",bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    (destination/"figure_provenance.json").write_text(json.dumps({"source":str(data_path.relative_to(ROOT)),"source_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"displayed":displayed,
        "metric":"Relative RMS on 65536 uniform midpoint samples, actual saved MLP",
        "width":"All hidden neurons including both halos","smoothing":"none"},indent=2)+"\n")
    print(destination/"quill_scaling.png")


if __name__=="__main__":main()
