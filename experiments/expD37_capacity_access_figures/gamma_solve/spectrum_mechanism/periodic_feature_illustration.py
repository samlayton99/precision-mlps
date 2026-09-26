"""Real-space illustration of Appendix A's periodic reference, not training."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism"


def periodic(d, gamma=4., length=2.):
    # Paired image sum from the collaborator note, Appendix A.
    result = np.tanh(gamma*d)
    for ell in range(1, 13):
        result = result + (-1)**ell * (
            np.tanh(gamma*(d-ell*length)) + np.tanh(gamma*(d+ell*length)))
    return result


def main():
    gamma, length = 4., 2.
    dense = np.linspace(-3, 3, 2401)
    x = np.linspace(-1, 1, 33)
    centers = -1 + np.arange(16)/8
    test = np.linspace(-3, 3, 301)
    assert np.max(abs(periodic(test+length)+periodic(test))) < 2e-14
    assert np.max(abs(periodic(test+2*length)-periodic(test))) < 2e-14
    actual = np.tanh(gamma*(x[:,None]-centers[None,:]))/np.sqrt(len(x))
    reference = periodic(x[:,None]-centers[None,:])/np.sqrt(len(x))
    fig = plt.figure(figsize=(12.4, 8.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1,1,.045],height_ratios=[.8,1],
                          left=.085,right=.95,bottom=.08,top=.84,wspace=.31,hspace=.43)
    for col, center in enumerate([0., .875]):
        ax = fig.add_subplot(grid[0,col])
        ax.axvspan(-1,1,color=".93",zorder=0)
        ax.plot(dense,np.tanh(gamma*(dense-center)),color="#443983",lw=2,label="Ordinary tanh")
        ax.plot(dense,periodic(dense-center),color="#21918c",lw=2,ls="--",label="Periodic replacement")
        ax.scatter(x,np.tanh(gamma*(x-center)),color="#443983",s=10,zorder=3)
        ax.scatter(x,periodic(x-center),edgecolor="#21918c",facecolor="white",s=15,zorder=4)
        ax.axvline(-1,color=".6",lw=.8);ax.axvline(1,color=".6",lw=.8)
        ax.set(xlim=(-3,3),ylim=(-1.13,1.13),xlabel="Input x",
               title=rf"Center $c={center:g}$; gray region is $[-1,1]$")
        if col == 0: ax.set_ylabel("Feature value, before normalization")
        ax.grid(alpha=.15)
        if col == 0:
            handles,labels=ax.get_legend_handles_labels()
    limit=1/np.sqrt(len(x))
    for col,(matrix,title) in enumerate([(actual,"Actual tanh feature matrix"),
                                        (reference,"Sampled periodic feature matrix")]):
        ax=fig.add_subplot(grid[1,col])
        im=ax.imshow(matrix,origin="upper",aspect="auto",cmap="viridis",vmin=-limit,vmax=limit,
                     extent=(-.5,len(centers)-.5,len(x)-.5,-.5),interpolation="nearest")
        ax.set(xlabel="Neuron column j (centers increase left to right)",title=title,
               xticks=[0,4,8,12,15],yticks=[0,8,16,24,32])
        if col == 0: ax.set_ylabel("Sample row i (inputs increase downward)")
    cb=fig.colorbar(im,cax=fig.add_subplot(grid[1,2]))
    cb.set_label(r"Matrix entry: feature value / $\sqrt{33}$")
    fig.suptitle(r"The periodic reference in real space: $L=2$, period $2L=4$, $\gamma=4$",y=.975,fontsize=14)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.943),ncol=2,frameon=False)
    fig.text(.5,.875,"Dots are sampled entries; lower panels contain 33 samples and 16 core neurons (bias omitted).",
             ha="center",fontsize=10)
    fig.savefig(OUT/"periodic_feature_illustration.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    print(OUT/"periodic_feature_illustration.png")
    print("Entry i=0, j=15: actual =",actual[0,-1],"periodic =",reference[0,-1])


if __name__ == "__main__":
    main()
