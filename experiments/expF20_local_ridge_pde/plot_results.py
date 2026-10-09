"""Figures and a machine-readable collection of the saved expF20 runs."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF20_local_ridge_pde'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mplconfig'))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(name):return json.loads((OUT/name).read_text())
def legend(ax):ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=8)
def style(ax):ax.grid(True,which='major',alpha=.2);ax.spines[['top','right']].set_visible(False)


def main():
    files=sorted(OUT.glob('*.json'))
    with (OUT/'metrics.jsonl').open('w') as handle:
        for file in files:handle.write(json.dumps(dict(artifact=file.name,**json.loads(file.read_text())))+'\n')
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.labelsize':10})
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    a=axes[0,0]
    rows=sorted([load(f.name) for f in OUT.glob('operator_m128_*.json')],key=lambda d:d['neurons'])
    x=[r['neurons'] for r in rows]
    for order,label in [(0,'Value'),(2,'Laplacian')]:
        a.semilogx(x,[r['products'][order]['forward_seconds']*1000 for r in rows],'-o',label=label)
    a.set(xlabel='Total neurons (128 directions fixed)',ylabel='Forward time at 1,024 points (ms)',ylim=(0,35))
    a.set_title('Center refinement barely changes evaluation time',pad=48);legend(a)
    paths=[('free_raw_m128_n8193_k1024_i100_seed813.json','Raw, linear','#b34739'),
           ('free_smooth_m128_n8193_k1024_i200_seed813.json','Smooth, linear','#287c8e'),
           ('free_smooth_m128_n8193_k8192_i200_seed813.json','Smooth, 8,192 points','#579e54'),
           ('free_smooth_m128_n8193_k1024_i200_seed813_nonlinear.json','Smooth, nonlinear','#7b5b9f')]
    for file,label,color in paths:
        d=load(file);hist=d['history'];its=[r['iterations'] for r in hist]
        axes[0,1].semilogy(its,[r['fresh_relative_l2'] for r in hist],'-o',label=label,color=color)
        axes[1,0].semilogy(its,[r['fresh_pde_rms'] for r in hist],'-o',label=label,color=color)
    axes[0,1].set(xlabel='Cumulative inner iterations',ylabel='Held-out relative field error',ylim=(1e-6,1))
    axes[0,1].set_title('1,048,704 independent neuron readouts',pad=48);legend(axes[0,1])
    axes[1,0].set(xlabel='Cumulative inner iterations',ylabel='Held-out PDE residual RMS',ylim=(1e-4,10))
    axes[1,0].set_title('Small field error still leaves derivative error',pad=48);legend(axes[1,0])
    a=axes[1,1]
    reduced=load('reduced_million.json')['rows']+load('reduced_million_refinement.json')['rows']
    for freq,label in [(1.3,'Smooth target'),(5.,'More oscillatory target')]:
        rs=sorted([r for r in reduced if r['target_frequency']==freq and '+u^3' in r['equation']],key=lambda r:r['independent_coordinates'])
        a.semilogy([r['independent_coordinates'] for r in rs],[r['heldout_field_relative_l2'] for r in rs],'-o',label=label)
    a.set(xlabel='Independent function coefficients',ylabel='Held-out relative field error',ylim=(1e-15,1))
    a.set_title('1,081,600 neurons, reduced nonlinear solves',pad=48);legend(a)
    for ax in axes.ravel():style(ax)
    fig.suptitle('Local ridge PDE feasibility: evaluation scales; solution selection matters',fontsize=14,y=.995)
    fig.text(.5,.01,'Float64 CPU. Free solves start at zero; no interior solution labels. Timings include local machine load.\nFree solves use 50-step LSMR chunks; reduced solves use dense QR in at most 276 coordinates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.97),h_pad=3.5,w_pad=2.5)
    fig.savefig(OUT/'feasibility.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    rows=[load(f'operator_m{m}_n{n}_k1024.json') for m,n in [(16,65537),(128,8193),(512,2049)]]
    for key,label in [('forward_seconds','Forward'),('adjoint_seconds','Transpose')]:
        axes[0].loglog([r['directions'] for r in rows],[r['products'][2][key]*1000 for r in rows],'-o',label=label)
    axes[0].set(xlabel='Directions (about 1.05 million neurons fixed)',ylabel='Laplacian operation time (ms)')
    axes[0].set_title('Direction count remains a scaling cost',pad=42);legend(axes[0])
    for order,label in [(0,'Value'),(2,'Laplacian')]:
        rs=sorted([load(f.name) for f in OUT.glob('operator_m128_*.json')],key=lambda r:r['neurons'])
        axes[1].loglog([r['neurons'] for r in rs],[r['dense_audit'][order]['speedup'] for r in rs],'-o',label=label)
    axes[1].set(xlabel='Neurons (128 directions fixed)',ylabel='Dense / local measured time, 16 points')
    axes[1].set_title('Speedup over direct all-center evaluation',pad=42);legend(axes[1])
    for ax in axes:style(ax)
    fig.tight_layout();fig.savefig(OUT/'operator_scaling.png');plt.close(fig)


if __name__=='__main__':main()
