"""Plot the retained frozen-sine weights; no report generation."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    with (root/'measurements.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    packs = np.load(root/'coefficients.npz')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    n, lam = 128, .5
    h = 2/n
    centers = -1+h*np.arange(-24, 153)
    styles = [('gd', .002, '#2563eb', 'o'), ('adam', .002, '#dc2626', 's'),
              ('adam', .0002, '#059669', '^')]
    for opt, eta, color, marker in styles:
        key = 'gd_n128_l0.5_s600000' if opt=='gd' else f'adam_n128_l0.5_e{eta}_s600000'
        axes[0].plot(centers, packs[key][:-1]/h, color=color, lw=1.2, label=f'{opt.upper()}, rate {eta}')
        selected = sorted((r for r in rows if r['optimizer']==opt and float(r['eta'])==eta
                           and float(r['lam'])==lam and int(r['step'])==600000), key=lambda r:int(r['n']))
        axes[1].plot([int(r['n']) for r in selected], [float(r['interior_rms_over_h']) for r in selected],
                     color=color, marker=marker, ms=6, lw=1.2, fillstyle='none', label=f'{opt.upper()}, rate {eta}')
    z = np.pi*np.pi/32
    axes[0].plot(centers, np.pi*np.sinh(z)/z*np.cos(2*np.pi*centers), '--', color='black', lw=1,
                 label='Construction interior density')
    for a,b in [(-1.375,-1),(1,1.375)]:
        axes[0].axvspan(a,b,color='gray',alpha=.12)
    axes[0].set(xlabel='Fixed feature center', ylabel='Physical readout / h',
                title='Sine, N=128, gamma=32; 600k updates', xlim=(-1.375,1.375))
    axes[0].legend(fontsize=7, loc='lower center', ncol=2)
    axes[1].set(xlabel='Core intervals N (h=2/N)', ylabel='Interior readout RMS / h',
                title='Refine h at fixed gamma h=0.5', xticks=[64,128,256])
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/'readout_scale.png', dpi=180)
    fig.savefig(root/'readout_scale.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
