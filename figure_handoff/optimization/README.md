# Optimization figures: style handoff

Start with [render.py](render.py). It regenerates the paper's Figures 3 and 4 from 370 kB of saved plotting arrays using NumPy and Matplotlib on a CPU. Typography and colors are collected at the top of the script; each figure has its own drawing function. The bundled PNGs reproduce the research originals pixel-for-pixel in the recorded environment.

## Rebuild

From the repository root, using Python 3.12 or newer:

```sh
python3.12 -m venv figure_handoff/optimization/.venv
figure_handoff/optimization/.venv/bin/python -m pip install -r figure_handoff/optimization/requirements.txt
figure_handoff/optimization/.venv/bin/python figure_handoff/optimization/render.py
```

This writes 600-DPI PNGs and vector PDFs to `figure_handoff/optimization/rendered/`. Pass `--output /path/to/figures` to choose another directory. The committed reference exports remain in [figures/](figures/).

![Figure 3: target-relevant kernel ratios and frozen-readout learning](figures/spectrum_readout.png)

![Figure 4: acquired slopes, output error, and population bandwidth](figures/joint_acquisition.png)

## Where to change the style

| Entry point | Controls |
| --- | --- |
| `STYLE` | Font family and sizes, title weight, tick sizing, axes, and PDF font embedding. |
| `OPTIMIZER_COLORS`, `BANDWIDTH_COLORS` | Consistent optimizer and bandwidth palettes. |
| `spectrum()` | Figure 3's two panels, direct rank labels, local legends, markers, and axis limits. |
| `acquisition()` | Figure 4's three panels, shared lower legend, neuron shading, and layout. |
| `DPI` and `figsize` | Raster resolution and physical figure dimensions. |

The current styling uses outlined markers, light horizontal grids, and no top/right spines. Figure 3 uses local legends because its panels encode different quantities. Figure 4 uses a shared legend below the panels because optimizer colors recur throughout. Direct labels identify the construction references and selected spectral ranks. Figure 3's design size is 5.5 × 2.0 inches; Figure 4's is 5.5 × 2.2 inches before tight cropping.

For style revisions, edit the drawing functions and shared settings, then rerun the command. Check the exports at the paper's text width, especially legends, reference labels, and shaded regions.

## What the plotted quantities mean

**Figure 3, left:** kernel eigenvalue ratios $\mu_i/\mu_1$ and theorem intervals at ranks 4, 14, and 30, for bandwidths $1/32$, $1/16$, $1/8$, and $1/4$. These ranks carry substantial target energy; the eigenvectors themselves change with bandwidth. Shading is a theorem interval.

**Figure 3, right:** executed frozen-readout GD error and the square root of its output-error lower bound. Solid lines and hollow markers indicate execution; dashed lines indicate the bound. The horizontal axis includes all five million updates and uses a linear region near zero followed by a logarithmic region.

**Figure 4, left:** final mean absolute slope versus total width, including halos. Faint curves show five seed means; outlined markers show the pooled mean. The dotted reference is the supplied $\lambda=1/4$ construction.

**Figure 4, middle:** width-512 relative training RMS error. Solid curves are joint training; dashed curves are frozen-readout training on the supplied dictionary. Blue is Adam and orange is GD. Joint curves summarize the five seeds after time binning; their shading retains seed and within-bin extrema, including transient spikes. Frozen Adam's shading retains within-bin extrema. These are executed trajectories.

**Figure 4, right:** mean bandwidth $\lambda=h_W|\gamma|$ for the same joint runs. Shading is one population standard deviation across all neurons of the five equal-sized seeds, with the lower edge clipped at zero. It measures neuron heterogeneity, not uncertainty in the population mean. The dotted supplied bandwidth is a comparison scale, not a universal necessary threshold.

Both figures use the normalized mixed-sine target $[\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(14\pi x)]/\sqrt{21/32}$ on $[-1,1]$. Keep the meaning of the bands and line styles consistent with the captions when changing the visual design.

## Data and provenance

- [data/spectrum_readout.npz](data/spectrum_readout.npz) contains the exact displayed ratios and intervals, GD curve samples and marker samples, and bound curves. Arrays indexed by bandwidth follow `bandwidths`; ratio rows follow `ranks`.
- [data/joint_acquisition.npz](data/joint_acquisition.npz) contains the exact displayed slope means, error summaries and extrema, population bandwidth moments, and frozen comparisons. Names are prefixed by optimizer. All `*_steps` arrays use raw update counts; the renderer converts Figure 4 to millions.
- [sources/section34_paired_figures.py](sources/section34_paired_figures.py) and [sources/figure4_figure.py](sources/figure4_figure.py) preserve the original research plotting programs, including their input checks and data aggregation. They are included to show how the plotted evidence was assembled; the standalone `render.py` is the entry point for this bundle.
- [provenance.json](provenance.json) records the research snapshot and file hashes. The original provenance records in `sources/` identify the full study inputs; those historical paths are not dependencies of the standalone renderer.
- [verification.json](verification.json) records the environment and comparison against the original PNGs. PDF metadata includes creation times, so equivalence was checked through raster pixels rather than PDF byte hashes.

The renderer consumes existing scientific results. Regenerating these figures does not train models, repeat recipe selection, or recompute theorem bounds.
