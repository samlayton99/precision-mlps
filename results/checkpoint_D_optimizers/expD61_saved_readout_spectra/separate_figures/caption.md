# Scale acquisition and readout spectrum: 24 functions

Both figures use three initialization panels. Every function is plotted individually.

**Scale acquisition:** restored exactly from the approved original figure. Blue, orange, and purple identify learning rates 0.01, 0.001, and 0.0001; shades distinguish functions within each family. Solid lines show per-run median absolute neuron slopes, dotted boundaries show the central 98% interval, and translucent fills show the interval. QUILL's reference slope is 18.375. Median annotations summarize the 72 final per-run medians in each panel.

**Readout spectrum:** learning rates 0.01, 0.001, and 0.0001 use solid, dashed, and dotted lines, respectively. Viridis colors identify the 10th, 20th, ..., 90th spectral percentiles (50th is the median). Each panel contains 24 functions × 3 learning rates × 9 percentiles. There is no aggregation across functions. Rates below the numerical resolution (5.17e-26) overlap at the bottom line; they are not assigned that value in the saved data. These percentiles describe the 197 readout modes, not confidence intervals or residual-energy distributions.

The 200k-step reference is rho = 2(1 - exp(-1/200000)), approximately 1e-5. Under frozen-readout GD with eta = 1/(2 mu_max), a residual component at this rate takes 200,000 updates to decrease to about 37% of its starting magnitude. This reference is separate from the joint-training learning rates indicated by line style.
