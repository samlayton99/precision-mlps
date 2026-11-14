# Scale acquisition and readout rates across 24 functions

Each column uses a different initialization scale; colors identify the joint-training learning rate, and matching shades identify functions in both rows. All 216 runs use 196 tanh neurons and 200,000 full-batch GD updates.

**Top:** median absolute neuron slope and central 98% interval for each run. Dotted lines mark the 1st and 99th percentiles. The QUILL reference is gamma = 18.375. Each median annotation summarizes the 72 final per-run medians in its panel.

**Bottom:** the 90th percentile of the normalized readout rates, rho_i = sigma_i^2 / sigma_max^2, computed from the actual hidden geometry at each saved checkpoint. Each curve is one run, with no pooling across functions. Rates below the numerical resolution (5.17e-26) are displayed on the bottom line. These are spectral percentiles, not confidence intervals or residual-energy percentiles.

The dashed 200k-step reference is rho = 2(1 - exp(-1/200000)), approximately 1e-5. At the theorem's reference frozen-readout GD step size, eta = 1/(2 mu_max), this rate takes 200,000 updates to reduce a residual component to exp(-1), or about 37%, of its starting magnitude. This reference is separate from the joint-training learning rates in the legend.

At the final checkpoint, every run's 90th-percentile rate is below 2.8e-9. Training keeps scales small while slow readout directions persist; the spectral curves alone do not measure how much residual error occupies those directions.
