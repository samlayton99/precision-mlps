# Empirical support for the small-scale regime

We tested 24 functions with width-196 tanh networks, three learning rates, and default, 2x, and 4x first-layer initialization. All 216 GD runs were trained for 200,000 steps.

The median neuron slopes remain close to their initial values. Even the upper edge of the central 98% interval stays below 7.32 throughout all runs, compared with the QUILL reference scale of 18.375. Increasing initialization scale shifts the starting distribution but does not produce population-wide growth toward the construction scale. This supports the empirical relevance and persistence of the limited-scale regime studied by the theory.

The construction scale is a reference, not a necessary accuracy threshold. The theorem's additional concentration and tracking conditions require separate checks.

[24-function scale-acquisition plot](../../../results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition/combined_24_functions/scale_acquisition.png)
