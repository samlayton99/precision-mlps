"""Tests for QIMlp and layer types.

Test cases:
- QIMlp forward pass produces correct output shape [batch, 1]
- GammaLinear: output = gamma * (x - centers)
- GammaExpLinear: output = exp(log_gamma)/h * (x - centers)
- StandardLinear: output = x @ weight + bias
- get_gamma() returns correct effective gamma for each layer type
- get_centers() returns correct centers
- features(x) returns Phi with correct shape [n_points, width]
- Model parameters are float64
- get_layer factory returns correct layer type
"""

# TODO: implement
