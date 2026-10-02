# ReLU readout experiment

Fixed centers, no training. The model includes an explicit affine term so neuron coefficients describe curvature without needing to encode an arbitrary linear trend. Every solve fits function values in continuous L2, never derivatives.

63 equally spaced interior knots on [-1,1], h=1/32; activation gamma=1. The plotted purple values are w_j/h, not raw w_j. For arbitrary positive gamma, the meaningful slope jump is gamma*v_j; changing gamma does not change a ReLU's spatial width.

A ReLU spline has distributional second derivative sum_j w_j delta(x-c_j). Thus w_j/h can approach f''(c_j) as a sampled curvature density, although the model's pointwise second derivative is zero between knots. Equality is not claimed for finite-spacing least squares.

| Target | Relative function L2 error | Relative readout-curvature l2 error | Max readout-curvature error |
|---|---:|---:|---:|
| Sine | 0.00144326 | 0.00158373 | 0.240954 |
| Runge | 0.00114087 | 0.000303217 | 0.0206036 |
| Mixed sine | 0.00615482 | 0.00517335 | 3.63306 |
| Quadratic | 0.00016276 | 1.07021e-12 | 7.29905e-12 |

Verification: independent local hat-basis solve uses the exact finite-element mass matrix, transforms nodal values to slope jumps, and compares against dense ReLU QR/SVD least squares. Separate order-48 quadrature checks integration. A gamma=4 solve verifies gamma*v invariance. Full numerical differences are in results.json.

The exact relation w_j=(s_{j+1}-2s_j+s_{j-1})/h uses solved nodal values s, not sampled target values. Replacing s by f(c_j) gives interpolation coefficients, a different optimization problem. Boundary effects are included in all plotted coefficients.
