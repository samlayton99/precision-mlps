# Checked arithmetic tolerance and remaining computation obligations

`NumericalPrecision.rounded_network_error` treats the actual dense shallow evaluator. If hidden weights, hidden biases, readouts, output bias, individual activation evaluations, and final summation each have absolute error at most δ≤1, input coordinates satisfy |x_k|≤R, exact activations have magnitude at most B, their relevant Lipschitz bound is L, and exact readout absolute mass is at most V, then output error is at most

`δ [2 + B W + (V+W)(L(dR+1)+1)]`.

The hidden-parameter contribution `(dR+1)δ` is derived from the actual dot products. The proof propagates every listed perturbation through the actual finite sum, including products between perturbed coefficients and features.

`workingBits_accuracy` proves that `b=ceil(log(C/ε)/log 2)` suffices for `C 2^(-b)≤ε`, for positive amplification C and tolerance ε. `workingBits_bound` gives the finite upper bound `b<max(0,log(C/ε)/log 2)+1`. These are absolute-accuracy bits; coefficient dynamic range and the acquisition of certified inputs still matter. This is not a statement about a fixed hardware mantissa or an implemented arbitrary-precision evaluator.

`readoutDensity_bound` derives the actual density bound from the spectral envelope:

`|ρ_B(x)| ≤ integral|pHat| · exp(A L)/c`.

`normalized_spoke_readout_mass` shows that normalized positive angular weights remove an artificial factor M from the raw readout mass: the bound is rowWidth times rowCoefficientScale times densityBound. Anchor contributions are to be added separately.

Still required for full T8: propagate target-sample errors through finite Chebyshev preparation and anchor evaluation; certify quadrature for profile/density acquisition and quadrature-node errors; and count the computation needed to obtain those inputs. The checked evaluator tolerance is one component of this numerical construction theorem, not a substitute for it. Exact-real neuron counts must remain separate from an end-to-end bit-complexity claim.
