**Table 1. High-precision approximation of difficult functions**

| Function on [-1,1] | Adam-trained MLP | Gated NPU | QUILL |
|:---|---:|---:|---:|
| \(\mathord{\operatorname{Ai}(-30x)\;\text{(Airy)}}\) | 7.4e-01 | 7.3e-01 | **6.1e-15** |
| \(\mathord{y-0.9\sin y=\pi x\;\text{(Kepler inversion)}}\) | 5.3e-04 | 4.0e-04 | **3.5e-15** |
| \(\mathord{K\!\left(\frac{0.99}{1+9x^2}\right)\;\text{(elliptic)}}\) | 1.4e-03 | 6.2e-03 | **6.7e-16** |
| \(\mathord{\frac{e^{-2x^2}\cos(30x+12x^2)}{1+100(x-0.2)^2}}\) | 1.0e-01 | 9.3e-02 | **1.6e-15** |

Independent test relative L2 errors across six hidden widths. MLP: 50,000 Adam steps. NPU: 50,000 Adam steps followed by up to 1,000 full-batch L-BFGS iterations; no readout refitting or RealNPU. All methods use Float64 and identical sample locations and function values at each width. Selected widths may differ between methods. Displayed NPU models are selected by the smallest sampled maximum absolute residual across the six widths, including probes near input zero crossings; their independent relative L2 errors are reported here. The Airy model uses the composed gated NPU variant. Adam and QUILL models retain validation-L2 selection.
