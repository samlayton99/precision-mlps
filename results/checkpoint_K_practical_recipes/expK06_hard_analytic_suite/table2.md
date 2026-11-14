**Table 2. QUILL accuracy on 14 additional functions**

| Function | Width | L2 Error | Function | Width | L2 Error |
|:---|---:|---:|:---|---:|---:|
| \(\mathord{b(x)\;\text{(compact bump)}}\) | 2,141 | 7.6e-14 | \(\mathord{E\!\left(\frac{0.999}{1+(10x+1)^2}\right)\;\text{(elliptic)}}\) | 4,225 | 3.0e-15 |
| \(\mathord{S(8x)\;\text{(Fresnel sine)}}\) | 1,089 | 2.2e-15 | \(\mathord{\frac{\log(1+400x^2)}{1+25(x-0.3)^2}}\) | 559 | 1.4e-15 |
| \(\mathord{J_0(60x)\;\text{(Bessel)}}\) | 559 | 2.7e-15 | \(\mathord{\Gamma(2.475x+2.525)}\) | 1,089 | 6.6e-15 |
| \(\mathord{J_6(60x)\;\text{(Bessel)}}\) | 559 | 3.0e-15 | \(\mathord{V(x-0.13;0.025,0.01)\;\text{(Voigt)}}\) | 559 | 1.3e-15 |
| \(\mathord{\operatorname{Si}(80x)\;\text{(sine integral)}}\) | 559 | 9.5e-16 | \(\mathord{{}_1F_1(\tfrac12;1;-150x^2)\;\text{(Kummer)}}\) | 559 | 1.2e-15 |
| \(\mathord{D(20x-2)\;\text{(Dawson)}}\) | 297 | 1.4e-15 | \(\mathord{e^{-40x^2}\sin(16\pi x)}\) | 559 | 4.2e-15 |
| \(\mathord{\operatorname{erf}(12x-1.8)}\) | 169 | 4.0e-15 | \(\mathord{1/\sqrt{x}}\) | 4,815 | 5.0e-14 |

Errors are relative L2. All functions use [-1,1] except 1/sqrt(x), which uses [0.0001,1]. That row uses the earlier 15-block multiscale cardinal-QI construction with a C17 extension, derivative samples, and 30-decimal construction; its exported network is evaluated in Float64. The other rows use sample-fitted uniform QUILL. The compact bump is \(b(x)=e^{-1/(1-(x/0.35)^2)}\) for \(|x|<0.35\), and zero otherwise; it is smooth but nonanalytic.
