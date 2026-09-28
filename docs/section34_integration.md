# Integrating Section 3.4 into submission draft (19)

Insert the proposed subsection after Section 3.3. It uses the existing construction's $h$, $\gamma$, and $\lambda=\gamma h$, and denotes kernel eigenvalues by $\mu_i$. The new theorem is an output-error lower bound for a frozen finite dictionary. The joint-training results are empirical. The full proof belongs in the appendix.

Remove the old Section 4 polynomial training-time argument and its schematic figure. Its statements about hidden-parameter motion and coarse-error equilibrium tracking are not established by the new frozen-readout theorem. The replacement figure uses executed training and separately identified theorem predictions. Renumber subsequent sections and update references to the arithmetic-circuit construction accordingly.

The introduction's second contribution currently claims a proof that gradient descent fails to reach the width-dependent regime because pre-activation parameters move polynomially slowly while the readout absorbs the residual. Replace that claim with the following argument:

> **Optimization access to high precision.** The construction identifies a relative-bandwidth regime whose slopes grow with width. We show that slope also controls readout optimization: smoothing at small slope produces weak kernel directions, and target energy in those directions yields a quantitative output-error lower bound for frozen-feature gradient descent. Matched training experiments test the bound and measure how joint Adam and GD acquire slopes and output accuracy. This separates the existence of accurate MLP weights from the cost of learning them.

The final sentences of the “Precision limits and high-accuracy neural training” related-work paragraph currently cite Section 4 and claim exponential-in-width suppression at constant slope. Replace them with:

> The construction also identifies a relative-bandwidth regime for high precision. Section 3.4 examines its optimization implications: a slope-dependent finite-kernel bound quantifies the output error that remains under frozen-readout gradient descent, while joint-training experiments measure the acquisition of the corresponding feature geometry.

The transition to arithmetic circuits should use the observed acquisition cost to motivate supplying accurate primitives directly. The scalar experiments do not need to establish a theorem about all language-model training procedures for that design choice to follow. Avoid converting a fixed-width experiment into evidence of a width-scaling law, or a finite training budget into permanent impossibility.

The figure's slope panel must be read with its output-error panel. A learned dictionary has heterogeneous slopes and centers, so its RMS or 99th-percentile slope cannot be compared with a uniform dictionary as if the latter supplied a universal accuracy threshold. The readout-restart and slope-intervention diagnostics determine how much of the measured gap can be attributed to the acquired features.

## Positioning relative to established spectral-bias results

The generic mechanism is established prior work. [Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*](https://papers.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf), connect Fourier-feature scale to the kernel spectrum and target-component convergence, including an explicit eigenmode learning law. [Xu et al. (2020), *Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks*](https://www.global-sci.com/cicp/article/view/6896), analyze exponentially frequency-suppressed tanh gradients at small input weights; Section 6 of the [author manuscript](https://arxiv.org/pdf/1901.06523) gives the small-weight gradient and relative loss-decay statements. [Rahaman et al. (2019), *On the Spectral Bias of Neural Networks*](https://proceedings.mlr.press/v97/rahaman19a/rahaman19a.pdf), establish frequency-dependent learning and connect high-frequency acquisition with parameter-scale growth, principally for ReLU networks.

An appropriate related-work sentence is:

> Spectral bias and the effect of feature scale on kernel-mediated learning are established phenomena (Rahaman et al., 2019; Xu et al., 2020; Tancik et al., 2020). Here we quantify this mechanism for QUILL's finite tanh geometry: explicit slope-dependent spectral bounds yield target-weighted output-error lower bounds for frozen-readout GD, which we compare with executed trajectories. The joint-training experiments test whether training acquires a useful geometry in the same high-precision setting.

This positions the finite-center quantitative analysis and its connection to the construction as the contribution. It does not claim that eigenmode-dependent learning, small-slope tanh frequency suppression, or feature-scale effects on kernel spectra are new. This is a check of close antecedents, not an exhaustive novelty assessment.
