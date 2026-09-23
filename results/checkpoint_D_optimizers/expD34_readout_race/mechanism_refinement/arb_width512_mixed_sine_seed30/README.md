# A non-moment rigorous prefix, and a failed longer certificate

For `mixed_sine`, seed 30, physical width $W=705$ and $N_{\rm ref}=512$, the unchanged Arb enclosure proves

$$
\max_j\lambda_j(n)<0.002,
\qquad 0\le n\le13{,}000,
\qquad \lambda_j=|a_j|/256,
$$

starting from the archived update-20,000 state. Thus this covers absolute updates 20,000–33,000, including every intermediate step. The predeclared attempt to cover **20,000 additional updates failed**: the enclosure radius became vacuous in the next block, updates 13,001–13,100. This is a failure of this bound, not evidence of scale acquisition by the actual network.

## What was verified

The certificate concerns exact real-arithmetic, full empirical GD on the archived binary64 inputs, target values, initial state, and step size `0.002`. It does not certify the rounding errors of the GPU implementation.

The original deployed helper was used unchanged, with SHA256
`8ce86d75c5bc57035a8402d00e2add13b4d4faf6444f1e07a5968d925094788d`.
Its hardcoded normalization is $h=1/64$, **not** the width-512 value $1/256$. The final completed helper prefix was bounded by approximately $0.007519292797$. Dividing the interval bound by the exact factor four gives

$$
\max_{n\le13{,}000,j}\lambda_j(n)
<0.002.
$$

The rescaled upper endpoint is approximately $0.001879823199222$.

The outward-rounded rescaling and original output rows are preserved in [outcome.json](outcome.json) and [partial_chunk100.json](partial_chunk100.json). The rows were extracted from the original helper's stdout after its failure; they are not a claim that the helper completed the requested horizon.

The reference was newly generated from this fork using the existing frozen-effective predictor. Endpoints were issued at every 100 additional updates. The Arb calculation treats their archived binary64 values as exact chord endpoints. This is not the same floating-point recurrence reference used by the earlier FP64 moving-tube panel. The compatibility archive has these endpoints at rows 99, 199, …, 19,999; its zero-filled unused rows must not be treated as reference states. Only chunk sizes 100 and 1,000 were used.

## Both attempts and the obstruction

| Chunk length | Last completed rigorous prefix | Last tube radius | Next failed block | One-CPU wall time |
|---:|---:|---:|---:|---:|
| 1,000 | 3,000 updates | 0.04467 | 3,001–4,000 | 23 seconds |
| 100 | 13,000 updates | 0.37833 | 13,001–13,100 | 697 seconds |

The helper does not print the exact iteration inside the failed block, so only that bracket is known. No later prefix is certified by these runs.

The finer chords extend the result substantially, but the positive-curvature bound on the chord defect remains costly. In the first 1,000-step block, the point defect is $1.82\times10^{-10}$ while its uniform chord bound is $7.78\times10^{-6}$. In the final completed 100-step block, these are $1.47\times10^{-8}$ and $7.91\times10^{-7}$. The nonlinear radius recurrence then becomes vacuous. Both columns and every completed radius are retained in the [1,000-step log](arb512-c1000-1291.out) and [100-step log](arb512-c100-1293.out).

This identifies an obstruction in the unchanged enclosure: a small point defect does not imply a comparably small defect bound over its whole reference segment and growing tube. It does not invalidate the much smaller FP64 reference error or establish that the proposed fine-force dynamics fail.

## Selection and reproducibility

This was a feasibility-selected, non-moment instance. Among the five generic targets in the completed width-512 seed-30 FP64 panel, `mixed_sine` had the smallest 20,000-update radius, about $0.00023494$; the others ranged from about $0.00297$ to $0.01093$. Selecting it was appropriate for attempting a second rigorous example, but it is not an unbiased cross-target success rate.

[preparation.json](preparation.json) records the original input hash, selected row 1, source hash, reference hash, and normalization convention. [input.npz](input.npz) and [chord_input.npz](chord_input.npz) suffice to rerun the original helper. The [reference manifest](reference_forecast/manifest.json), [preparation command](arb512-preparation.sbatch), [100-step invocation](arb512-c100.sbatch), and [output-curation command](curation.sbatch) are preserved. Jobs 1291 and 1293 used a total of 720 seconds on one CPU, with no GPU allocation and no change to the enclosure algorithm.

A distant-threshold exclusion alone should be compared with the [generic energy argument](../../../../../docs/d34_energy_baseline.md). This moving-reference calculation supplies a sharper instance-specific bound over its certified prefix; its failure at a longer horizon must not be concealed by that separate baseline.
