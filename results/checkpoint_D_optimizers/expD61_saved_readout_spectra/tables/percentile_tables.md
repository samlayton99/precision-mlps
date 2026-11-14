**Neuron scales — central 98% interval**

**Gamma goal (QUILL reference): γ = 18.375**

| Learning rate | Default initialization | 2× initialization | 4× initialization |
|---|---|---|---|
| 0.01 | 0.005 — 0.502 — 1.748 | 0.008 — 0.979 — 2.833 | 0.015 — 1.940 — 4.495 |
| 0.001 | 0.005 — 0.487 — 1.246 | 0.008 — 0.970 — 2.154 | 0.015 — 1.940 — 4.000 |
| 0.0001 | 0.004 — 0.484 — 1.018 | 0.008 — 0.970 — 1.995 | 0.015 — 1.940 — 3.984 |

Cells: **1st percentile — median — 99th percentile** of absolute neuron scales.

**Normalized readout rates — 10th, 50th, and 90th percentiles**

**200k-step rate goal: ρ ≈ 10⁻⁵**

| Learning rate | Default initialization | 2× initialization | 4× initialization |
|---|---|---|---|
| 0.01 | 0 — 0 — 5.92 × 10⁻²⁴ | 0 — 0 — 9.77 × 10⁻¹⁷ | 0 — 0 — 3.90 × 10⁻¹¹ |
| 0.001 | 0 — 0 — 8.22 × 10⁻²⁶ | 0 — 0 — 1.44 × 10⁻¹⁷ | 0 — 0 — 1.93 × 10⁻¹¹ |
| 0.0001 | 0 — 0 — 0 | 0 — 0 — 1.28 × 10⁻¹⁷ | 0 — 0 — 1.90 × 10⁻¹¹ |

Cells: **10th percentile — median — 90th percentile**. **0** denotes numerically unresolved rates (at or below 5.17 × 10⁻²⁶), not exact zero.

Each cell pools all 24 functions and all 409 saved checkpoints through 200k training steps, with equal weight per saved neuron/mode value. These are pooled distributions, not worst-case bounds for every function or checkpoint.

The 200k-step reference uses frozen-readout GD with step size 1/(2 μ_max): a mode at that rate decreases its error component to approximately 37% of its starting magnitude after 200k updates.
