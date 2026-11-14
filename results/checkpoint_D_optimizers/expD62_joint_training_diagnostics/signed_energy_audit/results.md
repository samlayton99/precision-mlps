# Signed-energy bound evaluated on saved GD trajectories

The signed-energy comparison in docs/review_process/iclr_2027/signed_energy_escape_bound.md was evaluated retrospectively on 24 default-initialization D62 trajectories and the small/default width-705 P5 pair. No training was launched. The analysis starts at 30k and obtains constant allowances from saved checkpoints through 50k.

The exact radial-energy identity, its expanded affine/nonaffine identity, and the moment-based one-step inequality passed at all 1,066 evaluated states. These checks do not prove persistence between checkpoints or beyond the window, and do not constitute an independent mathematical review.

For fairness, the table below evaluates both old and new recurrences with constant suffix maxima, rather than comparing an old interpolated allowance to a new maximum. “Doubling” refers to the computed parameter-energy ceiling reaching twice its starting energy, not the measured network energy doubling.

| Width | Function | LR | Original recurrence: updates to doubled energy ceiling | Signed-energy recurrence: updates to doubled energy ceiling |
|---:|---|---:|---:|---:|
| 196 | runge_100 | 0.001 | 281 | 28 |
| 196 | sine_4pi | 0.001 | 205 | 36 |
| 196 | square | 0.001 | 3134 | 28 |

All 24 default D62 cases reach the doubled-energy ceiling sooner with the proposed signed-energy comparison. The width-705 default P5 case does so in 9 updates, versus 101 with the old recurrence. The small-initialization P5 case retains the doubled-energy barrier for the available 20k continuation under both comparisons. Taking the minimum with the old bound prevents weakening the final result, but this candidate adds no useful duration in these default cases.

Example: width-196 sine at 30k, LR 0.001. Current energy is 131.4696. The next exact GD energy increase, computed from the current gradient, is 6.8052e-5. The new recurrence allows 0.95381, versus 0.32281 for the old constant-allowance recurrence. The measured energy at 50k is only 133.1476.

The signed self-output term is stabilizing here: its contribution to the exact energy increment is -1.2687e-5. Completing the square bounds that contribution by 1.4309e-7 when the actual aggregated remainder is used. Replacing the aggregated remainder with the theoretical per-neuron moment estimate raises its contribution to 0.36052 (0.37256 after the suffix concentration maximum). Thus the exact signed identity preserves helpful cancellation, but the subsequent absolute-moment bounds throw much of that information away.

The default initialization has individual hidden parameters of order one, and S6=sum_j r_j^6 is about 137.56 in this case. The squared-remainder term scales with S6 squared; it is not small in this regime. At width 705, smaller-initialization P5 has S6 approximately 0.000770, while default P5 has S6 approximately 507.44. Valid global polynomial remainder estimates can still be extremely loose.

The earlier 3.2-million-step statement remains an illustrative conditional calculation under explicitly small-energy, weak-target-loading allowances. It is not a result for PyTorch defaults. A useful default-regime improvement would have to control aggregate signed/remainder terms without losing their cancellation; that has not been derived.
