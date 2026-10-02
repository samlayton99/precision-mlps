# Second iteration: isolate direction count from normalized slope

Written before these extra runs, after some first-stage validation results were visible. This is an adaptive exploratory extension, not part of the original frozen eleven-arm design. No new held-out test scores have been evaluated.

At fixed width, increasing the number of directions from 24 to 64 reduces centers per direction from 21/22 to 8. At fixed lambda this also changes normalized slope gamma*A from 2.5/2.625 to .875. A better score could therefore come from additional directions, softer transitions, or both. Two added controls complete an approximate 2-by-2 comparison:

| Directions | Soft slope | Original slope |
|---|---|---|
| 24 | lambda=.0875: gamma*A=.875/.91875 | centered: lambda=.25, gamma*A=2.5/2.625 |
| 64 | directions64: lambda=.25, gamma*A=.875 | lambda=5/7: gamma*A=2.5 |

The small mismatch for the eight 22-center banks is explicit; their remaining sixteen banks match exactly. More directions also changes the random direction set. All other training controls stay fixed.

Eight new 10k-step runs: the same four screening tasks and seed0, training and validation only. Rank all thirteen candidates by the same four-task geometric mean of best validation MSE relative to original QI. Lock that choice before any confirmation test evaluation. All first-stage outputs and identities remain unchanged. The second-iteration source and override map are included in its run identities.
