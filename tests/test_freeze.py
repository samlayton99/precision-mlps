"""Tests for parameter freezing.

Test cases:
- FrozenParam is not matched by nnx.DiffState(0, nnx.Param)
- freeze_params converts Param -> FrozenParam
- unfreeze_params converts FrozenParam -> Param
- freeze_inner_layer freezes gamma and centers
- freeze_gamma only freezes gamma, leaves centers trainable
- Frozen params have zero gradient after value_and_grad
- get_trainable_param_count reflects freeze state
- Forward pass is unchanged after freezing
- freeze then unfreeze is identity (values preserved)
"""

# TODO: implement
