"""Tests for training loop and optimizers.

Test cases:
- train_step reduces loss on a simple problem
- Adam optimizer produces valid gradients
- build_schedule returns correct schedule type
- MetricsCollector.collect returns all expected keys
- _parse_eval_schedule correctly parses schedule strings
- mse loss matches manual computation
- lp_loss with p=2 matches mse
- get_loss_fn returns correct loss function
- TrainResult has correct structure after training
"""

# TODO: implement
