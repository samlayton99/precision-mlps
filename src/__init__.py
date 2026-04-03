"""precisionMLPs: Machine-precision MLP training via quasi-interpolant theory."""

import torch

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
