"""Experiment 3: Geometry Ladder Cascade.

7 levels from most constrained to least constrained:

Level 1: Full construction
  - gamma, centers, readout all from QI. No training. Baseline precision.

Level 2: Fixed geometry, exact readout
  - gamma, centers from QI (frozen). Readout solved via least squares.
  - Tests: does exact readout match full construction?

Level 3: Fixed geometry, trained readout
  - gamma, centers from QI (frozen). Readout trained with Adam -> LBFGS.
  - Tests: how much precision is lost by training readout vs solving exactly?

Level 4: Fixed gamma, free centers, exact readout
  - gamma from QI (frozen). Centers free (random init). Readout solved exactly.
  - Tests: does the grid structure matter, or just gamma scaling?

Level 5: Fixed gamma, free centers, trained readout
  - gamma from QI (frozen). Centers and readout trained.
  - Tests: can training find good centers when gamma is correct?

Level 6: Free gamma, free centers, exact readout
  - All inner params free. Readout solved exactly after training inner layer.
  - Tests: is the readout solve a sufficient crutch?

Level 7: Fully free
  - Standard end-to-end training. No construction. No freezing.
  - Baseline: how far does unconstrained training get?
"""

# TODO: implement
# LEVELS = [
#     {"freeze_gamma": True,  "freeze_centers": True,  "readout": "construction", "name": "full_construction"},
#     {"freeze_gamma": True,  "freeze_centers": True,  "readout": "lstsq",        "name": "fixed_geom_exact_readout"},
#     {"freeze_gamma": True,  "freeze_centers": True,  "readout": "train",         "name": "fixed_geom_trained_readout"},
#     {"freeze_gamma": True,  "freeze_centers": False, "readout": "lstsq",        "name": "fixed_gamma_exact_readout"},
#     {"freeze_gamma": True,  "freeze_centers": False, "readout": "train",         "name": "fixed_gamma_trained_readout"},
#     {"freeze_gamma": False, "freeze_centers": False, "readout": "lstsq",        "name": "free_geom_exact_readout"},
#     {"freeze_gamma": False, "freeze_centers": False, "readout": "train",         "name": "fully_free"},
# ]
