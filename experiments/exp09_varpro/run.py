"""Experiment 9: Variable Projection (VarPro) / Reduced Objective.

Core:
- Nonlinear params: theta = (lambda, delta_k) or (log_gamma, centers)
- Linear params v(theta) solved exactly via lstsq at each iteration
- Optimize reduced loss with Adam -> LBFGS (on nonlinear params only)
- Compare directly to matched end-to-end training from geometry ladder

Additional:
- Log reduced Jacobian/Hessian conditioning.
- If VarPro works dramatically better, that's evidence that raw
  end-to-end coordinates are the wrong optimization variables.
"""

# TODO: implement
# 1. Create QIMlp with gamma_exp layer type
# 2. Freeze readout (it will be solved, not trained)
# 3. Build VarProObjective(model, x_train, y_train)
# 4. Optimize nonlinear params with Adam -> LBFGS
#    (using VarProObjective as the loss function)
# 5. After optimization, call varpro.solve_and_update_readout()
# 6. Evaluate final model on eval data
# 7. Compare to exp03 level 7 (fully free, same width/seed)
