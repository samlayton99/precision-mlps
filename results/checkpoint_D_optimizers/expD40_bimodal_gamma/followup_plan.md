# D40 second range: a milder high mode

Recorded during the first-stage screen, after the ten Airfoil arms and the first two Kin8nm arms had completed, before any follow-up run or confirmation test evaluation. The original 30:1 mixture puts almost all squared weight norm in high rows. D39 already showed that high bandwidth can worsen validation; the completed Airfoil validation screen does not favor the wide mixture. This motivates a separate 10:1 range, not a retrospective replacement of the first range.

Keep low gamma at .1g and reduce high gamma from 3g to g. Test the half/half mixture and its own uniform RMS control sqrt((.1^2+1^2)/2)g, in both-layer and layer-2-only scopes, on the same four tasks, seed 0, 10k steps. This adds sixteen validation-only runs. The pure low and high endpoints already exist as the original low/middle arms. Reference geometry, masks, readout, minibatches and all training choices stay paired.

Extend the per-task mixture selection to these four possibilities (two ranges by two scopes), using minimum validation MSE on the same schedule. The corresponding homogeneous comparison considers all originally screened single scales plus the milder RMS scale in that scope. Confirmation still includes selected mixture, its matched RMS, best homogeneous control in the same scope, and centered reference, deduplicated. Lock the full 56-run screen and these choices before test evaluation. No claim that the adaptive screen is independent validation evidence.

Original training source files are unchanged. A separate followup.py records the new range, source hash and resolved override in each result. Its confirmation driver dispatches original and mild arms with their original recorded configurations.
