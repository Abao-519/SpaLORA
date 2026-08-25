# Method semantics and attribution contract

For a symmetric normalized graph Laplacian L, modality m is transformed by

`Y_m = (I + beta_m L)^(-1) (I + gamma_m L) X_m`, beta_m,gamma_m >= 0.

The sparse solve uses sixteen fixed Richardson iterations.  beta > gamma is low-pass, gamma > beta is bounded sharpening, and beta = gamma = 0 is exact identity.  Each modality's robust graph roughness is the median feature-wise energy `||X-SX||^2 / ||X||^2`; the common target is the geometric mean across the two observed modalities.  A fixed coefficient bank and distortion penalty choose beta/gamma from input statistics only.

All arms use the same adapters, zero-initialized residual around the retained representation, masked reconstruction, cross-view consistency, DEC-style prototype term, optimizer, 80 steps, 15-candidate bank and four registered endpoints.  Labels are unavailable to calibration, training, candidate generation and checkpoint locking.  The benchmark profile opens labels only after artifacts are locked.

Attribution requires FULL_RESPONSE_CALIBRATION to improve the same endpoint over MATCHED_BACKBONE and not be explained by low-pass-only, sharpen-only, shared-scale or swapped-response controls.  This condition failed; endpoint-specific gains are not a transferable representation claim.
