# Formula and selection contract

+Night-21B uses a clean-room sparse spaMGCN-style scaffold: two modality-specific AE/graph encoders, four registered sparse propagation orders, global order attention, and late linear fusion. The common endpoint is KMeans with known public K, `n_init=20`, endpoint seed 0.

+On registered spatial edges, carrier cosine determines the top-0.70 positive and bottom-0.30 boundary strata. Positive edges additionally require both raw-view cosines at or above their medians; boundary edges require both at or below their medians. Other/conflicting edges abstain. Positive loss is weighted `1-cos(z_i,z_j)`; boundary loss is weighted `relu(cos(z_i,z_j)-0.15)^2`. Per-stratum edge weights are normalized to mean one. B1 uses normalized pointwise MSE, B2 only positive relations, B3 only boundary relations, and FULL uses both.

+Discovery labels are opened only by the independent evaluator after NPZ partition, partition SHA, representation SHA, artifact SHA and checkpoint SHA are locked. The stable main board uses 700 optimization steps because the 300-step loss curves were not in a stable window. One post-lock mechanism revision increased positive weight from 0.25 to 1.0 and reduced boundary weight from 0.25 to 0.10; it was frozen before rerunning and failed its two-lane gate. Family transfer and multi-seed confirmation were therefore not authorized.
