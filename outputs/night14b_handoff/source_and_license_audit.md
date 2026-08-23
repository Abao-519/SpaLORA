# Night-14B source and license audit

- New project code is a clean-room implementation written for this sprint and remains under the repository MIT license.
- It imports PyTorch (BSD-style), NumPy (BSD), SciPy (BSD), scikit-learn (BSD), pandas (BSD), and the existing MIT-licensed SpaLORA project modules.
- No third-party model source was copied into the repository or compact.
- SEPAR and 3d-OT were used only to register protocol context from their paper/official documentation; neither external method was trained or bundled.
- TSPR is an engineering work name. The experiment did not establish independent novelty or an independent score increment for it.
