#!/usr/bin/env python3
"""Generate the explicit Night-14B full-training grid."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "configs/night14b"

BASE = {
    "backbone": "CR_BALANCED_XREC",
    "candidate_id": "C15_BAL_XREC_600_WEAK_ALIGN",
    "depth": 2,
    "dropout": 0.05,
    "gradient_clip": 5.0,
    "graph_k": 8,
    "hidden_dim": 96,
    "latent_dim": 64,
    "learning_rate": 0.001,
    "loss_weights": {
        "alignment": 0.04,
        "covariance": 0.0005,
        "cross_recon": 1.0,
        "graph_smooth": 0.008,
        "private_recon": 0.5,
        "topology_agreement": 0.015,
        "variance": 0.1,
    },
    "steps": 600,
    "weight_decay": 0.00001,
}


def candidate(identifier, mode, freeze, steps, low, post_k, post_beta,
              post_steps, head_k, head_anchor, head_iterations,
              cluster_ks,
              edge_reconstruction=.5, trusted_smoothness=.02,
              rejected_edge_retention=.02, learning_rate=5e-4):
    return {
        "candidate_id": identifier,
        "edge_mode": mode,
        "base_config": deepcopy(BASE),
        "initial_low_strength": low,
        "initial_high_strength": 0.0,
        "support_scale": 10.0,
        "support_center": 0.60,
        "conflict_scale": 12.0,
        "conflict_center": 0.18,
        "boundary_scale": 10.0,
        "boundary_center": 0.40,
        "edge_loss_weights": {
            "edge_reconstruction": edge_reconstruction,
            "trusted_smoothness": trusted_smoothness,
            "rejected_edge_retention": rejected_edge_retention,
        },
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 1e-5,
        "steps": steps,
        "gradient_clip": 5.0,
        "freeze_base": freeze,
        "post_graph_k": post_k,
        "post_beta": post_beta,
        "post_steps": post_steps,
        "head_graph_k": head_k,
        "head_anchor": head_anchor,
        "head_iterations": head_iterations,
        "cluster_ks": list(cluster_ks),
    }


CONFIGS = [
    # P22 numeric HPO: all four retain the same model class and information flow.
    candidate("U20_FIXED_P22", "FIXED_LOW", True, 300, .20, 12, .80, 1, 24, .20, 10, (9,)),
    candidate("U21_SUPPORT_P22", "SUPPORT_ONLY", True, 300, .30, 12, .80, 1, 24, .20, 10, (9,)),
    candidate("U22_TSPR_P22", "TSPR", True, 300, .35, 12, .80, 1, 24, .20, 10, (9,)),
    candidate("U23_TSPR_FINETUNE_P22", "TSPR", False, 500, .35, 12, .80, 1, 24, .20, 10, (9,),
              edge_reconstruction=.8, trusted_smoothness=.01,
              rejected_edge_retention=.05, learning_rate=2e-4),
    # MISAR numeric HPO uses the same core/loss/head types with smaller graph K.
    candidate("U24_FIXED_MISAR", "FIXED_LOW", True, 300, .20, 18, .80, 2, 18, .00, 0, (7, 12)),
    candidate("U25_SUPPORT_MISAR", "SUPPORT_ONLY", True, 300, .30, 18, .80, 2, 18, .00, 0, (7, 12)),
    candidate("U26_TSPR_MISAR", "TSPR", True, 300, .35, 18, .80, 2, 18, .00, 0, (7, 12)),
    candidate("U27_TSPR_FINETUNE_MISAR", "TSPR", False, 500, .35, 18, .80, 2, 18, .00, 0, (7, 12),
              edge_reconstruction=.8, trusted_smoothness=.01,
              rejected_edge_retention=.05, learning_rate=2e-4),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        path = OUTPUT / (config["candidate_id"] + ".json")
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    (OUTPUT / "training_grid.json").write_text(
        json.dumps({"config_count": len(CONFIGS), "configs": CONFIGS},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"config_count": len(CONFIGS), "output": str(OUTPUT)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
