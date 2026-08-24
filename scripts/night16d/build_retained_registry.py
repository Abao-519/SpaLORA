#!/usr/bin/env python3
"""Bounded exact-retained-anchor screen requested after synthetic-anchor plateau."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SpaLORA.night16d_cmbf_rl import CMBFRLConfig, config_sha256  # noqa: E402


def main() -> None:
    output = Path(sys.argv[1])
    base = CMBFRLConfig(
        latent_dim=64,
        hidden_dim=96,
        training_steps=60,
        learning_rate=0.0025,
        teacher_source="retained",
        teacher_scale=1.5,
        content_scale=0.6,
        residual_scale=0.35,
        lambda_support=0.3,
        lambda_boundary=0.4,
        lambda_conflict=0.2,
        lambda_anchor=1.0,
        lambda_gate_prior=0.2,
        endpoint_trust_threshold=0.7,
        endpoint_move_margin=0.005,
        operation_mode="full",
    )
    configs = [
        replace(base, operation_mode="teacher", training_steps=0, residual_scale=0.0),
        replace(base, operation_mode="teacher_head", training_steps=0, residual_scale=0.0),
        replace(base, operation_mode="generic", training_steps=0, residual_scale=0.10),
        replace(base, operation_mode="support", training_steps=40),
        replace(base, operation_mode="support_boundary", training_steps=40),
        replace(base, operation_mode="support_conflict", training_steps=40),
        replace(base, operation_mode="full_no_trust", trust_enabled=False, lambda_anchor=0.0, training_steps=40),
        replace(base, shuffle_edge_states=True, training_steps=40),
    ]
    seen = {config_sha256(x) for x in configs}
    for teacher_scale, content_scale, residual_scale, trust, margin, rank_mode in (
        (0.6, 0.8, 0.20, 0.55, 0.02, "global"),
        (0.6, 0.8, 0.45, 0.70, 0.005, "global"),
        (0.6, 0.8, 0.80, 0.85, 0.0, "node_local"),
        (1.0, 0.7, 0.20, 0.55, 0.02, "global"),
        (1.0, 0.7, 0.45, 0.70, 0.005, "node_local"),
        (1.0, 0.7, 0.80, 0.85, 0.0, "global"),
        (1.5, 0.6, 0.20, 0.55, 0.02, "global"),
        (1.5, 0.6, 0.45, 0.70, 0.005, "node_local"),
        (1.5, 0.6, 0.80, 0.85, 0.0, "global"),
        (2.2, 0.4, 0.20, 0.55, 0.02, "node_local"),
        (2.2, 0.4, 0.45, 0.70, 0.005, "global"),
        (2.2, 0.4, 0.80, 0.85, 0.0, "node_local"),
    ):
        config = replace(
            base,
            teacher_scale=teacher_scale,
            content_scale=content_scale,
            residual_scale=residual_scale,
            endpoint_trust_threshold=trust,
            endpoint_move_margin=margin,
            rank_mode=rank_mode,
        )
        if config_sha256(config) not in seen:
            configs.append(config); seen.add(config_sha256(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "night16d-exact-retained-anchor-registry-v4",
                "training_seeds": [0],
                "endpoint_seeds": [0],
                "configs": [asdict(x) for x in configs],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
