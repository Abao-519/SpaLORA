#!/usr/bin/env python3
"""Build a deterministic, compact Night-16D refinement registry."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night16d_cmbf_rl import CMBFRLConfig, config_sha256  # noqa: E402


def main() -> None:
    output = Path(sys.argv[1])
    base = CMBFRLConfig(training_steps=60, operation_mode="full")
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
    rng = np.random.default_rng(20260824)
    choices = {
        "learning_rate": [0.0015, 0.0025, 0.0040],
        "training_steps": [40, 60, 80],
        "residual_scale": [0.08, 0.16, 0.30, 0.55, 0.85],
        "lambda_support": [0.10, 0.30, 0.60],
        "lambda_boundary": [0.10, 0.30, 0.60],
        "lambda_conflict": [0.05, 0.20, 0.50],
        "lambda_anchor": [0.50, 1.00, 2.00],
        "lambda_gate_prior": [0.02, 0.10, 0.30],
        "boundary_margin": [0.50, 0.75, 1.00],
        "endpoint_trust_threshold": [0.35, 0.50, 0.65, 0.80],
        "endpoint_move_margin": [0.0, 0.005, 0.02, 0.05],
        "endpoint_core_quantile": [0.35, 0.50, 0.65],
        "rank_mode": ["global", "node_local"],
    }
    seen = {config_sha256(x) for x in configs}
    anchors = [
        replace(base, residual_scale=0.08, endpoint_trust_threshold=0.35, endpoint_move_margin=0.02, lambda_anchor=2.0),
        replace(base, residual_scale=0.30, endpoint_trust_threshold=0.65, endpoint_move_margin=0.005),
        replace(base, residual_scale=0.55, endpoint_trust_threshold=0.80, endpoint_move_margin=0.0, lambda_boundary=0.60),
        replace(base, residual_scale=0.30, endpoint_trust_threshold=0.65, endpoint_move_margin=0.005, rank_mode="node_local"),
    ]
    for config in anchors:
        if config_sha256(config) not in seen:
            configs.append(config); seen.add(config_sha256(config))
    while sum(x.operation_mode == "full" and not x.shuffle_edge_states for x in configs) < 32:
        config = replace(
            base,
            learning_rate=float(rng.choice(choices["learning_rate"])),
            training_steps=int(rng.choice(choices["training_steps"])),
            residual_scale=float(rng.choice(choices["residual_scale"])),
            lambda_support=float(rng.choice(choices["lambda_support"])),
            lambda_boundary=float(rng.choice(choices["lambda_boundary"])),
            lambda_conflict=float(rng.choice(choices["lambda_conflict"])),
            lambda_anchor=float(rng.choice(choices["lambda_anchor"])),
            lambda_gate_prior=float(rng.choice(choices["lambda_gate_prior"])),
            boundary_margin=float(rng.choice(choices["boundary_margin"])),
            endpoint_trust_threshold=float(rng.choice(choices["endpoint_trust_threshold"])),
            endpoint_move_margin=float(rng.choice(choices["endpoint_move_margin"])),
            endpoint_core_quantile=float(rng.choice(choices["endpoint_core_quantile"])),
            rank_mode=str(rng.choice(choices["rank_mode"])),
        )
        if config_sha256(config) not in seen:
            configs.append(config); seen.add(config_sha256(config))
    payload = {
        "schema": "night16d-refine-registry-v2",
        "training_seeds": [0],
        "endpoint_seeds": [0],
        "configs": [asdict(x) for x in configs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
