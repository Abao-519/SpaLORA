#!/usr/bin/env python3
"""Focused representation/endpoint screen after conservative no-op plateau."""

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
        training_steps=80,
        learning_rate=0.003,
        residual_scale=0.8,
        lambda_support=0.3,
        lambda_boundary=0.4,
        lambda_conflict=0.2,
        lambda_anchor=0.5,
        lambda_gate_prior=0.2,
        endpoint_trust_threshold=0.8,
        endpoint_move_margin=0.0,
        operation_mode="full",
    )
    configs = [replace(base, operation_mode="teacher", training_steps=0, residual_scale=0.0)]
    seen = {config_sha256(configs[0])}
    for teacher_scale in (1.25, 1.75, 2.25):
        for content_scale in (0.10, 0.30, 0.55):
            for residual_scale, trust_threshold, rank_mode in (
                (0.45, 0.65, "global"),
                (0.80, 0.80, "global"),
                (1.20, 0.92, "node_local"),
            ):
                config = replace(
                    base,
                    teacher_scale=teacher_scale,
                    content_scale=content_scale,
                    residual_scale=residual_scale,
                    endpoint_trust_threshold=trust_threshold,
                    rank_mode=rank_mode,
                )
                if config_sha256(config) not in seen:
                    configs.append(config); seen.add(config_sha256(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "night16d-aggressive-representation-registry-v3",
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
