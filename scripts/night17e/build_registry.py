#!/usr/bin/env python3
"""Build the preregistered small Night-17E mechanism grid."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


METHOD_STARTS = {
    "P22_K9": "PATH_UNIFORM__AUTHORITY_STRONG_START__L06",
    "MISAR_K7": "PATH_UNIFORM__AUTHORITY_STRONG_START__L06",
    "HUMAN_HIPPOCAMPUS_K7": "KMEANS_RETAINED_S4__UNIFORM_MASS_MATCHED",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--lane", required=True, choices=tuple(METHOD_STARTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--p0-only", action="store_true")
    args = parser.parse_args()
    parent = json.loads(Path(args.parent).read_text(encoding="utf-8"))
    direct = [x for x in parent["candidates"] if x["variant"] == "NIGHT15F_DIRECT"]
    if len(direct) != 1:
        raise ValueError("parent frozen direct config missing")
    base = copy.deepcopy(direct[0]["config"]["base"])
    parameter_grid = [
        ("L01_MIX025", 0.25, 1.0),
        ("L02_MIX050", 0.50, 1.0),
        ("L03_MIX075", 0.75, 1.0),
        ("L04_MIX050_POWER2", 0.50, 2.0),
    ]
    if args.p0_only:
        parameter_grid = [parameter_grid[1]]
    candidates = []
    for config_id, mix, power in parameter_grid:
        candidates.append(
            {
                "config_id": config_id,
                "config": {
                    "base": base,
                    "relation_mix": mix,
                    "relation_floor": 0.05,
                    "relation_power": power,
                    "uncertainty_scale": 1.0,
                },
            }
        )
    starts = [
        {
            "start_id": "METHOD_NIGHT16H_START",
            "source": "candidate_bank",
            "source_id": METHOD_STARTS[args.lane],
            "role": "NIGHT16H_LOCKED_METHOD_START",
        }
    ]
    if not args.p0_only:
        starts.extend(
            {
                "start_id": f"ROBUSTNESS_{seed}",
                "source": "carrier_start",
                "source_id": f"KMEANS_RETAINED_S{seed}",
                "role": "PREREGISTERED_KMEANS_ROBUSTNESS_START",
            }
            for seed in range(5)
        )
    registry = {
        "schema": "night17e-lrcc-registry-v1",
        "lane": args.lane,
        "formula_frozen_before_label_evaluation": True,
        "starts": starts,
        "arms": [
            "RELATION_DISABLED",
            "ZERO_RELATION",
            "UNIFORM_MASS_MATCHED",
            "PERMUTED_RELATION",
            "LEARNED_RELATION",
        ],
        "candidates": candidates,
        "selection_profiles": ["FIXED_GLOBAL", "DIRECT_PUBLIC_HPO", "STRICT_LOSO"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
