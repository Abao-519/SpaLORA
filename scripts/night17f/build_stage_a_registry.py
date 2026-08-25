#!/usr/bin/env python3
"""Build the frozen small Stage-A direct-posterior grid."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    args = parser.parse_args()
    parent = json.loads(Path(args.parent).read_text(encoding="utf-8"))
    direct = [item for item in parent["candidates"] if item.get("variant") == "NIGHT15F_DIRECT"]
    if len(direct) != 1:
        raise ValueError("parent must contain exactly one NIGHT15F_DIRECT config")
    base = copy.deepcopy(direct[0]["config"]["base"])
    base_sha256 = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidates = []
    for config_id, mix in (("D01_MIX025", 0.25), ("D02_MIX050", 0.50), ("D03_MIX075", 0.75)):
        candidates.append(
            {
                "config_id": config_id,
                "config": {
                    "base": base,
                    "relation_mix": mix,
                    "relation_floor": 0.05,
                    "relation_power": 1.0,
                    "uncertainty_scale": 1.0,
                },
            }
        )
    registry = {
        "schema": "night17f-stage-a-direct-posterior-registry-v1",
        "scientific_role": "IN_STUDY_TEACHER_CONSUMER_DIAGNOSTIC",
        "lane": args.lane,
        "bank_mode": "UNBIASED_BANK",
        "formula_frozen_before_label_evaluation": True,
        "night15f_direct_base_sha256": base_sha256,
        "start": {
            "start_id": "METHOD_NIGHT16H_START",
            "source_id": METHOD_STARTS[args.lane],
            "role": "NIGHT16H_LOCKED_METHOD_START",
        },
        "arms": [
            "RELATION_DISABLED",
            "UNIFORM_MASS_MATCHED",
            "PERMUTED_POSTERIOR",
            "ANALYTIC_UNWEIGHTED_POSTERIOR",
            "DIRECT_WEIGHTED_POSTERIOR",
        ],
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
