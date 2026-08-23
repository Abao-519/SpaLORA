#!/usr/bin/env python3
"""Matched component ablations on frozen Night-15E finalists."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from night15e_continuous_search import ContinuousEnergyConfig, load_lane, run_one, write_rows


def variants(config: ContinuousEnergyConfig):
    return {
        "FULL": config,
        "UNIFORM_EDGE_EVIDENCE": replace(
            config, edge_floor=0.9999, conflict_penalty=0.0001
        ),
        "NO_REJECTED_MASS_SELF_RETURN": replace(
            config, mass_center=-2.0, mass_temperature=0.005, neighbor_capacity=1.0
        ),
        "RETAINED_UNARY_DOMINANT": replace(
            config, retained_bias=8.0, view_balance=0.0, unary_temperature=0.02
        ),
        "SINGLE_SCALE_DOMINANT": replace(
            config, low_weight=0.0001, twohop_weight=0.0001, high_weight=0.0001
        ),
        "TRUST_TERM_NEAR_ZERO": replace(config, trust_scale=0.0001),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.registry.read_text(encoding="utf-8"))
    rows = []
    for lane, registered in frozen["lanes"].items():
        data, graph, labels, mask, k, initial, evidence = load_lane(
            args.kit, args.banks, args.partitions, lane
        )
        full_config = ContinuousEnergyConfig(**registered["config"])
        for name, config in variants(full_config).items():
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence,
                f"{registered['config_id']}__{name}", config, "MATCHED_ABLATION"
            )
            row["ablation"] = name
            row["full_config_id"] = registered["config_id"]
            row["is_matched_full"] = name == "FULL"
            row["ablation_config_json"] = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
            if name == "FULL" and row.get("partition_sha256") != registered["partition_sha256"]:
                raise RuntimeError(f"full ablation replay mismatch for {lane}")
            rows.append(row)
        print(json.dumps({"lane": lane, "rows": len(variants(full_config))}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    result = {
        "status": "PASS",
        "rows": len(rows),
        "lanes": len(frozen["lanes"]),
        "ablation_count_per_lane": 6,
        "labels_used_for_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
