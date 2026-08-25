#!/usr/bin/env python3
"""Apply the preregistered Night-17C strict scientific gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from SpaLORA.night17b_sfrd import exceeds_all_matched_controls


CONTROL_ARMS = (
    "FROZEN_RETAINED_SAME_HEAD",
    "FULL_BANK_SMOOTH_REFERENCE",
    "UNBIASED_SMOOTH_REFERENCE",
    "ZERO_RESIDUAL_CONTROL",
)


def load_metrics(paths):
    rows = []
    for path in paths:
        rows.extend(csv.DictReader(Path(path).open(encoding="utf-8")))
    return rows


def select(rows):
    lanes = sorted({row["lane"] for row in rows})
    configs = sorted({row["config_id"] for row in rows if row["arm"] == "UNBIASED_FULL"})
    candidates = []
    detailed = []
    for config_id in configs:
        passes = 0
        lane_details = []
        for lane in lanes:
            lane_rows = [row for row in rows if row["lane"] == lane]
            controls = [row for row in lane_rows if row["arm"] in CONTROL_ARMS]
            if len(controls) != len(CONTROL_ARMS):
                raise ValueError(f"{lane}: expected four unique matched controls")
            full = [row for row in lane_rows if row["arm"] == "UNBIASED_FULL" and row["config_id"] == config_id]
            permuted = [row for row in lane_rows if row["arm"] == "PERMUTED_RELATION" and row["config_id"] == config_id]
            if len(full) != 1 or len(permuted) != 1:
                raise ValueError(f"{lane}/{config_id}: missing or duplicate full/permuted row")
            full, permuted = full[0], permuted[0]
            control_pairs = [(float(row["ari"]), float(row["nmi"])) for row in controls]
            dual, strongest_ari, strongest_nmi = exceeds_all_matched_controls(
                float(full["ari"]), float(full["nmi"]), control_pairs
            )
            permuted_dominates = (
                float(permuted["ari"]) > float(full["ari"]) + 1e-12
                and float(permuted["nmi"]) > float(full["nmi"]) + 1e-12
            )
            passed = bool(dual and not permuted_dominates)
            passes += int(passed)
            detail = {
                "lane": lane,
                "config_id": config_id,
                "full_ari": float(full["ari"]),
                "full_nmi": float(full["nmi"]),
                "strongest_control_ari": strongest_ari,
                "strongest_control_nmi": strongest_nmi,
                "delta_ari": float(full["ari"]) - strongest_ari,
                "delta_nmi": float(full["nmi"]) - strongest_nmi,
                "permuted_ari": float(permuted["ari"]),
                "permuted_nmi": float(permuted["nmi"]),
                "permuted_dominates": permuted_dominates,
                "strict_lane_pass": passed,
            }
            lane_details.append(detail)
            detailed.append(detail)
        candidates.append({
            "config_id": config_id,
            "strict_pass_lanes": passes,
            "worst_joint_delta": min(min(row["delta_ari"], row["delta_nmi"]) for row in lane_details),
            "mean_delta_ari": sum(row["delta_ari"] for row in lane_details) / len(lane_details),
            "mean_delta_nmi": sum(row["delta_nmi"] for row in lane_details) / len(lane_details),
        })
    candidates.sort(key=lambda row: (
        -row["strict_pass_lanes"], -row["worst_joint_delta"], -row["mean_delta_ari"],
        -row["mean_delta_nmi"], row["config_id"],
    ))
    selected = candidates[0]
    return {
        "schema": "night17c-strict-selection-v1",
        "selected_config_id": selected["config_id"],
        "strict_pass_lanes": selected["strict_pass_lanes"],
        "required_pass_lanes": 2,
        "gate_passed": selected["strict_pass_lanes"] >= 2,
        "multi_seed_authorized": selected["strict_pass_lanes"] >= 2,
        "candidate_ranking": candidates,
        "lane_details": [row for row in detailed if row["config_id"] == selected["config_id"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = select(load_metrics(args.metrics))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
