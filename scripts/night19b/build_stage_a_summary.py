#!/usr/bin/env python3
"""Mechanical Stage-A gate for Night-19B."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FULL = "CSAD_FULL"
EQUIVALENCE_CONTROLS = ("SIMPLE_OPERATOR_AVERAGE", "CSAD_CONFLICT_DISABLED", "CSAD_MODALITY_EDGE_PERMUTED")
PRIMARY = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    thresholds = contract["stage_a_gate"]
    rows = []
    for path in args.evaluation:
        rows.extend(csv.DictReader(Path(path).open(encoding="utf-8")))
    by_lane_config = defaultdict(list)
    for row in rows:
        for key in ("ari", "nmi", "ami", "fmi"):
            row[key] = float(row[key])
        by_lane_config[(row["lane"], row["config_id"])].append(row)
    comparisons = []
    for (lane, config_id), group in sorted(by_lane_config.items()):
        fulls = [row for row in group if row["arm"] == FULL]
        controls = [row for row in group if row["arm"] != FULL]
        if len(fulls) != 1 or not controls:
            raise ValueError("each lane/config needs one full and matched controls")
        full = fulls[0]
        max_ari_control = max(controls, key=lambda row: (row["ari"], row["nmi"], row["arm"]))
        max_nmi_control = max(controls, key=lambda row: (row["nmi"], row["ari"], row["arm"]))
        equivalent = [
            row["arm"] for row in controls
            if row["arm"] in EQUIVALENCE_CONTROLS and row["partition_sha256"] == full["partition_sha256"]
        ]
        comparisons.append({
            "lane": lane, "config_id": config_id,
            "full_ari": full["ari"], "full_nmi": full["nmi"], "full_partition_sha256": full["partition_sha256"],
            "coordinate_max_control_ari": max_ari_control["ari"], "coordinate_max_control_ari_arm": max_ari_control["arm"],
            "coordinate_max_control_nmi": max_nmi_control["nmi"], "coordinate_max_control_nmi_arm": max_nmi_control["arm"],
            "delta_ari": full["ari"] - max_ari_control["ari"],
            "delta_nmi": full["nmi"] - max_nmi_control["nmi"],
            "equivalent_key_controls": equivalent,
        })
    lane_results = []
    for lane in sorted({row["lane"] for row in comparisons}):
        candidates = [row for row in comparisons if row["lane"] == lane]
        selected = max(candidates, key=lambda row: (min(row["delta_ari"], row["delta_nmi"]), row["delta_ari"], row["delta_nmi"], -len(row["equivalent_key_controls"]), row["config_id"]))
        passed = bool(
            selected["delta_ari"] >= float(thresholds["minimum_delta_ari"])
            and selected["delta_nmi"] >= float(thresholds["minimum_delta_nmi"])
            and not selected["equivalent_key_controls"]
        )
        lane_results.append({**selected, "lane_pass": passed})
    primary_rows = [row for row in lane_results if row["lane"] in PRIMARY]
    primary_pass = sum(int(row["lane_pass"]) for row in primary_rows)
    third_safe = all(not (row["delta_ari"] < -float(thresholds["maximum_third_lane_joint_drop"]) and row["delta_nmi"] < -float(thresholds["maximum_third_lane_joint_drop"])) for row in primary_rows)
    arm_statistics = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["lane"], row["arm"])].append(row)
    for (lane, arm), group in sorted(grouped.items()):
        for metric in ("ari", "nmi"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            arm_statistics.append({"lane": lane, "arm": arm, "metric": metric, "best": float(values.max()), "median": float(np.median(values)), "mean": float(values.mean()), "min": float(values.min()), "count": int(values.size)})
    authorized = bool(primary_pass >= int(thresholds["minimum_primary_lane_pass_count"]) and third_safe)
    result = {
        "schema": "night19b-stage-a-gate-v1",
        "selection_semantics": "LABEL_ASSISTED_BEST_PROFILE_BY_MAXIMIZING_MIN_COORDINATEWISE_DELTA_WITHIN_SAME_CONFIG",
        "comparisons": comparisons,
        "lane_results": lane_results,
        "primary_lane_pass_count": primary_pass,
        "third_lane_safety": third_safe,
        "family_stage_authorized": authorized,
        "trainable_unfolding_authorized": authorized,
        "arm_statistics": arm_statistics,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"primary_lane_pass_count": primary_pass, "third_lane_safety": third_safe, "family_stage_authorized": authorized, "lane_results": lane_results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
