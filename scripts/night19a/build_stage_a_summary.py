#!/usr/bin/env python3
"""Mechanical Stage-A scientific gate from locked evaluation tables."""

import argparse
import csv
import json
from pathlib import Path


PRIMARY = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
FULL = "EVIDENCE_CONDITIONED_ARBITRATION_FULL"
CONTROL = (
    "STRONG_START_NO_TRAIN", "STANDARD_WEIGHTED_SUM", "VANILLA_PCGRAD", "GLOBAL_MIN_NORM",
    "EVIDENCE_PERMUTED_MASS_MATCHED", "TOPOLOGY_DISABLED", "ARBITRATION_DISABLED_SAME_LOSSES",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    by_lane = {}
    all_rows = []
    for path in args.evaluation:
        rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
        for row in rows:
            row["ari"] = float(row["ari"]); row["nmi"] = float(row["nmi"])
            key = (row["lane"], row["profile"])
            if key in by_lane:
                raise ValueError("duplicate lane/profile")
            by_lane[key] = row; all_rows.append(row)
    lane_results = []
    for lane in PRIMARY:
        full = by_lane[(lane, FULL)]
        controls = [by_lane[(lane, profile)] for profile in CONTROL]
        strongest_ari = max(row["ari"] for row in controls)
        strongest_nmi = max(row["nmi"] for row in controls)
        strict = full["ari"] > strongest_ari and full["nmi"] > strongest_nmi
        lane_results.append({
            "lane": lane, "full_ari": full["ari"], "full_nmi": full["nmi"],
            "coordinatewise_strongest_control_ari": strongest_ari,
            "coordinatewise_strongest_control_nmi": strongest_nmi,
            "delta_ari_vs_coordinatewise_strongest": full["ari"] - strongest_ari,
            "delta_nmi_vs_coordinatewise_strongest": full["nmi"] - strongest_nmi,
            "strict_dual_pass": strict,
            "strongest_ari_control": max(controls, key=lambda row: row["ari"])["profile"],
            "strongest_nmi_control": max(controls, key=lambda row: row["nmi"])["profile"],
        })
    placenta_full = by_lane[("PLACENTA_K10", FULL)]
    placenta_standard = by_lane[("PLACENTA_K10", "STANDARD_WEIGHTED_SUM")]
    placenta_safety = (
        placenta_full["ari"] >= placenta_standard["ari"] - 0.01
        and placenta_full["nmi"] >= placenta_standard["nmi"] - 0.01
    )
    pass_count = sum(int(row["strict_dual_pass"]) for row in lane_results)
    multi_seed_authorized = pass_count >= 2 and placenta_safety
    decision = {
        "schema": "night19a-stage-a-gate-v1",
        "primary_strict_dual_pass_count": pass_count,
        "required_primary_strict_dual_pass_count": 2,
        "placenta_max_additional_drop_vs_standard_sum": 0.01,
        "placenta_safety_pass": placenta_safety,
        "multi_seed_authorized": multi_seed_authorized,
        "terminal_classification_if_not_authorized": "NO_INDEPENDENT_METHOD_SIGNAL",
        "lane_results": lane_results,
        "placenta": {
            "full_ari": placenta_full["ari"], "full_nmi": placenta_full["nmi"],
            "standard_ari": placenta_standard["ari"], "standard_nmi": placenta_standard["nmi"],
        },
    }
    Path(args.output).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
