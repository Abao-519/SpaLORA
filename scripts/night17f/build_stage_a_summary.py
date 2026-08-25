#!/usr/bin/env python3
"""Mechanically apply the preregistered Stage-A gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PRIMARY = "DIRECT_WEIGHTED_POSTERIOR"
CONTROLS = (
    "RELATION_DISABLED",
    "UNIFORM_MASS_MATCHED",
    "PERMUTED_POSTERIOR",
    "ANALYTIC_UNWEIGHTED_POSTERIOR",
)


def read(path: str):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [row for path in args.evaluation for row in read(path)]
    lanes = sorted({row["lane"] for row in rows})
    gate = []
    for lane in lanes:
        subset = [row for row in rows if row["lane"] == lane and row["status"] == "PASS"]
        baseline = [row for row in subset if row["arm"] == "INPUT_START"]
        if len(baseline) != 1:
            raise ValueError(f"{lane}: unique INPUT_START missing")
        primary = [row for row in subset if row["arm"] == PRIMARY]
        if not primary:
            raise ValueError(f"{lane}: primary arm missing")
        chosen = max(primary, key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"]), -int(row["config_id"].split("_")[0][1:])))
        matched = {
            arm: [row for row in subset if row["arm"] == arm and row["config_id"] == chosen["config_id"]]
            for arm in CONTROLS
        }
        if any(len(value) != 1 for value in matched.values()):
            raise ValueError(f"{lane}: same-config matched control missing")
        base_ari, base_nmi = float(baseline[0]["absolute_ari"]), float(baseline[0]["absolute_nmi"])
        ari, nmi = float(chosen["absolute_ari"]), float(chosen["absolute_nmi"])
        double_gain = ari > base_ari + 1e-12 and nmi > base_nmi + 1e-12
        independent = all(
            ari > float(value[0]["absolute_ari"]) + 1e-12
            and nmi > float(value[0]["absolute_nmi"]) + 1e-12
            for value in matched.values()
        )
        gate.append(
            {
                "lane": lane,
                "night16h_input_ari": base_ari,
                "night16h_input_nmi": base_nmi,
                "selected_config_id": chosen["config_id"],
                "direct_ari": ari,
                "direct_nmi": nmi,
                "delta_ari": ari - base_ari,
                "delta_nmi": nmi - base_nmi,
                "double_gain_vs_night16h": double_gain,
                "independent_of_all_matched_controls": independent,
                "stage_a_lane_pass": double_gain and independent,
                "direct_partition_sha256": chosen["partition_sha256"],
                "direct_changed_from_initial": chosen["changed_from_initial"],
                "direct_min_cluster_size_full": chosen["min_cluster_size_full"],
                **{
                    f"{arm.lower()}_ari": float(value[0]["absolute_ari"])
                    for arm, value in matched.items()
                },
                **{
                    f"{arm.lower()}_nmi": float(value[0]["absolute_nmi"])
                    for arm, value in matched.items()
                },
                **{
                    f"{arm.lower()}_partition_sha256": value[0]["partition_sha256"]
                    for arm, value in matched.items()
                },
            }
        )
    pass_count = sum(str(row["stage_a_lane_pass"]).lower() == "true" for row in gate)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write(output / "all_stage_a_evaluation.csv", rows)
    write(output / "stage_a_gate_table.csv", gate)
    decision = {
        "schema": "night17f-stage-a-gate-v1",
        "scientific_role": "IN_STUDY_TEACHER_CONSUMER_DIAGNOSTIC",
        "primary_lane_count": len(gate),
        "lane_pass_count": pass_count,
        "required_lane_pass_count": 2,
        "stage_a_passed": pass_count >= 2,
        "stage_b_authorized": pass_count >= 2,
        "classification_if_stopped": "SCIENTIFIC_NEGATIVE",
        "selection_rule": "max ARI then NMI within the three locked global mix configs; controls matched at selected config",
        "independence_rule": "primary must strictly exceed disabled, uniform-mass, within-scale permutation, and unweighted analytic in both ARI and NMI",
        "gate_rows": gate,
    }
    (output / "stage_a_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage_a_passed": decision["stage_a_passed"], "lane_pass_count": pass_count}, sort_keys=True))


if __name__ == "__main__":
    main()

