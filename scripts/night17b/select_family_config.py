#!/usr/bin/env python3
"""Transparent post-lock family-level benchmark HPO for Night-17B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from SpaLORA.night17b_sfrd import exceeds_all_matched_controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        rows.extend(csv.DictReader(Path(path).open(encoding="utf-8")))
    lanes = sorted({row["lane"] for row in rows})
    baselines = {}
    feasible_controls = {}
    for lane in lanes:
        matches = [row for row in rows if row["lane"] == lane and row["run_id"] == "BASELINE__FROZEN_RETAINED"]
        if len(matches) != 1:
            raise ValueError(f"baseline missing for {lane}")
        baselines[lane] = matches[0]
        matches = [row for row in rows if row["lane"] == lane and row["run_id"] == "BASELINE__FEASIBLE_RELATION_SMOOTH"]
        if len(matches) != 1:
            raise ValueError(f"feasible head-only control missing for {lane}")
        feasible_controls[lane] = matches[0]
    config_ids = sorted({row["config_id"] for row in rows if row["arm"] == "FULL_WEIGHTED"})
    ranking = []
    for config_id in config_ids:
        lane_rows = []
        for lane in lanes:
            matches = [row for row in rows if row["lane"] == lane and row["config_id"] == config_id and row["arm"] == "FULL_WEIGHTED"]
            if len(matches) != 1:
                raise ValueError(f"full row missing for {lane}/{config_id}")
            row = matches[0]
            passed, strongest_ari, strongest_nmi = exceeds_all_matched_controls(
                float(row["ari"]),
                float(row["nmi"]),
                [
                    (float(baselines[lane]["ari"]), float(baselines[lane]["nmi"])),
                    (float(feasible_controls[lane]["ari"]), float(feasible_controls[lane]["nmi"])),
                ],
            )
            delta_ari = float(row["ari"]) - strongest_ari
            delta_nmi = float(row["nmi"]) - strongest_nmi
            lane_rows.append((lane, delta_ari, delta_nmi, passed, strongest_ari, strongest_nmi))
        ranking.append(
            {
                "config_id": config_id,
                "dual_positive_lanes": sum(passed for _, _, _, passed, _, _ in lane_rows),
                "worst_joint_delta": min(min(delta_ari, delta_nmi) for _, delta_ari, delta_nmi, _, _, _ in lane_rows),
                "mean_delta_ari": sum(delta_ari for _, delta_ari, _, _, _, _ in lane_rows) / len(lane_rows),
                "mean_delta_nmi": sum(delta_nmi for _, _, delta_nmi, _, _, _ in lane_rows) / len(lane_rows),
                "lane_deltas": [
                    {
                        "lane": lane,
                        "delta_ari_vs_strongest_control": da,
                        "delta_nmi_vs_strongest_control": dn,
                        "strict_dual_pass": passed,
                        "strongest_control_ari": strongest_ari,
                        "strongest_control_nmi": strongest_nmi,
                    }
                    for lane, da, dn, passed, strongest_ari, strongest_nmi in lane_rows
                ],
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["dual_positive_lanes"],
            -row["worst_joint_delta"],
            -row["mean_delta_ari"],
            -row["mean_delta_nmi"],
            row["config_id"],
        )
    )
    selected = ranking[0]
    selected_rows = []
    arms = (
        "FROZEN_RETAINED_SAME_HEAD",
        "FEASIBLE_CONSENSUS_SAME_HEAD",
        "FULL_WEIGHTED",
        "UNWEIGHTED_RELATION",
        "UNBIASED_BANK_WEIGHTED",
        "RELATION_PERMUTED",
        "SINGLE_VIEW1",
        "SINGLE_VIEW2",
    )
    for lane in lanes:
        baseline = baselines[lane]
        for arm in arms:
            if arm == "FROZEN_RETAINED_SAME_HEAD":
                candidates = [baseline]
            elif arm == "FEASIBLE_CONSENSUS_SAME_HEAD":
                candidates = [row for row in rows if row["lane"] == lane and row["arm"] == arm]
            else:
                candidates = [row for row in rows if row["lane"] == lane and row["arm"] == arm and row["config_id"] == selected["config_id"]]
            if len(candidates) != 1:
                raise ValueError(f"matched arm missing for {lane}/{arm}")
            row = dict(candidates[0])
            row["delta_ari_vs_frozen"] = float(row["ari"]) - float(baseline["ari"])
            row["delta_nmi_vs_frozen"] = float(row["nmi"]) - float(baseline["nmi"])
            row["delta_ari_vs_strongest_control"] = float(row["ari"]) - max(
                float(baseline["ari"]), float(feasible_controls[lane]["ari"])
            )
            row["delta_nmi_vs_strongest_control"] = float(row["nmi"]) - max(
                float(baseline["nmi"]), float(feasible_controls[lane]["nmi"])
            )
            selected_rows.append(row)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    decision = {
        "schema": "night17b-family-config-selection-v1",
        "selection_semantics": "post-lock transparent label-assisted family benchmark HPO; strict gate uses coordinate-wise strongest frozen-carrier and feasible-head controls",
        "selected_config_id": selected["config_id"],
        "selected_dual_positive_lanes": selected["dual_positive_lanes"],
        "gate_passed": selected["dual_positive_lanes"] >= 2,
        "ranking": ranking,
    }
    Path(args.output_json).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
