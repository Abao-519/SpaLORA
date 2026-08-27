"""Mechanical Stage-B method-contribution gate over independent evaluation tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FULL = "FULL_XBED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--main-lanes", nargs="+", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-decision", required=True)
    args = parser.parse_args()
    rows = []
    for value in args.inputs:
        with Path(value).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    by_lane = {}
    for row in rows:
        row["ari"] = float(row["ari"])
        row["nmi"] = float(row["nmi"])
        by_lane.setdefault(row["lane"], {})[row["candidate_id"]] = row
    summary = []
    for lane, candidates in sorted(by_lane.items()):
        full = candidates[FULL]
        controls = [row for key, row in candidates.items() if key != FULL]
        max_ari = max(float(row["ari"]) for row in controls)
        max_nmi = max(float(row["nmi"]) for row in controls)
        dual_beats_each = all(full["ari"] > row["ari"] and full["nmi"] > row["nmi"] for row in controls)
        summary.append(
            {
                "lane": lane,
                "role": "PRIMARY" if lane in args.main_lanes else "SECONDARY",
                "full_ari": full["ari"],
                "full_nmi": full["nmi"],
                "strongest_control_ari": max_ari,
                "strongest_control_nmi": max_nmi,
                "delta_ari_vs_coordinatewise_strongest": full["ari"] - max_ari,
                "delta_nmi_vs_coordinatewise_strongest": full["nmi"] - max_nmi,
                "dual_beats_every_control": dual_beats_each,
                "full_partition_sha256": full["partition_sha256"],
                "min_cluster_size_full": full["min_cluster_size_full"],
            }
        )
    output_table = Path(args.output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    primary = [row for row in summary if row["role"] == "PRIMARY"]
    passes = sum(bool(row["dual_beats_every_control"]) for row in primary)
    macro_ari = sum(float(row["delta_ari_vs_coordinatewise_strongest"]) for row in primary) / len(primary)
    macro_nmi = sum(float(row["delta_nmi_vs_coordinatewise_strongest"]) for row in primary) / len(primary)
    remaining = [row for row in primary if not row["dual_beats_every_control"]]
    safety = all(
        float(row["delta_ari_vs_coordinatewise_strongest"]) >= -0.02
        and float(row["delta_nmi_vs_coordinatewise_strongest"]) >= -0.02
        for row in remaining
    )
    authorized = passes >= 2 and macro_ari > 0 and macro_nmi > 0 and safety
    decision = {
        "schema": "night23a-stage-b-decision-v1",
        "main_lane_passes": passes,
        "main_lane_total": len(primary),
        "macro_delta_ari_vs_coordinatewise_strongest": macro_ari,
        "macro_delta_nmi_vs_coordinatewise_strongest": macro_nmi,
        "remaining_lane_safety": safety,
        "placenta_confirmation_authorized": authorized,
        "classification_if_stopped": None if authorized else "RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN",
        "no_posthoc_hpo": True,
    }
    Path(args.output_decision).write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
