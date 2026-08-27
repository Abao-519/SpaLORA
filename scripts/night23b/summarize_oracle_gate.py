"""Mechanically choose the oracle-identifiable consumer and evaluate the frozen Stage-A gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-decision", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    rows = []
    for path in args.inputs:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        for key in ("teacher_recovery_ari", "teacher_recovery_nmi", "relation_scale"):
            row[key] = float(row[key])
    primary = [lane for lane, value in contract["lanes"].items() if value["role"] == "PRIMARY"]
    candidates = sorted({(row["mode"], row["relation_scale"]) for row in rows if row["mode"] != "CARRIER_ONLY"})
    ranked = []
    for mode, scale in candidates:
        selected = [row for row in rows if row["lane"] in primary and row["mode"] == mode and row["relation_scale"] == scale]
        score = [min(row["teacher_recovery_ari"], row["teacher_recovery_nmi"]) for row in selected]
        ranked.append({"mode": mode, "relation_scale": scale, "worst": min(score), "mean": sum(score) / len(score)})
    ranked.sort(key=lambda row: (-row["worst"], -row["mean"], row["relation_scale"], 0 if row["mode"] == "ORACLE_BINARY_POSITIVE" else 1))
    chosen = ranked[0]
    summary = []
    gate = contract["identifiability_gate"]
    for lane in contract["lanes"]:
        carrier = next(row for row in rows if row["lane"] == lane and row["mode"] == "CARRIER_ONLY")
        full = next(row for row in rows if row["lane"] == lane and row["mode"] == chosen["mode"] and row["relation_scale"] == chosen["relation_scale"])
        dual_gain = min(full["teacher_recovery_ari"] - carrier["teacher_recovery_ari"], full["teacher_recovery_nmi"] - carrier["teacher_recovery_nmi"])
        lane_pass = full["teacher_recovery_ari"] >= gate["per_lane_minimum_ari"] and full["teacher_recovery_nmi"] >= gate["per_lane_minimum_nmi"] and dual_gain >= gate["minimum_dual_gain_over_carrier"]
        summary.append({"lane": lane, "role": contract["lanes"][lane]["role"], "selected_mode": chosen["mode"], "selected_relation_scale": chosen["relation_scale"], "carrier_ari": carrier["teacher_recovery_ari"], "carrier_nmi": carrier["teacher_recovery_nmi"], "oracle_ari": full["teacher_recovery_ari"], "oracle_nmi": full["teacher_recovery_nmi"], "dual_gain_over_carrier": dual_gain, "lane_identifiable": lane_pass, "oracle_partition_sha256": full["partition_sha256"]})
    output = Path(args.output_table); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    passes = sum(row["lane_identifiable"] for row in summary if row["role"] == "PRIMARY")
    authorized = passes >= int(gate["minimum_primary_passes"])
    decision = {"schema": "night23b-oracle-consumer-gate-v1", "selected_consumer": chosen, "primary_identifiable_count": passes, "primary_total": len(primary), "stage_b_authorized": authorized, "classification_if_stopped": None if authorized else "RELATION_CONSUMER_NOT_IDENTIFIABLE", "benchmark_reference_labels_read": 0, "oracle_is_diagnostic_only": True, "ranking": ranked}
    Path(args.output_decision).write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
