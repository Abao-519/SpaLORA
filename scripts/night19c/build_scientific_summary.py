#!/usr/bin/env python3
"""Mechanically recover Night-17C authority and apply the Night-19C seed0 gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from SpaLORA.night19c_zero_start_transfer import file_sha256


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
RUNS = {
    "FROZEN_RETAINED": "BASELINE__FROZEN_RETAINED",
    "RELATION_SMOOTH": "BASELINE__UNBIASED_SMOOTH",
    "ZERO_RESIDUAL": "BASELINE__ZERO_RESIDUAL",
    "PERMUTED_RELATION": "Z01_CONSERVATIVE__PERMUTED_RELATION__S{seed}",
    "Z01_FULL": "Z01_CONSERVATIVE__UNBIASED_FULL__S{seed}",
}


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-work", required=True)
    parser.add_argument("--old-compact", required=True)
    parser.add_argument("--placenta-dir", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    old = Path(args.old_work)
    old_compact = Path(args.old_compact)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    formula = json.loads(Path(args.formula).read_text(encoding="utf-8"))
    for relative, expected in formula["source_sha256"].items():
        if relative == "taskbook":
            continue
        path = Path("/root/SpaLORA-night16h") / relative
        if file_sha256(path) != expected:
            raise ValueError(f"active source differs from frozen formula: {relative}")

    recovery_rows = []
    authority_checks = []
    for seed in (0, 1, 2):
        root = old / ("formal" if seed == 0 else f"confirmation/seed{seed}")
        for lane in LANES:
            lane_root = root / lane
            producer = json.loads((lane_root / "producer.producer.json").read_text())
            replay = json.loads((lane_root / "fresh_replay.json").read_text())
            evaluator = json.loads((lane_root / "metrics.evaluator.json").read_text())
            checks = {
                "artifact_sha_exact": file_sha256(lane_root / "producer.npz") == producer["artifact_sha256"],
                "checkpoint_sha_exact": file_sha256(lane_root / "checkpoint.pt") == producer["checkpoint_sha256"],
                "fresh_representation_exact": bool(replay["all_representation_exact"]),
                "fresh_partition_exact": bool(replay["all_partition_exact"]),
                "metrics_sha_exact": file_sha256(lane_root / "metrics.csv") == evaluator["metrics_sha256"],
                "producer_label_reads_zero": producer["producer_label_reads"] == 0,
            }
            if not all(checks.values()):
                raise ValueError(f"Night-17C authority failed: seed={seed} lane={lane} {checks}")
            authority_checks.append({"lane": lane, "seed": seed, **checks,
                                     "artifact_sha256": producer["artifact_sha256"],
                                     "checkpoint_sha256": producer["checkpoint_sha256"],
                                     "carrier_sha256": producer["carrier_sha256"]})
            by_run = {row["run_id"]: row for row in read_csv(lane_root / "metrics.csv")}
            for arm, template in RUNS.items():
                run_id = template.format(seed=seed)
                row = by_run[run_id]
                recovery_rows.append({
                    "lane": lane, "seed": seed, "arm": arm, "run_id": run_id,
                    "ari": row["ari"], "nmi": row["nmi"], "ami": row["ami"], "fmi": row["fmi"],
                    "min_cluster_size": row["min_cluster_size"], "cluster_sizes": row["cluster_sizes"],
                    "partition_sha256": row["partition_sha256"],
                    "representation_sha256": row["representation_sha256"],
                })
    write_csv(out / "night17c_three_lane_authority_recovery.csv", recovery_rows)
    (out / "night17c_authority_recovery_audit.json").write_text(json.dumps({
        "schema": "night19c-night17c-authority-recovery-v1",
        "night17c_compact_index_sha256": file_sha256(old_compact / "compact_delivery_index.json"),
        "night17c_core_sha256": formula["source_sha256"]["SpaLORA/night17c_zero_start.py"],
        "checked_lane_seed_units": len(authority_checks),
        "all_authority_checks_pass": True,
        "checks": authority_checks,
        "scientific_results_rerun": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    placenta_dir = Path(args.placenta_dir)
    placenta_rows = read_csv(placenta_dir / "metrics.csv")
    by_run = {row["run_id"]: row for row in placenta_rows}
    controls = [by_run[name] for name in ("FROZEN_RETAINED", "RELATION_SMOOTH", "ZERO_RESIDUAL", "PERMUTED_RELATION")]
    full = by_run["Z01_FULL"]
    strongest_ari = max(float(row["ari"]) for row in controls)
    strongest_nmi = max(float(row["nmi"]) for row in controls)
    delta_ari = float(full["ari"]) - strongest_ari
    delta_nmi = float(full["nmi"]) - strongest_nmi
    no_structure_failure = int(full["min_cluster_size"]) > 1
    gate_pass = delta_ari >= 0.005 and delta_nmi >= 0.005 and no_structure_failure
    gate = {
        "schema": "night19c-placenta-seed0-gate-v1",
        "full_ari": float(full["ari"]), "full_nmi": float(full["nmi"]),
        "coordinate_wise_strongest_control_ari": strongest_ari,
        "coordinate_wise_strongest_control_nmi": strongest_nmi,
        "delta_ari": delta_ari, "delta_nmi": delta_nmi,
        "required_delta_ari": 0.005, "required_delta_nmi": 0.005,
        "full_min_cluster_size": int(full["min_cluster_size"]),
        "no_singleton_or_empty": no_structure_failure,
        "seed0_transfer_gate_passed": gate_pass,
        "seeds_1_2_authorized": gate_pass,
        "stage_c_authorized": gate_pass,
        "classification": "LOCAL_SIGNAL" if gate_pass else "SCIENTIFIC_NEGATIVE",
    }
    if gate["seeds_1_2_authorized"] != gate["seed0_transfer_gate_passed"]:
        raise AssertionError("gate authorization is internally inconsistent")
    (out / "placenta_seed0_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out / "placenta_absolute_metrics_and_controls.csv", placenta_rows)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
