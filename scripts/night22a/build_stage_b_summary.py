"""Mechanically summarize locked Night-22A junction candidates and gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FAMILY = {
    "A1_K10": "RNA_PROTEIN",
    "TONSIL_S1_K4": "RNA_PROTEIN",
    "P22_K9": "RNA_CHROMATIN",
    "PLACENTA_K10": "RNA_CHROMATIN",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--stage-a-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stage_a = {row["lane"]: row for row in read_csv(args.stage_a_summary)}
    all_rows = []
    manifests = {}
    for evaluation in sorted(Path(args.evaluation_dir).glob("*.csv")):
        bank = Path(args.bank_dir) / evaluation.with_suffix(".npz").name
        manifest = json.loads(bank.with_suffix(".json").read_text(encoding="utf-8"))
        key = (manifest["lane"], manifest["start_candidate"])
        manifests[key] = manifest
        for row in read_csv(evaluation):
            row["start_candidate"] = manifest["start_candidate"]
            row["training_seed"] = str(manifest["seed"])
            row["family"] = FAMILY[row["lane"]]
            row["candidate_bank_sha256"] = manifest["partition_bank_sha256"]
            row["checkpoint_bank_sha256"] = manifest["checkpoint_bank_sha256"]
            row["changed_spots_vs_start"] = str(
                manifest["candidate_diagnostics"][row["candidate_id"]].get("changed_spots_vs_start", 0)
            )
            all_rows.append(row)
    if not all_rows:
        raise RuntimeError("no junction evaluations")
    write_csv(output / "junction_all_candidate_ledger.csv", all_rows)

    contribution = []
    lane_best = {}
    for lane in sorted(FAMILY):
        full_rows = [row for row in all_rows if row["lane"] == lane and "__FULL__" in row["candidate_id"]]
        for full in full_rows:
            profile = full["candidate_id"].split("__")[0]
            atomic = [
                row
                for row in all_rows
                if row["lane"] == lane
                and row["start_candidate"] == full["start_candidate"]
                and (row["candidate_id"] == "INPUT_GEOMETRY_START" or row["candidate_id"].startswith(profile + "__"))
                and "__FULL__" not in row["candidate_id"]
            ]
            strongest_ari = max(float(row["ari"]) for row in atomic)
            strongest_nmi = max(float(row["nmi"]) for row in atomic)
            strict = all(
                float(full["ari"]) > float(row["ari"]) and float(full["nmi"]) > float(row["nmi"])
                for row in atomic
            )
            a = stage_a[lane]
            row = {
                "lane": lane,
                "family": FAMILY[lane],
                "start_candidate": full["start_candidate"],
                "profile_id": profile,
                "full_candidate_id": full["candidate_id"],
                "full_ari": full["ari"],
                "full_nmi": full["nmi"],
                "input_start_ari": next(x["ari"] for x in atomic if x["candidate_id"] == "INPUT_GEOMETRY_START"),
                "input_start_nmi": next(x["nmi"] for x in atomic if x["candidate_id"] == "INPUT_GEOMETRY_START"),
                "strongest_atomic_ari": strongest_ari,
                "strongest_atomic_nmi": strongest_nmi,
                "delta_ari_vs_coordinatewise_atomic": float(full["ari"]) - strongest_ari,
                "delta_nmi_vs_coordinatewise_atomic": float(full["nmi"]) - strongest_nmi,
                "stage_a_best_head_ari": a["stage_a_best_ari"],
                "stage_a_best_head_nmi": a["stage_a_best_nmi"],
                "delta_ari_vs_stage_a_best_head": float(full["ari"]) - float(a["stage_a_best_ari"]),
                "delta_nmi_vs_stage_a_best_head": float(full["nmi"]) - float(a["stage_a_best_nmi"]),
                "changed_spots_vs_start": full["changed_spots_vs_start"],
                "min_cluster_size": full["min_cluster_size_full"],
                "strict_full_double_beats_each_matched_atomic": strict,
                "independent_lane_pass": bool(
                    strict
                    and int(full["changed_spots_vs_start"]) > 0
                    and float(full["ari"]) > float(a["stage_a_best_ari"])
                    and float(full["nmi"]) > float(a["stage_a_best_nmi"])
                ),
                "partition_sha256": full["candidate_partition_sha256"],
            }
            contribution.append(row)
        lane_rows = [row for row in contribution if row["lane"] == lane]
        eligible = [row for row in lane_rows if row["independent_lane_pass"]]
        lane_best[lane] = max(
            eligible or lane_rows,
            key=lambda row: (float(row["full_ari"]), float(row["full_nmi"]), row["profile_id"], row["start_candidate"]),
        )
    write_csv(output / "junction_matched_contribution_board.csv", contribution)
    write_csv(output / "junction_best_full_by_lane.csv", [lane_best[lane] for lane in sorted(lane_best)])

    independent = [row for row in lane_best.values() if row["independent_lane_pass"]]
    covered_families = sorted(set(row["family"] for row in independent))
    macro_delta_ari = sum(float(row["delta_ari_vs_stage_a_best_head"]) for row in lane_best.values()) / len(lane_best)
    gate = len(independent) >= 2 and len(covered_families) == 2 and macro_delta_ari > 0.01
    decision = {
        "schema": "night22a-stage-b-mechanical-decision-v1",
        "independent_lane_pass_count": len(independent),
        "independent_lane_passes": [row["lane"] for row in independent],
        "covered_families": covered_families,
        "study_balanced_macro_delta_ari_vs_stage_a_best_head": macro_delta_ari,
        "family_frozen_confirmation_authorized": gate,
        "classification_if_no_confirmation": "NO_PARTITION_JUNCTION_SIGNAL" if not independent else "LOCAL_PARTITION_JUNCTION_SIGNAL",
        "labels_opened_only_by_independent_evaluator_after_candidate_lock": True,
    }
    (output / "stage_b_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
