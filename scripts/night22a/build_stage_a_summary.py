"""Summarize locked Night-22A geometry ceilings after independent evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows_from(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--parent-evaluations", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    geometry_dir = Path(args.geometry_dir)
    output_dir = Path(args.output_dir)
    rows = []
    for path in sorted(evaluation_dir.glob("*.csv")):
        rows.extend(rows_from(path))
    if not rows:
        raise RuntimeError("no geometry evaluations")
    for row in rows:
        row["evidence_board"] = "STAGE_A_LABEL_ASSISTED_GEOMETRY_CEILING"
    write_rows(output_dir / "geometry_ceiling_all.csv", rows)

    parent = rows_from(Path(args.parent_evaluations))
    lanes = sorted(set(row["lane"] for row in rows))
    summary = []
    selection = {
        "schema": "night22a-stage-b-label-assisted-development-start-selection-v1",
        "selection_rule": "maximum ARI; tie by NMI; then lexical representation source and candidate ID",
        "labels_used_only_after_geometry_banks_locked": True,
        "role": "transparent development start; not a family-frozen or label-free transfer rule",
        "lanes": {},
    }
    for lane in lanes:
        lane_rows = [row for row in rows if row["lane"] == lane]
        best = max(
            lane_rows,
            key=lambda row: (
                float(row["ari"]),
                float(row["nmi"]),
                row["representation_source"],
                row["candidate_id"],
            ),
        )
        parent_rows = [row for row in parent if row["lane"] == lane]
        parent_best = max(parent_rows, key=lambda row: (float(row["ari"]), float(row["nmi"])))
        bank_path = geometry_dir / f"{lane}__{best['representation_source']}.npz"
        manifest = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
        summary.append(
            {
                "lane": lane,
                "stage_a_best_representation_source": best["representation_source"],
                "stage_a_best_candidate_id": best["candidate_id"],
                "stage_a_best_ari": best["ari"],
                "stage_a_best_nmi": best["nmi"],
                "stage_a_min_cluster_size": best["min_cluster_size_full"],
                "night21c_endpoint_best_representation_source": parent_best["representation_source"],
                "night21c_endpoint_best_candidate_id": parent_best["candidate_id"],
                "night21c_endpoint_best_ari": parent_best["ari"],
                "night21c_endpoint_best_nmi": parent_best["nmi"],
                "delta_ari_vs_night21c_endpoint_best": float(best["ari"]) - float(parent_best["ari"]),
                "delta_nmi_vs_night21c_endpoint_best": float(best["nmi"]) - float(parent_best["nmi"]),
                "geometry_bank_sha256": manifest["candidate_bank_sha256"],
                "partition_sha256": best["candidate_partition_sha256"],
            }
        )
        selection["lanes"][lane] = {
            "representation_source": best["representation_source"],
            "candidate_id": best["candidate_id"],
            "geometry_bank": str(bank_path),
            "embedding_path": manifest["embedding_path"],
            "parent_bank_path": manifest["parent_bank_path"],
            "carrier_path": manifest["carrier_path"],
            "partition_sha256": best["candidate_partition_sha256"],
        }
    write_rows(output_dir / "geometry_ceiling_summary.csv", summary)
    (output_dir / "stage_b_development_start_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
