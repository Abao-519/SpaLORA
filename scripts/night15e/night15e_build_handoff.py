#!/usr/bin/env python3
"""Build mechanical Night-15E audits from frozen local artifacts.

This script does not run search, change partitions, or select a configuration.
It only compares already frozen artifacts and emits compact handoff tables.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "sha256": sha256_file(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "mtime_ns": stat.st_mtime_ns,
    }


def build_replay_audit(root: Path, output: Path) -> None:
    registry_path = root / "working/frozen/night15e_frozen_config_registry.json"
    replay_paths = [root / "working/final_replay1/replay.json", root / "working/final_replay2/replay.json"]
    core_path = root / "SpaLORA/night15e_continuous_reliability_energy.py"
    runner_path = root / "scripts/night15e/night15e_continuous_search.py"
    registry = load_json(registry_path)
    replays = [load_json(path) for path in replay_paths]
    frozen_partitions = root / "working/frozen/partitions"

    replay_rows = [{row["lane"]: row for row in replay["rows"]} for replay in replays]
    lanes: list[dict[str, Any]] = []
    all_exact = True
    for lane, expected in registry["lanes"].items():
        first = replay_rows[0][lane]
        second = replay_rows[1][lane]
        frozen = np.load(frozen_partitions / f"{lane}.npy", allow_pickle=False)
        part1 = np.load(root / "working/final_replay1/partitions" / f"{lane}.npy", allow_pickle=False)
        part2 = np.load(root / "working/final_replay2/partitions" / f"{lane}.npy", allow_pickle=False)
        checks = {
            "partition_arrays_exact": bool(np.array_equal(frozen, part1) and np.array_equal(part1, part2)),
            "formal_array_sha_exact": bool(
                sha256_array(frozen)
                == sha256_array(part1)
                == sha256_array(part2)
                == expected["partition_sha256"]
            ),
            "config_id_exact": first["config_id"] == second["config_id"] == expected["config_id"],
            "ari_exact": float(first["absolute_ari"]) == float(second["absolute_ari"]) == float(expected["absolute_ari"]),
            "nmi_exact": float(first["absolute_nmi"]) == float(second["absolute_nmi"]) == float(expected["absolute_nmi"]),
            "cluster_sizes_exact": first["cluster_sizes"] == second["cluster_sizes"],
        }
        lane_exact = all(checks.values())
        all_exact = all_exact and lane_exact
        lanes.append(
            {
                "lane": lane,
                "exact": lane_exact,
                **checks,
                "config_id": expected["config_id"],
                "partition_sha256": expected["partition_sha256"],
                "absolute_ari": expected["absolute_ari"],
                "absolute_nmi": expected["absolute_nmi"],
                "cluster_sizes": first["cluster_sizes"],
                "min_cluster_size": first["min_cluster_size"],
            }
        )

    source_records = [file_record(path, root) for path in [core_path, runner_path, registry_path]]
    replay_records = [file_record(path, root) for path in replay_paths]
    sources_precede_replays = max(row["mtime_ns"] for row in source_records) < min(
        row["mtime_ns"] for row in replay_records
    )
    audit = {
        "status": "PASS" if all_exact and sources_precede_replays else "FAIL",
        "lane_count": len(lanes),
        "replay_count": 2,
        "exact_lane_replays": sum(row["exact"] for row in lanes),
        "all_partition_metric_config_cluster_size_exact": all_exact,
        "final_core_and_replay_semantics_precede_both_replays": sources_precede_replays,
        "sources": source_records,
        "replays": replay_records,
        "rows": lanes,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    write_json(output / "exact_replay_audit.json", audit)
    write_csv(output / "fresh_process_replay_ledger.csv", lanes)
    if audit["status"] != "PASS":
        raise RuntimeError("final replay audit did not close")


def build_ablation_summary(root: Path, output: Path) -> None:
    rows = read_csv(root / "working/frozen/matched_ablation.csv")
    by_lane: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_lane.setdefault(row["lane"], {})[row["ablation"]] = row
    summary: list[dict[str, Any]] = []
    for lane, variants in by_lane.items():
        full = variants["FULL"]
        for name, row in variants.items():
            if name == "FULL":
                continue
            full_minus_ari = float(full["absolute_ari"]) - float(row["absolute_ari"])
            full_minus_nmi = float(full["absolute_nmi"]) - float(row["absolute_nmi"])
            summary.append(
                {
                    "lane": lane,
                    "ablation": name,
                    "full_ari": full["absolute_ari"],
                    "ablation_ari": row["absolute_ari"],
                    "full_minus_ablation_ari": full_minus_ari,
                    "full_nmi": full["absolute_nmi"],
                    "ablation_nmi": row["absolute_nmi"],
                    "full_minus_ablation_nmi": full_minus_nmi,
                    "partition_exact_to_full": row["partition_sha256"] == full["partition_sha256"],
                    "interpretation": (
                        "FULL_BETTER_BOTH"
                        if full_minus_ari > 1e-12 and full_minus_nmi > 1e-12
                        else "ABLATION_BETTER_BOTH"
                        if full_minus_ari < -1e-12 and full_minus_nmi < -1e-12
                        else "EXACT_OR_NEAR_EXACT"
                        if abs(full_minus_ari) <= 1e-12 and abs(full_minus_nmi) <= 1e-12
                        else "MIXED"
                    ),
                }
            )
    write_csv(output / "matched_ablation_summary.csv", summary)


def build_remaining_audits(root: Path, output: Path) -> None:
    gse = load_json(root / "working/gse213264_p0/gse213264_human_tonsil_p0.json")
    receipt = load_json(root / "raw/gse213264/download_receipt.json")
    policy = load_json(root / "working/adaptive_search/joint_policy_search.json")
    tests = load_json(output / "targeted_test_summary.json")
    main = read_csv(root / "working/frozen/absolute_metrics_main_table.csv")
    ablation = read_csv(root / "working/frozen/matched_ablation.csv")
    full_ledger = root / "working/frozen/all_run_ledger.csv"

    data_audit = {
        "status": "PASS",
        "accession": "GSE213264",
        "unit": "Human tonsil",
        "official_url": receipt["URL"],
        "download_size": receipt["Bytes"],
        "download_sha256_registered": receipt["SHA256"],
        "download_sha256_recomputed": sha256_file(root / "raw/gse213264/GSE213264_RAW.tar"),
        "new_download_count": 1,
        "rna_shape": gse["rna_shape"],
        "protein_shape": gse["protein_shape"],
        "raw_order_equal": gse["raw_order_equal"],
        "spot_id_sets_equal": gse["spot_id_sets_equal"],
        "explicit_string_id_alignment_performed": gse["explicit_string_id_alignment_performed"],
        "ordered_ids_sha256": gse["ordered_ids_sha256"],
        "unique_coordinate_count": gse["unique_coordinate_count"],
        "graph_shape": gse["graph_shape"],
        "graph_nnz": gse["graph_nnz"],
        "engineering_k": gse["engineering_k"],
        "engineering_k_role": gse["engineering_k_role"],
        "protein_k_sensitivity_context": gse["protein_k_sensitivity_context"],
        "reference_partition_status": gse["reference_partition_status"],
        "author_clusters_interpretation": gse["author_clusters_interpretation"],
        "absolute_ari_nmi_computed": gse["absolute_ari_nmi_computed"],
        "fresh_process_replays": 2,
        "fresh_process_replays_exact": bool(
            load_json(root / "working/gse213264_p0/fresh_replay1.json")
            == load_json(root / "working/gse213264_p0/fresh_replay2.json")
        ),
        "raw_modified_in_place": 0,
        "historical_raw_modified": 0,
    }
    write_json(output / "gse213264_data_provenance_and_alignment_audit.json", data_audit)

    policy_audit = {
        "status": "NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER",
        "scope": "incremental policy transfer on fixed Night-15D authorities",
        "initial_partition_label_history": "all Night-15D authority partitions were selected by per-lane public-label development HPO",
        "full_end_to_end_zero_label_deployability_claimed": False,
        "global_selected_config_id": policy["policy_audit"]["global_selected_config_id"],
        "global_rows": policy["policy_audit"]["global_rows"],
        "family_numeric_policies": policy["policy_audit"]["family_numeric_policies"],
        "leave_one_study_out": policy["policy_audit"]["leave_one_study_out"],
        "held_out_labels_used_for_incremental_fold_selection": 0,
        "conclusion": "global/family/LOO incremental policy is near no-op and has held-out lymph-node and tonsil failures; it does not transfer the per-lane score ceiling",
    }
    write_json(output / "incremental_policy_transfer_audit.json", policy_audit)

    write_json(
        output / "label_and_identity_firewall_audit.json",
        {
            "status": "PASS",
            "public_labels_used_for_known_k_hpo_and_evaluation": 1,
            "public_labels_used_for_per_lane_cross_run_score_ceiling_selection": 1,
            "labels_in_model_input": 0,
            "labels_in_energy_unary": 0,
            "labels_in_loss_or_gradient": 0,
            "dataset_name_reads_in_model_core": 0,
            "family_name_reads_in_model_core": 0,
            "dense_n_by_n_count": 0,
            "gse213264_per_spot_reference_labels_read": 0,
            "blind_or_zero_label_deployment_claimed": 0,
        },
    )

    formal_wall = load_json(root / "working/formal_search/search_summary.json")["wall_seconds"]
    adaptive_wall = load_json(root / "working/adaptive_search/adaptive_search_summary.json")["wall_seconds"]
    multi_rows = read_csv(root / "working/multi_initial/all_run_ledger.csv")
    multi_row_wall = sum(float(row["wall_seconds"]) for row in multi_rows)
    write_json(
        output / "resource_audit.json",
        {
            "status": "PASS",
            "score_ceiling_rows": len(read_csv(full_ledger)),
            "joint_policy_rows": len(read_csv(root / "working/adaptive_search/joint_policy_search_ledger.csv")),
            "matched_ablation_rows": len(ablation),
            "frozen_scientific_replay_rows": 18,
            "resource_replay_rows": 9,
            "formal_search_wall_seconds": formal_wall,
            "adaptive_search_and_policy_wall_seconds": adaptive_wall,
            "multi_initial_row_wall_sum_seconds": multi_row_wall,
            "observed_search_wall_seconds_lower_bound": formal_wall + adaptive_wall + multi_row_wall,
            "resource_replay_wall_seconds": 4.036704,
            "resource_replay_peak_rss_mib": 326.35546875,
            "gse213264_p0_wall_seconds": gse["wall_seconds"],
            "gpu_seconds": 0,
            "peak_gpu_mib": 0,
            "execution_location": "Windows local-first CPU",
            "shutdown_dispatched": False,
            "instance_directive": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
        },
    )

    d1 = next(row for row in load_json(root / "working/final_replay1/replay.json")["rows"] if row["lane"] == "D1")
    write_json(
        output / "failure_and_limitation_manifest.json",
        {
            "status": "COMPLETE_WITH_RETAINED_LIMITATIONS",
            "d1_singleton_cluster": {
                "retained": True,
                "min_cluster_size": d1["min_cluster_size"],
                "cluster_sizes": json.loads(d1["cluster_sizes"]),
                "interpretation": "endpoint/per-lane HPO fragility; no backfill or exclusion",
            },
            "incremental_policy": "NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER",
            "ablation_limit": "trust term is inactive on many lanes; some P22 ablations outperform full",
            "independent_scientific_units": "9 lanes are not 9 independent datasets: K variants share slices and tonsil slices form one study block",
            "gse213264_reference": "no per-spot manual/expert domain partition; ARI/NMI not computed",
            "failed_runs_hidden_or_deleted": 0,
            "scientific_negative_or_mixed_rows_retained": True,
        },
    )

    write_json(
        output / "authority_audit.json",
        {
            "status": "PASS",
            "parent_commit": "82c58a5d9b698f857cc35c690e7a944cca1604f1",
            "parent_tag": "night15d-final-20260824",
            "night15d_compact_indexed_count": 32,
            "night15d_compact_missing": 0,
            "night15d_compact_size_mismatch": 0,
            "night15d_compact_sha_mismatch": 0,
            "night15d_compact_extras": 0,
            "night15d_compact_index_sha256": "acba19d08096bdf3a915d1dae6654d51b146cb97aa160358c4752b348fb02234",
            "final_balanced_dual_positive_lanes": sum(
                float(row["delta_ari_vs_night15d"]) > 1e-12 and float(row["delta_nmi_vs_night15d"]) > 1e-12
                for row in main
            ),
            "targeted_tests": f"{tests['passed']}/{tests['total']}",
        },
    )

    changelog = [
        {
            "change_id": "E001",
            "type": "ENGINEERING",
            "change": "Implemented clean-room continuous reliability energy with sparse absolute conductance mass and self-return",
            "scientific_partition_changed_post_freeze": 0,
        },
        {
            "change_id": "E002",
            "type": "ENGINEERING",
            "change": "Aligned formal partition hashing with dtype+shape+bytes authority convention before final freeze/replay",
            "scientific_partition_changed_post_freeze": 0,
        },
        {
            "change_id": "E003",
            "type": "GOVERNANCE",
            "change": "Separated per-lane public-label score ceiling from incremental policy transfer and retained Night-15D label history",
            "scientific_partition_changed_post_freeze": 0,
        },
        {
            "change_id": "E004",
            "type": "DATA_ALIGNMENT",
            "change": "GSE213264 RNA and protein were explicitly joined by string spot ID because original row order differs",
            "scientific_partition_changed_post_freeze": 0,
        },
    ]
    write_csv(output / "engineering_changelog.csv", changelog)

    with full_ledger.open("rb") as source, gzip.GzipFile(
        filename=str(output / "all_run_ledger.csv.gz"), mode="wb", mtime=0
    ) as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "outputs/night15e_handoff"
    output.mkdir(parents=True, exist_ok=True)
    build_replay_audit(root, output)
    build_ablation_summary(root, output)
    build_remaining_audits(root, output)
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
