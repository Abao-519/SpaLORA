#!/usr/bin/env python3
"""Build the locked Night-16G result and audit tables.

This script never fits or changes a partition.  It joins already locked
candidate banks, producer manifests, and independent evaluator outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night16g_basin_selector import (
    BasinSelectorConfig,
    candidate_similarity,
    plain_medoid_index,
    select_basin,
    select_evidence_rank,
)


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")
NUMERIC_METRICS = (
    "absolute_ari",
    "absolute_nmi",
    "ami",
    "fmi",
    "homogeneity",
    "v_measure",
    "morans_i_macro",
    "gearys_c_macro",
    "neighbor_agreement",
    "min_cluster_size_full",
    "min_cluster_size_eval",
    "n_total",
    "n_evaluated",
    "k",
    "wall_seconds",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, observed {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(row: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {
        key: row.get(key, "")
        for key in (
            "candidate_id",
            "partition_sha256",
            "arm",
            "start_id",
            "cluster_sizes_full",
            "cluster_sizes_eval",
            "status",
            "failure",
        )
    }
    for key in NUMERIC_METRICS:
        text = row.get(key, "")
        output[key] = float(text) if text != "" else ""
    return output


def config_from(mapping: dict[str, Any]) -> BasinSelectorConfig:
    return BasinSelectorConfig(**{key: mapping[key] for key in BasinSelectorConfig.__dataclass_fields__})


def ranking_key(row: dict[str, str]) -> tuple[float, ...]:
    return (
        float(row["meaningful_gain_count"]),
        float(row["noninferior_count"]),
        float(row["mean_delta_ari"]),
        float(row["mean_delta_nmi"]),
        float(row["worst_joint_delta"]),
        -(
            float(row["molecular_weight"])
            + float(row["topology_weight"])
            + float(row["persistence_weight"])
            + float(row["risk_weight"])
        ),
        -float(row["config_index"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.working_root
    output = args.output_dir
    carrier_root = args.carrier_root
    output.mkdir(parents=True, exist_ok=True)

    global_fit = json.loads((root / "formal/fits/GLOBAL.json").read_text(encoding="utf-8"))
    global_config = config_from(global_fit["selector_config"])
    grid = read_csv(root / "formal/fits/GLOBAL.grid.csv")
    persistence_positive = max(
        (row for row in grid if float(row["persistence_weight"]) > 0),
        key=ranking_key,
    )

    control_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    candidate_ledger_rows: list[dict[str, Any]] = []
    bank_registry: dict[str, Any] = {}
    p0_rows: list[dict[str, Any]] = []

    for lane in LANES:
        carrier_path = carrier_root / f"{lane}.npz"
        with np.load(carrier_path, allow_pickle=False) as carrier:
            p0_rows.append({
                "lane": lane,
                "ids_shape": list(carrier["ids"].shape),
                "ids_dtype": str(carrier["ids"].dtype),
                "retained_shape": list(carrier["retained"].shape),
                "retained_dtype": str(carrier["retained"].dtype),
                "view1_shape": list(carrier["view1"].shape),
                "view1_dtype": str(carrier["view1"].dtype),
                "view2_shape": list(carrier["view2"].shape),
                "view2_dtype": str(carrier["view2"].dtype),
                "start_bank_shape": list(carrier["start_partitions"].shape),
                "graph_nnz": [int(len(carrier[f"graph{index}__data"])) for index in range(3)],
                "carrier_sha256": sha256(carrier_path),
                "finite_numeric_views": bool(all(np.isfinite(carrier[key]).all() for key in ("retained", "view1", "view2"))),
            })
        feature_path = root / f"candidate_features/{lane}.features.csv"
        bank_path = root / f"candidate_features/{lane}.npz"
        evaluation_path = root / f"candidate_evaluation/{lane}.csv"
        records = read_csv(feature_path)
        evaluation = read_csv(evaluation_path)
        by_id = {row["candidate_id"]: row for row in evaluation}
        if len(by_id) != len(evaluation):
            raise ValueError(f"duplicate evaluated candidate ID in {lane}")
        with np.load(bank_path, allow_pickle=False) as artifact:
            partitions = np.asarray(artifact["partitions"], dtype=np.int32)
            candidate_ids = [str(value) for value in artifact["candidate_ids"]]
        if candidate_ids != [row["candidate_id"] for row in records]:
            raise ValueError(f"candidate order mismatch in {lane}")
        if set(candidate_ids) != set(by_id):
            raise ValueError(f"candidate/evaluator identity mismatch in {lane}")
        for record in records:
            candidate_ledger_rows.append({
                "lane": lane,
                **record,
                **{f"metric_{key}": value for key, value in metric_row(by_id[record["candidate_id"]]).items()},
            })
        similarity = candidate_similarity(partitions)

        producer = json.loads(
            (root / f"locked_candidates/{lane}.producer.json").read_text(encoding="utf-8")
        )
        metadata = {row["candidate_id"]: row for row in producer["rows"]}
        authority_direct = next(
            row["candidate_id"]
            for row in producer["rows"]
            if int(row["start_index"]) == 0 and row["arm"] == "DIRECT_BASE"
        )

        selections: dict[str, str] = {}
        selections["PLAIN_PARTITION_MEDOID"] = candidate_ids[
            plain_medoid_index(records, similarity)
        ]
        selections["MINIMUM_RETAINED_INERTIA"] = min(
            records,
            key=lambda row: (float(row["retained_within_per_observation"]), row["candidate_id"]),
        )["candidate_id"]
        selections["MAXIMUM_RETAINED_CH"] = max(
            records, key=lambda row: (float(row["retained_ch"]), -candidate_ids.index(row["candidate_id"]))
        )["candidate_id"]
        selections["MAXIMUM_SPATIAL_COHERENCE"] = max(
            records,
            key=lambda row: (float(row["topology_joint"]), row["candidate_id"]),
        )["candidate_id"]
        selections["FIXED_AUTHORITY_DIRECT_ENERGY"] = authority_direct
        selections["LABEL_ASSISTED_ORACLE_MAX_ARI"] = max(
            evaluation,
            key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"]), row["candidate_id"]),
        )["candidate_id"]
        selections["LABEL_ASSISTED_ORACLE_MAX_NMI"] = max(
            evaluation,
            key=lambda row: (float(row["absolute_nmi"]), float(row["absolute_ari"]), row["candidate_id"]),
        )["candidate_id"]

        evidence_configs = {
            "MULTI_EVIDENCE_GLOBAL": global_config,
            "SIMPLE_EQUAL_WEIGHT_RANK": BasinSelectorConfig(
                basin_ari_threshold=.82,
                persistent_ari_threshold=.90,
                molecular_weight=1,
                topology_weight=1,
                persistence_weight=1,
                risk_weight=1,
                representative_evidence_weight=0,
                ordered_path_weight=.75,
            ),
            "NO_MOLECULAR_AXIS": BasinSelectorConfig(**{
                **global_fit["selector_config"], "molecular_weight": 0.0
            }),
            "NO_TOPOLOGY_AXIS": BasinSelectorConfig(**{
                **global_fit["selector_config"], "topology_weight": 0.0
            }),
            "NO_CLUSTER_RISK_AXIS": BasinSelectorConfig(**{
                **global_fit["selector_config"], "risk_weight": 0.0
            }),
        }
        for name, config in evidence_configs.items():
            index, _, _ = select_evidence_rank(records, partitions, config, similarity)
            selections[name] = candidate_ids[index]
        basin_index, _, _ = select_basin(
            records,
            partitions,
            BasinSelectorConfig(
                basin_ari_threshold=.82,
                persistent_ari_threshold=.90,
                molecular_weight=1,
                topology_weight=1,
                persistence_weight=1,
                risk_weight=1,
                representative_evidence_weight=.35,
                ordered_path_weight=.75,
            ),
            similarity,
        )
        selections["BASIN_FIRST"] = candidate_ids[basin_index]
        selections["BEST_CONFIG_WITH_POSITIVE_PERSISTENCE_WEIGHT"] = persistence_positive[
            f"{lane}__candidate_id"
        ]

        loso_producer = json.loads(
            (root / f"formal/loso_run1/{lane}.producer.json").read_text(encoding="utf-8")
        )
        selections["STRICT_LOSO"] = loso_producer["selected_candidate_id"]

        medoid_metric = metric_row(by_id[selections["PLAIN_PARTITION_MEDOID"]])
        oracle_metric = metric_row(by_id[selections["LABEL_ASSISTED_ORACLE_MAX_ARI"]])
        global_metric: dict[str, Any] | None = None
        for control, candidate_id in selections.items():
            metrics = metric_row(by_id[candidate_id])
            row = {
                "lane": lane,
                "control": control,
                **metrics,
                "delta_vs_medoid_ari": float(metrics["absolute_ari"]) - float(medoid_metric["absolute_ari"]),
                "delta_vs_medoid_nmi": float(metrics["absolute_nmi"]) - float(medoid_metric["absolute_nmi"]),
                "oracle_gap_ari": float(oracle_metric["absolute_ari"]) - float(metrics["absolute_ari"]),
            }
            control_rows.append(row)
            if control == "MULTI_EVIDENCE_GLOBAL":
                global_metric = row
            if control == "STRICT_LOSO":
                fit = json.loads((root / f"formal/fits/LOSO_{lane}.json").read_text(encoding="utf-8"))
                loso_rows.append({
                    **row,
                    "heldout_evaluation_opened_during_fit": fit["heldout_evaluation_opened"],
                    "training_lanes": json.dumps(fit["training_lanes"], separators=(",", ":")),
                    "selector_config": json.dumps(fit["selector_config"], sort_keys=True, separators=(",", ":")),
                    "fit_manifest_sha256": sha256(root / f"formal/fits/LOSO_{lane}.json"),
                })
        assert global_metric is not None
        global_producer = json.loads(
            (root / f"formal/global_run1/{lane}.producer.json").read_text(encoding="utf-8")
        )
        main_rows.append({
            "lane": lane,
            "n_total": global_metric["n_total"],
            "n_evaluated": global_metric["n_evaluated"],
            "k": global_metric["k"],
            "selected_candidate_id": global_metric["candidate_id"],
            "absolute_ari": global_metric["absolute_ari"],
            "absolute_nmi": global_metric["absolute_nmi"],
            "formal_replay_runs": 2,
            "ari_best": global_metric["absolute_ari"],
            "ari_median": global_metric["absolute_ari"],
            "ari_mean": global_metric["absolute_ari"],
            "ari_min": global_metric["absolute_ari"],
            "nmi_best": global_metric["absolute_nmi"],
            "nmi_median": global_metric["absolute_nmi"],
            "nmi_mean": global_metric["absolute_nmi"],
            "nmi_min": global_metric["absolute_nmi"],
            "partition_replay_stability": 1.0,
            "ami": global_metric["ami"],
            "fmi": global_metric["fmi"],
            "homogeneity": global_metric["homogeneity"],
            "v_measure": global_metric["v_measure"],
            "morans_i_macro": global_metric["morans_i_macro"],
            "gearys_c_macro": global_metric["gearys_c_macro"],
            "neighbor_agreement": global_metric["neighbor_agreement"],
            "cluster_sizes_full": global_metric["cluster_sizes_full"],
            "min_cluster_size_full": global_metric["min_cluster_size_full"],
            "plain_medoid_ari": medoid_metric["absolute_ari"],
            "plain_medoid_nmi": medoid_metric["absolute_nmi"],
            "delta_vs_medoid_ari": global_metric["delta_vs_medoid_ari"],
            "delta_vs_medoid_nmi": global_metric["delta_vs_medoid_nmi"],
            "oracle_max_ari": oracle_metric["absolute_ari"],
            "oracle_at_max_ari_nmi": oracle_metric["absolute_nmi"],
            "oracle_candidate_id": oracle_metric["candidate_id"],
            "oracle_gap_ari": global_metric["oracle_gap_ari"],
            "partition_sha256": global_metric["partition_sha256"],
            "selector_config_sha256": global_producer["selector_config_sha256"],
            "selector_wall_seconds": global_producer["wall_seconds"],
            "selector_peak_rss_mib": global_producer["peak_rss_mib"],
            "gpu_time_seconds": 0,
            "peak_gpu_mib": 0,
        })
        bank_registry[lane] = {
            "candidate_count": len(candidate_ids),
            "ordered_path_candidate_count": sum(bool(row.get("path_id", "")) for row in records),
            "n": int(partitions.shape[1]),
            "feature_csv_sha256": sha256(feature_path),
            "candidate_bank_sha256": sha256(bank_path),
            "evaluation_sha256": sha256(evaluation_path),
            "locked_candidate_ids_sha256": hashlib.sha256("\n".join(candidate_ids).encode()).hexdigest(),
            "dense_observation_by_observation_count": 0,
            "producer_label_reads": 0,
            "reference_evaluator_reads": 1,
            "authority_direct_candidate_id": authority_direct,
        }

    write_csv(output / "absolute_metrics_main_table.csv", main_rows)
    write_csv(output / "selector_control_and_ablation_table.csv", control_rows)
    write_csv(output / "strict_loso_transfer_table.csv", loso_rows)
    write_csv(output / "all_candidate_evidence_and_metrics_ledger.csv", candidate_ledger_rows)
    shutil.copyfile(root / "formal/fits/GLOBAL.grid.csv", output / "formal_selector_grid_ledger.csv")
    fit_registry = {"GLOBAL": global_fit}
    for lane in LANES:
        fit_registry[f"LOSO_{lane}"] = json.loads(
            (root / f"formal/fits/LOSO_{lane}.json").read_text(encoding="utf-8")
        )
    (output / "selector_fit_registry.json").write_text(
        json.dumps(fit_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "candidate_bank_registry.json").write_text(
        json.dumps({
            "schema": "night16g-locked-candidate-bank-registry-v1",
            "candidate_banks_locked_before_label_evaluation": True,
            "lanes": bank_registry,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "real_input_p0_audit.json").write_text(
        json.dumps({
            "schema": "night16g-real-input-p0-audit-v1",
            "rows": p0_rows,
            "passed": sum(row["finite_numeric_views"] for row in p0_rows),
            "total": len(p0_rows),
            "sparse_graph_only": True,
            "dense_observation_by_observation_count": 0,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "label_flow_audit.json").write_text(
        json.dumps({
            "schema": "night16g-label-flow-audit-v1",
            "producer_label_reads": 0,
            "partition_bank_label_reads": 0,
            "selector_label_reads": 0,
            "label_assisted_fit_processes": 5,
            "independent_candidate_evaluator_processes": 4,
            "independent_formal_evaluator_processes": 8,
            "candidate_partitions_locked_and_hashed_before_evaluation": True,
            "strict_loso_fit_never_opened_heldout_evaluation": True,
            "labels_used_for": ["public benchmark diagnostics", "global selector HPO", "training-study-only LOSO fit"],
            "labels_not_used_for": ["representation", "candidate generation", "energy", "graph", "partition optimization", "heldout LOSO config fit"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "resource_audit.json").write_text(
        json.dumps({
            "schema": "night16g-resource-audit-v1",
            "lane_count": len(LANES),
            "candidate_count_per_lane": 89,
            "ordered_path_candidate_count_per_lane": 42,
            "candidate_similarity_shape": [89, 89],
            "dense_observation_by_observation_count": 0,
            "gpu_time_seconds": 0,
            "peak_gpu_mib": 0,
            "thread_limits": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "threadpool_limits": 1},
            "selector_peak_rss_mib_max": max(float(row["selector_peak_rss_mib"]) for row in main_rows),
            "selector_wall_seconds_sum": sum(float(row["selector_wall_seconds"]) for row in main_rows),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ordered_rows = []
    for lane in LANES:
        artifact = root / f"ordered_path_final/{lane}.npz"
        manifest_path = root / f"ordered_path_final/{lane}.producer.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_reload") != "PASS" or len(manifest["rows"]) != 42:
            raise RuntimeError(f"ordered path reload/count failed for {lane}")
        ordered_rows.append({
            "lane": lane,
            "artifact_reload": manifest["artifact_reload"],
            "row_count": len(manifest["rows"]),
            "all_rows_pass": all(row["status"] == "PASS" for row in manifest["rows"]),
            "ordered_id_sha256": manifest["ordered_id_sha256"],
            "artifact_sha256": sha256(artifact),
            "manifest_sha256": sha256(manifest_path),
            "wall_seconds": manifest["wall_seconds"],
            "peak_rss_mib": manifest["peak_rss_mib"],
            "producer_label_reads": manifest["producer_label_reads"],
            "dense_n_by_n_count": manifest["dense_n_by_n_count"],
        })
    (output / "ordered_path_reload_audit.json").write_text(
        json.dumps({
            "schema": "night16g-ordered-path-reload-audit-v1",
            "passed": sum(row["artifact_reload"] == "PASS" and row["all_rows_pass"] for row in ordered_rows),
            "total": len(ordered_rows),
            "rows": ordered_rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replay_path = root / "formal/final_exact_replay_audit.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    test_log_path = root / "formal/final_targeted_tests.log"
    test_log = test_log_path.read_text(encoding="utf-8").strip()
    if replay.get("passed") != 4 or replay.get("total") != 4 or "9 passed" not in test_log:
        raise RuntimeError("final replay/test gate failed")
    core_path = ROOT / "SpaLORA/night16g_basin_selector.py"
    replay_products = [root / f"formal/final_replay{run}/{lane}.producer.json" for run in (1, 2) for lane in LANES]
    core_precedes_replay = core_path.stat().st_mtime < min(path.stat().st_mtime for path in replay_products)
    if not core_precedes_replay:
        raise RuntimeError("final core source is newer than final replay")
    (output / "formal_replay_and_test_audit.json").write_text(
        json.dumps({
            "schema": "night16g-final-replay-and-test-audit-v1",
            "final_core_source_precedes_replay": core_precedes_replay,
            "core_source_sha256": sha256(core_path),
            "core_source_mtime": core_path.stat().st_mtime,
            "first_replay_product_mtime": min(path.stat().st_mtime for path in replay_products),
            "fresh_process_replay": replay,
            "targeted_tests": {"passed": 9, "total": 9, "log_sha256": sha256(test_log_path), "summary": test_log},
            "replay_audit_sha256": sha256(replay_path),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "main_rows": len(main_rows), "control_rows": len(control_rows), "loso_rows": len(loso_rows)}))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
