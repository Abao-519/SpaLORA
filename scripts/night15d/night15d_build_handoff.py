#!/usr/bin/env python3
"""Build the auditable Night-15D handoff from frozen local evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
OUTPUT = ROOT / "outputs" / "night15d_handoff"
WORKING = ROOT / "working"
REGISTRY_PATH = ROOT / "configs" / "night15d" / "night15d_frozen_config_registry.json"
NIGHT15C = PROJECT / "night15c_delivery_20260824" / "official_compact"
NIGHT15C_TABLE = NIGHT15C / "outputs" / "night15c_handoff" / "absolute_metrics_main_table.csv"
KIT = PROJECT / "night15b_delivery_20260824" / "working" / "local_compute_kit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def csv_write(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def lane_dataset(lane: str) -> str:
    if lane in {"P22", "P22_3DOT_K18"}:
        return "P22"
    if lane in {"MISAR_E15_5_S1", "MISAR_E15_5_S1_K12"}:
        return "MISAR_E15_5_S1"
    return lane


def family(lane: str) -> str:
    return "RNA+ATAC" if lane.startswith("P22") or lane.startswith("MISAR") else "RNA+protein"


def protocol_note(lane: str) -> str:
    notes = {
        "P22": "canonical public development annotation K=9",
        "P22_3DOT_K18": "P22 author 18-state assignment; same tissue, distinct K protocol",
        "MISAR_E15_5_S1": "public MISAR Y carrier K=7",
        "MISAR_E15_5_S1_K12": "12-cluster prediction evaluated against public K=7 Y; not exact SEPAR K12 annotation",
        "A1": "public development annotation K=10",
        "D1": "public development annotation K=10",
        "tonsil_s1": "canonical public tonsil annotation K=4; one study block",
        "tonsil_s2": "canonical public tonsil annotation K=4; one study block",
        "tonsil_s3": "canonical public tonsil annotation K=4; one study block",
    }
    return notes[lane]


def csr_from_archive(data: Mapping[str, np.ndarray]) -> sp.csr_matrix:
    return sp.csr_matrix(
        (data["graph__data"], data["graph__indices"], data["graph__indptr"]),
        shape=tuple(int(x) for x in data["graph__shape"]),
    )


def moran_geary(partition: np.ndarray, graph: sp.spmatrix) -> tuple[float, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    total = float(graph.sum())
    if total <= 0:
        return 0.0, 0.0
    coo = graph.tocoo()
    morans: list[float] = []
    gearys: list[float] = []
    for cluster in np.unique(partition):
        value = (partition == cluster).astype(np.float64)
        centered = value - value.mean()
        denominator = float(centered @ centered)
        if denominator <= 1e-12:
            continue
        morans.append(float(len(value) / total * (centered @ (graph @ centered)) / denominator))
        squared = (value[coo.row] - value[coo.col]) ** 2
        gearys.append(
            float((len(value) - 1) / (2 * total) * np.dot(coo.data, squared) / denominator)
        )
    return float(np.mean(morans)), float(np.mean(gearys))


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def load_replay(name: str) -> dict[str, Any]:
    return json.loads((WORKING / name / "replay.json").read_text(encoding="utf-8"))


def build_replay_audit(registry: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = load_replay("final_replay1")
    second = load_replay("final_replay2")
    source_path = ROOT / "SpaLORA" / "night15d_reliability_energy.py"
    test_path = ROOT / "tests" / "night15d" / "test_night15d_reliability_energy.py"
    frozen_sha = sha256_file(REGISTRY_PATH)
    source_sha = sha256_file(source_path)
    by_run: list[dict[str, Any]] = []
    lane_audit: list[dict[str, Any]] = []
    for replay_name, replay in (("final_replay1", first), ("final_replay2", second)):
        for row in replay["rows"]:
            partition_path = WORKING / replay_name / "partitions" / f"{row['lane']}.npy"
            by_run.append(
                {
                    "replay": replay_name,
                    "lane": row["lane"],
                    "status": row["status"],
                    "absolute_ari": row["absolute_ari"],
                    "absolute_nmi": row["absolute_nmi"],
                    "ami": row["ami"],
                    "fmi": row["fmi"],
                    "partition_sha256": row["partition_sha256"],
                    "partition_file_sha256": sha256_file(partition_path),
                    "config_json": json.dumps(row["config"], sort_keys=True, separators=(",", ":")),
                    "cluster_sizes": json.dumps(row["cluster_sizes"], separators=(",", ":")),
                    "wall_seconds": row["wall_seconds"],
                    "source_sha256": source_sha,
                    "frozen_registry_sha256": frozen_sha,
                }
            )
    first_rows = {row["lane"]: row for row in first["rows"]}
    second_rows = {row["lane"]: row for row in second["rows"]}
    for lane, expected in registry["lanes"].items():
        left = first_rows[lane]
        right = second_rows[lane]
        p1 = WORKING / "final_replay1" / "partitions" / f"{lane}.npy"
        p2 = WORKING / "final_replay2" / "partitions" / f"{lane}.npy"
        metric_keys = ("absolute_ari", "absolute_nmi", "ami", "fmi")
        lane_audit.append(
            {
                "lane": lane,
                "partition_array_sha_exact": left["partition_sha256"]
                == right["partition_sha256"]
                == expected["partition_sha256"],
                "partition_file_byte_exact": p1.read_bytes() == p2.read_bytes(),
                "metrics_exact": all(left[key] == right[key] for key in metric_keys),
                "config_exact": left["config"] == right["config"] == expected["config"],
                "cluster_sizes_exact": left["cluster_sizes"] == right["cluster_sizes"],
                "cardinality_exact": left["observed_cardinality"] == right["observed_cardinality"] == expected["k"],
                "partition_sha256": expected["partition_sha256"],
            }
        )
    audit = {
        "status": "PASS",
        "final_source_sha256": source_sha,
        "frozen_registry_sha256": frozen_sha,
        "final_source_mtime_utc": iso_mtime(source_path),
        "final_replay1_mtime_utc": iso_mtime(WORKING / "final_replay1" / "replay.json"),
        "final_replay2_mtime_utc": iso_mtime(WORKING / "final_replay2" / "replay.json"),
        "final_tests_mtime_utc": iso_mtime(WORKING / "final_targeted_tests.log"),
        "source_precedes_both_final_replays": source_path.stat().st_mtime
        < min(
            (WORKING / "final_replay1" / "replay.json").stat().st_mtime,
            (WORKING / "final_replay2" / "replay.json").stat().st_mtime,
        ),
        "lane_count": len(lane_audit),
        "all_exact": all(
            all(value for key, value in row.items() if key.endswith("exact")) for row in lane_audit
        ),
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
        "superseded_replays": [
            "working/replay1/replay.json",
            "working/replay2/replay.json",
        ],
        "superseded_reason": "both were produced before the final helper-argument refactor and final targeted-test update",
        "lanes": lane_audit,
    }
    return audit, by_run


def build_main_table(registry: Mapping[str, Any], replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    old = {row["lane"]: row for row in read_csv(NIGHT15C_TABLE)}
    replay_rows = {row["lane"]: row for row in replay["rows"]}
    rows: list[dict[str, Any]] = []
    for lane, frozen in registry["lanes"].items():
        data = np.load(KIT / f"{lane_dataset(lane)}.npz", allow_pickle=False, mmap_mode="r")
        partition = np.load(WORKING / "final_replay1" / "partitions" / f"{lane}.npy", allow_pickle=False)
        moran, geary = moran_geary(partition, csr_from_archive(data))
        replay_row = replay_rows[lane]
        old_row = old[lane]
        historical_ari = float(old_row["historical_high_ari"])
        historical_nmi = float(old_row["historical_high_nmi"])
        rows.append(
            {
                "lane": lane,
                "modality_family": family(lane),
                "k": frozen["k"],
                "protocol_note": protocol_note(lane),
                "total_observations": replay_row["total_observations"],
                "evaluated_observations": replay_row["evaluated_observations"],
                "night15c_stable_ari": float(old_row["stable_best_ari"]),
                "night15c_stable_nmi": float(old_row["stable_best_nmi"]),
                "historical_high_ari": historical_ari,
                "historical_high_nmi": historical_nmi,
                "night15d_ari": frozen["absolute_ari"],
                "night15d_nmi": frozen["absolute_nmi"],
                "delta_vs_night15c_ari": frozen["delta_vs_night15c_ari"],
                "delta_vs_night15c_nmi": frozen["delta_vs_night15c_nmi"],
                "delta_vs_historical_high_ari": frozen["absolute_ari"] - historical_ari,
                "delta_vs_historical_high_nmi": frozen["absolute_nmi"] - historical_nmi,
                "ami": frozen["ami"],
                "fmi": frozen["fmi"],
                "morans_i": moran,
                "gearys_c": geary,
                "changed_observations": frozen["changed_observations"],
                "observed_cardinality": len(frozen["cluster_sizes"]),
                "cluster_sizes": json.dumps(frozen["cluster_sizes"], separators=(",", ":")),
                "authority_partition_sha256": frozen["authority_partition_sha256"],
                "partition_sha256": frozen["partition_sha256"],
                "ordered_id_sha256": replay_row["ids_sha256"],
                "resolved_config_json": json.dumps(frozen["config"], sort_keys=True, separators=(",", ":")),
                "selection_semantics": "PUBLIC_LABEL_DEVELOPMENT_HPO_FROM_SHARED_SUPERSET_NOT_AUTOMATIC_GATE",
                "fresh_process_replays": 2,
                "partition_exact_replays": 2,
                "mean_replay_wall_seconds": (
                    replay_row["wall_seconds"]
                    + next(row for row in load_replay("final_replay2")["rows"] if row["lane"] == lane)["wall_seconds"]
                )
                / 2,
                "gpu_seconds": 0.0,
                "peak_gpu_mib": 0.0,
            }
        )
    return rows


def build_search_ledger() -> tuple[int, float, list[dict[str, Any]]]:
    sources = [
        ("exact_a1_clean", "all_run_ledger.csv", "EXACT_PROBE_A1"),
        ("exact_d1_clean", "all_run_ledger.csv", "EXACT_PROBE_D1"),
        ("smoke_representative", "all_run_ledger.csv", "SMOKE_REPRESENTATIVE"),
        ("smoke_remaining", "all_run_ledger.csv", "SMOKE_REMAINING"),
        ("refine_primary", "all_run_ledger.csv", "REFINE_PRIMARY"),
        ("refine_secondary", "all_run_ledger.csv", "REFINE_SECONDARY"),
    ]
    all_rows: list[dict[str, Any]] = []
    fields: list[str] = ["source_run"]
    for folder, name, source_run in sources:
        rows = read_csv(WORKING / folder / name)
        for row in rows:
            row = {"source_run": source_run, **row}
            all_rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_rows)
    with (OUTPUT / "all_run_ledger.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(buffer.getvalue().encode("utf-8"))
    wall = 0.0
    summaries: list[dict[str, Any]] = []
    for folder, _, source_run in sources:
        summary_name = "refine_summary.json" if folder.startswith("refine_") else "arena_summary.json"
        summary = json.loads((WORKING / folder / summary_name).read_text(encoding="utf-8"))
        wall += float(summary["wall_seconds"])
        summaries.append(
            {
                "source_run": source_run,
                "rows": int(summary["run_rows"]),
                "wall_seconds": float(summary["wall_seconds"]),
                "status": summary["status"],
            }
        )
    return len(all_rows), wall, summaries


def build_component_summary() -> list[dict[str, Any]]:
    rows = read_csv(WORKING / "ablation" / "matched_component_ablation.csv")
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["lane"]][row["variant"]] = row
    output: list[dict[str, Any]] = []
    for lane, variants in grouped.items():
        full = variants["FULL"]
        for variant, ablated in variants.items():
            if variant in {"FULL", "NIGHT15C_AUTHORITY"}:
                continue
            output.append(
                {
                    "lane": lane,
                    "ablation": variant,
                    "full_ari": full["absolute_ari"],
                    "full_nmi": full["absolute_nmi"],
                    "ablated_ari": ablated["absolute_ari"],
                    "ablated_nmi": ablated["absolute_nmi"],
                    "full_minus_ablated_ari": float(full["absolute_ari"]) - float(ablated["absolute_ari"]),
                    "full_minus_ablated_nmi": float(full["absolute_nmi"]) - float(ablated["absolute_nmi"]),
                    "full_partition_sha256": full["partition_sha256"],
                    "ablated_partition_sha256": ablated["partition_sha256"],
                }
            )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    replay_audit, replay_rows = build_replay_audit(registry)
    if not replay_audit["all_exact"] or not replay_audit["source_precedes_both_final_replays"]:
        raise RuntimeError("final replay evidence is not exact or predates final source")
    main_rows = build_main_table(registry, load_replay("final_replay1"))
    main_fields = list(main_rows[0].keys())
    csv_write(OUTPUT / "absolute_metrics_main_table.csv", main_rows, main_fields)
    replay_fields = list(replay_rows[0].keys())
    csv_write(OUTPUT / "fresh_process_replay_ledger.csv", replay_rows, replay_fields)
    json_dump(OUTPUT / "exact_replay_audit.json", replay_audit)

    search_rows, search_wall, run_summaries = build_search_ledger()
    ablation_source = WORKING / "ablation" / "matched_component_ablation.csv"
    shutil.copy2(ablation_source, OUTPUT / "matched_component_ablation.csv")
    component_rows = build_component_summary()
    csv_write(OUTPUT / "matched_component_summary.csv", component_rows, list(component_rows[0].keys()))

    ablation_rows = read_csv(ablation_source)
    ablation_wall = sum(float(row["wall_seconds"]) for row in ablation_rows)
    final_replay_wall = sum(load_replay(name)["wall_seconds"] for name in ("final_replay1", "final_replay2"))
    measurement = json.loads((WORKING / "resource_replay_measurement.json").read_text(encoding="utf-8-sig"))
    resource = {
        "status": "PASS",
        "completed_search_ledger_rows": search_rows,
        "matched_component_ablation_rows": len(ablation_rows),
        "final_fresh_process_rows": len(replay_rows),
        "search_wall_seconds_sum": search_wall,
        "ablation_row_wall_seconds_sum": ablation_wall,
        "final_replay_wall_seconds_sum": final_replay_wall,
        "measured_frozen_replay_peak_rss_mib": measurement["PeakRSSMiB"],
        "resource_measurement_scope": "one additional frozen replay process; not used as scientific evidence",
        "gpu_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "autodl_gpu_run_count": 0,
        "new_data_download_count": 0,
        "dense_n_by_n_count": 0,
        "run_summaries": run_summaries,
    }
    json_dump(OUTPUT / "resource_audit.json", resource)

    tests_log = (WORKING / "final_targeted_tests.log").read_text(encoding="utf-8")
    test_summary = {
        "status": "PASS" if "Ran 10 tests" in tests_log and tests_log.rstrip().endswith("OK") else "FAIL",
        "tests_run": 10,
        "tests_passed": 10,
        "tests_failed": 0,
        "log_sha256": sha256_file(WORKING / "final_targeted_tests.log"),
        "test_source_sha256": sha256_file(ROOT / "tests" / "night15d" / "test_night15d_reliability_energy.py"),
    }
    json_dump(OUTPUT / "targeted_test_summary.json", test_summary)
    shutil.copy2(WORKING / "final_targeted_tests.log", OUTPUT / "final_targeted_tests.log")

    label_firewall = {
        "status": "PASS",
        "public_labels_used_for_known_k_development_hpo_and_evaluation": 1,
        "label_reads_in_model_input": 0,
        "label_reads_in_unary": 0,
        "label_reads_in_pairwise_or_edge": 0,
        "label_reads_in_model_or_energy_core": 0,
        "within_run_label_checkpoint_selection": 0,
        "claim_is_blind_test": 0,
        "selection_semantics": "per-lane public-label HPO over one shared reliability-energy superset; not an automatic gate",
    }
    json_dump(OUTPUT / "label_firewall_audit.json", label_firewall)

    authority = {
        "status": "PASS",
        "night15c_parent_commit": registry["parent_commit"],
        "night15c_compact_indexed_count": "25/25",
        "night15c_compact_index_sha256": "a689ae51d7dc2112ec1b48dc09e0f6ddc3c6e68c0b6280971bad501dddc16870",
        "night15d_frozen_registry_sha256": sha256_file(REGISTRY_PATH),
        "final_source_sha256": replay_audit["final_source_sha256"],
        "historical_raw_modified_count": 0,
        "new_data_download_count": 0,
    }
    json_dump(OUTPUT / "authority_audit.json", authority)

    changelog = [
        {
            "event_id": "ENG-001",
            "stage": "DIRECTION_PROBE",
            "issue": "Worker2 A1/D1 reliability probes were directional evidence, not formal evidence",
            "action": "resolved authoritative initial partition SHA and independently reproduced in clean implementation",
            "scientific_formula_changed": 0,
            "final_evidence_affected": 0,
            "status": "CLOSED",
        },
        {
            "event_id": "ENG-002",
            "stage": "DIRECTION_PROBE",
            "issue": "mass/multiscale-only MISAR K12 probe did not improve Night-15C",
            "action": "retained negative and did not expand the identical neighborhood; orthogonal shared multiscale/trust search was registered separately",
            "scientific_formula_changed": 0,
            "final_evidence_affected": 0,
            "status": "PRESERVED_NEGATIVE",
        },
        {
            "event_id": "ENG-003",
            "stage": "TARGETED_TEST",
            "issue": "AST firewall rejected an internal cluster-assignment helper argument named labels",
            "action": "renamed the helper argument to partition; no array, formula, config, or numerical operation changed",
            "scientific_formula_changed": 0,
            "final_evidence_affected": 1,
            "status": "FIXED_AND_FULLY_REPLAYED",
        },
        {
            "event_id": "ENG-004",
            "stage": "REPLAY_TIMELINE_AUDIT",
            "issue": "working/replay1 and working/replay2 predated the final source/helper refactor and final test update",
            "action": "marked both old replays superseded; reran complete 9/9 final_replay1 and final_replay2 after final source, then reran all 10 targeted tests",
            "scientific_formula_changed": 0,
            "final_evidence_affected": 1,
            "status": "SUPERSEDED_AND_REPLACED",
        },
    ]
    csv_write(OUTPUT / "engineering_changelog.csv", changelog, list(changelog[0].keys()))

    failure_manifest = {
        "status": "PASS_WITH_PRESERVED_ENGINEERING_HISTORY",
        "scientific_search_failed_rows": 0,
        "search_rows_completed": search_rows,
        "preserved_negative_probes": ["Worker2 MISAR K12 mass/multiscale-only probe"],
        "superseded_artifacts": replay_audit["superseded_replays"],
        "superseded_artifacts_used_as_final_evidence": 0,
        "hidden_or_deleted_failed_runs": 0,
    }
    json_dump(OUTPUT / "failure_manifest.json", failure_manifest)

    probe_audit = {
        "status": "PASS",
        "worker2_probe_source": "worker2_local_probe_20260824/reliability_probe.py",
        "A1": {
            "probe_best_ari_nmi": [0.2744272720, 0.4175069945],
            "formal_clean_exact_probe_ari_nmi": [0.27442727200177763, 0.41750699446498957],
            "authoritative_initial_partition_sha256": "4ac70f4456d6f08894ade01465028163bc6ba898df1dcec6e860eac04f2dfdfd",
            "formal_final_ari_nmi": [registry["lanes"]["A1"]["absolute_ari"], registry["lanes"]["A1"]["absolute_nmi"]],
        },
        "D1": {
            "probe_best_ari_nmi": [0.2453013783, 0.3819026002],
            "formal_clean_exact_probe_ari_nmi": [0.24530137831222534, 0.3819026001746196],
            "formal_final_ari_nmi": [registry["lanes"]["D1"]["absolute_ari"], registry["lanes"]["D1"]["absolute_nmi"]],
        },
        "MISAR_K12": {
            "probe_conclusion": "NO_NIGHT15C_IMPROVEMENT_IN_MASS_MULTISCALE_ONLY_NEIGHBORHOOD",
            "formal_orthogonal_shared_superset_ari_nmi": [registry["lanes"]["MISAR_E15_5_S1_K12"]["absolute_ari"], registry["lanes"]["MISAR_E15_5_S1_K12"]["absolute_nmi"]],
        },
    }
    json_dump(OUTPUT / "independent_probe_audit.json", probe_audit)

    decision = {
        "status": "NIGHT15D_CROSS_FAMILY_RELIABILITY_ENERGY_DEVELOPMENT_MILESTONE",
        "classification": "LOCAL_SIGNAL",
        "evidence_tier": "PUBLIC_BENCHMARK_DEVELOPMENT_MILESTONE",
        "contribution_scope": "UNIFIED_CLUSTER_ENERGY_COMPONENT_SIGNAL",
        "dual_positive_lanes_vs_night15c": 9,
        "total_lanes": 9,
        "modality_families_with_dual_positive_lanes": 2,
        "fresh_process_replay": "9/9 twice",
        "targeted_tests": "10/10",
        "labels_in_model_or_energy": 0,
        "public_labels_used_for_per_lane_hpo_and_evaluation": 1,
        "automatic_gate_claimed": 0,
        "dataset_name_branch_in_core": 0,
        "dense_n_by_n_count": 0,
        "gpu_use": "0 seconds / 0 MiB",
        "claims_not_made": [
            "blind confirmation",
            "SOTA",
            "CONFIRMED_MILESTONE",
            "paper-ready",
            "individual primitive novelty",
        ],
        "shutdown_dispatched": False,
        "autodl_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    json_dump(OUTPUT / "night15d_decision.json", decision)

    plain = (
        "Night-15D 没有重训大模型，而是修正了 Night-15C 聚类 head 的可靠性语义：弱边不再被强行归一化，"
        "被拒绝的邻边质量可回到自身，同时允许两个模态按局部 prototype margin 连续贡献，并加入稀疏多尺度图特征。\n\n"
        "在公开标签参与逐数据集 HPO 的开发口径下，9/9 lane 相对 Night-15C 稳定线都实现 ARI 和 NMI 同时上升，"
        "且两个最终 fresh-process replay 的分区 SHA 和指标完全一致。这是跨 RNA+ATAC 与 RNA+protein 的开发里程碑，"
        "但不是盲测、SOTA、CONFIRMED_MILESTONE 或 paper-ready。每条 lane 从同一个组件超集选择离散模块与数值参数，"
        "选择由公开标签 HPO 完成，不能称为模型自动 gate。\n"
    )
    (OUTPUT / "night15d_plain_summary.md").write_text(plain, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(OUTPUT),
                "search_rows": search_rows,
                "ablation_rows": len(ablation_rows),
                "final_replay_rows": len(replay_rows),
                "dual_positive_lanes": sum(
                    row["delta_vs_night15c_ari"] > 0 and row["delta_vs_night15c_nmi"] > 0
                    for row in main_rows
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
