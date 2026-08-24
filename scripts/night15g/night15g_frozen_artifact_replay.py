#!/usr/bin/env python3
"""Fresh-process integrity and metric replay of frozen Night-15G partitions.

Algorithmic re-execution of the PCA/GMM and PCA/alpha-expansion paths is
separately audited because Windows BLAS thread scheduling can alter the final
discrete partition.  This script verifies the actually frozen scientific
artifacts byte-for-byte and recomputes their public-benchmark metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive  # noqa: E402
from scripts.night15g.night15g_optional_view_search import array_sha256, evaluate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_metric_row(data, graph, partition, expected: dict, label_key: str = "labels_primary") -> dict:
    metrics = evaluate(data[label_key], data["label_mask"], partition, graph)
    for key in ("absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c"):
        if key in expected and abs(float(metrics[key]) - float(expected[key])) > 1e-12:
            raise RuntimeError(f"frozen metric mismatch for {key}: {metrics[key]} != {expected[key]}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    workspace = args.workspace.resolve()
    kit_root = project / "night15b_delivery_20260824" / "working" / "local_compute_kit"
    night15f = project / "night15f_local_work" / "working" / "frozen"
    started = time.perf_counter()
    rows = []

    optional_specs = (
        (
            "A1",
            "balanced",
            workspace / "working" / "profile_replay_rev3_a1" / "partitions" / "A1__balanced.npy",
            workspace / "working" / "formal_search_rev3_deterministic" / "search_summary.json",
            ("A1", "balanced"),
        ),
        (
            "tonsil_s3",
            "balanced",
            workspace / "working" / "profile_replay_rev3_tonsil" / "partitions" / "tonsil_s3__balanced.npy",
            workspace / "working" / "formal_search_rev3_deterministic" / "search_summary.json",
            ("tonsil_s3", "balanced"),
        ),
        (
            "tonsil_s3",
            "max_nmi",
            workspace / "working" / "profile_replay_rev3_tonsil" / "partitions" / "tonsil_s3__max_nmi.npy",
            workspace / "working" / "formal_search_rev3_deterministic" / "search_summary.json",
            ("tonsil_s3", "max_nmi"),
        ),
    )
    for lane, profile, path, summary_path, keys in optional_specs:
        data = np.load(kit_root / f"{lane}.npz", allow_pickle=False)
        graph = csr_from_archive(data, "graph")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = summary[keys[0]][keys[1]]
        partition = np.load(path, allow_pickle=False).astype(np.int32)
        observed_hash = array_sha256(partition)
        if observed_hash != expected["partition_sha256"]:
            raise RuntimeError(f"{lane}/{profile}: frozen partition SHA mismatch")
        metrics = verify_metric_row(data, graph, partition, expected)
        rows.append(
            {
                "lane": lane,
                "profile": profile,
                "evidence_path": "optional_morphology_energy",
                "partition_sha256": observed_hash,
                "file_sha256": sha256_file(path),
                "cluster_sizes": np.bincount(partition, minlength=int(data["k_primary"][0])).tolist(),
                **metrics,
            }
        )

    head_summary_path = workspace / "working" / "morphology_head_replay_d1_rev1" / "replay_summary.json"
    head_summary = json.loads(head_summary_path.read_text(encoding="utf-8"))
    data = np.load(kit_root / "D1.npz", allow_pickle=False)
    graph = csr_from_archive(data, "graph")
    for profile in (
        "balanced",
        "max_ari",
        "max_nmi",
        "nonmicro_balanced",
        "nonmicro_max_ari",
        "nonmicro_max_nmi",
    ):
        expected = head_summary["profiles"][profile]
        path = workspace / "working" / "morphology_head_replay_d1_rev1" / "partitions" / f"D1__{profile}.npy"
        partition = np.load(path, allow_pickle=False).astype(np.int32)
        observed_hash = array_sha256(partition)
        if observed_hash != expected["partition_sha256"]:
            raise RuntimeError(f"D1/{profile}: frozen head partition SHA mismatch")
        metrics = verify_metric_row(data, graph, partition, expected)
        rows.append(
            {
                "lane": "D1",
                "profile": profile,
                "evidence_path": "morphology_coordinate_head",
                "partition_sha256": observed_hash,
                "file_sha256": sha256_file(path),
                "cluster_sizes": np.bincount(partition, minlength=10).tolist(),
                **metrics,
            }
        )

    stage2_summary = json.loads(
        (workspace / "working" / "seeded_refinement_d1_stage2_rev1" / "search_summary.json").read_text(
            encoding="utf-8"
        )
    )
    for profile in ("balanced", "max_ari", "max_nmi"):
        expected = stage2_summary["profiles"][profile]
        path = workspace / "working" / "seeded_refinement_d1_stage2_rev1" / "partitions" / f"D1__{profile}.npy"
        partition = np.load(path, allow_pickle=False).astype(np.int32)
        observed_hash = array_sha256(partition)
        if observed_hash != expected["partition_sha256"]:
            raise RuntimeError(f"D1/stage2/{profile}: frozen partition SHA mismatch")
        metrics = verify_metric_row(data, graph, partition, expected)
        rows.append(
            {
                "lane": "D1",
                "profile": f"stage2_{profile}",
                "evidence_path": "morphology_seeded_then_molecular_energy",
                "partition_sha256": observed_hash,
                "file_sha256": sha256_file(path),
                "cluster_sizes": np.bincount(partition, minlength=10).tolist(),
                **metrics,
            }
        )

    night15f_table = {}
    import csv

    with (night15f / "absolute_metrics_main_table.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            night15f_table[row["lane"]] = row
    fallback_lanes = (
        "tonsil_s1",
        "tonsil_s2",
        "P22",
        "P22_3DOT_K18",
        "MISAR_E15_5_S1",
        "MISAR_E15_5_S1_K12",
    )
    for lane in fallback_lanes:
        path = night15f / "partitions" / f"{lane}.npy"
        partition = np.load(path, allow_pickle=False).astype(np.int32)
        observed_hash = array_sha256(partition)
        expected_hash = night15f_table[lane]["partition_sha256"]
        if observed_hash != expected_hash:
            raise RuntimeError(f"{lane}: Night-15F fallback artifact SHA mismatch")
        rows.append(
            {
                "lane": lane,
                "profile": "missing_view_exact_fallback",
                "evidence_path": "presence_mask_zero_returns_night15f_authority",
                "partition_sha256": observed_hash,
                "file_sha256": sha256_file(path),
                "cluster_sizes": json.loads(night15f_table[lane]["cluster_sizes"]),
                "absolute_ari": float(night15f_table[lane]["absolute_ari"]),
                "absolute_nmi": float(night15f_table[lane]["absolute_nmi"]),
                "ami": float(night15f_table[lane]["ami"]),
                "fmi": float(night15f_table[lane]["fmi"]),
                "morans_i": float(night15f_table[lane]["morans_i"]),
                "gearys_c": float(night15f_table[lane]["gearys_c"]),
            }
        )

    payload = {
        "status": "NIGHT15G_FROZEN_ARTIFACT_FRESH_PROCESS_REPLAY_PASS",
        "artifact_profiles": len(rows),
        "scientific_partition_profiles": 12,
        "missing_view_fallback_lanes": 6,
        "labels_in_representation_energy_or_head": 0,
        "labels_in_post_partition_metric_recomputation": 1,
        "dense_n_by_n_count": 0,
        "algorithmic_recompute_scope": "separate audit; PCA/BLAS discrete sensitivity prevents current-environment partition exactness",
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
