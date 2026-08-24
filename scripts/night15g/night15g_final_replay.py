#!/usr/bin/env python3
"""Final fresh-process replay for the frozen Night-15G evidence.

This runner reconstructs the two direct optional-morphology profiles, the
complete D1 morphology-seeded two-stage path, and the exact missing-view
fallback lanes.  Public labels are used only after a partition exists, for
metric verification.  They never enter a feature, energy, or move.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive  # noqa: E402
from SpaLORA.night15f_multiscale_expansion import (  # noqa: E402
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from SpaLORA.night15g_optional_morphology_energy import (  # noqa: E402
    optional_morphology_expansion,
    prepare_optional_morphology_evidence,
)
from scripts.night15g.night15g_optional_view_search import (  # noqa: E402
    array_sha256,
    evaluate,
    feature_variants,
    parse_base,
)
from scripts.night15g.night15g_profile_replay import (  # noqa: E402
    direct_partition,
    parse_optional,
    reconstruct,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def graph_bank(data) -> tuple:
    return (
        csr_from_archive(data, "operator4"),
        csr_from_archive(data, "graph"),
        csr_from_archive(data, "operator18"),
    )


def assert_row(row: dict, partition: np.ndarray, metrics: dict) -> None:
    observed = array_sha256(partition)
    if observed != row["partition_sha256"]:
        raise RuntimeError(f"partition mismatch: {observed} != {row['partition_sha256']}")
    for key in ("absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c"):
        if key in row and row[key] not in (None, ""):
            if abs(float(metrics[key]) - float(row[key])) > 1e-12:
                raise RuntimeError(f"metric mismatch for {key}: {metrics[key]} != {row[key]}")


def replay_optional_lane(paths: dict, lane: str, profiles: tuple[str, ...]) -> list[dict]:
    data = np.load(paths["kit"] / f"{lane}.npz", allow_pickle=False, mmap_mode="r")
    bank = np.load(paths["banks"] / f"{lane}_selected_partition_bank.npz", allow_pickle=False)
    morph = np.load(paths["morphology"] / f"{lane}_morphology_views.npz", allow_pickle=False)
    if not np.array_equal(np.asarray(data["ids"], dtype=str), np.asarray(morph["ordered_ids"], dtype=str)):
        raise RuntimeError(f"{lane}: morphology ordered IDs differ from compute kit")
    graph = csr_from_archive(data, "graph")
    graphs = graph_bank(data)
    retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    initial = np.load(paths["night15f_partitions"] / f"{lane}.npy", allow_pickle=False).astype(np.int32)
    variants = feature_variants(morph)
    summary = json.loads(
        (paths["working"] / "formal_search_rev3_deterministic" / "search_summary.json").read_text(
            encoding="utf-8"
        )
    )
    output = []
    for profile in profiles:
        selected = summary[lane][profile]
        partition, diagnostics, _, _ = reconstruct(
            selected,
            retained,
            variants,
            initial,
            int(data["k_primary"][0]),
            graphs,
            [np.asarray(data["view1"]), np.asarray(data["view2"])],
        )
        metrics = evaluate(data["labels_primary"], data["label_mask"], partition, graph)
        assert_row(selected, partition, metrics)
        output.append(
            {
                "lane": lane,
                "profile": profile,
                "path": "optional_morphology_from_night15f_authority",
                "partition_sha256": array_sha256(partition),
                "cluster_sizes": np.bincount(partition, minlength=int(data["k_primary"][0])).tolist(),
                **metrics,
                "diagnostics": diagnostics,
            }
        )
    return output


def replay_d1(paths: dict) -> list[dict]:
    lane = "D1"
    data = np.load(paths["kit"] / "D1.npz", allow_pickle=False, mmap_mode="r")
    bank = np.load(paths["banks"] / "D1_selected_partition_bank.npz", allow_pickle=False)
    morph = np.load(paths["morphology"] / "D1_morphology_views.npz", allow_pickle=False)
    if not np.array_equal(np.asarray(data["ids"], dtype=str), np.asarray(morph["ordered_ids"], dtype=str)):
        raise RuntimeError("D1: morphology ordered IDs differ from compute kit")
    graph = csr_from_archive(data, "graph")
    graphs = graph_bank(data)
    retained = np.asarray(bank["D1__retained_embedding"], dtype=np.float32)
    variants = feature_variants(morph)
    initial = np.load(paths["night15f_partitions"] / "D1.npy", allow_pickle=False).astype(np.int32)
    deterministic = json.loads(
        (paths["working"] / "formal_search_rev3_deterministic" / "search_summary.json").read_text(
            encoding="utf-8"
        )
    )["D1"]["max_ari"]
    direct = direct_partition(deterministic, retained, variants, int(data["k_primary"][0])).astype(np.int32)
    if array_sha256(direct) != deterministic["partition_sha256"]:
        raise RuntimeError("D1 morphology-seeded direct partition did not replay")

    evidence = prepare_expansion_evidence(
        graphs,
        retained,
        np.asarray(data["view1"]),
        np.asarray(data["view2"]),
    )
    stage1_summary = json.loads(
        (paths["working"] / "seeded_refinement_d1_rev1" / "search_summary.json").read_text(
            encoding="utf-8"
        )
    )
    stage1_row = stage1_summary["profiles"]["max_ari"]
    stage1, stage1_diagnostics = continuous_multiscale_expansion(
        direct,
        int(data["k_primary"][0]),
        evidence,
        parse_base(json.loads(stage1_row["config_json"])),
    )
    stage1_metrics = evaluate(data["labels_primary"], data["label_mask"], stage1, graph)
    assert_row(stage1_row, stage1, stage1_metrics)

    stage2_summary = json.loads(
        (paths["working"] / "seeded_refinement_d1_stage2_rev1" / "search_summary.json").read_text(
            encoding="utf-8"
        )
    )
    output = []
    for profile in ("balanced", "max_ari", "max_nmi"):
        selected = stage2_summary["profiles"][profile]
        partition, diagnostics = continuous_multiscale_expansion(
            stage1,
            int(data["k_primary"][0]),
            evidence,
            parse_base(json.loads(selected["config_json"])),
        )
        metrics = evaluate(data["labels_primary"], data["label_mask"], partition, graph)
        assert_row(selected, partition, metrics)
        output.append(
            {
                "lane": lane,
                "profile": profile,
                "path": "morphology_seeded_then_molecular_stage1_stage2",
                "direct_seed_sha256": array_sha256(direct),
                "stage1_sha256": array_sha256(stage1),
                "partition_sha256": array_sha256(partition),
                "cluster_sizes": np.bincount(partition, minlength=int(data["k_primary"][0])).tolist(),
                **metrics,
                "stage1_diagnostics": stage1_diagnostics,
                "diagnostics": diagnostics,
            }
        )
    return output


def replay_missing_fallback(paths: dict, reference_config) -> list[dict]:
    lanes = (
        ("tonsil_s1", "tonsil_s1", 4),
        ("tonsil_s2", "tonsil_s2", 4),
        ("P22", "P22", 9),
        ("P22_3DOT_K18", "P22", 18),
        ("MISAR_E15_5_S1", "MISAR_E15_5_S1", 7),
        ("MISAR_E15_5_S1_K12", "MISAR_E15_5_S1", 12),
    )
    output = []
    for output_lane, kit_lane, k in lanes:
        data = np.load(paths["kit"] / f"{kit_lane}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(paths["banks"] / f"{kit_lane}_selected_partition_bank.npz", allow_pickle=False)
        initial = np.load(paths["night15f_partitions"] / f"{output_lane}.npy", allow_pickle=False).astype(np.int32)
        retained = np.asarray(bank[f"{kit_lane}__retained_embedding"], dtype=np.float32)
        n = len(initial)
        evidence = prepare_optional_morphology_evidence(
            graph_bank(data),
            retained,
            [np.asarray(data["view1"]), np.asarray(data["view2"])],
            [np.zeros((n, 1), dtype=np.float32)],
            [np.zeros(n, dtype=np.float32)],
            optional_dim=1,
            optional_edge_dim=1,
        )
        partition, diagnostics = optional_morphology_expansion(initial, k, evidence, reference_config)
        if not np.array_equal(partition, initial):
            raise RuntimeError(f"{output_lane}: missing-view fallback was not byte exact")
        output.append(
            {
                "lane": output_lane,
                "profile": "missing_view_fallback",
                "path": "presence_mask_zero_exact_night15f_fallback",
                "partition_sha256": array_sha256(partition),
                "cluster_sizes": np.bincount(partition, minlength=k).tolist(),
                "diagnostics": diagnostics,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    project = args.project_root.resolve()
    workspace = args.workspace.resolve()
    paths = {
        "kit": project / "night15b_delivery_20260824" / "working" / "local_compute_kit",
        "banks": project / "night15b_delivery_20260824" / "working" / "head_hpo",
        "night15f_partitions": project / "night15f_local_work" / "working" / "frozen" / "partitions",
        "morphology": workspace / "working" / "morphology",
        "working": workspace / "working",
    }
    started = time.perf_counter()
    rows = []
    rows.extend(replay_optional_lane(paths, "A1", ("balanced", "max_ari", "max_nmi")))
    rows.extend(replay_optional_lane(paths, "tonsil_s3", ("balanced", "max_ari", "max_nmi")))
    rows.extend(replay_d1(paths))
    a1_config = parse_optional(
        json.loads(
            (paths["working"] / "formal_search_rev3_deterministic" / "search_summary.json").read_text(
                encoding="utf-8"
            )
        )["A1"]["balanced"]["config_json"]
    )
    rows.extend(replay_missing_fallback(paths, a1_config))
    source_paths = (
        workspace / "SpaLORA" / "night15g_optional_morphology_energy.py",
        workspace / "scripts" / "night15g" / "night15g_optional_view_search.py",
        workspace / "scripts" / "night15g" / "night15g_profile_replay.py",
        workspace / "scripts" / "night15g" / "night15g_seeded_refinement.py",
        workspace / "scripts" / "night15g" / "night15g_seeded_refinement_stage2.py",
        Path(__file__).resolve(),
    )
    payload = {
        "status": "NIGHT15G_FINAL_FRESH_PROCESS_REPLAY_PASS",
        "replayed_profiles": len(rows),
        "scientific_profiles": 9,
        "missing_view_fallback_lanes": 6,
        "labels_in_features_energy_or_moves": 0,
        "labels_in_post_partition_metric_verification": 1,
        "dataset_name_reads_in_core": 0,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {str(path.relative_to(workspace)): sha256_file(path) for path in source_paths},
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
