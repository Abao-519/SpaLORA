#!/usr/bin/env python3
"""Freeze Night-16B label-assisted benchmark profiles and replay recipes.

All candidate partitions have already been serialized before public-reference
evaluation.  This script applies the registered, public benchmark HPO rule to
those immutable rows, records supplementary profiles, and emits recipes that
can regenerate every selected partition in a fresh process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from SpaLORA.night16b_unified_structured_decoder import (
    RepairConfig,
    generic_repair,
    partition_sha256,
)
from scripts.night16b.night16b_candidate_pipeline import (
    LANE_DATASET,
    _feature_bank,
    _start_bank,
    graph_from_archive,
    lane_k,
)


PRIMARY = ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"]
SECONDARY = ["P22_3DOT_K18", "MISAR_E15_5_S1_K12"]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(frame: pd.DataFrame, n: int, k: int, fraction: float) -> pd.DataFrame:
    threshold = max(5, int(np.ceil(float(fraction) * n / k))) if fraction > 0 else 0
    return frame[
        (frame.status == "PASS")
        & (frame.exact_k == 1)
        & (frame.finite == 1)
        & (frame.min_cluster_size >= threshold)
    ].copy()


def choose_headline(frame: pd.DataFrame, n: int, k: int, fraction: float = 0.01) -> pd.Series:
    pool = eligible(frame, n, k, fraction)
    if pool.empty:
        raise RuntimeError("no candidate satisfies the registered guard")
    return pool.sort_values(
        ["absolute_ari", "absolute_nmi", "complexity", "min_cluster_size", "candidate_id"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_nmi(frame: pd.DataFrame, n: int, k: int) -> pd.Series:
    pool = eligible(frame, n, k, 0.01)
    return pool.sort_values(
        ["absolute_nmi", "absolute_ari", "complexity", "min_cluster_size", "candidate_id"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).iloc[0]


def pareto(frame: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    pool = eligible(frame, n, k, 0.01).sort_values(
        ["absolute_ari", "absolute_nmi"], ascending=[False, False]
    )
    keep = []
    best_nmi = -np.inf
    for index, row in pool.iterrows():
        if float(row.absolute_nmi) > best_nmi + 1e-15:
            keep.append(index)
            best_nmi = float(row.absolute_nmi)
    return pool.loc[keep]


def load_sources(entries: list[str]) -> tuple[pd.DataFrame, dict[str, Path]]:
    frames = []
    source_dirs: dict[str, Path] = {}
    for entry in entries:
        name, raw = entry.split("=", 1)
        directory = Path(raw)
        ledger = pd.read_csv(directory / "evaluated_ledger.csv")
        ledger["source_stage"] = name
        ledger["source_directory"] = str(directory)
        frames.append(ledger)
        source_dirs[name] = directory
    return pd.concat(frames, ignore_index=True, sort=False), source_dirs


def row_dict(row: pd.Series) -> dict[str, object]:
    result = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = None if not np.isfinite(value) else float(value)
        elif pd.isna(value):
            result[key] = None
        else:
            result[key] = value
    return result


def trace_chain(selected: pd.Series, lane_frame: pd.DataFrame) -> list[pd.Series]:
    by_id = {str(row.candidate_id): row for _, row in lane_frame.iterrows()}
    chain = [selected]
    seen = {str(selected.candidate_id)}
    parent = selected.get("parent_candidate_id")
    while parent is not None and not pd.isna(parent) and str(parent):
        parent = str(parent)
        if parent in seen or parent not in by_id:
            raise RuntimeError(f"broken candidate lineage at {parent}")
        chain.append(by_id[parent])
        seen.add(parent)
        parent = by_id[parent].get("parent_candidate_id")
    return list(reversed(chain))


def resolve_initial(
    lane: str,
    first: pd.Series,
    args: argparse.Namespace,
    n: int,
    k: int,
    bank: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, str]:
    starts, lineage = _start_bank(
        lane,
        n,
        k,
        bank,
        args.frontier,
        args.night15f,
        args.extra_start_root,
    )
    expected = str(first.start_partition_sha256)
    for name, partition in starts.items():
        if partition_sha256(partition) == expected:
            return np.asarray(partition, dtype=np.int32), name
    sources = [item for item in lineage if item["partition_sha256"] == expected]
    raise RuntimeError(f"{lane}: start SHA not resolvable; lineage={sources}")


def replay_chain(
    initial: np.ndarray,
    feature_bank: dict[str, np.ndarray],
    graph,
    k: int,
    configs: list[RepairConfig],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    partition = np.asarray(initial, dtype=np.int32)
    diagnostics = []
    for index, config in enumerate(configs):
        feature = feature_bank.get(config.feature_mode, feature_bank["retained"])
        partition, detail = generic_repair(partition, feature, graph, k, config)
        diagnostics.append({"step": index, "config": asdict(config), **detail})
    return partition, diagnostics


def try_consolidate(
    initial: np.ndarray,
    target_sha: str,
    feature_bank: dict[str, np.ndarray],
    graph,
    k: int,
    configs: list[RepairConfig],
) -> tuple[list[RepairConfig], bool]:
    if len(configs) <= 1 or not all(config.enabled for config in configs):
        return configs, False
    dictionaries = [asdict(config) for config in configs]
    for field in dictionaries[0]:
        if field == "boundary_refine_sweeps":
            continue
        if any(item[field] != dictionaries[0][field] for item in dictionaries[1:]):
            return configs, False
    merged = dict(dictionaries[0])
    merged["boundary_refine_sweeps"] = int(sum(item["boundary_refine_sweeps"] for item in dictionaries))
    candidate = RepairConfig(**merged)
    replayed, _ = replay_chain(initial, feature_bank, graph, k, [candidate])
    if partition_sha256(replayed) == target_sha:
        return [candidate], True
    return configs, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="name=directory")
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--night15f", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--extra-start-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    partition_dir = args.output / "partitions"
    start_dir = args.output / "starts"
    partition_dir.mkdir(exist_ok=True)
    start_dir.mkdir(exist_ok=True)

    all_runs, source_dirs = load_sources(args.source)
    all_runs.to_csv(args.output / "all_candidate_hpo_ledger.csv", index=False)
    selected_rows = []
    max_nmi_rows = []
    guard_rows = []
    pareto_rows = []
    sensitivity_rows = []
    resource_rows = []
    recipes: dict[str, object] = {}
    dependencies = []

    for lane in PRIMARY + SECONDARY:
        lane_frame = all_runs[all_runs.lane == lane].copy()
        if lane_frame.empty:
            raise RuntimeError(f"missing lane: {lane}")
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        n, k = len(data["ids"]), lane_k(data, lane)
        graph = graph_from_archive(data)
        features, feature_audit = _feature_bank(data, bank, lane, args.morphology)
        selected = choose_headline(lane_frame, n, k)
        selected_rows.append(row_dict(selected))
        max_nmi_rows.append(row_dict(choose_nmi(lane_frame, n, k)))
        for fraction in (0.0, 0.01, 0.02, 0.05):
            row = row_dict(choose_headline(lane_frame, n, k, fraction))
            row["guard_fraction_of_equal"] = fraction
            row["guard_threshold"] = max(5, int(np.ceil(fraction * n / k))) if fraction else 0
            guard_rows.append(row)
        for _, row in pareto(lane_frame, n, k).iterrows():
            pareto_rows.append(row_dict(row))

        chain_rows = trace_chain(selected, lane_frame)
        initial, resolved_start = resolve_initial(lane, chain_rows[0], args, n, k, bank)
        configs = [RepairConfig(**json.loads(str(row.config_json))) for row in chain_rows]
        target_sha = str(selected.partition_sha256)
        replayed, diagnostics = replay_chain(initial, features, graph, k, configs)
        if partition_sha256(replayed) != target_sha:
            raise RuntimeError(f"{lane}: full lineage replay mismatch")
        replay_configs, consolidated = try_consolidate(
            initial, target_sha, features, graph, k, configs
        )
        replayed, replay_diagnostics = replay_chain(initial, features, graph, k, replay_configs)
        if partition_sha256(replayed) != target_sha:
            raise RuntimeError(f"{lane}: frozen replay mismatch")
        np.save(start_dir / f"{lane}.npy", initial.astype(np.int32), allow_pickle=False)
        np.save(partition_dir / f"{lane}.npy", replayed.astype(np.int32), allow_pickle=False)
        recipes[lane] = {
            "lane": lane,
            "dataset": dataset,
            "n": n,
            "k": k,
            "selected_candidate_id": str(selected.candidate_id),
            "selected_source_stage": str(selected.source_stage),
            "selected_start_name": str(selected.start_name),
            "resolved_start_name": resolved_start,
            "initial_partition_sha256": partition_sha256(initial),
            "expected_partition_sha256": target_sha,
            "candidate_lineage": [str(row.candidate_id) for row in chain_rows],
            "original_config_chain": [json.loads(str(row.config_json)) for row in chain_rows],
            "replay_config_chain": [asdict(config) for config in replay_configs],
            "consolidated_exact": consolidated,
            "generation_diagnostics": replay_diagnostics,
            "feature_audit": feature_audit,
            "graph_shape": [n, n],
            "graph_nnz": int(graph.nnz),
        }
        dependencies.append(
            {
                "lane": lane,
                "dataset": dataset,
                "n": n,
                "k": k,
                "kit_path": str(args.kit / f"{dataset}.npz"),
                "start_source": resolved_start,
                "start_partition_sha256": partition_sha256(initial),
                "candidate_lineage": ">".join(str(row.candidate_id) for row in chain_rows),
                "operator_path": "common start bank > shared numeric repair/structured decoder > guard > evaluator",
                "feature_graph_energy_label_reads": 0,
                "label_contact": "independent evaluator and cross-run benchmark HPO only",
                "final_partition_sha256": target_sha,
            }
        )

        # The mechanical refine batch around a selected parent is the most
        # faithful local sensitivity surface.  For a coarse winner, use the
        # same start and feature mode in its fixed coarse batch.
        config = json.loads(str(selected.config_json))
        if str(selected.source_stage).startswith("refine"):
            neighbourhood = lane_frame[
                (lane_frame.source_stage == selected.source_stage)
                & (lane_frame.parent_candidate_id.astype(str) == str(selected.parent_candidate_id))
            ].copy()
        else:
            neighbourhood = lane_frame[
                (lane_frame.source_stage == selected.source_stage)
                & (lane_frame.start_name == selected.start_name)
            ].copy()
            neighbourhood = neighbourhood[
                neighbourhood.config_json.map(
                    lambda raw: json.loads(str(raw)).get("feature_mode", "retained")
                    == config.get("feature_mode", "retained")
                )
            ]
        for _, row in neighbourhood.iterrows():
            record = row_dict(row)
            parsed = json.loads(str(row.config_json))
            for key, value in parsed.items():
                record[f"parameter__{key}"] = value
            record["delta_ari_vs_headline"] = float(row.absolute_ari - selected.absolute_ari)
            record["delta_nmi_vs_headline"] = float(row.absolute_nmi - selected.absolute_nmi)
            sensitivity_rows.append(record)

        resource_rows.append(
            {
                "lane": lane,
                "candidate_rows": int(len(lane_frame)),
                "passed": int((lane_frame.status == "PASS").sum()),
                "failed": int((lane_frame.status != "PASS").sum()),
                "unique_partitions": int(lane_frame.partition_sha256.nunique()),
                "candidate_cpu_wall_seconds_sum": float(lane_frame.wall_seconds.fillna(0).sum()),
                "gpu_seconds": 0.0,
                "dense_n_by_n_count": 0,
            }
        )

    pd.DataFrame(selected_rows).to_csv(args.output / "headline_tuned_profile.csv", index=False)
    pd.DataFrame(max_nmi_rows).to_csv(args.output / "max_nmi_profile.csv", index=False)
    pd.DataFrame(guard_rows).to_csv(args.output / "guard_sensitivity.csv", index=False)
    pd.DataFrame(pareto_rows).to_csv(args.output / "ari_nmi_pareto.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.output / "parameter_sensitivity.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(args.output / "hpo_resource_table.csv", index=False)
    pd.DataFrame(dependencies).to_csv(args.output / "frontier_dependency_graph.csv", index=False)
    (args.output / "frontier_dependency_graph.json").write_text(
        json.dumps(dependencies, indent=2) + "\n", encoding="utf-8"
    )
    registry = {
        "status": "NIGHT16B_FORMAL_PROFILE_FROZEN",
        "selection_protocol": "label-assisted benchmark HPO",
        "headline_rule": "exact K, finite, min cluster >= max(5,ceil(0.01*N/K)); maximize ARI, then NMI, then lower complexity, then larger minimum cluster",
        "labels_in_candidate_producer": 0,
        "labels_in_cross_run_hpo": True,
        "lanes": recipes,
        "source_ledgers": {key: str(value / "evaluated_ledger.csv") for key, value in source_dirs.items()},
    }
    (args.output / "night16b_frozen_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS",
        "lanes": len(recipes),
        "primary_lanes": len(PRIMARY),
        "secondary_lanes": len(SECONDARY),
        "all_candidate_rows": int(len(all_runs)),
        "all_candidate_failures": int((all_runs.status != "PASS").sum()),
        "all_candidate_ledger_sha256": file_sha(args.output / "all_candidate_hpo_ledger.csv"),
        "frozen_registry_sha256": file_sha(args.output / "night16b_frozen_registry.json"),
        "dense_n_by_n_count": 0,
    }
    (args.output / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
