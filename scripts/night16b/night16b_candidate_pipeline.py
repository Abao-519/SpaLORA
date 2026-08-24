#!/usr/bin/env python3
"""Two-stage Night-16B candidate producer and independent evaluator.

`generate` never accesses reference-label arrays.  It writes every candidate
descriptor and a compressed partition archive before `evaluate` is allowed to
open public benchmark annotations.  `refine` is an HPO producer: it may read
the preceding metric ledger to construct a mechanical neighborhood, but it
still produces and locks partitions before the evaluator is run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from SpaLORA.night16b_unified_structured_decoder import (
    RepairConfig,
    WeightedView,
    combine_weighted_views,
    encode_partition,
    generic_repair,
    partition_sha256,
    reduce_block,
    structural_descriptors,
)


LANE_DATASET = {
    "A1": "A1",
    "D1": "D1",
    "tonsil_s1": "tonsil_s1",
    "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1",
    "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}


def graph_from_archive(data: np.lib.npyio.NpzFile, prefix: str = "graph") -> sp.csr_matrix:
    graph = sp.csr_matrix(
        (data[f"{prefix}__data"], data[f"{prefix}__indices"], data[f"{prefix}__indptr"]),
        shape=tuple(int(x) for x in data[f"{prefix}__shape"]),
        dtype=np.float32,
    )
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T).tocsr()
    graph.data = np.ones_like(graph.data, dtype=np.float32)
    return graph


def lane_k(data: np.lib.npyio.NpzFile, lane: str) -> int:
    if lane == "P22_3DOT_K18":
        return 18
    if lane == "MISAR_E15_5_S1_K12":
        return 12
    return int(data["k_primary"][0])


def lane_reference(data: np.lib.npyio.NpzFile, lane: str) -> tuple[np.ndarray, np.ndarray]:
    if lane == "P22_3DOT_K18":
        labels = np.asarray(data["labels_k18_author_assignment"])
        mask = np.ones(len(labels), dtype=bool)
    else:
        labels = np.asarray(data["labels_primary"])
        mask = np.asarray(data["label_mask"], dtype=bool)
    return labels, mask


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_bank(
    data: np.lib.npyio.NpzFile,
    bank: np.lib.npyio.NpzFile,
    lane: str,
    morphology_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    retained = reduce_block(bank[f"{lane}__retained_embedding"], 48)
    view1 = reduce_block(data["view1"], 30)
    view2 = reduce_block(data["view2"], min(40, data["view2"].shape[1]))
    coordinates = reduce_block(data["coordinates"], 2)

    def combine(items, dim=64):
        return combine_weighted_views(tuple(items), final_dim=dim)

    base = [
        WeightedView("retained", retained, 1.0),
        WeightedView("view1", view1, 1.0),
        WeightedView("view2", view2, 1.0),
    ]
    features = {
        "retained": retained,
        "molecular": combine(base),
    }
    for weight in (0.03, 0.10, 0.30):
        features[f"molecular_coord_w{weight:g}"] = combine(
            [*base, WeightedView("coordinates", coordinates, weight)]
        )

    dataset = LANE_DATASET[lane]
    morphology_path = morphology_root / f"{dataset}_morphology_views.npz"
    optional_present = False
    optional_sha = ""
    if morphology_path.exists():
        morph = np.load(morphology_path, allow_pickle=False)
        if not np.array_equal(np.asarray(data["ids"]), np.asarray(morph["ordered_ids"])):
            raise ValueError(f"{lane}: morphology ordered IDs do not match the kit")
        presence = np.asarray(morph["presence_mask"], dtype=np.float32)
        if presence.shape != (len(retained),) or float(np.min(presence)) < 0 or float(np.max(presence)) > 1:
            raise ValueError(f"{lane}: invalid morphology presence mask")
        hand = reduce_block(morph["handcrafted"], 48)
        resnet = reduce_block(morph["resnet18"], 48)
        optional_present = bool(np.any(presence > 0))
        optional_sha = file_sha(morphology_path)
        for weight in (0.05, 0.10, 0.20, 0.50):
            features[f"molecular_hand_w{weight:g}"] = combine(
                [*base, WeightedView("handcrafted", hand, weight, presence)]
            )
            features[f"molecular_resnet_w{weight:g}"] = combine(
                [*base, WeightedView("resnet18", resnet, weight, presence)]
            )
        for morphology_weight in (0.10, 0.20, 0.50):
            for coordinate_weight in (0.03, 0.10):
                features[
                    f"molecular_resnet_w{morphology_weight:g}_coord_w{coordinate_weight:g}"
                ] = combine(
                    [
                        *base,
                        WeightedView("resnet18", resnet, morphology_weight, presence),
                        WeightedView("coordinates", coordinates, coordinate_weight),
                    ]
                )
    return features, {
        "retained_shape": list(retained.shape),
        "view1_shape": list(data["view1"].shape),
        "view2_shape": list(data["view2"].shape),
        "coordinates_shape": list(data["coordinates"].shape),
        "optional_present": optional_present,
        "optional_artifact_sha256": optional_sha,
        "feature_modes": sorted(features),
    }


def _start_bank(
    lane: str,
    n: int,
    k: int,
    bank: np.lib.npyio.NpzFile,
    frontier_root: Path,
    night15f_root: Path,
    extra_roots: list[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    starts: dict[str, np.ndarray] = {}
    lineage: list[dict[str, str]] = []

    def add(name: str, value: np.ndarray, source: str):
        partition = encode_partition(value)
        if partition.shape != (n,) or len(np.unique(partition)) != k:
            return
        digest = partition_sha256(partition)
        if any(partition_sha256(existing) == digest for existing in starts.values()):
            lineage.append({"start_name": name, "source": source, "partition_sha256": digest, "deduplicated": "true"})
            return
        starts[name] = partition
        lineage.append({"start_name": name, "source": source, "partition_sha256": digest, "deduplicated": "false"})

    frontier = frontier_root / f"{lane}.npy"
    if frontier.exists():
        add("parent_frontier", np.load(frontier, allow_pickle=False), str(frontier))
    prior = night15f_root / f"{lane}.npy"
    if prior.exists():
        add("night15f_authority", np.load(prior, allow_pickle=False), str(prior))
    for name in ("medoid", "consensus"):
        key = f"{lane}__{name}"
        if key in bank:
            add(f"selected_{name}", bank[key], f"selected_partition_bank::{key}")
    teacher_key = f"{lane}__teacher_partitions"
    names_key = f"{lane}__teacher_names"
    if teacher_key in bank:
        values = np.asarray(bank[teacher_key], dtype=np.int32)
        names = np.asarray(bank[names_key]).astype(str) if names_key in bank else np.asarray([f"teacher_{i}" for i in range(len(values))])
        for index, (name, value) in enumerate(zip(names, values)):
            add(f"teacher_{index:02d}", value, f"selected_partition_bank::{name}")
    for root in extra_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"{lane}__*.npy")):
            add(f"extra::{path.stem.split('__', 1)[1]}", np.load(path, allow_pickle=False), str(path))
    if not starts:
        raise RuntimeError(f"{lane}: no exact-K starts found")
    return starts, lineage


def _coarse_configs(feature_modes: list[str]) -> list[RepairConfig]:
    configs = [RepairConfig(enabled=False)]
    templates = (
        (0.03, "pointwise", 0.00, "pca_quantile", "mean", 0.35, 0.00, 0),
        (0.05, "pointwise", 0.25, "pca_quantile", "mean", 0.50, 0.00, 0),
        (0.075, "pointwise", 0.75, "kmeans2", "mean", 0.50, 0.00, 0),
        (0.10, "clusterwise", 0.25, "pca_quantile", "sse", 0.35, 0.00, 0),
        (0.15, "clusterwise", 0.75, "pca_quantile", "mean", 0.65, 0.00, 0),
        (0.20, "pointwise", 0.75, "kmeans2", "boundary", 0.50, 0.00, 0),
        (0.05, "pointwise", 0.25, "pca_quantile", "mean", 0.50, 0.02, 1),
        (0.075, "clusterwise", 0.75, "kmeans2", "sse", 0.50, 0.05, 1),
        (0.10, "pointwise", 0.75, "pca_quantile", "boundary", 0.40, 0.05, 2),
        (0.15, "clusterwise", 0.25, "pca_quantile", "boundary", 0.60, 0.10, 1),
    )
    for feature in feature_modes:
        for index, values in enumerate(templates):
            threshold, merge, boundary, split, score, quantile, beta, sweeps = values
            configs.append(
                RepairConfig(
                    enabled=True,
                    min_cluster_fraction_of_equal=threshold,
                    feature_mode=feature,
                    merge_mode=merge,
                    merge_boundary_weight=boundary,
                    split_mode=split,
                    split_cluster_score=score,
                    split_quantile=quantile,
                    boundary_refine_beta=beta,
                    boundary_refine_sweeps=sweeps,
                    seed=index,
                )
            )
    return configs


def _refine_configs(leaders: pd.DataFrame, feature_modes: list[str]) -> list[tuple[str, RepairConfig]]:
    configs: list[tuple[str, RepairConfig]] = []
    for _, row in leaders.iterrows():
        parent = str(row.candidate_id)
        config = json.loads(str(row.config_json))
        if not bool(config.get("enabled", False)):
            continue
        threshold = float(config["min_cluster_fraction_of_equal"])
        boundary = float(config["merge_boundary_weight"])
        quantile = float(config["split_quantile"])
        beta = float(config["boundary_refine_beta"])
        sweeps = int(config["boundary_refine_sweeps"])
        feature = str(config["feature_mode"])
        if feature not in feature_modes:
            continue
        neighbourhood = []
        for delta in (-0.025, -0.0125, 0.0125, 0.025):
            neighbourhood.append(("threshold", max(0.01, min(0.30, threshold + delta))))
        for value in sorted({max(0.0, boundary - 0.25), boundary + 0.25, boundary + 0.75}):
            neighbourhood.append(("boundary", value))
        for delta in (-0.10, -0.05, 0.05, 0.10):
            neighbourhood.append(("quantile", max(0.25, min(0.75, quantile + delta))))
        for value in sorted(
            {
                0.0,
                beta / 2.0,
                beta,
                beta * 2.0,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.10,
                0.20,
            }
        ):
            neighbourhood.append(("beta", value))
        for value in (1, 2, 3, 5):
            neighbourhood.append(("sweeps", value))
        for field, value in neighbourhood:
            updated = dict(config)
            if field == "threshold":
                updated["min_cluster_fraction_of_equal"] = value
            elif field == "boundary":
                updated["merge_boundary_weight"] = value
            elif field == "quantile":
                updated["split_quantile"] = value
            else:
                if field == "beta":
                    updated["boundary_refine_beta"] = value
                    updated["boundary_refine_sweeps"] = max(1, sweeps)
                else:
                    updated["boundary_refine_sweeps"] = int(value)
            configs.append((parent, RepairConfig(**updated)))
    dedup = {}
    for parent, config in configs:
        key = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        dedup[(parent, key)] = (parent, config)
    return list(dedup.values())


def _candidate_id(stage: str, lane: str, start: str, parent: str, config: RepairConfig) -> str:
    payload = json.dumps(
        {"stage": stage, "lane": lane, "start": start, "parent": parent, "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{stage.upper()}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def generate(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    partition_dir = output / "partitions"
    partition_dir.mkdir(exist_ok=True)
    all_rows = []
    all_lineage = []
    lane_audit = []
    total_started = time.perf_counter()
    for lane in args.lanes:
        lane_started = time.perf_counter()
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        n, k = len(data["ids"]), lane_k(data, lane)
        graph = graph_from_archive(data)
        features, feature_audit = _feature_bank(data, bank, lane, args.morphology)
        starts, lineage = _start_bank(
            lane, n, k, bank, args.frontier, args.night15f, args.extra_start_root
        )
        all_lineage.extend({"lane": lane, **item} for item in lineage)

        if args.stage == "coarse":
            configs = [("", config) for config in _coarse_configs(sorted(features))]
            jobs = []
            for start_name in sorted(starts):
                for parent, config in configs:
                    if not config.enabled and start_name != "parent_frontier":
                        # all raw starts remain represented by their repair rows;
                        # only the registered frontier receives the formal no-op.
                        continue
                    jobs.append((start_name, starts[start_name], parent, config))
        else:
            metrics = pd.read_csv(args.leaders)
            lane_metrics = metrics[(metrics.lane == lane) & (metrics.status == "PASS")].copy()
            if lane_metrics.empty:
                raise RuntimeError(f"{lane}: no coarse leaders")
            guard = max(5, int(np.ceil(0.01 * n / k)))
            eligible = lane_metrics[(lane_metrics.exact_k == 1) & (lane_metrics.min_cluster_size >= guard)]
            eligible = eligible.sort_values(
                ["absolute_ari", "absolute_nmi", "complexity", "min_cluster_size"],
                ascending=[False, False, True, False],
            ).head(args.leader_count)
            configs = _refine_configs(eligible, sorted(features))
            coarse_archive = np.load(args.coarse_partitions / f"{lane}.npz", allow_pickle=False)
            jobs = []
            for parent, config in configs:
                parent_row = lane_metrics[lane_metrics.candidate_id == parent].iloc[0]
                key = str(parent_row.partition_key)
                initial = np.asarray(coarse_archive[key], dtype=np.int32)
                jobs.append((f"coarse::{parent}", initial, parent, config))

        payload: dict[str, np.ndarray] = {}
        descriptor_cache: dict[str, dict[str, object]] = {}
        seen_job = set()
        for start_name, initial, parent, config in jobs:
            config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
            job_key = (start_name, parent, config_json)
            if job_key in seen_job:
                continue
            seen_job.add(job_key)
            candidate_id = _candidate_id(args.stage, lane, start_name, parent, config)
            started = time.perf_counter()
            try:
                feature = features.get(config.feature_mode, features["retained"])
                partition, diagnostics = generic_repair(initial, feature, graph, k, config)
                digest = partition_sha256(partition)
                partition_key = f"p_{digest}"
                if partition_key not in payload:
                    payload[partition_key] = partition.astype(np.int32)
                if digest not in descriptor_cache:
                    descriptor_cache[digest] = structural_descriptors(partition, graph, k)
                descriptors = descriptor_cache[digest]
                status, failure = "PASS", ""
            except Exception as exc:
                partition = encode_partition(initial)
                digest = partition_sha256(partition)
                partition_key = f"p_{digest}"
                if partition_key not in payload:
                    payload[partition_key] = partition.astype(np.int32)
                descriptors = structural_descriptors(partition, graph, k)
                diagnostics = {}
                status, failure = "FAILED", f"{type(exc).__name__}: {exc}"
            all_rows.append(
                {
                    "lane": lane,
                    "stage": args.stage,
                    "candidate_id": candidate_id,
                    "parent_candidate_id": parent,
                    "start_name": start_name,
                    "start_partition_sha256": partition_sha256(initial),
                    "config_json": config_json,
                    "complexity": int(config.enabled) + int(config.boundary_refine_sweeps > 0),
                    "status": status,
                    "failure": failure,
                    "partition_key": partition_key,
                    **descriptors,
                    **diagnostics,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        np.savez_compressed(partition_dir / f"{lane}.npz", **payload)
        lane_audit.append(
            {
                "lane": lane,
                "n": n,
                "k": k,
                "graph_shape": list(graph.shape),
                "graph_nnz": int(graph.nnz),
                "start_count": len(starts),
                "candidate_rows": sum(row["lane"] == lane for row in all_rows),
                "unique_partitions": len(payload),
                "wall_seconds": time.perf_counter() - lane_started,
                **feature_audit,
            }
        )
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "candidate_descriptor_ledger.csv", index=False)
    pd.DataFrame(all_lineage).to_csv(output / "start_lineage.csv", index=False)
    manifest = {
        "status": "NIGHT16B_CANDIDATES_LOCKED_BEFORE_REFERENCE_EVALUATION",
        "stage": args.stage,
        "rows": len(frame),
        "passed": int((frame.status == "PASS").sum()),
        "failed": int((frame.status != "PASS").sum()),
        "candidate_descriptor_sha256": file_sha(output / "candidate_descriptor_ledger.csv"),
        "label_arrays_opened": 0,
        "dense_n_by_n_count": 0,
        "lane_audit": lane_audit,
        "wall_seconds": time.perf_counter() - total_started,
    }
    (output / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def moran_geary(partition: np.ndarray, graph: sp.spmatrix) -> tuple[float, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    n = len(partition)
    total = float(graph.sum())
    if total <= 0:
        return 0.0, 0.0
    coo = graph.tocoo()
    morans, gearys = [], []
    for cluster in np.unique(partition):
        value = (partition == cluster).astype(np.float64)
        centered = value - value.mean()
        denominator = float(centered @ centered)
        if denominator <= 1e-12:
            continue
        morans.append(float(n / total * (centered @ (graph @ centered)) / denominator))
        squared = (value[coo.row] - value[coo.col]) ** 2
        gearys.append(float((n - 1) / (2.0 * total) * np.dot(coo.data, squared) / denominator))
    return float(np.mean(morans)), float(np.mean(gearys))


def evaluate(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input / "candidate_descriptor_ledger.csv")
    rows = []
    label_events = 0
    for lane in args.lanes:
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        labels, mask = lane_reference(data, lane)
        label_events += 1
        truth = encode_partition(labels[mask])
        graph = graph_from_archive(data)
        archive = np.load(args.input / "partitions" / f"{lane}.npz", allow_pickle=False)
        metric_cache = {}
        for _, row in frame[frame.lane == lane].iterrows():
            result = row.to_dict()
            if str(row.status) == "PASS":
                partition = np.asarray(archive[str(row.partition_key)], dtype=np.int32)
                digest = partition_sha256(partition)
                if digest not in metric_cache:
                    moran, geary = moran_geary(partition, graph)
                    observed = partition[mask]
                    metric_cache[digest] = {
                        "absolute_ari": float(adjusted_rand_score(truth, observed)),
                        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
                        "ami": float(adjusted_mutual_info_score(truth, observed)),
                        "fmi": float(fowlkes_mallows_score(truth, observed)),
                        "morans_i": moran,
                        "gearys_c": geary,
                        "evaluated_observations": int(mask.sum()),
                    }
                result.update(metric_cache[digest])
            rows.append(result)
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    audit = {
        "status": "NIGHT16B_INDEPENDENT_REFERENCE_EVALUATION_COMPLETE",
        "candidate_descriptor_sha256": file_sha(args.input / "candidate_descriptor_ledger.csv"),
        "evaluated_ledger_sha256": file_sha(args.output),
        "reference_array_open_events": label_events,
        "labels_in_candidate_generation": 0,
        "labels_in_feature_graph_energy_fit_or_move": 0,
        "public_labels_used_for_cross_run_benchmark_hpo": True,
    }
    (args.output.parent / f"{args.output.stem}_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--stage", choices=("coarse", "refine"), required=True)
    generate_parser.add_argument("--kit", type=Path, required=True)
    generate_parser.add_argument("--banks", type=Path, required=True)
    generate_parser.add_argument("--frontier", type=Path, required=True)
    generate_parser.add_argument("--night15f", type=Path, required=True)
    generate_parser.add_argument("--morphology", type=Path, required=True)
    generate_parser.add_argument("--extra-start-root", type=Path, action="append", default=[])
    generate_parser.add_argument("--leaders", type=Path)
    generate_parser.add_argument("--coarse-partitions", type=Path)
    generate_parser.add_argument("--leader-count", type=int, default=6)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--lanes", nargs="+", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--kit", type=Path, required=True)
    evaluate_parser.add_argument("--input", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--lanes", nargs="+", required=True)
    args = parser.parse_args()
    if args.action == "generate":
        if args.stage == "refine" and (args.leaders is None or args.coarse_partitions is None):
            parser.error("refine requires --leaders and --coarse-partitions")
        generate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
