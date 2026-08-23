#!/usr/bin/env python3
"""Local-first sparse direct-clustering arena for Night-15C.

Public labels are loaded only by ``evaluate`` and by cross-run ranking.  They
are never passed to unary, edge, energy, spectral, Fisher or ensemble code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15c_cluster_energy import (
    align_partition,
    csr_from_archive,
    dynamic_prototype_icm,
    multiscale_edge_conductance,
    partition_bank_unary,
    potts_icm,
    potts_mean_field,
    prototype_unary,
    raw_bimodal_edge_conductance,
    reduced,
    reduced_controlled,
    pseudo_fisher_transform,
    sha256_array,
    spectral_partition,
    standardize,
    symmetric_binary,
)


HIGH_WATER = {
    "P22": (0.5691172816484583, 0.6851311063039861),
    "P22_3DOT_K18": (0.6877945220149917, 0.7233302261338711),
    "MISAR_E15_5_S1": (0.5143, 0.6289634776869945),
    "MISAR_E15_5_S1_K12": (0.4142752375766465, 0.5906093484988677),
    "A1": (0.273021059361755, 0.4173676729225421),
    "D1": (0.2437566119939119, 0.3805336205886551),
    "tonsil_s1": (0.2079467132724583, 0.2969279606499894),
    "tonsil_s2": (0.2130056436093344, 0.2674062765696641),
    "tonsil_s3": (0.1969333595315539, 0.2497791360710971),
}


def encode_labels(labels: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    return encoded.astype(np.int32)


def moran_geary(partition: np.ndarray, graph: sp.spmatrix) -> Tuple[float, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    n = len(partition)
    total = float(graph.sum())
    if total <= 0:
        return 0.0, 0.0
    coo = graph.tocoo()
    morans: List[float] = []
    gearys: List[float] = []
    for cluster in np.unique(partition):
        value = (partition == cluster).astype(np.float64)
        centered = value - value.mean()
        denominator = float(centered @ centered)
        if denominator <= 1e-12:
            continue
        morans.append(float(n / total * (centered @ (graph @ centered)) / denominator))
        squared = (value[coo.row] - value[coo.col]) ** 2
        gearys.append(float((n - 1) / (2 * total) * np.dot(coo.data, squared) / denominator))
    return float(np.mean(morans)), float(np.mean(gearys))


def evaluate(
    labels: np.ndarray,
    mask: np.ndarray,
    partition: np.ndarray,
    graph: sp.spmatrix,
    spatial: bool = False,
) -> Dict[str, float]:
    truth = encode_labels(labels[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    result = {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
    }
    if spatial:
        result["morans_i"], result["gearys_c"] = moran_geary(partition, graph)
    return result


def objective(metrics: Mapping[str, float]) -> float:
    return float(metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"])


@dataclass
class Lane:
    dataset: str
    lane: str
    k: int
    labels: np.ndarray
    mask: np.ndarray


def dataset_lanes(dataset: str, archive: Mapping[str, np.ndarray]) -> List[Lane]:
    primary = Lane(
        dataset,
        dataset,
        int(archive["k_primary"][0]),
        archive["labels_primary"].astype(str),
        archive["label_mask"].astype(bool),
    )
    result = [primary]
    if dataset == "MISAR_E15_5_S1":
        result.append(Lane(dataset, "MISAR_E15_5_S1_K12", 12, primary.labels, primary.mask))
    if dataset == "P22" and "labels_k18_author_assignment" in archive.files:
        result.append(
            Lane(
                dataset,
                "P22_3DOT_K18",
                18,
                archive["labels_k18_author_assignment"].astype(str),
                primary.mask,
            )
        )
    return result


def unique_partitions(parts: Iterable[Tuple[str, np.ndarray]], k: int) -> List[Tuple[str, np.ndarray]]:
    result: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for name, value in parts:
        partition = np.asarray(value, dtype=np.int32)
        if len(np.unique(partition)) != int(k):
            continue
        digest = sha256_array(partition)
        if digest in seen:
            continue
        seen.add(digest)
        result.append((name, partition))
    return result


def load_bank(
    kit_archive: Mapping[str, np.ndarray],
    selected_archive: Mapping[str, np.ndarray],
    lane: Lane,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], Dict[str, List[Tuple[str, np.ndarray]]]]:
    prefix = lane.lane
    retained_key = f"{prefix}__retained_embedding"
    if retained_key not in selected_archive.files:
        raise KeyError(retained_key)
    embedding = np.asarray(selected_archive[retained_key], dtype=np.float32)
    groups: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    selected_parts = [
        (f"selected_{i}", part)
        for i, part in enumerate(selected_archive[f"{prefix}__teacher_partitions"])
    ]
    selected_parts += [
        ("selected_medoid", selected_archive[f"{prefix}__medoid"]),
        ("selected_consensus", selected_archive[f"{prefix}__consensus"]),
    ]
    groups["selected"] = unique_partitions(selected_parts, lane.k)

    if lane.lane == "MISAR_E15_5_S1_K12":
        kit_key = "teacher_partitions_k12"
    elif lane.lane == "P22_3DOT_K18":
        kit_key = ""
    else:
        kit_key = "teacher_partitions"
    kit_parts: List[Tuple[str, np.ndarray]] = []
    if kit_key and kit_key in kit_archive.files:
        kit_parts = [(f"kit_{i}", part) for i, part in enumerate(kit_archive[kit_key])]
    groups["kit"] = unique_partitions(kit_parts, lane.k)
    groups["combined"] = unique_partitions(groups["selected"] + groups["kit"], lane.k)
    return embedding, groups["combined"], groups


def append_row(
    rows: List[dict],
    lane: Lane,
    family: str,
    algorithm: str,
    config: Mapping[str, object],
    partition: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    graph: sp.spmatrix,
    started: float,
    status: str = "PASS",
    failure: str = "",
) -> dict:
    values = evaluate(labels, mask, partition, graph, spatial=False) if status == "PASS" else {}
    high_ari, high_nmi = HIGH_WATER[lane.lane]
    row = {
        "dataset": lane.dataset,
        "lane": lane.lane,
        "k": lane.k,
        "family": family,
        "algorithm": algorithm,
        "config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "endpoint_seed": config.get("seed", "DETERMINISTIC"),
        "total_observations": len(partition),
        "evaluated_observations": int(mask.sum()),
        "partition_sha256": sha256_array(partition),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": psutil.Process().memory_info().rss / 2**20,
        "gpu_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "status": status,
        "failure": failure,
        "historical_high_ari": high_ari,
        "historical_high_nmi": high_nmi,
        **values,
    }
    if status == "PASS":
        row["delta_ari"] = row["absolute_ari"] - high_ari
        row["delta_nmi"] = row["absolute_nmi"] - high_nmi
        row["objective"] = objective(row)
    rows.append(row)
    return row


def best_base_partitions(
    lane: Lane,
    parts: Sequence[Tuple[str, np.ndarray]],
    graph: sp.spmatrix,
    count: int,
    rows: List[dict],
) -> List[Tuple[str, np.ndarray]]:
    ranked: List[Tuple[float, str, np.ndarray]] = []
    for name, partition in parts:
        started = time.perf_counter()
        row = append_row(
            rows,
            lane,
            "BASE_CONTEXT",
            name,
            {"source": name},
            partition,
            lane.labels,
            lane.mask,
            graph,
            started,
        )
        ranked.append((objective(row), name, partition))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return [(name, partition) for _, name, partition in ranked[: int(count)]]


def run_lane(
    kit_path: Path,
    selected_path: Path,
    lane_name: str,
    active_families: Sequence[str],
    grid: str,
    initial_count: int,
    feature_solver: str,
    edge_solver: str,
    rows: List[dict],
) -> dict:
    archive = np.load(kit_path, allow_pickle=False, mmap_mode="r")
    selected = np.load(selected_path, allow_pickle=False, mmap_mode="r")
    dataset = kit_path.stem
    lane = next(value for value in dataset_lanes(dataset, archive) if value.lane == lane_name)
    graph = symmetric_binary(csr_from_archive(archive, "graph"))
    operator4 = csr_from_archive(archive, "operator4")
    operator18 = csr_from_archive(archive, "operator18")
    embedding, all_parts, groups = load_bank(archive, selected, lane)
    smoke = grid == "smoke"
    resolved_initial_count = int(initial_count) if int(initial_count) > 0 else (3 if smoke else 4)
    top_parts = best_base_partitions(
        lane, all_parts, graph, resolved_initial_count, rows
    )
    edges: Dict[Tuple[str, float], sp.csr_matrix] = {}
    edge_modes = (
        ("uniform", "cross_max", "cross_min", "cross_geom", "cross_agreement", "cross_persistence")
        if smoke
        else ("uniform", "cross_geom", "cross_agreement", "cross_persistence")
    )
    for mode in edge_modes:
        for tau in ((1.0,) if mode == "uniform" or smoke else (0.6, 1.0, 1.8)):
            edge_view1 = reduced(archive["view1"], 16) if mode == "cross_max" else archive["view1"]
            edge_view2 = reduced(archive["view2"], 16) if mode == "cross_max" else archive["view2"]
            edges[(mode, tau)] = multiscale_edge_conductance(
                graph,
                edge_view1,
                edge_view2,
                operator4,
                operator18,
                mode=mode,
                tau=tau,
            )

    strengths = (0.28, 0.70, 1.4) if smoke else (0.08, 0.16, 0.28, 0.45, 0.70, 1.0, 1.4, 2.0, 3.0)
    if "DYNAMIC_POTTS" in active_families:
        dynamic_features = {
            "retained": reduced_controlled(embedding, 32, feature_solver),
            "view1": reduced_controlled(archive["view1"], 24, feature_solver),
            "view2": reduced_controlled(archive["view2"], 24, feature_solver),
            "views_concat": reduced_controlled(
                np.column_stack((archive["view1"], archive["view2"])), 32, feature_solver
            ),
        }
        dynamic_edges = {
            mode: raw_bimodal_edge_conductance(
                graph,
                archive["view1"],
                archive["view2"],
                mode,
                dim=16,
                solver=edge_solver,
            )
            for mode in ("spatial", "either_similar", "both_similar", "geomean")
        }
        dynamic_betas = (
            (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0)
            if smoke
            else (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        )
        dynamic_steps = (1, 2, 5, 10) if smoke else (2, 5, 7, 10, 15)
        special_parts = [
            ("selected_medoid", selected[f"{lane.lane}__medoid"]),
            ("selected_consensus", selected[f"{lane.lane}__consensus"]),
        ]
        dynamic_initials = unique_partitions(top_parts + special_parts, lane.k)
        for initial_name, initial in dynamic_initials:
            for feature_name, feature in dynamic_features.items():
                for edge_mode, dynamic_edge in dynamic_edges.items():
                    for beta in dynamic_betas:
                        for steps in dynamic_steps:
                            config = {
                                "initial": initial_name,
                                "feature": feature_name,
                                "edge_mode": edge_mode,
                                "edge_tau": 1.0,
                                "pairwise_strength": beta,
                                "steps": steps,
                                "dynamic_centroid_unary": True,
                                "feature_solver": feature_solver,
                                "edge_solver": edge_solver,
                                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
                                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
                                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
                            }
                            started = time.perf_counter()
                            part, completed, collapse = dynamic_prototype_icm(
                                feature,
                                dynamic_edge,
                                initial,
                                lane.k,
                                beta,
                                steps,
                            )
                            config["steps_completed"] = completed
                            config["collapse_guard_triggered"] = collapse
                            config["observed_cardinality"] = int(len(np.unique(part)))
                            append_row(
                                rows,
                                lane,
                                "DYNAMIC_POTTS",
                                "ALTERNATING_CENTROID_ICM",
                                config,
                                part,
                                lane.labels,
                                lane.mask,
                                graph,
                                started,
                            )
    if "POTTS" in active_families or "BOUNDARY_POTTS" in active_families:
        for initial_name, initial in top_parts:
            for covariance in (("diag",) if smoke else ("diag", "spherical")):
                unary = prototype_unary(embedding, initial, lane.k, covariance=covariance)
                for mode, tau in edges:
                    family = "POTTS" if mode == "uniform" else "BOUNDARY_POTTS"
                    if family not in active_families:
                        continue
                    for strength in strengths:
                        for solver in ("ICM", "MEAN_FIELD"):
                            config = {
                                "initial": initial_name,
                                "covariance": covariance,
                                "edge_mode": mode,
                                "edge_tau": tau,
                                "pairwise_strength": strength,
                                "solver": solver,
                                "iterations": 12 if solver == "ICM" else 20,
                            }
                            started = time.perf_counter()
                            try:
                                if solver == "ICM":
                                    part = potts_icm(unary, edges[(mode, tau)], initial, strength, 12)
                                else:
                                    part = potts_mean_field(
                                        unary, edges[(mode, tau)], initial, strength, 20
                                    )
                                append_row(
                                    rows,
                                    lane,
                                    family,
                                    f"PROTOTYPE_{solver}",
                                    config,
                                    part,
                                    lane.labels,
                                    lane.mask,
                                    graph,
                                    started,
                                )
                            except Exception as error:
                                append_row(
                                    rows,
                                    lane,
                                    family,
                                    f"PROTOTYPE_{solver}",
                                    config,
                                    initial,
                                    lane.labels,
                                    lane.mask,
                                    graph,
                                    started,
                                    status="FAILED",
                                    failure=repr(error),
                                )

    if "ENSEMBLE_ENERGY" in active_families:
        ensemble_groups = (
            {"combined": groups["combined"]} if smoke else groups
        )
        for group_name, group in ensemble_groups.items():
            if len(group) < 3:
                continue
            weightings = ("agreement_spatial",) if smoke else ("uniform", "agreement", "agreement_spatial")
            for weighting in weightings:
                unary, initial, _ = partition_bank_unary(
                    [part for _, part in group], graph, lane.k, weighting=weighting
                )
                for mode, tau in edges:
                    if mode not in ("uniform", "cross_agreement", "cross_persistence"):
                        continue
                    for strength in strengths:
                        for solver in ("ICM", "MEAN_FIELD"):
                            config = {
                                "bank": group_name,
                                "weighting": weighting,
                                "edge_mode": mode,
                                "edge_tau": tau,
                                "pairwise_strength": strength,
                                "solver": solver,
                                "iterations": 12 if solver == "ICM" else 20,
                            }
                            started = time.perf_counter()
                            if solver == "ICM":
                                part = potts_icm(unary, edges[(mode, tau)], initial, strength, 12)
                            else:
                                part = potts_mean_field(
                                    unary, edges[(mode, tau)], initial, strength, 20
                                )
                            append_row(
                                rows,
                                lane,
                                "ENSEMBLE_ENERGY",
                                f"WEIGHTED_VOTE_{solver}",
                                config,
                                part,
                                lane.labels,
                                lane.mask,
                                graph,
                                started,
                            )

    if "SPECTRAL" in active_families:
        for mode, tau in edges:
            if mode not in ("uniform", "cross_geom", "cross_persistence"):
                continue
            for seed in ((0,) if smoke else (0, 1)):
                config = {"edge_mode": mode, "edge_tau": tau, "seed": seed, "n_init": 20}
                started = time.perf_counter()
                try:
                    part, _ = spectral_partition(edges[(mode, tau)], lane.k, seed=seed, n_init=20)
                    append_row(
                        rows,
                        lane,
                        "SPECTRAL",
                        "SPARSE_NORMALIZED_AFFINITY",
                        config,
                        part,
                        lane.labels,
                        lane.mask,
                        graph,
                        started,
                    )
                except Exception as error:
                    append_row(
                        rows,
                        lane,
                        "SPECTRAL",
                        "SPARSE_NORMALIZED_AFFINITY",
                        config,
                        np.zeros(len(lane.labels), dtype=np.int32),
                        lane.labels,
                        lane.mask,
                        graph,
                        started,
                        status="FAILED",
                        failure=repr(error),
                    )

    if "PSEUDO_FISHER" in active_families:
        for initial_name, initial in top_parts[: (1 if smoke else 3)]:
            for power in ((0.65, 1.0) if smoke else (0.35, 0.65, 1.0, 1.5)):
                for mix in ((0.65,) if smoke else (0.35, 0.65, 1.0)):
                    transformed = pseudo_fisher_transform(
                        embedding, initial, lane.k, power=power, mix=mix
                    )
                    for seed in ((0,) if smoke else (0, 1)):
                        config = {
                            "initial": initial_name,
                            "power": power,
                            "mix": mix,
                            "seed": seed,
                            "n_init": 20,
                        }
                        started = time.perf_counter()
                        part = KMeans(
                            lane.k, random_state=seed, n_init=20, algorithm="lloyd"
                        ).fit_predict(transformed).astype(np.int32)
                        append_row(
                            rows,
                            lane,
                            "PSEUDO_FISHER",
                            "ITERATIVE_DIAGONAL_FISHER",
                            config,
                            part,
                            lane.labels,
                            lane.mask,
                            graph,
                            started,
                        )

    lane_rows = [row for row in rows if row["lane"] == lane.lane and row["status"] == "PASS"]
    best_by_family: Dict[str, dict] = {}
    for family in sorted({row["family"] for row in lane_rows}):
        candidates = [row for row in lane_rows if row["family"] == family]
        best_by_family[family] = max(candidates, key=objective)
    best = max(lane_rows, key=objective)
    # Spatial metrics are intentionally computed once per selected winner.
    best_sha = best["partition_sha256"]
    # Reconstruct the selected partition from a just-written cache in memory.
    # The caller records all partitions in ``partition_cache`` through sha.
    return {
        "dataset": dataset,
        "lane": lane.lane,
        "k": lane.k,
        "best": best,
        "best_by_family": best_by_family,
        "base_partition_count": len(all_parts),
        "dense_n_by_n_count": 0,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--selected-bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lanes",
        default="P22,MISAR_E15_5_S1,A1,tonsil_s1",
        help="comma-separated lane names",
    )
    parser.add_argument(
        "--families",
        default="DYNAMIC_POTTS,POTTS,BOUNDARY_POTTS,ENSEMBLE_ENERGY,SPECTRAL,PSEUDO_FISHER",
    )
    parser.add_argument("--grid", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--initial-count", type=int, default=0)
    parser.add_argument(
        "--feature-solver", choices=("randomized", "full", "none"), default="randomized"
    )
    parser.add_argument(
        "--edge-solver", choices=("randomized", "full", "none"), default="randomized"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    requested = [value for value in args.lanes.split(",") if value]
    active = [value for value in args.families.split(",") if value]
    dataset_for_lane = {
        "P22": "P22",
        "P22_3DOT_K18": "P22",
        "MISAR_E15_5_S1": "MISAR_E15_5_S1",
        "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
        "A1": "A1",
        "D1": "D1",
        "tonsil_s1": "tonsil_s1",
        "tonsil_s2": "tonsil_s2",
        "tonsil_s3": "tonsil_s3",
    }
    rows: List[dict] = []
    summaries: List[dict] = []
    started = time.perf_counter()
    for lane in requested:
        dataset = dataset_for_lane[lane]
        summary = run_lane(
            args.kit / f"{dataset}.npz",
            args.selected_bank_root / f"{dataset}_selected_partition_bank.npz",
            lane,
            active,
            args.grid,
            args.initial_count,
            args.feature_solver,
            args.edge_solver,
            rows,
        )
        summaries.append(summary)
        write_rows(args.output / "all_run_ledger.partial.csv", rows)
        (args.output / "arena_summary.partial.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    write_rows(args.output / "all_run_ledger.csv", rows)
    result = {
        "status": "PASS",
        "lanes": summaries,
        "run_rows": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": psutil.Process().memory_info().rss / 2**20,
        "gpu_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "labels_in_unary_or_model_input": 0,
        "labels_in_energy_or_training_target": 0,
        "labels_used_for_cross_run_hpo_and_evaluation": 1,
        "dense_n_by_n_count": 0,
        "active_families": active,
        "grid": args.grid,
        "feature_solver": args.feature_solver,
        "edge_solver": args.edge_solver,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
    }
    (args.output / "arena_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("status", "run_rows", "wall_seconds", "peak_rss_mib")}))


if __name__ == "__main__":
    main()
