#!/usr/bin/env python3
"""Fresh-process exact replay for one registered Night-15C Potts config."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15c_cluster_energy import (
    csr_from_archive,
    dynamic_prototype_icm,
    raw_bimodal_edge_conductance,
    reduced,
    reduced_controlled,
    sha256_array,
)


def encode(value: np.ndarray) -> np.ndarray:
    _, result = np.unique(value.astype(str), return_inverse=True)
    return result.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--feature", choices=("retained", "view1", "view2", "views_concat"), required=True)
    parser.add_argument("--edge", choices=("spatial", "either_similar", "both_similar", "geomean"), required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--feature-solver", choices=("randomized", "full", "none"), default="randomized")
    parser.add_argument("--edge-solver", choices=("randomized", "full", "none"), default="randomized")
    parser.add_argument(
        "--initial-source",
        default="medoid",
        help="medoid, consensus, selected:N, or kit:N",
    )
    parser.add_argument("--partition-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    data = np.load(args.dataset, allow_pickle=False)
    bank = np.load(args.bank, allow_pickle=False)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    prefix = args.lane
    if args.initial_source == "medoid":
        initial = np.asarray(bank[f"{prefix}__medoid"], dtype=np.int32)
    elif args.initial_source == "consensus":
        initial = np.asarray(bank[f"{prefix}__consensus"], dtype=np.int32)
    elif args.initial_source.startswith("selected:"):
        index = int(args.initial_source.split(":", 1)[1])
        initial = np.asarray(
            bank[f"{prefix}__teacher_partitions"][index], dtype=np.int32
        )
    elif args.initial_source.startswith("kit:"):
        index = int(args.initial_source.split(":", 1)[1])
        key = "teacher_partitions_k12" if args.lane == "MISAR_E15_5_S1_K12" else "teacher_partitions"
        if key not in data.files:
            raise KeyError(f"{key} is unavailable for {args.lane}")
        initial = np.asarray(data[key][index], dtype=np.int32)
    else:
        raise ValueError(f"unknown initial source: {args.initial_source}")
    selected = np.asarray(bank[f"{prefix}__teacher_partitions"], dtype=np.int32)
    matching_selected = [
        index for index, value in enumerate(selected)
        if sha256_array(value) == sha256_array(initial)
    ]
    if args.lane == "P22_3DOT_K18":
        labels = data["labels_k18_author_assignment"].astype(str)
        k = 18
    elif args.lane == "MISAR_E15_5_S1_K12":
        labels = data["labels_primary"].astype(str)
        k = 12
    else:
        labels = data["labels_primary"].astype(str)
        k = int(data["k_primary"][0])
    mask = data["label_mask"].astype(bool)
    if args.feature == "retained":
        feature = reduced_controlled(bank[f"{prefix}__retained_embedding"], 32, args.feature_solver)
    elif args.feature == "view1":
        feature = reduced_controlled(data["view1"], 24, args.feature_solver)
    elif args.feature == "view2":
        feature = reduced_controlled(data["view2"], 24, args.feature_solver)
    else:
        feature = reduced_controlled(
            np.column_stack((data["view1"], data["view2"])), 32, args.feature_solver
        )
    graph = csr_from_archive(data, "graph")
    conductance = raw_bimodal_edge_conductance(
        graph,
        data["view1"],
        data["view2"],
        args.edge,
        dim=16,
        solver=args.edge_solver,
    )
    observed, completed, collapse = dynamic_prototype_icm(
        feature, conductance, initial, k, args.beta, args.steps
    )
    truth = encode(labels[mask])
    predicted = observed[mask]
    graph_binary = graph.maximum(graph.T).tocsr()
    graph_binary.data[:] = 1.0
    graph_binary.setdiag(0)
    graph_binary.eliminate_zeros()
    total_weight = float(graph_binary.sum())
    coo = graph_binary.tocoo()
    morans = []
    gearys = []
    for cluster in range(k):
        value = (observed == cluster).astype(np.float64)
        centered = value - value.mean()
        denominator = float(centered @ centered)
        if denominator <= 1e-12 or total_weight <= 0:
            continue
        morans.append(
            float(len(value) / total_weight * (centered @ (graph_binary @ centered)) / denominator)
        )
        squared = (value[coo.row] - value[coo.col]) ** 2
        gearys.append(
            float((len(value) - 1) / (2 * total_weight) * np.dot(coo.data, squared) / denominator)
        )
    result = {
        "status": "PASS",
        "lane": args.lane,
        "total_observations": int(len(observed)),
        "evaluated_observations": int(mask.sum()),
        "registered_ordered_id_sha256": metadata["ordered_id_sha256"],
        "ids_array_sha256": sha256_array(data["ids"]),
        "k_requested": int(k),
        "initial_cardinality": int(len(np.unique(initial))),
        "observed_cardinality": int(len(np.unique(observed))),
        "initial_cluster_sizes": np.bincount(initial, minlength=k).astype(int).tolist(),
        "observed_cluster_sizes": np.bincount(observed, minlength=k).astype(int).tolist(),
        "changed_observations": int(np.sum(initial != observed)),
        "initial_partition_sha256": sha256_array(initial),
        "initial_source": args.initial_source,
        "matching_selected_teacher_indices": matching_selected,
        "partition_sha256": sha256_array(observed),
        "feature": args.feature,
        "edge": args.edge,
        "feature_solver": args.feature_solver,
        "edge_solver": args.edge_solver,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
        "beta": args.beta,
        "steps_requested": args.steps,
        "steps_completed": completed,
        "collapse_guard_triggered": collapse,
        "absolute_ari": float(adjusted_rand_score(truth, predicted)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ami": float(adjusted_mutual_info_score(truth, predicted)),
        "fmi": float(fowlkes_mallows_score(truth, predicted)),
        "morans_i": float(np.mean(morans)),
        "gearys_c": float(np.mean(gearys)),
        "labels_in_unary_or_energy": 0,
        "labels_used_for_evaluation": 1,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if result["observed_cardinality"] != k:
        raise RuntimeError("cardinality mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.partition_output is not None:
        args.partition_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.partition_output, observed, allow_pickle=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
