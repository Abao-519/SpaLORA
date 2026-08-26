#!/usr/bin/env python3
"""Label-free Night-19A D0 producer for real RNA+ATAC lanes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night17b_sfrd import (
    RelationPosterior,
    canonical_pair_bank,
    csr_from_carrier,
    deterministic_relation_smooth,
    encode_partition,
    relation_posterior,
    same_head_partition,
    sha256_array,
)
from SpaLORA.night17c_zero_start import node_trust_gate, reload_zero_start
from SpaLORA.night19a_gradient_d0 import (
    build_model,
    evidence_support_strata,
    prepare_tensors,
    state_sha256,
    train_standard_sum_d0,
)


TASKBOOK_SHA256 = "19107b229b1fbd06a6c6dafee797b469057ec8d75eab7dcec6e6ae22b8ad2e62"
CONFIG = {
    "config_id": "D0_NIGHT17C_Z01_STANDARD_SUM",
    "hidden_dim": 32,
    "residual_scale": 0.05,
    "learning_rate": 0.0007,
    "steps": 40,
    "relation_weight": 1.0,
    "anchor_weight": 4.0,
    "self_return_weight": 4.0,
    "consistency_weight": 0.5,
    "variance_weight": 0.1,
    "relation_smooth_alpha": 0.2,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def graph_sha256(graph: sp.csr_matrix) -> str:
    digest = hashlib.sha256()
    for value in (graph.data, graph.indices, graph.indptr, np.asarray(graph.shape, dtype=np.int64)):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def has_internal_edge_per_cluster(partition: np.ndarray, graph: sp.csr_matrix, k: int) -> bool:
    labels = encode_partition(partition)
    upper = sp.triu(graph, k=1).tocoo()
    supported = np.zeros(k, dtype=bool)
    same = labels[upper.row] == labels[upper.col]
    for cluster in np.unique(labels[upper.row[same]]):
        supported[int(cluster)] = True
    return bool(np.all(supported))


def uniform_feasible_posterior(
    partitions: np.ndarray,
    graph: sp.csr_matrix,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    k: int,
) -> tuple[RelationPosterior, np.ndarray]:
    eligible = []
    for index, raw in enumerate(partitions):
        partition = encode_partition(raw)
        sizes = np.bincount(partition, minlength=k)
        if np.unique(partition).size != k or int(sizes.min()) < 2:
            continue
        if not has_internal_edge_per_cluster(partition, graph, k):
            continue
        eligible.append(index)
    indices = np.asarray(eligible, dtype=np.int64)
    if indices.size < 2:
        raise ValueError("uniform stress-lane bank has fewer than two structurally feasible candidates")
    weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    same = partitions[indices][:, pair_i] == partitions[indices][:, pair_j]
    probability = np.asarray(weights @ same.astype(np.float64), dtype=np.float64)
    eps = 1e-12
    entropy = -(probability * np.log(probability + eps) + (1.0 - probability) * np.log(1.0 - probability + eps))
    uncertainty = np.clip(entropy / np.log(2.0), 0.0, 1.0)
    confidence = np.abs(2.0 * probability - 1.0) * (1.0 - uncertainty)
    all_weights = np.zeros(partitions.shape[0], dtype=np.float64)
    all_weights[indices] = weights
    return (
        RelationPosterior(
            probability_same=probability.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            positive_weight=(confidence * (probability > 0.5)).astype(np.float32),
            negative_weight=(confidence * (probability < 0.5)).astype(np.float32),
            candidate_weights=all_weights,
            selected_candidate_count=int(indices.size),
        ),
        indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--relation-source", choices=("NIGHT16H_UNBIASED_WEIGHTED", "UNIFORM_FEASIBLE_STRESS"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    execution_contract_path = Path(args.execution_contract)
    execution_contract = json.loads(execution_contract_path.read_text(encoding="utf-8"))
    if execution_contract.get("schema") != "night19a-gradient-d0-formula-freeze-rev1":
        raise ValueError("Night-19A REV1 execution contract schema mismatch")
    execution_contract_sha256 = file_sha256(execution_contract_path)
    carrier_path = Path(args.carrier)
    bank_path = Path(args.candidate_bank)
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "ids", "view1", "view2", "retained",
        "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape",
    }
    if not required.issubset(carrier):
        raise ValueError("carrier is missing required numeric fields")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in archive.files}
    if "ids" not in bank or "partitions" not in bank:
        raise ValueError("candidate bank lacks ids/partitions")
    if not np.array_equal(carrier["ids"], bank["ids"]):
        raise ValueError("carrier/candidate ordered IDs differ")
    if args.relation_source == "NIGHT16H_UNBIASED_WEIGHTED":
        if args.feasibility is None:
            raise ValueError("Night-16H relation source requires feasibility evidence")
        feasibility_path = Path(args.feasibility)
        records = list(csv.DictReader(feasibility_path.open(encoding="utf-8")))
        if "candidate_ids" not in bank:
            raise ValueError("Night-16H bank lacks candidate IDs")
        if [row["candidate_id"] for row in records] != bank["candidate_ids"].astype(str).tolist():
            raise ValueError("candidate evidence and partition order differ")
        posterior = relation_posterior(
            bank["partitions"], records, pair_i, pair_j, bank_mode="UNBIASED_BANK", weighted=True
        )
        eligible_indices = np.flatnonzero(posterior.candidate_weights > 0)
        feasibility_sha = file_sha256(feasibility_path)
    else:
        posterior, eligible_indices = uniform_feasible_posterior(
            bank["partitions"], graph, pair_i, pair_j, args.k
        )
        feasibility_sha = None
    strata, strata_diagnostics = evidence_support_strata(posterior, is_spatial)
    anchor = deterministic_relation_smooth(
        carrier["retained"], pair_i, pair_j, posterior, alpha=float(CONFIG["relation_smooth_alpha"])
    )
    gate, gate_diagnostics = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, posterior
    )
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    tensors = prepare_tensors(
        carrier["view1"], carrier["view2"], carrier["retained"], anchor, gate,
        pair_i, pair_j, is_spatial, posterior, strata, device,
    )
    model = build_model(
        carrier["view1"].shape[1], carrier["view2"].shape[1], carrier["retained"].shape[1],
        CONFIG, args.training_seed, device,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        result = train_standard_sum_d0(model, tensors, CONFIG)
    final = np.asarray(result["final_representation"], dtype=np.float32)
    partition = same_head_partition(final, args.k, seed=0)
    if np.unique(partition).size != args.k:
        raise RuntimeError("D0 endpoint violates exact K")
    replay = reload_zero_start(
        result["state_dict"], carrier["view1"], carrier["view2"], carrier["retained"],
        anchor, gate, CONFIG, device=device,
    )
    if not np.array_equal(replay, final):
        raise RuntimeError("strict checkpoint reload representation mismatch")
    artifact_path = output / "artifact.npz"
    np.savez_compressed(
        artifact_path,
        ids=carrier["ids"],
        representation=final,
        partition=partition.astype(np.int32),
        anchor_representation=anchor.astype(np.float32),
    )
    with np.load(artifact_path, allow_pickle=False) as reloaded:
        if not np.array_equal(reloaded["ids"], carrier["ids"]):
            raise RuntimeError("artifact reload IDs mismatch")
        if not np.array_equal(reloaded["representation"], final):
            raise RuntimeError("artifact reload representation mismatch")
        if not np.array_equal(reloaded["partition"], partition):
            raise RuntimeError("artifact reload partition mismatch")
    checkpoint_path = output / "checkpoint.pt"
    authority = {
        "config": CONFIG,
        "taskbook_sha256": TASKBOOK_SHA256,
        "execution_contract_sha256": execution_contract_sha256,
        "lane": args.lane,
        "training_seed": int(args.training_seed),
        "carrier_sha256": file_sha256(carrier_path),
        "candidate_bank_sha256": file_sha256(bank_path),
        "feasibility_sha256": feasibility_sha,
        "ordered_ids_sha256": sha256_array(carrier["ids"]),
        "anchor_representation_sha256": sha256_array(anchor),
        "trust_gate_sha256": sha256_array(gate),
        "relation_probability_sha256": sha256_array(posterior.probability_same),
    }
    torch.save({"state_dict": result["state_dict"], "authority": authority}, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device)
    if loaded["authority"] != authority:
        raise RuntimeError("checkpoint authority strict reload mismatch")
    if state_sha256(loaded["state_dict"]) != result["final_state_sha256"]:
        raise RuntimeError("checkpoint state strict reload mismatch")
    sizes = np.bincount(encode_partition(partition), minlength=args.k).astype(int)
    manifest = {
        "schema": "night19a-gradient-d0-producer-v1",
        "lane": args.lane,
        "n": int(carrier["ids"].size),
        "k": int(args.k),
        "training_seed": int(args.training_seed),
        "relation_source": args.relation_source,
        "selected_relation_candidate_count": int(posterior.selected_candidate_count),
        "eligible_candidate_indices_sha256": sha256_array(eligible_indices),
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "graph_sha256": graph_sha256(graph),
        "pair_count": int(pair_i.size),
        "carrier_sha256": authority["carrier_sha256"],
        "candidate_bank_sha256": authority["candidate_bank_sha256"],
        "feasibility_sha256": feasibility_sha,
        "ordered_ids_sha256": authority["ordered_ids_sha256"],
        "anchor_representation_sha256": authority["anchor_representation_sha256"],
        "trust_gate_sha256": authority["trust_gate_sha256"],
        "relation_probability_sha256": authority["relation_probability_sha256"],
        "artifact_sha256": file_sha256(artifact_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "partition_sha256": sha256_array(partition),
        "representation_sha256": sha256_array(final),
        "cluster_sizes": sizes.tolist(),
        "min_cluster_size": int(sizes.min()),
        "exact_k": bool(np.unique(partition).size == args.k),
        "config": CONFIG,
        "taskbook_sha256": TASKBOOK_SHA256,
        "execution_contract_sha256": execution_contract_sha256,
        "core_source_sha256": file_sha256(Path(__file__).resolve().parents[2] / "SpaLORA" / "night19a_gradient_d0.py"),
        "producer_source_sha256": source_sha256(),
        "producer_label_reads": 0,
        "annotation_columns_accessed": [],
        "zero_start_exact_anchor": True,
        "step0_semantics": "ZERO_START_BOUNDARY_NOT_USED_FOR_PERSISTENT_CONFLICT_GATE",
        "zero_norm_cosine_semantics": "NULL_NA_NOT_ZERO",
        "parameter_coordinate_semantics": "FIXED_NAMED_PARAMETER_ORDER_UNUSED_FILLED_WITH_SAME_SHAPE_ZERO",
        "parameter_l2_change": result["parameter_l2_change"],
        "initial_state_sha256": result["initial_state_sha256"],
        "final_state_sha256": result["final_state_sha256"],
        "trajectory": result["trajectory"],
        "evidence_strata": strata_diagnostics,
        "node_trust_gate": gate_diagnostics,
        "strict_checkpoint_reload": "PASS",
        "artifact_reload": "PASS",
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)) if device.startswith("cuda") else 0.0,
        "thread_limits": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "threadpool_limits": 1},
    }
    manifest_path = output / "producer.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "lane", "training_seed", "n", "k", "pair_count", "parameter_l2_change",
        "partition_sha256", "representation_sha256", "wall_seconds", "peak_gpu_mib"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
