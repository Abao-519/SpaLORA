#!/usr/bin/env python3
"""Label-isolated placenta producer for the frozen Night-17C Z01 core."""

from __future__ import annotations

import argparse
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

import SpaLORA.night17c_zero_start as frozen_core
import SpaLORA.night19c_zero_start_transfer as adapter
from SpaLORA.night17b_sfrd import (
    canonical_pair_bank,
    deterministic_relation_smooth,
    row_normalize,
    same_head_partition,
    sha256_array,
    standardize,
)
from SpaLORA.night17c_zero_start import (
    node_trust_gate,
    reload_zero_start,
    stratified_permute_relation,
    train_zero_start,
)


def load_numeric_carrier(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        discovered = sorted(archive.files)
        needed = [
            "ids", "view1", "view2", "retained",
            "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape",
        ]
        missing = [key for key in needed if key not in archive.files]
        if missing:
            raise ValueError(f"carrier numeric allow-list keys missing: {missing}")
        carrier = {key: np.asarray(archive[key]) for key in needed}
    annotation_like = [key for key in needed if any(token in key.lower() for token in ("label", "truth", "annotation"))]
    if annotation_like:
        raise RuntimeError(f"annotation-like key entered producer allow-list: {annotation_like}")
    return carrier, discovered, needed


def csr_from_carrier(carrier):
    return sp.csr_matrix(
        (carrier["graph0__data"], carrier["graph0__indices"], carrier["graph0__indptr"]),
        shape=tuple(int(value) for value in carrier["graph0__shape"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bank-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--taskbook", required=True)
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    carrier_path = Path(args.carrier)
    bank_path = Path(args.bank)
    bank_manifest_path = Path(args.bank_manifest)
    carrier, discovered_keys, accessed_keys = load_numeric_carrier(carrier_path)
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if adapter.file_sha256(bank_path) != bank_manifest["output_bank_sha256"]:
        raise ValueError("relation bank SHA mismatch")
    if bank_manifest["candidate_count"] != 16 or bank_manifest["labels_read"] != 0:
        raise ValueError("relation bank authority contract mismatch")
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in ("ids", "candidate_ids", "partitions", "candidate_weights")}
    if not np.array_equal(carrier["ids"], bank["ids"]):
        raise ValueError("carrier/relation-bank ordered IDs differ")
    if bank["candidate_ids"].astype(str).tolist() != bank_manifest["candidate_ids"]:
        raise ValueError("relation-bank candidate order differs from manifest")
    if [sha256_array(row) for row in bank["partitions"]] != bank_manifest["partition_sha256"]:
        raise ValueError("relation-bank partition authority mismatch")
    if not np.array_equal(bank["candidate_weights"], np.full(16, 1.0 / 16.0)):
        raise ValueError("relation-bank candidate weights are not exact equal weights")

    graph = csr_from_carrier(carrier)
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    posterior = adapter.equal_weight_relation_posterior(bank["partitions"], pair_i, pair_j)
    permuted = stratified_permute_relation(posterior, is_spatial)
    smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, posterior, alpha=0.2)
    permuted_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, permuted, alpha=0.2)
    gate, gate_diag = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    permuted_gate, permuted_gate_diag = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, permuted
    )
    retained = row_normalize(standardize(carrier["retained"]))
    config = dict(adapter.Z01_CONSERVATIVE)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    representations = []
    partitions = []
    run_ids = []
    diagnostics = []

    def append(run_id: str, representation: np.ndarray, diag: dict) -> None:
        representation = np.asarray(representation, dtype=np.float32)
        partition = same_head_partition(representation, 10, seed=0)
        if np.unique(partition).size != 10:
            raise RuntimeError(f"{run_id} violates exact K=10")
        run_ids.append(run_id)
        representations.append(representation)
        partitions.append(partition.astype(np.int32))
        diagnostics.append({
            "run_id": run_id,
            "representation_sha256": sha256_array(representation),
            "partition_sha256": sha256_array(partition),
            "cluster_sizes": np.bincount(partition, minlength=10).astype(int).tolist(),
            "min_cluster_size": int(np.bincount(partition, minlength=10).min()),
            **diag,
        })

    append("FROZEN_RETAINED", retained, {"actual_optimizer_steps": 0})
    append("RELATION_SMOOTH", smooth, {"actual_optimizer_steps": 0})
    append("ZERO_RESIDUAL", smooth, {"actual_optimizer_steps": 0, "step0_exact_smooth": True})
    with threadpool_limits(limits=1):
        permuted_result = train_zero_start(
            carrier["view1"], carrier["view2"], carrier["retained"], permuted_smooth,
            permuted_gate, pair_i, pair_j, is_spatial, permuted, config, args.training_seed,
            device=device,
        )
        full_result = train_zero_start(
            carrier["view1"], carrier["view2"], carrier["retained"], smooth, gate,
            pair_i, pair_j, is_spatial, posterior, config, args.training_seed, device=device,
        )
    append("PERMUTED_RELATION", permuted_result.representation, {
        **dict(permuted_result.diagnostics), "node_gate": permuted_gate_diag,
        "relation_probability_sha256": sha256_array(permuted.probability_same),
        "relation_uncertainty_sha256": sha256_array(permuted.uncertainty),
        "node_gate_sha256": sha256_array(permuted_gate),
        "smooth_carrier_sha256": sha256_array(permuted_smooth),
        "relation_gate_smooth_same_source": True,
    })
    append("Z01_FULL", full_result.representation, {
        **dict(full_result.diagnostics), "node_gate": gate_diag,
        "relation_probability_sha256": sha256_array(posterior.probability_same),
        "relation_uncertainty_sha256": sha256_array(posterior.uncertainty),
        "node_gate_sha256": sha256_array(gate),
        "smooth_carrier_sha256": sha256_array(smooth),
        "relation_gate_smooth_same_source": True,
    })
    for result, state_smooth, state_gate in (
        (permuted_result, permuted_smooth, permuted_gate), (full_result, smooth, gate)
    ):
        replayed = reload_zero_start(
            result.state_dict, carrier["view1"], carrier["view2"], carrier["retained"],
            state_smooth, state_gate, config, device=device,
        )
        if not np.array_equal(replayed, result.representation):
            raise RuntimeError("in-process strict checkpoint replay mismatch")

    checkpoint_path = output / "checkpoint.pt"
    torch.save({
        "schema": "night19c-placenta-zero-start-checkpoint-v1",
        "training_seed": args.training_seed,
        "config": config,
        "permuted_state_dict": permuted_result.state_dict,
        "full_state_dict": full_result.state_dict,
    }, checkpoint_path)
    artifact_path = output / "producer.npz"
    np.savez_compressed(
        artifact_path,
        ids=carrier["ids"],
        run_ids=np.asarray(run_ids, dtype="U"),
        partitions=np.asarray(partitions, dtype=np.int32),
        representations=np.asarray(representations, dtype=np.float32),
        representation_sha256=np.asarray([sha256_array(value) for value in representations], dtype="U64"),
        pair_i=pair_i.astype(np.int32), pair_j=pair_j.astype(np.int32),
        pair_is_spatial=is_spatial.astype(np.uint8),
        primary_relation_probability=posterior.probability_same,
        primary_relation_uncertainty=posterior.uncertainty,
        primary_node_gate=gate,
        primary_smooth=smooth,
        permuted_smooth=permuted_smooth,
        permuted_node_gate=permuted_gate,
    )
    manifest = {
        "schema": "night19c-placenta-zero-start-producer-v1",
        "lane": "PLACENTA_K10", "n": int(carrier["ids"].size), "k": 10,
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "pair_count": int(pair_i.size), "spatial_graph_nnz": int(graph.nnz),
        "training_seed": args.training_seed, "device": device,
        "z01_config": config,
        "candidate_count": 16, "candidate_ids": bank["candidate_ids"].astype(str).tolist(),
        "candidate_partition_sha256": bank_manifest["partition_sha256"],
        "candidate_weights": bank["candidate_weights"].tolist(),
        "bank_authority_sha256": bank_manifest["bank_authority_sha256"],
        "bank_sha256": adapter.file_sha256(bank_path),
        "bank_manifest_sha256": adapter.file_sha256(bank_manifest_path),
        "carrier_sha256": adapter.file_sha256(carrier_path),
        "carrier_discovered_keys": discovered_keys,
        "carrier_accessed_keys": accessed_keys,
        "carrier_annotation_arrays_accessed": 0,
        "producer_label_reads": 0,
        "run_ids": run_ids,
        "partition_sha256": [sha256_array(value) for value in partitions],
        "representation_sha256": [sha256_array(value) for value in representations],
        "run_diagnostics": diagnostics,
        "artifact_sha256": adapter.file_sha256(artifact_path),
        "checkpoint_sha256": adapter.file_sha256(checkpoint_path),
        "source_sha256": {
            "night17c_frozen_core": adapter.file_sha256(Path(frozen_core.__file__)),
            "night19c_adapter": adapter.file_sha256(Path(adapter.__file__)),
            "night19c_producer": adapter.file_sha256(Path(__file__)),
            "taskbook": adapter.file_sha256(Path(args.taskbook)),
        },
        "zero_start_formula_modified": False,
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "thread_limits": {"OMP": 1, "MKL": 1, "OPENBLAS": 1, "threadpoolctl": 1},
    }
    (output / "producer.producer.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "pair_count", "wall_seconds", "peak_gpu_mib")}, indent=2))


if __name__ == "__main__":
    main()
