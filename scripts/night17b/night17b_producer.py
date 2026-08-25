#!/usr/bin/env python3
"""Label-free real-data producer for the Night-17B SFRD P0."""

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
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night17b_sfrd import (
    canonical_pair_bank,
    csr_from_carrier,
    deterministic_relation_smooth,
    permute_relation,
    relation_posterior,
    reload_representation,
    row_normalize,
    same_head_partition,
    sha256_array,
    standardize,
    train_residual,
)


CONFIGS = (
    {
        "config_id": "F01_BALANCED",
        "hidden_dim": 32,
        "residual_scale": 0.15,
        "learning_rate": 0.001,
        "steps": 40,
        "relation_weight": 1.0,
        "anchor_weight": 2.0,
        "consistency_weight": 0.5,
        "variance_weight": 0.1,
        "margin": 1.0,
    },
    {
        "config_id": "F02_RELATION",
        "hidden_dim": 32,
        "residual_scale": 0.25,
        "learning_rate": 0.001,
        "steps": 40,
        "relation_weight": 2.0,
        "anchor_weight": 1.0,
        "consistency_weight": 0.5,
        "variance_weight": 0.1,
        "margin": 1.0,
    },
    {
        "config_id": "F03_ANCHORED",
        "hidden_dim": 32,
        "residual_scale": 0.10,
        "learning_rate": 0.0007,
        "steps": 40,
        "relation_weight": 1.0,
        "anchor_weight": 4.0,
        "consistency_weight": 1.0,
        "variance_weight": 0.1,
        "margin": 1.0,
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-id", default="ALL")
    parser.add_argument("--train-arms", default="ALL")
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    carrier_path = Path(args.carrier)
    bank_path = Path(args.candidate_bank)
    feasibility_path = Path(args.feasibility)
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in archive.files}
    records = list(csv.DictReader(feasibility_path.open(encoding="utf-8")))
    if not np.array_equal(carrier["ids"], bank["ids"]):
        raise ValueError("carrier/candidate ordered IDs differ")
    if len(records) != bank["partitions"].shape[0]:
        raise ValueError("candidate record count differs from partition bank")
    if [row["candidate_id"] for row in records] != bank["candidate_ids"].tolist():
        raise ValueError("candidate metadata order differs from partition bank")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    full_weighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "FULL_BANK", True)
    full_unweighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "FULL_BANK", False)
    unbiased_weighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", True)
    permuted = permute_relation(full_weighted)
    retained = row_normalize(standardize(carrier["retained"]))
    run_ids = []
    arm_ids = []
    config_ids = []
    partitions = []
    representation_hashes = []
    diagnostics = []

    def append(run_id: str, arm: str, config_id: str, representation: np.ndarray, diag: dict) -> None:
        partition = same_head_partition(representation, args.k, seed=0)
        if np.unique(partition).size != args.k:
            raise RuntimeError("same head violated exact K")
        run_ids.append(run_id)
        arm_ids.append(arm)
        config_ids.append(config_id)
        partitions.append(partition)
        representation_hashes.append(sha256_array(representation))
        diagnostics.append(
            {
                "run_id": run_id,
                "arm": arm,
                "config_id": config_id,
                "partition_sha256": sha256_array(partition),
                "representation_sha256": sha256_array(representation),
                "min_cluster_size": int(np.bincount(partition, minlength=args.k).min()),
                "cluster_sizes": np.bincount(partition, minlength=args.k).astype(int).tolist(),
                **diag,
            }
        )

    append(
        "BASELINE__FROZEN_RETAINED",
        "FROZEN_RETAINED_SAME_HEAD",
        "BASELINE",
        retained,
        {"actual_optimizer_steps": 0, "parameter_l2_change": 0.0},
    )
    smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, full_weighted, alpha=0.2)
    append(
        "BASELINE__FEASIBLE_RELATION_SMOOTH",
        "FEASIBLE_CONSENSUS_SAME_HEAD",
        "BASELINE",
        smooth,
        {"actual_optimizer_steps": 0, "parameter_l2_change": 0.0},
    )
    checkpoint_states = {}
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    arm_specs = (
        ("FULL_WEIGHTED", full_weighted, (1.0, 1.0)),
        ("UNWEIGHTED_RELATION", full_unweighted, (1.0, 1.0)),
        ("UNBIASED_BANK_WEIGHTED", unbiased_weighted, (1.0, 1.0)),
        ("RELATION_PERMUTED", permuted, (1.0, 1.0)),
        ("SINGLE_VIEW1", full_weighted, (1.0, 0.0)),
        ("SINGLE_VIEW2", full_weighted, (0.0, 1.0)),
    )
    active_configs = tuple(config for config in CONFIGS if args.config_id == "ALL" or config["config_id"] == args.config_id)
    if not active_configs:
        raise ValueError(f"unknown config id {args.config_id}")
    requested_arms = None if args.train_arms == "ALL" else set(args.train_arms.split(","))
    active_arms = tuple(spec for spec in arm_specs if requested_arms is None or spec[0] in requested_arms)
    if not active_arms:
        raise ValueError("no requested training arm")
    with threadpool_limits(limits=1):
        for config in active_configs:
            for arm, posterior, view_mask in active_arms:
                result = train_residual(
                    carrier["view1"],
                    carrier["view2"],
                    carrier["retained"],
                    pair_i,
                    pair_j,
                    is_spatial,
                    posterior,
                    config,
                    args.training_seed,
                    view_mask=view_mask,
                    device=device,
                )
                run_id = f"{config['config_id']}__{arm}__S{args.training_seed}"
                append(run_id, arm, config["config_id"], result.representation, dict(result.diagnostics))
                if arm == "FULL_WEIGHTED":
                    checkpoint_states[config["config_id"]] = result.state_dict
                    replayed = reload_representation(
                        result.state_dict,
                        carrier["view1"],
                        carrier["view2"],
                        carrier["retained"],
                        config,
                        view_mask=view_mask,
                        device=device,
                    )
                    if not np.array_equal(replayed, result.representation):
                        raise RuntimeError("strict in-process checkpoint round-trip mismatch")
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "schema": "night17b-sfrd-checkpoint-v1",
            "lane": args.lane,
            "training_seed": args.training_seed,
            "configs": list(active_configs),
            "full_weighted_state_dict": checkpoint_states,
        },
        checkpoint_path,
    )
    artifact_path = output / "producer.npz"
    np.savez_compressed(
        artifact_path,
        ids=carrier["ids"],
        run_ids=np.asarray(run_ids, dtype="U"),
        arm_ids=np.asarray(arm_ids, dtype="U"),
        config_ids=np.asarray(config_ids, dtype="U"),
        partitions=np.asarray(partitions, dtype=np.int32),
        representation_sha256=np.asarray(representation_hashes, dtype="U64"),
        pair_i=pair_i.astype(np.int32),
        pair_j=pair_j.astype(np.int32),
        pair_is_spatial=is_spatial.astype(np.uint8),
        relation_probability=full_weighted.probability_same,
        relation_uncertainty=full_weighted.uncertainty,
    )
    peak_gpu = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    manifest = {
        "schema": "night17b-sfrd-producer-v1",
        "lane": args.lane,
        "k": args.k,
        "n": int(carrier["ids"].size),
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "spatial_graph_nnz": int(graph.nnz),
        "pair_count": int(pair_i.size),
        "spatial_pair_count": int(is_spatial.sum()),
        "feature_or_union_pair_count": int((~is_spatial).sum()),
        "full_bank_candidate_count": full_weighted.selected_candidate_count,
        "unbiased_bank_candidate_count": unbiased_weighted.selected_candidate_count,
        "producer_label_reads": 0,
        "candidate_weight_inputs": ["molecular_joint", "topology_joint", "persistence"],
        "carrier_sha256": file_sha256(carrier_path),
        "candidate_bank_sha256": file_sha256(bank_path),
        "feasibility_sha256": file_sha256(feasibility_path),
        "artifact_sha256": file_sha256(artifact_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "run_diagnostics": diagnostics,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_mb": peak_gpu,
        "device": device,
        "thread_limits": {"OMP": 1, "MKL": 1, "OPENBLAS": 1, "threadpoolctl": 1},
    }
    manifest_path = output / "producer.producer.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "pair_count", "full_bank_candidate_count", "unbiased_bank_candidate_count", "wall_seconds", "peak_gpu_mb")}, indent=2))


if __name__ == "__main__":
    main()
