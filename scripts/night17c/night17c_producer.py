#!/usr/bin/env python3
"""Label-free producer for Night-17C zero-start relation refinement."""

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
    relation_posterior,
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


CONFIGS = (
    {
        "config_id": "Z01_CONSERVATIVE",
        "hidden_dim": 32,
        "residual_scale": 0.05,
        "learning_rate": 0.0007,
        "steps": 40,
        "relation_weight": 1.0,
        "anchor_weight": 4.0,
        "self_return_weight": 4.0,
        "consistency_weight": 0.5,
        "variance_weight": 0.1,
    },
    {
        "config_id": "Z02_BALANCED",
        "hidden_dim": 32,
        "residual_scale": 0.10,
        "learning_rate": 0.001,
        "steps": 40,
        "relation_weight": 1.0,
        "anchor_weight": 3.0,
        "self_return_weight": 4.0,
        "consistency_weight": 0.5,
        "variance_weight": 0.1,
    },
    {
        "config_id": "Z03_ACTIVE",
        "hidden_dim": 32,
        "residual_scale": 0.15,
        "learning_rate": 0.001,
        "steps": 40,
        "relation_weight": 2.0,
        "anchor_weight": 2.0,
        "self_return_weight": 3.0,
        "consistency_weight": 0.5,
        "variance_weight": 0.1,
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
    if [row["candidate_id"] for row in records] != bank["candidate_ids"].tolist():
        raise ValueError("candidate evidence order differs from locked partitions")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    full_weighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "FULL_BANK", True)
    unbiased_weighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", True)
    unbiased_unweighted = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", False)
    permuted = stratified_permute_relation(unbiased_weighted, is_spatial)
    full_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, full_weighted, alpha=0.2)
    unbiased_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, unbiased_weighted, alpha=0.2)
    unweighted_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, unbiased_unweighted, alpha=0.2)
    permuted_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, permuted, alpha=0.2)
    primary_gate, primary_gate_diag = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, unbiased_weighted
    )
    full_gate, full_gate_diag = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, full_weighted)
    unweighted_gate, unweighted_gate_diag = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, unbiased_unweighted
    )
    permuted_gate, permuted_gate_diag = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, permuted
    )
    retained = row_normalize(standardize(carrier["retained"]))
    run_ids, arm_ids, config_ids = [], [], []
    partitions, representation_hashes, diagnostics = [], [], []

    def append(run_id: str, arm: str, config_id: str, representation: np.ndarray, diag: dict) -> None:
        representation = np.asarray(representation, dtype=np.float32)
        partition = same_head_partition(representation, args.k, seed=0)
        if np.unique(partition).size != args.k:
            raise RuntimeError("same endpoint violated exact K")
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
                "representation_sha256": sha256_array(representation),
                "partition_sha256": sha256_array(partition),
                "min_cluster_size": int(np.bincount(partition, minlength=args.k).min()),
                "cluster_sizes": np.bincount(partition, minlength=args.k).astype(int).tolist(),
                **diag,
            }
        )

    append("BASELINE__FROZEN_RETAINED", "FROZEN_RETAINED_SAME_HEAD", "BASELINE", retained, {"actual_optimizer_steps": 0})
    append("BASELINE__FULL_BANK_SMOOTH", "FULL_BANK_SMOOTH_REFERENCE", "BASELINE", full_smooth, {"actual_optimizer_steps": 0})
    append("BASELINE__UNBIASED_SMOOTH", "UNBIASED_SMOOTH_REFERENCE", "BASELINE", unbiased_smooth, {"actual_optimizer_steps": 0})
    append("BASELINE__ZERO_RESIDUAL", "ZERO_RESIDUAL_CONTROL", "BASELINE", unbiased_smooth, {"actual_optimizer_steps": 0, "step0_exact_smooth": True})

    arm_specs = (
        ("UNBIASED_FULL", unbiased_weighted, unbiased_smooth, primary_gate, (1.0, 1.0), primary_gate_diag),
        ("UNWEIGHTED_RELATION", unbiased_unweighted, unweighted_smooth, unweighted_gate, (1.0, 1.0), unweighted_gate_diag),
        ("FULL_BANK_SENSITIVITY", full_weighted, full_smooth, full_gate, (1.0, 1.0), full_gate_diag),
        ("PERMUTED_RELATION", permuted, permuted_smooth, permuted_gate, (1.0, 1.0), permuted_gate_diag),
        ("ONE_RAW_VIEW1_ADAPTER", unbiased_weighted, unbiased_smooth, primary_gate, (1.0, 0.0), primary_gate_diag),
        ("ONE_RAW_VIEW2_ADAPTER", unbiased_weighted, unbiased_smooth, primary_gate, (0.0, 1.0), primary_gate_diag),
    )
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint_states = {}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        for config in CONFIGS:
            for arm, posterior, smooth, gate, view_mask, gate_diag in arm_specs:
                result = train_zero_start(
                    carrier["view1"],
                    carrier["view2"],
                    carrier["retained"],
                    smooth,
                    gate,
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
                append(
                    run_id,
                    arm,
                    config["config_id"],
                    result.representation,
                    {
                        **dict(result.diagnostics),
                        "node_gate": gate_diag,
                        "relation_probability_sha256": sha256_array(posterior.probability_same),
                        "relation_uncertainty_sha256": sha256_array(posterior.uncertainty),
                        "node_gate_sha256": sha256_array(gate),
                        "smooth_carrier_sha256": sha256_array(smooth),
                        "relation_gate_smooth_same_source": True,
                    },
                )
                if arm == "UNBIASED_FULL":
                    checkpoint_states[config["config_id"]] = result.state_dict
                    replayed = reload_zero_start(
                        result.state_dict,
                        carrier["view1"],
                        carrier["view2"],
                        carrier["retained"],
                        smooth,
                        gate,
                        config,
                        view_mask=view_mask,
                        device=device,
                    )
                    if not np.array_equal(replayed, result.representation):
                        raise RuntimeError("in-process strict checkpoint replay mismatch")
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "schema": "night17c-zero-start-checkpoint-v1",
            "lane": args.lane,
            "training_seed": args.training_seed,
            "configs": list(CONFIGS),
            "unbiased_full_state_dict": checkpoint_states,
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
        primary_relation_probability=unbiased_weighted.probability_same,
        primary_relation_uncertainty=unbiased_weighted.uncertainty,
        primary_node_gate=primary_gate,
        full_smooth=full_smooth,
        unbiased_smooth=unbiased_smooth,
        unweighted_smooth=unweighted_smooth,
        permuted_smooth=permuted_smooth,
    )
    peak_gpu = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    manifest = {
        "schema": "night17c-zero-start-producer-v1",
        "lane": args.lane,
        "k": args.k,
        "n": int(carrier["ids"].size),
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "pair_count": int(pair_i.size),
        "spatial_graph_nnz": int(graph.nnz),
        "unbiased_candidate_count": unbiased_weighted.selected_candidate_count,
        "full_candidate_count": full_weighted.selected_candidate_count,
        "primary_gate_diagnostics": primary_gate_diag,
        "producer_label_reads": 0,
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
        "training_seed": args.training_seed,
        "thread_limits": {"OMP": 1, "MKL": 1, "OPENBLAS": 1, "threadpoolctl": 1},
    }
    (output / "producer.producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "pair_count", "unbiased_candidate_count", "wall_seconds", "peak_gpu_mb")}, indent=2))


if __name__ == "__main__":
    main()
