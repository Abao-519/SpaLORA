#!/usr/bin/env python3
"""Label-free producer for Night-19A Stage-A matched arbitration arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night17b_sfrd import (
    canonical_pair_bank,
    csr_from_carrier,
    deterministic_relation_smooth,
    encode_partition,
    relation_posterior,
    same_head_partition,
    sha256_array,
)
from SpaLORA.night17c_zero_start import node_trust_gate
from SpaLORA.night19a_gradient_d0 import build_model, evidence_support_strata, prepare_tensors, state_sha256, zero_start_representation
from SpaLORA.night19a_sparse_arbitration import (
    ALL_ARMS,
    TRAINED_ARMS,
    make_topology_disabled_tensors,
    projection_groups_and_strengths,
    train_stage_a_arm,
)
from scripts.night19a.night19a_d0_producer import (
    CONFIG,
    TASKBOOK_SHA256,
    file_sha256,
    graph_sha256,
    uniform_feasible_posterior,
)


def source_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def load_lane_inputs(args):
    carrier_path = Path(args.carrier)
    bank_path = Path(args.candidate_bank)
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"ids", "view1", "view2", "retained", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"}
    if not required.issubset(carrier) or not {"ids", "partitions"}.issubset(bank):
        raise ValueError("carrier/candidate fields missing")
    if not np.array_equal(carrier["ids"], bank["ids"]):
        raise ValueError("carrier/candidate ordered IDs differ")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    if args.relation_source == "NIGHT16H_UNBIASED_WEIGHTED":
        if args.feasibility is None:
            raise ValueError("feasibility required")
        feasibility_path = Path(args.feasibility)
        records = list(csv.DictReader(feasibility_path.open(encoding="utf-8")))
        if "candidate_ids" not in bank:
            raise ValueError("candidate IDs missing")
        if [row["candidate_id"] for row in records] != bank["candidate_ids"].astype(str).tolist():
            raise ValueError("candidate evidence order differs")
        posterior = relation_posterior(bank["partitions"], records, pair_i, pair_j, bank_mode="UNBIASED_BANK", weighted=True)
        feasibility_sha = file_sha256(feasibility_path)
    else:
        posterior, _ = uniform_feasible_posterior(bank["partitions"], graph, pair_i, pair_j, args.k)
        feasibility_sha = None
    strata, strata_diagnostics = evidence_support_strata(posterior, is_spatial)
    anchor = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, posterior, alpha=float(CONFIG["relation_smooth_alpha"]))
    gate, gate_diagnostics = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    return carrier, bank, graph, pair_i, pair_j, is_spatial, posterior, strata, strata_diagnostics, anchor, gate, gate_diagnostics, feasibility_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--relation-source", choices=("NIGHT16H_UNBIASED_WEIGHTED", "UNIFORM_FEASIBLE_STRESS"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--d0-gate", required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    if args.training_seed != 0:
        raise ValueError("Stage-A freeze permits seed0 only before gate")
    contract_path = Path(args.execution_contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "night19a-sparse-evidence-gradient-arbitration-stage-a-freeze-rev1":
        raise ValueError("Stage-A contract mismatch")
    if contract.get("model") != CONFIG:
        raise ValueError("Stage-A contract model differs from identified D0 configuration")
    d0_gate_path = Path(args.d0_gate)
    d0_gate = json.loads(d0_gate_path.read_text(encoding="utf-8"))
    if d0_gate.get("schema") != "night19a-gradient-d0-gate-rev1" or not d0_gate.get("stage_a_authorized"):
        raise ValueError("D0 REV1 gate did not authorize Stage A")
    values = load_lane_inputs(args)
    carrier, bank, graph, pair_i, pair_j, is_spatial, posterior, strata, strata_diag, anchor, gate, gate_diag, feasibility_sha = values
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    tensors = prepare_tensors(carrier["view1"], carrier["view2"], carrier["retained"], anchor, gate, pair_i, pair_j, is_spatial, posterior, strata, device)
    topology_disabled = make_topology_disabled_tensors(tensors, is_spatial)
    relation_weight = tensors.relation_weight.detach().cpu().numpy()
    groups, strengths, full_group_diag = projection_groups_and_strengths(
        carrier["ids"], pair_i, pair_j, is_spatial, relation_weight, strata, permuted=False
    )
    permuted_groups, permuted_strengths, perm_group_diag = projection_groups_and_strengths(
        carrier["ids"], pair_i, pair_j, is_spatial, relation_weight, strata, permuted=True
    )
    if perm_group_diag["maximum_absolute_mass_error"] > 1e-8:
        raise RuntimeError("permuted evidence mass gate failed")
    representations = [np.asarray(anchor, dtype=np.float32)]
    states = {}
    arm_diagnostics = {"STRONG_START_NO_TRAIN": {"parameter_l2_change": 0.0}}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        for arm in TRAINED_ARMS:
            result = train_stage_a_arm(
                arm, tensors, topology_disabled, CONFIG, args.training_seed,
                groups, strengths, permuted_groups, permuted_strengths, device,
            )
            representations.append(result.pop("final_representation"))
            states[arm] = result.pop("state_dict")
            arm_diagnostics[arm] = result
    representations = np.stack(representations, axis=0).astype(np.float32)
    partitions = np.stack([same_head_partition(value, args.k, seed=0) for value in representations], axis=0).astype(np.int32)
    if any(np.unique(partition).size != args.k for partition in partitions):
        raise RuntimeError("Stage-A endpoint violates exact K")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "artifact.npz"
    np.savez_compressed(artifact_path, ids=carrier["ids"], profile_ids=np.asarray(ALL_ARMS), representations=representations, partitions=partitions)
    authority = {
        "lane": args.lane,
        "training_seed": int(args.training_seed),
        "config": CONFIG,
        "taskbook_sha256": TASKBOOK_SHA256,
        "execution_contract_sha256": file_sha256(contract_path),
        "d0_gate_sha256": file_sha256(d0_gate_path),
        "carrier_sha256": file_sha256(Path(args.carrier)),
        "candidate_bank_sha256": file_sha256(Path(args.candidate_bank)),
        "feasibility_sha256": feasibility_sha,
        "ordered_ids_sha256": sha256_array(carrier["ids"]),
        "anchor_representation_sha256": sha256_array(anchor),
        "groups_sha256": sha256_array(groups),
        "permuted_groups_sha256": sha256_array(permuted_groups),
    }
    checkpoint_path = output / "checkpoints.pt"
    torch.save({"states": states, "authority": authority}, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device)
    if loaded["authority"] != authority or set(loaded["states"]) != set(TRAINED_ARMS):
        raise RuntimeError("checkpoint strict reload authority mismatch")
    for arm in TRAINED_ARMS:
        if state_sha256(loaded["states"][arm]) != state_sha256(states[arm]):
            raise RuntimeError("checkpoint strict reload state mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["representations"], representations) or not np.array_equal(archive["partitions"], partitions):
            raise RuntimeError("artifact strict reload mismatch")
    cluster_sizes = [np.bincount(encode_partition(partition), minlength=args.k).astype(int).tolist() for partition in partitions]
    manifest = {
        "schema": "night19a-stage-a-producer-v1",
        "lane": args.lane,
        "n": int(carrier["ids"].size),
        "k": int(args.k),
        "training_seed": int(args.training_seed),
        "profile_ids": list(ALL_ARMS),
        "view1_shape": list(carrier["view1"].shape),
        "view2_shape": list(carrier["view2"].shape),
        "retained_shape": list(carrier["retained"].shape),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "graph_sha256": graph_sha256(graph),
        "pair_count": int(pair_i.size),
        "spatial_pair_count": int(np.sum(is_spatial)),
        "feature_pair_count": int(np.sum(~is_spatial)),
        **authority,
        "artifact_sha256": file_sha256(artifact_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "representation_sha256": [sha256_array(value) for value in representations],
        "partition_sha256": [sha256_array(value) for value in partitions],
        "cluster_sizes": cluster_sizes,
        "min_cluster_size": [min(value) for value in cluster_sizes],
        "exact_k_all": True,
        "evidence_strata": strata_diag,
        "node_trust_gate": gate_diag,
        "full_projection_groups": full_group_diag,
        "permuted_projection_groups": perm_group_diag,
        "arm_diagnostics": arm_diagnostics,
        "strict_checkpoint_reload": "PASS",
        "artifact_reload": "PASS",
        "producer_label_reads": 0,
        "annotation_columns_accessed": [],
        "zero_start_anchor_semantics": "REGISTERED_RELATION_SMOOTHED_CARRIER_NOT_RAW_RETAINED",
        "core_source_sha256": file_sha256(Path(__file__).resolve().parents[2] / "SpaLORA" / "night19a_sparse_arbitration.py"),
        "producer_source_sha256": source_sha256(),
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1024.0 ** 2) if device.startswith("cuda") else 0.0,
    }
    (output / "producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lane": args.lane, "profiles": len(ALL_ARMS), "wall_seconds": manifest["wall_seconds"], "artifact_sha256": manifest["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
