#!/usr/bin/env python3
"""Build an 89-candidate Night-16H-budget carrier from the frozen learned representation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from SpaLORA.night17b_sfrd import (
    canonical_pair_bank,
    csr_from_carrier,
    deterministic_relation_smooth,
    relation_posterior,
    sha256_array,
)
from SpaLORA.night17c_zero_start import (
    node_trust_gate,
    reload_zero_start,
    stratified_permute_relation,
    train_zero_start,
)
from scripts.night17c.night17c_replay import safe_torch_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-bank")
    parser.add_argument("--feasibility")
    parser.add_argument("--config-id", default="Z01_CONSERVATIVE")
    parser.add_argument("--representation-arm", choices=("UNBIASED_FULL", "ZERO_RESIDUAL_CONTROL", "PERMUTED_RELATION"), default="UNBIASED_FULL")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.artifact, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(carrier["ids"], artifact["ids"]):
        raise ValueError("carrier/artifact ordered IDs differ")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = safe_torch_load(Path(args.checkpoint), device)
    configs = {str(value["config_id"]): value for value in checkpoint["configs"]}
    if args.representation_arm == "UNBIASED_FULL":
        state = checkpoint["unbiased_full_state_dict"][args.config_id]
        learned = reload_zero_start(
            state,
            carrier["view1"], carrier["view2"], carrier["retained"],
            artifact["unbiased_smooth"], artifact["primary_node_gate"],
            configs[args.config_id], device=device,
        )
    elif args.representation_arm == "ZERO_RESIDUAL_CONTROL":
        learned = np.asarray(artifact["unbiased_smooth"], dtype=np.float32)
    else:
        if not args.candidate_bank or not args.feasibility:
            raise ValueError("permuted carrier requires candidate bank and feasibility evidence")
        with np.load(args.candidate_bank, allow_pickle=False) as archive:
            bank = {key: np.asarray(archive[key]) for key in archive.files}
        records = list(csv.DictReader(Path(args.feasibility).open(encoding="utf-8")))
        graph = csr_from_carrier(carrier, "graph0")
        pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
        primary = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", True)
        permuted = stratified_permute_relation(primary, is_spatial)
        smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, permuted, alpha=0.2)
        gate, _ = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, permuted)
        learned = train_zero_start(
            carrier["view1"], carrier["view2"], carrier["retained"], smooth, gate,
            pair_i, pair_j, is_spatial, permuted, configs[args.config_id],
            int(checkpoint["training_seed"]), device=device,
        ).representation
    run_id = f"{args.config_id}__{args.representation_arm}__S{checkpoint['training_seed']}"
    if args.representation_arm == "ZERO_RESIDUAL_CONTROL":
        run_id = "BASELINE__ZERO_RESIDUAL"
    index = int(np.flatnonzero(artifact["run_ids"] == run_id)[0])
    if sha256_array(learned) != str(artifact["representation_sha256"][index]):
        raise RuntimeError("learned representation does not match locked producer")
    smooth_index = int(np.flatnonzero(artifact["run_ids"] == "BASELINE__ZERO_RESIDUAL")[0])
    starts = [np.asarray(artifact["partitions"][smooth_index], dtype=np.int32)]
    start_ids = ["ZERO_RESIDUAL_SMOOTH_START"]
    with threadpool_limits(limits=1):
        for seed in range(5):
            starts.append(KMeans(n_clusters=args.k, n_init=20, random_state=seed).fit_predict(learned).astype(np.int32))
            start_ids.append(f"LEARNED_KMEANS_S{seed}")
    for partition in starts:
        if np.unique(partition).size != args.k:
            raise RuntimeError("integration start violated exact K")
    payload = dict(carrier)
    payload["retained"] = learned.astype(np.float32)
    payload["start_ids"] = np.asarray(start_ids, dtype="U")
    payload["start_partitions"] = np.stack(starts).astype(np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["retained"], learned) or not np.array_equal(replay["start_partitions"], payload["start_partitions"]):
            raise RuntimeError("integration carrier reload mismatch")
    manifest = {
        "schema": "night17c-night16h-budget-integration-carrier-v1",
        "config_id": args.config_id,
        "representation_arm": args.representation_arm,
        "training_seed": int(checkpoint["training_seed"]),
        "representation_sha256": sha256_array(learned),
        "start_ids": start_ids,
        "start_partition_sha256": [sha256_array(value) for value in starts],
        "expected_night16f_candidates": 47,
        "expected_ordered_path_candidates": 42,
        "expected_total_budget": 89,
        "artifact_reload": "PASS",
        "producer_label_reads": 0,
    }
    output.with_suffix(".carrier.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
