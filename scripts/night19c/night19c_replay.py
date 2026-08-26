#!/usr/bin/env python3
"""Fresh-process exact replay for Night-19C placenta Z01 transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night17b_sfrd import canonical_pair_bank, deterministic_relation_smooth, same_head_partition, sha256_array
from SpaLORA.night17c_zero_start import node_trust_gate, reload_zero_start, stratified_permute_relation
from SpaLORA.night19c_zero_start_transfer import Z01_CONSERVATIVE, equal_weight_relation_posterior, file_sha256
from scripts.night19c.night19c_producer import csr_from_carrier, load_numeric_carrier


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(args.artifact) != producer["artifact_sha256"]:
        raise ValueError("locked artifact SHA mismatch")
    if file_sha256(args.checkpoint) != producer["checkpoint_sha256"]:
        raise ValueError("locked checkpoint SHA mismatch")
    carrier, _, _ = load_numeric_carrier(Path(args.carrier))
    if file_sha256(args.carrier) != producer["carrier_sha256"]:
        raise ValueError("carrier SHA mismatch")
    with np.load(args.bank, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in ("ids", "candidate_ids", "partitions", "candidate_weights")}
    if file_sha256(args.bank) != producer["bank_sha256"]:
        raise ValueError("bank SHA mismatch")
    with np.load(args.artifact, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(carrier["ids"], bank["ids"]) or not np.array_equal(carrier["ids"], artifact["ids"]):
        raise ValueError("ordered IDs differ in fresh process")
    graph = csr_from_carrier(carrier)
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    posterior = equal_weight_relation_posterior(bank["partitions"], pair_i, pair_j)
    permuted = stratified_permute_relation(posterior, is_spatial)
    smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, posterior, alpha=0.2)
    permuted_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, permuted, alpha=0.2)
    gate, _ = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    permuted_gate, _ = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, permuted)
    for name, value in (("pair_i", pair_i), ("pair_j", pair_j), ("primary_smooth", smooth),
                        ("permuted_smooth", permuted_smooth), ("primary_node_gate", gate),
                        ("permuted_node_gate", permuted_gate)):
        if not np.array_equal(value, artifact[name]):
            raise RuntimeError(f"fresh-process authority mismatch: {name}")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = safe_torch_load(Path(args.checkpoint), device)
    if checkpoint["config"] != Z01_CONSERVATIVE or checkpoint["config"] != producer["z01_config"]:
        raise ValueError("checkpoint Z01 configuration differs from frozen authority")
    rows = []
    with threadpool_limits(limits=1):
        for run_id, state, state_smooth, state_gate in (
            ("PERMUTED_RELATION", checkpoint["permuted_state_dict"], permuted_smooth, permuted_gate),
            ("Z01_FULL", checkpoint["full_state_dict"], smooth, gate),
        ):
            representation = reload_zero_start(
                state, carrier["view1"], carrier["view2"], carrier["retained"],
                state_smooth, state_gate, checkpoint["config"], device=device,
            )
            index = int(np.flatnonzero(artifact["run_ids"] == run_id)[0])
            partition = same_head_partition(representation, 10, seed=0)
            rep_exact = sha256_array(representation) == str(artifact["representation_sha256"][index])
            partition_exact = np.array_equal(partition, artifact["partitions"][index])
            if not rep_exact or not partition_exact:
                raise RuntimeError(f"fresh-process replay mismatch: {run_id}")
            rows.append({
                "run_id": run_id,
                "representation_sha256": sha256_array(representation),
                "partition_sha256": sha256_array(partition),
                "representation_exact": rep_exact,
                "partition_exact": partition_exact,
            })
    result = {
        "schema": "night19c-placenta-fresh-replay-v1", "lane": "PLACENTA_K10",
        "replayed_trainable_arms": len(rows),
        "all_representation_exact": all(row["representation_exact"] for row in rows),
        "all_partition_exact": all(row["partition_exact"] for row in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
