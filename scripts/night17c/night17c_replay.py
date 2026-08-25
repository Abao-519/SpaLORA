#!/usr/bin/env python3
"""Fresh-process exact replay for locked Night-17C primary representations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    same_head_partition,
    sha256_array,
)
from SpaLORA.night17c_zero_start import node_trust_gate, reload_zero_start


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    artifact_path = Path(args.artifact)
    checkpoint_path = Path(args.checkpoint)
    if file_sha256(artifact_path) != producer["artifact_sha256"]:
        raise ValueError("artifact hash differs from locked producer manifest")
    if file_sha256(checkpoint_path) != producer["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from locked producer manifest")
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.candidate_bank, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in archive.files}
    records = list(csv.DictReader(Path(args.feasibility).open(encoding="utf-8")))
    if not np.array_equal(artifact["ids"], carrier["ids"]) or not np.array_equal(bank["ids"], carrier["ids"]):
        raise ValueError("ordered IDs differ during replay")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    posterior = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", True)
    smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, posterior, alpha=0.2)
    gate, _ = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    if not np.array_equal(pair_i, artifact["pair_i"]) or not np.array_equal(pair_j, artifact["pair_j"]):
        raise RuntimeError("pair bank is not deterministic")
    if not np.array_equal(smooth, artifact["unbiased_smooth"]):
        raise RuntimeError("smooth carrier is not exact on fresh process")
    if not np.array_equal(gate, artifact["primary_node_gate"]):
        raise RuntimeError("node gate is not exact on fresh process")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = safe_torch_load(checkpoint_path, device)
    config_map = {str(config["config_id"]): config for config in checkpoint["configs"]}
    rows = []
    with threadpool_limits(limits=1):
        for config_id, state in checkpoint["unbiased_full_state_dict"].items():
            representation = reload_zero_start(
                state, carrier["view1"], carrier["view2"], carrier["retained"], smooth, gate,
                config_map[str(config_id)], device=device,
            )
            run_id = f"{config_id}__UNBIASED_FULL__S{checkpoint['training_seed']}"
            indices = np.flatnonzero(artifact["run_ids"] == run_id)
            if indices.size != 1:
                raise RuntimeError(f"locked primary run is not unique: {run_id}")
            index = int(indices[0])
            partition = same_head_partition(representation, args.k, seed=0)
            representation_exact = sha256_array(representation) == str(artifact["representation_sha256"][index])
            partition_exact = np.array_equal(partition, artifact["partitions"][index])
            if not representation_exact or not partition_exact:
                raise RuntimeError(f"fresh-process replay mismatch: {run_id}")
            rows.append({
                "run_id": run_id,
                "representation_sha256": sha256_array(representation),
                "partition_sha256": sha256_array(partition),
                "representation_exact": representation_exact,
                "partition_exact": partition_exact,
            })
    result = {
        "schema": "night17c-zero-start-fresh-replay-v1",
        "lane": producer["lane"],
        "replayed_primary_configs": len(rows),
        "all_representation_exact": all(row["representation_exact"] for row in rows),
        "all_partition_exact": all(row["partition_exact"] for row in rows),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
