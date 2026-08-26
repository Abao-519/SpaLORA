#!/usr/bin/env python3
"""Fresh-process checkpoint/representation/partition replay for Night-19A D0."""

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
from SpaLORA.night19a_gradient_d0 import build_model, state_sha256
from scripts.night19a.night19a_d0_producer import uniform_feasible_posterior


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    carrier_path = Path(args.carrier)
    bank_path = Path(args.candidate_bank)
    artifact_path = Path(args.artifact)
    checkpoint_path = Path(args.checkpoint)
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    for path, key in (
        (carrier_path, "carrier_sha256"),
        (bank_path, "candidate_bank_sha256"),
        (artifact_path, "artifact_sha256"),
        (checkpoint_path, "checkpoint_sha256"),
    ):
        if file_sha256(path) != producer[key]:
            raise ValueError("replay authority hash mismatch: " + key)
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(carrier["ids"], bank["ids"]) or not np.array_equal(carrier["ids"], artifact["ids"]):
        raise ValueError("fresh replay ordered IDs differ")
    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    if producer["relation_source"] == "NIGHT16H_UNBIASED_WEIGHTED":
        feasibility_path = Path(args.feasibility)
        if file_sha256(feasibility_path) != producer["feasibility_sha256"]:
            raise ValueError("fresh replay feasibility hash mismatch")
        records = list(csv.DictReader(feasibility_path.open(encoding="utf-8")))
        posterior = relation_posterior(
            bank["partitions"], records, pair_i, pair_j, bank_mode="UNBIASED_BANK", weighted=True
        )
    else:
        posterior, _ = uniform_feasible_posterior(
            bank["partitions"], graph, pair_i, pair_j, int(producer["k"])
        )
    config = producer["config"]
    anchor = deterministic_relation_smooth(
        carrier["retained"], pair_i, pair_j, posterior, alpha=float(config["relation_smooth_alpha"])
    )
    gate, _ = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["authority"]["taskbook_sha256"] != producer["taskbook_sha256"]:
        raise ValueError("checkpoint taskbook authority mismatch")
    model = build_model(
        carrier["view1"].shape[1], carrier["view2"].shape[1], carrier["retained"].shape[1],
        config, int(producer["training_seed"]), device,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if state_sha256(checkpoint["state_dict"]) != producer["final_state_sha256"]:
        raise ValueError("checkpoint state SHA mismatch")
    with threadpool_limits(limits=1):
        representation = reload_zero_start(
            checkpoint["state_dict"], carrier["view1"], carrier["view2"], carrier["retained"],
            anchor, gate, config, device=device,
        )
        partition = same_head_partition(representation, int(producer["k"]), seed=0)
    representation_exact = bool(np.array_equal(representation, artifact["representation"]))
    partition_exact = bool(np.array_equal(partition, artifact["partition"]))
    if not representation_exact or not partition_exact:
        raise RuntimeError("fresh process output differs from locked artifact")
    audit = {
        "schema": "night19a-gradient-d0-fresh-replay-v1",
        "lane": producer["lane"],
        "training_seed": int(producer["training_seed"]),
        "representation_exact": representation_exact,
        "partition_exact": partition_exact,
        "representation_sha256": sha256_array(representation),
        "partition_sha256": sha256_array(partition),
        "checkpoint_strict_load": "PASS",
        "ordered_ids_exact": True,
        "label_reads": 0,
    }
    Path(args.output).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

