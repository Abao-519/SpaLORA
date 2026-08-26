#!/usr/bin/env python3
"""Independent fresh-process replay of Night-19A Stage-A artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night17b_sfrd import same_head_partition, sha256_array
from SpaLORA.night19a_gradient_d0 import build_model, prepare_tensors, zero_start_representation
from SpaLORA.night19a_sparse_arbitration import ALL_ARMS, TRAINED_ARMS
from scripts.night19a.night19a_d0_producer import CONFIG, file_sha256
from scripts.night19a.night19a_stage_a_producer import load_lane_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--relation-source", choices=("NIGHT16H_UNBIASED_WEIGHTED", "UNIFORM_FEASIBLE_STRESS"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(Path(args.artifact)) != producer["artifact_sha256"] or file_sha256(Path(args.checkpoint)) != producer["checkpoint_sha256"]:
        raise ValueError("locked Stage-A artifact/checkpoint SHA mismatch")
    carrier, _, _, pair_i, pair_j, is_spatial, posterior, strata, _, anchor, gate, _, _ = load_lane_inputs(args)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    tensors = prepare_tensors(carrier["view1"], carrier["view2"], carrier["retained"], anchor, gate, pair_i, pair_j, is_spatial, posterior, strata, device)
    loaded = torch.load(args.checkpoint, map_location=device)
    if loaded["authority"]["execution_contract_sha256"] != producer["execution_contract_sha256"]:
        raise ValueError("checkpoint/producer execution contract mismatch")
    representations = [np.asarray(anchor, dtype=np.float32)]
    for arm in TRAINED_ARMS:
        model = build_model(carrier["view1"].shape[1], carrier["view2"].shape[1], carrier["retained"].shape[1], CONFIG, args.training_seed, device)
        model.load_state_dict(loaded["states"][arm], strict=True)
        representations.append(zero_start_representation(model, tensors))
    representations = np.stack(representations, axis=0)
    partitions = np.stack([same_head_partition(value, args.k, seed=0) for value in representations], axis=0).astype(np.int32)
    with np.load(args.artifact, allow_pickle=False) as archive:
        ids_exact = np.array_equal(archive["ids"], carrier["ids"])
        profiles_exact = np.array_equal(archive["profile_ids"].astype(str), np.asarray(ALL_ARMS))
        representations_exact = np.array_equal(archive["representations"], representations)
        partitions_exact = np.array_equal(archive["partitions"], partitions)
    result = {
        "schema": "night19a-stage-a-fresh-replay-v1",
        "lane": args.lane,
        "training_seed": args.training_seed,
        "ids_exact": ids_exact,
        "profiles_exact": profiles_exact,
        "representations_exact": representations_exact,
        "partitions_exact": partitions_exact,
        "representation_sha256": [sha256_array(value) for value in representations],
        "partition_sha256": [sha256_array(value) for value in partitions],
        "label_reads": 0,
    }
    if not all((ids_exact, profiles_exact, representations_exact, partitions_exact)):
        raise RuntimeError("Stage-A fresh-process replay mismatch")
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
