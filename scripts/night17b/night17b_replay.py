#!/usr/bin/env python3
"""Fresh-process strict replay for SFRD full-weighted representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night17b_sfrd import reload_representation, same_head_partition, sha256_array


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(Path(args.artifact)) != producer["artifact_sha256"]:
        raise ValueError("artifact hash mismatch")
    if file_sha256(Path(args.checkpoint)) != producer["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    with np.load(args.artifact, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    checkpoint = load_checkpoint(Path(args.checkpoint))
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    results = []
    for config in checkpoint["configs"]:
        config_id = config["config_id"]
        run_id = f"{config_id}__FULL_WEIGHTED__S{checkpoint['training_seed']}"
        matches = np.flatnonzero(artifact["run_ids"] == run_id)
        if matches.size != 1:
            raise ValueError(f"missing locked run {run_id}")
        index = int(matches[0])
        representation = reload_representation(
            checkpoint["full_weighted_state_dict"][config_id],
            carrier["view1"],
            carrier["view2"],
            carrier["retained"],
            config,
            device=device,
        )
        partition = same_head_partition(representation, producer["k"], seed=0)
        representation_match = sha256_array(representation) == str(artifact["representation_sha256"][index])
        partition_match = np.array_equal(partition, artifact["partitions"][index])
        if not representation_match or not partition_match:
            raise RuntimeError(f"fresh replay mismatch for {run_id}")
        results.append(
            {
                "run_id": run_id,
                "representation_sha256": sha256_array(representation),
                "partition_sha256": sha256_array(partition),
                "representation_exact": representation_match,
                "partition_exact": partition_match,
            }
        )
    output = {
        "schema": "night17b-sfrd-replay-v1",
        "lane": producer["lane"],
        "fresh_process": True,
        "strict_checkpoint_load": True,
        "device": device,
        "full_weighted_runs": results,
        "status": "PASS",
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
