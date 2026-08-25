#!/usr/bin/env python3
"""Fresh-process exact replay for a locked Night-17A CEUP artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from SpaLORA.night17a_ceup import sha256_array
from scripts.night17a.night17a_producer import run_pipeline


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result, manifest = run_pipeline(
        Path(args.carrier), Path(args.config), Path(args.checkpoint), args.k, args.lane, "replay", args.device
    )
    authority_path = Path(args.authority)
    with np.load(authority_path, allow_pickle=False) as authority:
        checks = {
            "ids": np.array_equal(result["ids"], authority["ids"]),
            "edge_i": np.array_equal(result["edge_i"], authority["edge_i"]),
            "edge_j": np.array_equal(result["edge_j"], authority["edge_j"]),
            "edge_w": np.array_equal(result["edge_w"], authority["edge_w"]),
            "utility_raw_four": np.array_equal(result["utility_raw_four"], authority["utility_raw_four"]),
            "utility_q_four": np.array_equal(result["utility_q_four"], authority["utility_q_four"]),
            "arm_partitions": np.array_equal(result["arm_partitions"], authority["arm_partitions"]),
        }
        authority_hashes = {
            "utility": sha256_array(authority["utility_raw_four"]),
            "partitions": sha256_array(authority["arm_partitions"]),
        }
    if not all(checks.values()):
        raise RuntimeError(f"fresh-process replay mismatch: {checks}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "night17a-ceup-p0-replay-v1",
        "lane": args.lane,
        "status": "PASS",
        "checks": checks,
        "authority_path": str(authority_path.resolve()),
        "authority_sha256": file_sha256(authority_path),
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "replay_utility_sha256": sha256_array(result["utility_raw_four"]),
        "replay_partitions_sha256": sha256_array(result["arm_partitions"]),
        "authority_hashes": authority_hashes,
        "labels_accessed": 0,
        "annotation_keys_accessed": 0,
        "replay_wall_seconds": manifest["wall_seconds"],
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lane": args.lane, "status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

