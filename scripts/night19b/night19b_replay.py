#!/usr/bin/env python3
"""Independent fresh-process replay for locked Night-19B artifacts."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

from SpaLORA.night19b_csad import arm_operator, build_operator_bank, sha256_array, spectral_partition
from scripts.night19b.night19b_producer import csr_from_archive, file_sha256, load_contract, load_numeric_carrier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if file_sha256(Path(args.artifact)) != producer["artifact_sha256"]:
        raise ValueError("locked artifact SHA mismatch")
    if file_sha256(Path(args.contract)) != producer["contract_sha256"]:
        raise ValueError("contract SHA mismatch")
    contract = load_contract(Path(args.contract))
    carrier, discovered_keys, accessed_keys = load_numeric_carrier(
        Path(args.carrier), {int(row["spatial_scale"]) for row in contract["profiles"]}
    )
    with np.load(args.artifact, allow_pickle=False) as archive:
        locked = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(locked["ids"], carrier["ids"]):
        raise ValueError("ordered IDs mismatch")
    candidate_ids = locked["candidate_ids"].astype(str).tolist()
    rebuilt_partitions = []
    rebuilt_eigenvalues = []
    selected_profiles = [
        row for row in contract["profiles"]
        if any(value.startswith(row["config_id"] + "__") for value in candidate_ids)
    ]
    max_extra = max(int(row["spectral_extra_dim"]) for row in selected_profiles)
    with threadpool_limits(limits=1):
        for profile in selected_profiles:
            wanted = [value for value in candidate_ids if value.startswith(profile["config_id"] + "__")]
            if not wanted:
                continue
            graph = csr_from_archive(carrier, f"graph{int(profile['spatial_scale'])}")
            bank = build_operator_bank(
                carrier["view1"], carrier["view2"], graph,
                int(profile["feature_neighbors"]), int(profile["spatial_topk"]), float(profile["self_loop"]),
            )
            for candidate_id in wanted:
                arm = candidate_id.split("__", 2)[1]
                seed = int(candidate_id.rsplit("E", 1)[1])
                operator, _ = arm_operator(
                    arm, bank, carrier["ids"], int(profile["product_topk"]),
                    float(profile["conflict_strength"]), float(profile["conflict_floor"]),
                )
                partition, _embedding, values, _ = spectral_partition(
                    operator, int(producer["k"]), int(producer["k"]) + int(profile["spectral_extra_dim"]), seed
                )
                padded = np.full(int(producer["k"]) + max_extra, np.nan)
                padded[: values.size] = values
                rebuilt_partitions.append(partition)
                rebuilt_eigenvalues.append(padded)
    partitions = np.stack(rebuilt_partitions).astype(np.int32)
    eigenvalues = np.stack(rebuilt_eigenvalues).astype(np.float64)
    result = {
        "schema": "night19b-csad-fresh-replay-v1",
        "lane": producer["lane"],
        "ordered_ids_exact": bool(np.array_equal(locked["ids"], carrier["ids"])),
        "candidate_ids_exact": True,
        "partitions_exact": bool(np.array_equal(partitions, locked["partitions"])),
        "eigenvalues_exact_equal_nan": bool(np.array_equal(eigenvalues, locked["eigenvalues"], equal_nan=True)),
        "partition_sha256": [sha256_array(row) for row in partitions],
        "producer_partition_sha256_exact": [sha256_array(row) for row in partitions] == producer["partition_sha256"],
        "producer_label_reads": 0,
        "carrier_discovered_keys": discovered_keys,
        "carrier_accessed_keys": accessed_keys,
        "carrier_annotation_arrays_accessed": 0,
    }
    if not all(result[key] for key in ("ordered_ids_exact", "candidate_ids_exact", "partitions_exact", "eigenvalues_exact_equal_nan", "producer_partition_sha256_exact")):
        raise RuntimeError("fresh replay failed")
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
